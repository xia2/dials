from __future__ import annotations

import collections
import itertools
import math
import types

import wx
from wx.lib.intctrl import IntCtrl

from cctbx import uctbx
from cctbx.miller import index_generator
from dxtbx.imageset import ImageSet
from dxtbx.model.detector_helpers import get_detector_projection_2d_axes
from dxtbx.model.experiment_list import ExperimentList, ExperimentListFactory
from libtbx.utils import flat_list
from scitbx import matrix
from wxtbx import bitmaps, icons
from wxtbx.phil_controls import EVT_PHIL_CONTROL
from wxtbx.phil_controls.floatctrl import FloatCtrl
from wxtbx.phil_controls.intctrl import IntCtrl as PhilIntCtrl
from wxtbx.phil_controls.ints import IntsCtrl
from wxtbx.phil_controls.strctrl import StrCtrl

from dials.algorithms.image.threshold import (
    DispersionExtendedThresholdDebug,
    DispersionThresholdDebug,
)
from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as find_spots_phil_scope
from dials.extensions import SpotFinderThreshold
from dials.util import masking
from dials.util.image_viewer.mask_frame import MaskSettingsFrame
from dials.util.image_viewer.spotfinder_wrap import chooser_wrapper

from .slip_viewer import pyslip
from .slip_viewer.frame import MASK_VAL, XrayFrame
from .viewer_tools import (
    EVT_ZEROMQ_EVENT,
    ImageChooserControl,
    ImageCollectionWithSelection,
    LegacyChooserAdapter,
)

try:
    from typing import Optional
except ImportError:
    pass

SpotfinderData = collections.namedtuple(
    "SpotfinderData",
    [
        "all_pix_data",
        "all_foreground_circles",
        "ctr_mass_data",
        "max_pix_data",
        "miller_indices_data",
        "predictions_data",
        "shoebox_data",
        "vector_data",
        "vector_text_data",
    ],
)

myEVT_LOADIMG = wx.NewEventType()
EVT_LOADIMG = wx.PyEventBinder(myEVT_LOADIMG, 1)


class LoadImageEvent(wx.PyCommandEvent):
    """Event to signal that an image should be loaded"""

    def __init__(self, etype, eid, filename=None):
        """Creates the event object"""
        wx.PyCommandEvent.__init__(self, etype, eid)
        self._filename = filename

    def get_filename(self):
        return self._filename


def create_load_image_event(destination, filename):
    wx.PostEvent(destination, LoadImageEvent(myEVT_LOADIMG, -1, filename))


class RadialProfileThresholdDebug:
    # The radial_profile threshold algorithm does not have an associated
    # 'Debug' class. It does not create the same set of intermediate images
    # as the dispersion algorithms, so we can delegate to a
    # DispersionThresholdDebug object for those, while overriding the final_mask
    # method. This wrapper class handles that.
    def __init__(self, imageset, n_iqr, blur, n_bins):
        self.imageset = imageset
        params = find_spots_phil_scope.extract()
        params.spotfinder.threshold.radial_profile.blur = blur
        params.spotfinder.threshold.radial_profile.n_bins = n_bins
        params.spotfinder.threshold.radial_profile.n_iqr = n_iqr
        self.radial_profile = SpotFinderThreshold.load("radial_profile")(params)
        self._i_panel = 0

    def __call__(self, *args):
        dispersion = DispersionThresholdDebug(*args)
        image = args[0]
        mask = args[1]
        dispersion._final_mask = self.radial_profile.compute_threshold(
            image, mask, imageset=self.imageset, i_panel=self._i_panel
        )
        dispersion.final_mask = types.MethodType(lambda x: x._final_mask, dispersion)
        self._i_panel += 1
        return dispersion


def calculate_isoresolution_lines(
    spacings,
    beam,
    detector,
    flex_image,
    add_text=True,
    n_rays=720,
    binning=1,
):
    # Calculate 2θ angles
    wavelength = beam.get_wavelength()
    twotheta = uctbx.d_star_sq_as_two_theta(uctbx.d_as_d_star_sq(spacings), wavelength)

    # Get beam vector and two orthogonal vectors
    beamvec = matrix.col(beam.get_s0())
    bor1 = beamvec.ortho()
    bor2 = beamvec.cross(bor1)

    ring_data = []
    resolution_text_data = []
    for tt, d in zip(twotheta, spacings):
        # Generate rays at 2θ
        cone_base_centre = beamvec * math.cos(tt)
        cone_base_radius = (beamvec * math.sin(tt)).length()
        rad1 = bor1.normalize() * cone_base_radius
        rad2 = bor2.normalize() * cone_base_radius
        ticks = (2 * math.pi / n_rays) * flex.double_range(n_rays)
        offset1 = flex.vec3_double(n_rays, rad1) * flex.cos(ticks)
        offset2 = flex.vec3_double(n_rays, rad2) * flex.sin(ticks)
        rays = flex.vec3_double(n_rays, cone_base_centre) + offset1 + offset2

        # Get the ray intersections. Need to set a dummy phi value
        rt = flex.reflection_table.empty_standard(n_rays)
        rt["s1"] = rays
        rt["phi"] = flex.double(n_rays, 0)
        from dials.algorithms.spot_prediction import ray_intersection

        intersect = ray_intersection(detector, rt)
        rt = rt.select(intersect)
        if len(rt) == 0:
            continue

        curr_panel_id = rt[0]["panel"]
        panel = detector[curr_panel_id]

        # Split the intersections into sets of vertices in separate paths
        paths = []
        vertices = []
        for ref in rt.rows():
            if ref["panel"] != curr_panel_id:
                # close off the current path and reset the vertices
                paths.append(vertices)
                vertices = []
                curr_panel_id = ref["panel"]
                panel = detector[curr_panel_id]
            x, y = panel.millimeter_to_pixel(ref["xyzcal.mm"][0:2])
            try:
                # Multi-panel case
                y, x = flex_image.tile_readout_to_picture(
                    curr_panel_id, y - 0.5, x - 0.5
                )
            except AttributeError:
                # Single panel FlexImage
                pass
            vertices.append((x / binning, y / binning))
        paths.append(vertices)

        # For each path, convert vertices to segments and add to the ring data
        segments = []
        for vertices in paths:
            for i in range(len(vertices) - 1):
                # Avoid long segments along the image edges
                dx = abs(vertices[i + 1][0] - vertices[i][0])
                dy = abs(vertices[i + 1][1] - vertices[i][1])
                if dx > 30 or dy > 30:
                    continue
                segments.append((vertices[i], vertices[i + 1]))
        ring_data.extend(segments)

        # Add labels to the iso-resolution lines
        if add_text:
            cb1 = beamvec.rotate_around_origin(axis=bor1, angle=tt)
            for angle in (45, 135, 225, 315):
                txtvec = cb1.rotate_around_origin(
                    axis=beamvec, angle=math.radians(angle)
                )
                try:
                    panel_id, txtpos = detector.get_ray_intersection(txtvec)
                except RuntimeError:
                    continue
                txtpos = detector[panel_id].millimeter_to_pixel(txtpos)
                try:
                    # Multi-panel case
                    x, y = flex_image.tile_readout_to_picture(
                        panel_id, txtpos[1], txtpos[0]
                    )[::-1]
                except AttributeError:
                    # Single panel FlexImage
                    x, y = txtpos
                resolution_text_data.append(
                    (
                        x / binning,
                        y / binning,
                        f"{d:.2f}",
                    )
                )

    return (ring_data, resolution_text_data)


class SpotFrame(XrayFrame):
    def __init__(self, *args, **kwds):
        self.experiments = kwds.pop("experiments")
        self.reflections = kwds.pop("reflections")

        self.imagesets = list(
            itertools.chain(*[x.imagesets() for x in self.experiments])
        )
        self.crystals = list(itertools.chain(*[x.crystals() for x in self.experiments]))
        if len(self.imagesets) == 0:
            raise RuntimeError("No imageset could be constructed")

        # Store the list of images we can view
        self.images = ImageCollectionWithSelection()

        super().__init__(*args, **kwds)

        # Precalculate best-fit frame for image display if required
        for experiment_list in self.experiments:
            for experiment in experiment_list:
                detector = experiment.detector
                if not detector:
                    self.params.projection = None
                    continue
                if detector.has_projection_2d():
                    self.params.projection = None
                    continue
                else:
                    detector.projection_2d_axes = get_detector_projection_2d_axes(
                        detector
                    )
                detector.projection = self.params.projection

        self.viewing_stills = True
        self.viewing_still_scans = True
        for experiment_list in self.experiments:
            if any(exp.scan or exp.goniometer for exp in experiment_list):
                self.viewing_stills = False
                break
        for experiment_list in self.experiments:
            if not all(
                exp.scan and (exp.scan.get_oscillation()[1] == 0.0)
                for exp in experiment_list
            ):
                self.viewing_still_scans = False
                break

        if self.viewing_stills:
            is_multi_shot_exp = any(len(exp_list) > 1 for exp_list in self.experiments)
            if is_multi_shot_exp:
                assert len(self.experiments) == 1
                if self.reflections:
                    assert len(self.reflections) == 1
                new_experiments = []
                new_reflections = []
                for i_expt, expt in enumerate(self.experiments[0]):
                    print(
                        "Preparing experiments (%d / %d)"
                        % (i_expt + 1, len(self.experiments[0]))
                    )
                    exp_list = ExperimentList()
                    exp_list.append(expt)
                    new_experiments.append(exp_list)
                    if self.reflections:
                        refls = self.reflections[0].select(
                            self.reflections[0]["id"] == i_expt
                        )
                        refls["id"] = flex.int(len(refls), 0)
                        new_reflections.append(refls)
                self.experiments = new_experiments
                self.reflections = new_reflections
            else:
                new_reflections = []
                for refls in self.reflections:
                    refls["id"] = flex.int(len(refls), 0)
                    new_reflections.append(refls)
                self.reflections = new_reflections

        # If we have only one imageset, unindexed filtering becomes easier
        self.have_one_imageset = len(set(self.imagesets)) <= 1
        if self.viewing_stills:
            self.have_one_imageset = True

        self.viewer.reflections = self.reflections
        self.viewer.frames = self.imagesets
        self.dials_spotfinder_layers = []
        self.shoebox_layer = None
        self.ctr_mass_layer = None
        self.max_pix_layer = None
        self.predictions_layer = None
        self.miller_indices_layer = None
        self.vector_layer = None
        self.vector_text_layer = None
        self._ring_layer = None
        self._resolution_text_layer = None
        self.sel_image_polygon_layer = None
        self.sel_image_circle_layers = []
        self.mask_input = self.params.mask
        self.mask_image_viewer = None
        self._mask_frame = None

        self.display_foreground_circles_patch = False  # hard code this option, for now

        self._dispersion_debug_list_hash = 0

        if (
            self.experiments is not None
            and not self.reflections
            and self.params.predict_reflections
        ):
            self.reflections = self.predict()

        if self.params.d_min is not None and len(self.reflections):
            reflections = []
            for expt, refl in zip(self.experiments, self.reflections):
                if "rlp" not in refl:
                    if "xyzobs.mm.value" not in refl:
                        if "xyzobs.px.value" not in refl:
                            refl["xyzobs.px.value"] = refl["xyzcal.px"]
                            refl["xyzobs.px.variance"] = flex.vec3_double(
                                len(refl), (1, 1, 1)
                            )
                        refl.centroid_px_to_mm(ExperimentList([expt]))
                        refl.map_centroids_to_reciprocal_space(ExperimentList([expt]))

                d_spacings = 1 / refl["rlp"].norms()
                refl = refl.select(d_spacings > self.params.d_min)
                reflections.append(refl)
            self.reflections = reflections
        self.Bind(EVT_LOADIMG, self.load_file_event)

        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUIMask, id=self._id_mask)
        self.Bind(EVT_ZEROMQ_EVENT, self.OnZeroMQEvent)

    def setup_toolbar(self):
        btn = self.toolbar.AddTool(
            toolId=-1,
            label="Load file",
            bitmap=icons.hkl_file.GetBitmap(),
            shortHelp="Load file",
            kind=wx.ITEM_NORMAL,
        )
        self.Bind(wx.EVT_MENU, self.OnLoadFile, btn)
        # btn = self.toolbar.AddTool(toolId=-1,
        # label="Settings",
        # bitmap=icons.advancedsettings.GetBitmap(),
        # shortHelp="Settings",
        # kind=wx.ITEM_NORMAL)
        # self.Bind(wx.EVT_MENU, self.OnShowSettings, btn)
        # btn = self.toolbar.AddTool(toolId=-1,
        # label="Zoom",
        # bitmap=icons.search.GetBitmap(),
        # shortHelp="Zoom",
        # kind=wx.ITEM_NORMAL)
        # self.Bind(wx.EVT_MENU, self.OnZoom, btn
        btn = self.toolbar.AddTool(
            toolId=wx.ID_SAVEAS,
            label="Save As...",
            bitmap=bitmaps.fetch_icon_bitmap("actions", "save_all", 32),
            shortHelp="Save As...",
            kind=wx.ITEM_NORMAL,
        )
        self.Bind(wx.EVT_MENU, self.OnSaveAs, btn)
        txt = wx.StaticText(self.toolbar, -1, "Image:")
        self.toolbar.AddControl(txt)

        # Because parent classes (e.g. XRayFrame) depends on this control explicitly
        # created in this subclass, we create an adapter to connect to the new design
        self.image_chooser = LegacyChooserAdapter(self.images, self.load_image)

        # Create a sub-control with our image selection slider and label
        # Manually tune the height for now - don't understand toolbar sizing
        panel = ImageChooserControl(self.toolbar, size=(300, 60))
        # The Toolbar doesn't call layout for its children?!
        panel.Layout()
        # Platform support for slider events seems a little inconsistent
        # with wxPython 3, so we just trap all EVT_SLIDER events.
        panel.Bind(wx.EVT_SLIDER, self.OnChooseImage)
        # These events appear to be a more reliable indicator?
        panel.Bind(wx.EVT_SCROLL_CHANGED, self.OnChooseImage)
        # Finally, add our new control to the toolbar
        self.toolbar.AddControl(panel)
        self.image_chooser_panel = panel

        btn = self.toolbar.AddTool(
            toolId=wx.ID_BACKWARD,
            label="Previous",
            bitmap=bitmaps.fetch_icon_bitmap("actions", "1leftarrow"),
            shortHelp="Previous",
            kind=wx.ITEM_NORMAL,
        )
        self.Bind(wx.EVT_MENU, self.OnPrevious, btn)
        btn = self.toolbar.AddTool(
            toolId=wx.ID_FORWARD,
            label="Next",
            bitmap=bitmaps.fetch_icon_bitmap("actions", "1rightarrow"),
            shortHelp="Next",
            kind=wx.ITEM_NORMAL,
        )
        self.Bind(wx.EVT_MENU, self.OnNext, btn)

        txt = wx.StaticText(self.toolbar, -1, "Jump:")
        self.toolbar.AddControl(txt)

        self.jump_to_image = PhilIntCtrl(self.toolbar, -1, name="image", size=(65, -1))
        self.jump_to_image.SetMin(1)
        self.jump_to_image.SetValue(1)
        self.toolbar.AddControl(self.jump_to_image)
        self.Bind(EVT_PHIL_CONTROL, self.OnJumpToImage, self.jump_to_image)

        txt = wx.StaticText(self.toolbar, -1, "Stack:")
        self.toolbar.AddControl(txt)

        self.stack = PhilIntCtrl(self.toolbar, -1, name="stack", size=(65, -1))
        self.stack.SetMin(1)
        self.stack.SetValue(self.params.stack_images)
        self.toolbar.AddControl(self.stack)
        self.Bind(EVT_PHIL_CONTROL, self.OnStack, self.stack)

    def setup_menus(self):
        super().setup_menus()

        # XXX Placement
        self._id_mask = wx.NewId()
        item = self._actions_menu.Append(self._id_mask, " ")
        self.Bind(wx.EVT_MENU, self.OnMask, source=item)

    def OnMask(self, event):
        if not self._mask_frame:
            self._mask_frame = MaskSettingsFrame(
                self,
                wx.ID_ANY,
                "Mask tool",
                style=wx.CAPTION | wx.CLOSE_BOX | wx.RESIZE_BORDER,
            )
            self._mask_frame.Show()
            self._mask_frame.Raise()
        else:
            self._mask_frame.Destroy()

    def OnUpdateUIMask(self, event):
        # Toggle the menu item text depending on the state of the tool.

        if self._mask_frame:
            event.SetText("Hide mask tool")
        else:
            event.SetText("Show mask tool")

    def OnChooseImage(self, event):
        # Whilst scrolling and choosing, show what we are looking at
        selected_image = self.images[self.image_chooser_panel.GetValue() - 1]
        # Always show the current 'loaded' image as such
        if selected_image == self.images.selected:
            self.image_chooser_panel.set_label(self.get_key(selected_image))
        else:
            self.image_chooser_panel.set_temporary_label(self.get_key(selected_image))

        # Don't update whilst dragging the slider
        if event.EventType == wx.EVT_SLIDER.typeId:
            if wx.GetMouseState().LeftIsDown():
                return

        # Once we've stopped scrolling, load the selected item
        self.load_image(selected_image)

    def OnPrevious(self, event):
        super().OnPrevious(event)
        # Parent function moves - now update the UI to match
        self.jump_to_image.SetValue(self.images.selected_index + 1)

    def OnNext(self, event):
        super().OnNext(event)
        # Parent function moves - now update the UI to match
        self.jump_to_image.SetValue(self.images.selected_index + 1)

    def OnJumpToImage(self, event):
        phil_value = self.jump_to_image.GetPhilValue()
        if self.images.selected_index != (phil_value - 1):
            self.load_image(self.images[phil_value - 1])

    def OnStack(self, event):
        value = self.stack.GetPhilValue()

        if value == 1:
            for button in self.settings_frame.panel.dispersion_buttons:
                button.Enable()
        else:
            for button in self.settings_frame.panel.dispersion_buttons:
                button.Disable()

        if value != self.params.stack_images:
            self.params.stack_images = value
            self.reload_image()

    def GetBoxCorners(self, layer, p1, p2):
        """Get list of points inside box.

        layer  reference to layer object we are working on
        p1     one corner point of selection box
        p2     opposite corner point of selection box

        We have to figure out which corner is which.

        Return a list of (lon, lat) of points inside box.
        Return None (no selection) or list [((lon, lat), data), ...]
        of points inside the selection box.
        """

        # TODO: speed this up?  Do we need to??
        # get canonical box limits
        (p1x, p1y) = p1
        (p2x, p2y) = p2
        lx = min(p1x, p2x)  # left x coord
        rx = max(p1x, p2x)
        ty = max(p1y, p2y)  # top y coord
        by = min(p1y, p2y)

        return [(lx, by), (lx, ty), (rx, ty), (rx, by)]

    def boxSelect(self, event):
        """Select event from pyslip."""

        point = event.point
        assert event.evtype == pyslip.EventBoxSelect

        if point:
            assert len(point) == 4

            x0, y0 = point[0]
            x1, y1 = point[2]

            assert point == [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]

            point = [
                self.pyslip.tiles.map_relative_to_picture_fast_slow(*p) for p in point
            ]

            point_ = []
            panel_id = None
            for p in point:
                p1, p0, p_id = self.pyslip.tiles.flex_image.picture_to_readout(
                    p[1], p[0]
                )
                assert p_id >= 0, "Point must be within a panel"
                if panel_id is not None:
                    assert (
                        panel_id == p_id
                    ), "All points must be contained within a single panel"
                panel_id = p_id
                point_.append((p0, p1))
            point = point_

            region = masking.phil_scope.extract().untrusted[0]
            region.polygon = flat_list(point)
            region.panel = panel_id

            self.settings.untrusted.append(region)

        self.drawUntrustedPolygons()

        return True

    def drawUntrustedPolygons(self):
        # remove any previous selection
        if self.sel_image_polygon_layer:
            self.pyslip.DeleteLayer(self.sel_image_polygon_layer)
            self.sel_image_polygon_layer = None
        for layer in self.sel_image_circle_layers:
            self.pyslip.DeleteLayer(layer)
        self.sel_image_circle_layers = []

        if not len(self.settings.untrusted):
            return

        polygon_data = []
        circle_data = []
        d = {}
        for region in self.settings.untrusted:
            polygon = None
            circle = None
            if region.rectangle is not None:
                x0, x1, y0, y1 = region.rectangle
                polygon = [x0, y0, x1, y0, x1, y1, x0, y1]
            elif region.polygon is not None:
                polygon = region.polygon
            elif region.circle is not None:
                circle = region.circle

            if polygon is not None:
                assert len(polygon) % 2 == 0, "Polygon must contain 2D coords"
                vertices = []
                for i in range(int(len(polygon) / 2)):
                    x = polygon[2 * i]
                    y = polygon[2 * i + 1]
                    vertices.append((x, y))

                vertices = [
                    self.pyslip.tiles.flex_image.tile_readout_to_picture(
                        int(region.panel), v[1], v[0]
                    )
                    for v in vertices
                ]
                vertices = [(v[1], v[0]) for v in vertices]

                points_rel = [
                    self.pyslip.tiles.picture_fast_slow_to_map_relative(*v)
                    for v in vertices
                ]

                points_rel.append(points_rel[0])
                for i in range(len(points_rel) - 1):
                    polygon_data.append(((points_rel[i], points_rel[i + 1]), d))

            if circle is not None:
                x, y, r = circle
                y, x = self.pyslip.tiles.flex_image.tile_readout_to_picture(
                    int(region.panel), y, x
                )
                x, y = self.pyslip.tiles.picture_fast_slow_to_map_relative(x, y)
                center = matrix.col((x, y))
                e1 = matrix.col((1, 0))
                e2 = matrix.col((0, 1))
                circle_data.append(
                    (
                        center + r * (e1 + e2),
                        center + r * (e1 - e2),
                        center + r * (-e1 - e2),
                        center + r * (-e1 + e2),
                        center + r * (e1 + e2),
                    )
                )

        if polygon_data:
            self.sel_image_polygon_layer = self.pyslip.AddPolygonLayer(
                polygon_data,
                map_rel=True,
                color="#00ffff",
                radius=5,
                visible=True,
                # show_levels=[3,4],
                name="<boxsel_pt_layer>",
            )
        if circle_data:
            for circle in circle_data:
                self.sel_image_circle_layers.append(
                    self.pyslip.AddEllipseLayer(
                        circle,
                        map_rel=True,
                        color="#00ffff",
                        radius=5,
                        visible=True,
                        # show_levels=[3,4],
                        name="<boxsel_pt_layer>",
                    )
                )

    def add_file_name_or_data(self, image_data):
        """
        Adds an image to the viewer's list of images.

        This just adds the image to the virtual list, and updates the UI
        where necessary e.g. image chooser maximums. If the image already
        exists, will not add a duplicate.

        AKA and better named something like 'add_image' but we can't rename
        whilst tied to rstbx.

        :param image_data: The image metadata object
        :type  image_data: chooser_wrapper
        :returns: The index of the image in the list of images
        :rtype:   int
        """

        assert isinstance(image_data, chooser_wrapper)

        # If this is already loaded, then return the index
        if image_data in self.images:
            return self.images.index(image_data)

        self.images.add(image_data)
        self.image_chooser_panel.SetMax(len(self.images))
        self.jump_to_image.SetMax(len(self.images))
        self.stack.SetMax(len(self.images))
        return len(self.images) - 1

    def load_file_event(self, evt):
        self.load_image(evt.get_filename())

    def reload_image(self):
        """Re-load the currently displayed image"""
        with wx.BusyCursor():
            self.load_image(self.images.selected, refresh=True)

    def load_image(self, file_name_or_data, refresh=False):
        """
        Load and display an image.

        Given either a filename or a pre-existing image data object, loads the
        image from disk, displays it, and updates the UI to reflect the new image.

        :param file_name_or_data: The image item to load
        :param refresh: Should the image be reloaded if currently selected?
        :type file_name_or_data: str or chooser_wrapper
        """

        # If this image is already loaded, then don't reload it
        if not refresh and file_name_or_data == self.images.selected:
            return

        # If given a string, we need to load and convert to a chooser_wrapper
        if isinstance(file_name_or_data, str):
            experiments = ExperimentListFactory.from_filenames([file_name_or_data])
            assert len(experiments) == 1
            imagesets = experiments.imagesets()
            imageset = imagesets[0]
            file_name_or_data = chooser_wrapper(imageset, imageset.indices()[0])
            self.add_file_name_or_data(file_name_or_data)

        assert isinstance(file_name_or_data, chooser_wrapper)

        # We are never called without first calling add_file_name_or_data?
        assert file_name_or_data in self.images

        show_untrusted = False
        if self.params.show_mask:
            show_untrusted = True

        previously_selected_image = self.images.selected
        self.images.selected = file_name_or_data
        # Do the actual data/image loading and update the viewer
        super().load_image(
            file_name_or_data,
            get_image_data=self.get_image_data,
            show_untrusted=show_untrusted,
        )

        # Update the navigation UI controls to reflect this loaded image
        self.image_chooser_panel.SetValue(self.images.selected_index + 1)
        self.image_chooser_panel.set_label(self.get_key(file_name_or_data))
        self.jump_to_image.SetValue(self.images.selected_index + 1)

        # Destroy the cached data for the previous image
        if (
            previously_selected_image
            and previously_selected_image != self.images.selected
        ):
            previously_selected_image.set_image_data(None)

    def OnShowSettings(self, event):
        if self.settings_frame is None:
            frame_rect = self.GetRect()
            display_rect = wx.GetClientDisplayRect()
            x_start = frame_rect[0] + frame_rect[2]
            if x_start > (display_rect[2] - 400):
                x_start = display_rect[2] - 400
            y_start = frame_rect[1]
            self.settings_frame = SpotSettingsFrame(
                self,
                -1,
                "Settings",
                style=wx.CAPTION | wx.MINIMIZE_BOX,
                pos=(x_start, y_start),
            )
        self.settings_frame.Show()

    def _choose_text_colour(self):
        """Choose a text colour contrasting with the image pixels"""
        if self.settings.color_scheme > 1:  # heatmap or invert
            return "#cdcdcd"  # light grey
        else:
            return "#3b3b3b"  # dark grey

    def _draw_resolution_polygons(
        self, spacings, beam, detector, unit_cell, space_group
    ):
        """Draw resolution rings for arbitrary detector geometry using a polygon path"""

        add_text = unit_cell is None and space_group is None
        segments, res_labels = calculate_isoresolution_lines(
            spacings, beam, detector, self.pyslip.tiles.flex_image, add_text
        )

        ring_data = []
        metadata = {
            "color": "red",
            "closed": False,
            "width": self.pyslip.DefaultPolygonWidth,
        }
        for segment in segments:
            coord1 = self.pyslip.tiles.picture_fast_slow_to_map_relative(*segment[0])
            coord2 = self.pyslip.tiles.picture_fast_slow_to_map_relative(*segment[1])
            ring_data.append(((coord1, coord2), metadata))

        resolution_text_data = []
        metadata = {
            "placement": "cc",
            "colour": "red",
        }
        for txt_x, txt_y, txt_str in res_labels:
            x, y = self.pyslip.tiles.picture_fast_slow_to_map_relative(txt_x, txt_y)
            resolution_text_data.append((x, y, txt_str, metadata))

        # Remove the old ring layer, and draw a new one.
        if hasattr(self, "_ring_layer") and self._ring_layer is not None:
            self.pyslip.DeleteLayer(self._ring_layer, update=False)
            self._ring_layer = None

        self._ring_layer = self.pyslip.AddPolygonLayer(
            ring_data,
            name="<ring_layer>",
            show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
            update=False,
        )

        # Remove the old resolution text layer, and draw a new one.
        if (
            hasattr(self, "_resolution_text_layer")
            and self._resolution_text_layer is not None
        ):
            self.pyslip.DeleteLayer(self._resolution_text_layer, update=False)
            self._resolution_text_layer = None
        if resolution_text_data:
            self._resolution_text_layer = self.pyslip.AddTextLayer(
                resolution_text_data,
                map_rel=True,
                visible=True,
                show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                selectable=False,
                name="<resolution_text_layer>",
                fontsize=self.settings.fontsize,
                textcolour=self._choose_text_colour(),
                update=False,
            )

        return

    def draw_resolution_rings(self, unit_cell=None, space_group=None):
        image = self.image_chooser.GetClientData(
            self.image_chooser.GetSelection()
        ).image_set
        detector = image.get_detector()
        beam = image.get_beam()

        d_min = detector.get_max_resolution(beam.get_s0())
        d_star_sq_max = uctbx.d_as_d_star_sq(d_min)

        if unit_cell is not None and space_group is not None:
            unit_cell = space_group.average_unit_cell(unit_cell)
            generator = index_generator(unit_cell, space_group.type(), False, d_min)
            indices = generator.to_array()
            spacings = flex.sorted(unit_cell.d(indices))

        else:
            n_rings = 5
            step = d_star_sq_max / n_rings
            spacings = flex.double(
                [uctbx.d_star_sq_as_d((i + 1) * step) for i in range(0, n_rings)]
            )

        # For non-coplanar detectors use a polygon method rather than ellipses
        if detector.has_projection_2d():
            return self._draw_resolution_polygons(
                spacings,
                beam,
                detector,
                unit_cell,
                space_group,
            )

        wavelength = beam.get_wavelength()
        twotheta = uctbx.d_star_sq_as_two_theta(
            uctbx.d_as_d_star_sq(spacings), wavelength
        )

        # Get beam vector and two orthogonal vectors
        beamvec = matrix.col(beam.get_s0())
        bor1 = beamvec.ortho()
        bor2 = beamvec.cross(bor1)

        resolution_text_data = []
        ring_data = []
        # FIXME Currently assuming that all panels are in same plane
        p_id = detector.get_panel_intersection(beam.get_s0())
        if p_id == -1:
            # XXX beam doesn't intersect with any panels - is there a better solution?
            p_id = 0
        pan = detector[p_id]

        for tt, d in zip(twotheta, spacings):
            try:
                # Find 4 rays for given d spacing / two theta angle
                cb1 = beamvec.rotate_around_origin(axis=bor1, angle=tt)
                cb2 = beamvec.rotate_around_origin(axis=bor1, angle=-tt)
                cb3 = beamvec.rotate_around_origin(axis=bor2, angle=tt)
                cb4 = beamvec.rotate_around_origin(axis=bor2, angle=-tt)

                # Find intersection points with panel plane
                dp1 = pan.get_ray_intersection_px(cb1)
                dp2 = pan.get_ray_intersection_px(cb2)
                dp3 = pan.get_ray_intersection_px(cb3)
                dp4 = pan.get_ray_intersection_px(cb4)

                # If all four points are in positive beam direction, draw an ellipse.
                # Otherwise it's a hyperbola (not implemented yet)
            except RuntimeError:
                continue

            # find ellipse centre, the only point equidistant to each axial pair
            xs1 = dp1[0] + dp2[0]
            xs2 = dp3[0] + dp4[0]
            ys1 = dp1[1] + dp2[1]
            ys2 = dp3[1] + dp4[1]
            xd1 = dp2[0] - dp1[0]
            xd2 = dp4[0] - dp3[0]
            yd1 = dp1[1] - dp2[1]
            yd2 = dp3[1] - dp4[1]
            if abs(xd1) < 0.00001:
                cy = ys1 / 2
            elif abs(xd2) < 0.00001:
                cy = ys2 / 2
            else:
                t2 = (xs1 - xs2 + (ys2 - ys1) * yd1 / xd1) / (yd2 - xd2 * yd1 / xd1)
                t1 = (ys2 + t2 * xd2 - ys1) / xd1
                cy = (ys1 + t1 * xd1) / 2
                assert abs(cy - (ys2 + t2 * xd2) / 2) < 0.1
            if abs(yd1) < 0.00001:
                cx = xs1 / 2
            elif abs(yd2) < 0.00001:
                cx = xs2 / 2
            else:
                t2 = (xs1 - xs2 + (ys2 - ys1) * yd1 / xd1) / (yd2 - xd2 * yd1 / xd1)
                t1 = (ys2 + t2 * xd2 - ys1) / xd1
                cx = (xs1 + t1 * yd1) / 2
                assert abs(cx - (xs2 + t2 * yd2) / 2) < 0.1

            centre = self.pyslip.tiles.flex_image.tile_readout_to_picture(p_id, cy, cx)[
                ::-1
            ]
            dp1 = self.pyslip.tiles.flex_image.tile_readout_to_picture(
                p_id, dp1[1], dp1[0]
            )[::-1]
            dp3 = self.pyslip.tiles.flex_image.tile_readout_to_picture(
                p_id, dp3[1], dp3[0]
            )[::-1]

            # translate ellipse centre and four points to map coordinates
            centre = self.pyslip.tiles.picture_fast_slow_to_map_relative(*centre)
            dp1 = self.pyslip.tiles.picture_fast_slow_to_map_relative(dp1[0], dp1[1])
            dp3 = self.pyslip.tiles.picture_fast_slow_to_map_relative(dp3[0], dp3[1])

            # Determine eccentricity, cf. https://en.wikipedia.org/wiki/Eccentricity_(mathematics)
            ecc = math.sin(matrix.col(pan.get_normal()).angle(beamvec)) / math.sin(
                math.pi / 2 - tt
            )

            # Assuming that one detector axis is aligned with a major axis of
            # the ellipse, obtain the semimajor axis length a to calculate the
            # semiminor axis length b using the eccentricity ecc.
            ldp1 = math.hypot(dp1[0] - centre[0], dp1[1] - centre[1])
            ldp3 = math.hypot(dp3[0] - centre[0], dp3[1] - centre[1])
            if ldp1 >= ldp3:
                major = dp1
                a = ldp1
            else:
                major = dp3
                a = ldp3
            b = math.sqrt(a * a * (1 - (ecc * ecc)))
            # since e = f / a and f = sqrt(a^2 - b^2), cf. https://en.wikipedia.org/wiki/Ellipse

            # calculate co-vertex
            minor = (
                matrix.col([-centre[1] - dp1[1], centre[0] - dp1[0]]).normalize() * b
            )
            minor = (minor[0] + centre[0], minor[1] + centre[1])

            p = (centre, major, minor)
            ring_data.append(
                (
                    p,
                    self.pyslip.DefaultPolygonPlacement,
                    self.pyslip.DefaultPolygonWidth,
                    "red",
                    True,
                    self.pyslip.DefaultPolygonFilled,
                    self.pyslip.DefaultPolygonFillcolour,
                    self.pyslip.DefaultPolygonOffsetX,
                    self.pyslip.DefaultPolygonOffsetY,
                    None,
                )
            )

            if unit_cell is None and space_group is None:
                for angle in (45, 135, 225, 315):
                    txtvec = cb1.rotate_around_origin(
                        axis=beamvec, angle=math.radians(angle)
                    )
                    txtpos = pan.get_ray_intersection_px(txtvec)
                    txtpos = self.pyslip.tiles.flex_image.tile_readout_to_picture(
                        p_id, txtpos[1], txtpos[0]
                    )[::-1]
                    x, y = self.pyslip.tiles.picture_fast_slow_to_map_relative(
                        txtpos[0], txtpos[1]
                    )
                    resolution_text_data.append(
                        (
                            x,
                            y,
                            f"{d:.2f}",
                            {
                                "placement": "cc",
                                "colour": "red",
                            },
                        )
                    )

        # XXX Transparency?
        # Remove the old ring layer, and draw a new one.
        if hasattr(self, "_ring_layer") and self._ring_layer is not None:
            self.pyslip.DeleteLayer(self._ring_layer, update=False)
            self._ring_layer = None
        if ring_data:
            self._ring_layer = self.pyslip.AddLayer(
                self.pyslip.DrawLightweightEllipticalSpline,
                ring_data,
                True,
                True,
                show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                selectable=False,
                type=self.pyslip.TypeEllipse,
                name="<ring_layer>",
                update=False,
            )

        # Remove the old resolution text layer, and draw a new one.
        if (
            hasattr(self, "_resolution_text_layer")
            and self._resolution_text_layer is not None
        ):
            self.pyslip.DeleteLayer(self._resolution_text_layer, update=False)
            self._resolution_text_layer = None
        if resolution_text_data:
            self._resolution_text_layer = self.pyslip.AddTextLayer(
                resolution_text_data,
                map_rel=True,
                visible=True,
                show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                selectable=False,
                name="<resolution_text_layer>",
                fontsize=self.settings.fontsize,
                textcolour=self._choose_text_colour(),
                update=False,
            )

    def stack_images(self):
        mode = self.params.stack_mode
        if self.params.stack_images > 1:
            self.settings.display = "image"
            image = self.pyslip.tiles.raw_image
            image_data = image.get_image_data()
            if not isinstance(image_data, tuple):
                image_data = (image_data,)

            if self.params.show_mask:
                masks = tuple(i == MASK_VAL for i in image_data)

            i_frame = self.image_chooser.GetSelection()
            for i in range(1, self.params.stack_images):
                if (i_frame + i) >= len(self.images):
                    break

                image_data_i = self.images[i_frame + i].get_image_data()
                for j, rd in enumerate(image_data):
                    data = image_data_i[j]

                    if mode == "max":
                        sel = data > rd
                        rd = rd.as_1d().set_selected(sel.as_1d(), data.as_1d())
                    else:
                        rd += data

                if self.params.show_mask:
                    image_masks = self.images[i_frame + i].get_mask()
                    for merged_mask, image_mask in zip(masks, image_masks):
                        merged_mask.set_selected(~image_mask, True)

            # /= stack_images to put on consistent scale with single image
            # so that -1 etc. handled correctly (mean mode)
            if mode == "mean":
                image_data = tuple(i / self.params.stack_images for i in image_data)

            if self.params.show_mask:
                for rd, mask in zip(image_data, masks):
                    rd = rd.as_1d().set_selected(mask.as_1d(), MASK_VAL)

            # Don't show summed images with overloads
            self.pyslip.tiles.set_image_data(image_data, show_saturated=False)

            self.pyslip.ZoomToLevel(self.pyslip.tiles.zoom_level)
            self.update_statusbar()  # XXX Not always working?
            self.Layout()

    def get_image_data(self, image):
        image.set_image_data(None)
        if self.settings.display == "image":
            if self.settings.image_type == "corrected":
                image_data = image.get_image_data()
            else:
                image_data = image.get_image_data(corrected=False)
            if isinstance(image_data, tuple):
                image_data = tuple(id.as_double() for id in image_data)
            else:
                image_data = (image_data.as_double(),)

        else:
            dispersion_debug_list = self._calculate_dispersion_debug(image)

            if self.settings.display == "mean":
                mean = [dispersion.mean() for dispersion in dispersion_debug_list]
                image_data = mean
            elif self.settings.display == "variance":
                variance = [
                    dispersion.variance() for dispersion in dispersion_debug_list
                ]
                image_data = variance
            elif self.settings.display == "dispersion":
                cv = [
                    dispersion.index_of_dispersion()
                    for dispersion in dispersion_debug_list
                ]
                image_data = cv
            elif self.settings.display == "sigma_b":
                cv = [
                    dispersion.index_of_dispersion()
                    for dispersion in dispersion_debug_list
                ]
                cv_mask = [dispersion.cv_mask() for dispersion in dispersion_debug_list]
                cv_mask = [mask.as_1d().as_double() for mask in cv_mask]
                for i, mask in enumerate(cv_mask):
                    mask.reshape(cv[i].accessor())
                image_data = cv_mask
            elif self.settings.display == "sigma_s":
                cv = [
                    dispersion.index_of_dispersion()
                    for dispersion in dispersion_debug_list
                ]
                value_mask = [
                    dispersion.value_mask() for dispersion in dispersion_debug_list
                ]
                value_mask = [mask.as_1d().as_double() for mask in value_mask]
                for i, mask in enumerate(value_mask):
                    mask.reshape(cv[i].accessor())
                image_data = value_mask
            elif self.settings.display == "global":
                cv = [
                    dispersion.index_of_dispersion()
                    for dispersion in dispersion_debug_list
                ]
                global_mask = [
                    dispersion.global_mask() for dispersion in dispersion_debug_list
                ]
                global_mask = [mask.as_1d().as_double() for mask in global_mask]
                for i, mask in enumerate(global_mask):
                    mask.reshape(cv[i].accessor())
                image_data = global_mask
            elif self.settings.display == "threshold":
                cv = [
                    dispersion.index_of_dispersion()
                    for dispersion in dispersion_debug_list
                ]
                final_mask = [
                    dispersion.final_mask() for dispersion in dispersion_debug_list
                ]
                final_mask = [mask.as_1d().as_double() for mask in final_mask]
                for i, mask in enumerate(final_mask):
                    mask.reshape(cv[i].accessor())
                image_data = final_mask

            if self.settings.display in ("sigma_b", "sigma_s", "global", "threshold"):
                image_data = (500 * d for d in image_data)

        image_data = tuple(image_data)
        if self.params.show_mask:
            self.mask_image_data(image_data)
        return image_data

    def _calculate_dispersion_debug(self, image):
        # hash current settings
        dispersion_debug_list_hash = hash(
            (
                image.index,
                self.images.selected.image_set,
                self.settings.gain,
                self.settings.nsigma_b,
                self.settings.nsigma_s,
                self.settings.global_threshold,
                self.settings.min_local,
                tuple(self.settings.kernel_size),
                self.settings.threshold_algorithm,
                self.settings.n_iqr,
                self.settings.blur,
                self.settings.n_bins,
            )
        )

        # compare current settings to last calculation of "dispersion debug" list
        if dispersion_debug_list_hash == self._dispersion_debug_list_hash:
            return self._dispersion_debug_list

        detector = image.get_detector()
        image_mask = self.get_mask(image)
        image_data = image.get_image_data()
        assert self.settings.gain > 0
        gain_map = [
            flex.double(image_data[i].accessor(), self.settings.gain)
            for i in range(len(detector))
        ]
        if self.settings.threshold_algorithm == "dispersion_extended":
            algorithm = DispersionExtendedThresholdDebug
        elif self.settings.threshold_algorithm == "dispersion":
            algorithm = DispersionThresholdDebug
        else:
            algorithm = RadialProfileThresholdDebug(
                image, self.settings.n_iqr, self.settings.blur, self.settings.n_bins
            )

        dispersion_debug_list = []
        for i_panel in range(len(detector)):
            dispersion_debug_list.append(
                algorithm(
                    image_data[i_panel].as_double(),
                    image_mask[i_panel],
                    gain_map[i_panel],
                    self.settings.kernel_size,
                    self.settings.nsigma_b,
                    self.settings.nsigma_s,
                    self.settings.global_threshold,
                    self.settings.min_local,
                )
            )

        self._dispersion_debug_list = dispersion_debug_list
        self._dispersion_debug_list_hash = dispersion_debug_list_hash
        return dispersion_debug_list

    def show_filters(self):
        image_data = self.get_image_data(self.pyslip.tiles.raw_image)
        show_saturated = (
            self.settings.display == "image" and self.settings.image_type == "corrected"
        )
        self.pyslip.tiles.set_image_data(image_data, show_saturated)
        self.pyslip.ZoomToLevel(self.pyslip.tiles.zoom_level)
        self.update_statusbar()  # XXX Not always working?
        self.Layout()

    def update_settings(self, layout=True):
        # super(SpotFrame, self).update_settings(layout=layout)
        new_brightness = self.settings.brightness
        new_color_scheme = self.settings.color_scheme
        if (
            new_brightness is not self.pyslip.tiles.current_brightness
            or new_color_scheme is not self.pyslip.tiles.current_color_scheme
        ):
            self.pyslip.tiles.update_brightness(new_brightness, new_color_scheme)

        detector = self.pyslip.tiles.raw_image.get_detector()
        detector.projection = self.params.projection

        if self.settings.show_beam_center:
            if self.beam_layer is None and hasattr(self, "beam_center_cross_data"):
                self.beam_layer = self.pyslip.AddPolygonLayer(
                    self.beam_center_cross_data,
                    name="<beam_layer>",
                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                    update=False,
                )
        elif self.beam_layer is not None:
            self.pyslip.DeleteLayer(self.beam_layer, update=False)
            self.beam_layer = None

        if self.settings.show_dials_spotfinder_spots:
            spotfinder_data = self.get_spotfinder_data()
            shoebox_data = spotfinder_data.shoebox_data
            all_pix_data = spotfinder_data.all_pix_data
            all_foreground_circles = spotfinder_data.all_foreground_circles
            ctr_mass_data = spotfinder_data.ctr_mass_data
            max_pix_data = spotfinder_data.max_pix_data
            predictions_data = spotfinder_data.predictions_data
            miller_indices_data = spotfinder_data.miller_indices_data
            vector_data = spotfinder_data.vector_data
            vector_text_data = spotfinder_data.vector_text_data
            if len(self.dials_spotfinder_layers) > 0:
                for layer in self.dials_spotfinder_layers:
                    self.pyslip.DeleteLayer(layer, update=False)
                self.dials_spotfinder_layers = []
            if self.shoebox_layer is not None:
                self.pyslip.DeleteLayer(self.shoebox_layer, update=False)
                self.shoebox_layer = None
            if self.ctr_mass_layer is not None:
                self.pyslip.DeleteLayer(self.ctr_mass_layer, update=False)
                self.ctr_mass_layer = None
            if self.max_pix_layer is not None:
                self.pyslip.DeleteLayer(self.max_pix_layer, update=False)
                self.max_pix_layer = None
            if self.predictions_layer is not None:
                self.pyslip.DeleteLayer(self.predictions_layer, update=False)
                self.predictions_layer = None
            if self.miller_indices_layer is not None:
                self.pyslip.DeleteLayer(self.miller_indices_layer, update=False)
                self.miller_indices_layer = None
            if self.vector_layer is not None:
                self.pyslip.DeleteLayer(self.vector_layer, update=False)
                self.vector_layer = None
            if self.vector_text_layer is not None:
                self.pyslip.DeleteLayer(self.vector_text_layer, update=False)
                self.vector_text_layer = None
            if self._ring_layer is not None:
                self.pyslip.DeleteLayer(self._ring_layer, update=False)
                self._ring_layer = None
            if self._resolution_text_layer is not None:
                self.pyslip.DeleteLayer(self._resolution_text_layer, update=False)
                self._resolution_text_layer = None

            if self.settings.show_miller_indices and len(miller_indices_data):
                self.miller_indices_layer = self.pyslip.AddTextLayer(
                    miller_indices_data,
                    map_rel=True,
                    visible=True,
                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                    selectable=False,
                    fontsize=self.settings.fontsize,
                    textcolour=self._choose_text_colour(),
                    name="<miller_indices_layer>",
                    update=False,
                )
            if self.settings.show_predictions and len(predictions_data):
                self.predictions_layer = self.pyslip.AddPointLayer(
                    predictions_data,
                    name="<predictions_layer>",
                    radius=3,
                    renderer=self.pyslip.DrawPointLayer,
                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                    update=False,
                )
            if self.settings.show_all_pix:
                if len(all_pix_data) > 1:
                    if not self.display_foreground_circles_patch:
                        for key, value in all_pix_data.items():
                            base_color = self.prediction_colours[key][1:]
                            # dim the color so it stands apart from the prediction
                            r = base_color[0:2]
                            g = base_color[2:4]
                            b = base_color[4:6]
                            r = max(int(r, 16) - int("50", 16), 0)
                            g = max(int(g, 16) - int("50", 16), 0)
                            b = max(int(b, 16) - int("50", 16), 0)
                            color = f"#{r:02x}{g:02x}{b:02x}"
                            self.dials_spotfinder_layers.append(
                                self.pyslip.AddPointLayer(
                                    value,
                                    color=color,
                                    name="<all_pix_layer_%d>" % key,
                                    radius=2,
                                    renderer=self.pyslip.LightweightDrawPointLayer2,
                                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                                    update=False,
                                )
                            )
                    else:
                        e1 = matrix.col((1.0, 0.0))
                        e2 = matrix.col((0.0, 1.0))
                        for key, value in all_foreground_circles.items():
                            base_color = self.prediction_colours[key][1:]
                            positions = [i["position"] for i in value]
                            good_radius = flex.mean(
                                flex.double([i["radius"] for i in value])
                            )
                            vertices = []
                            for model_center in positions:
                                for vertex in [
                                    model_center + good_radius * (e1 + e2),
                                    model_center + good_radius * (e1 - e2),
                                    model_center + good_radius * (-e1 - e2),
                                    model_center + good_radius * (-e1 + e2),
                                    model_center + good_radius * (e1 + e2),
                                ]:
                                    vertices.append(vertex)

                            self.dials_spotfinder_layers.append(
                                self.pyslip.AddEllipseLayer(
                                    vertices,
                                    color=f"#{base_color}",
                                    name="<all_foreground_circles_%d>" % key,
                                    width=2,
                                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                                    update=False,
                                )
                            )
                            print(
                                "Circles: center of foreground masks for the %d spots actually integrated"
                                % (len(vertices) // 5)
                            )
                else:
                    if len(all_pix_data) > 0:
                        self.dials_spotfinder_layers.append(
                            self.pyslip.AddPointLayer(
                                all_pix_data[list(all_pix_data.keys())[0]],
                                color="green",
                                name="<all_pix_layer>",
                                radius=2,
                                renderer=self.pyslip.LightweightDrawPointLayer2,
                                show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                                update=False,
                            )
                        )
            if self.settings.show_shoebox and len(shoebox_data):
                self.shoebox_layer = self.pyslip.AddPolygonLayer(
                    shoebox_data,
                    map_rel=True,
                    visible=True,
                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                    selectable=False,
                    name="<shoebox_layer>",
                    update=False,
                )
            if self.settings.show_ctr_mass and len(ctr_mass_data):
                self.ctr_mass_layer = self.pyslip.AddPolygonLayer(
                    ctr_mass_data,
                    map_rel=True,
                    visible=True,
                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                    selectable=False,
                    name="<ctr_mass_layer>",
                    update=False,
                )
            if self.settings.show_max_pix and len(max_pix_data):
                self.max_pix_layer = self.pyslip.AddPointLayer(
                    max_pix_data,
                    color="pink",
                    name="<max_pix_layer>",
                    radius=2,
                    renderer=self.pyslip.LightweightDrawPointLayer,
                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                    update=False,
                )
            if len(vector_data) and len(vector_text_data):
                self.vector_layer = self.pyslip.AddPolygonLayer(
                    vector_data,
                    map_rel=True,
                    visible=True,
                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                    selectable=False,
                    name="<vector_layer>",
                    update=False,
                )
                self.vector_text_layer = self.pyslip.AddTextLayer(
                    vector_text_data,
                    map_rel=True,
                    visible=True,
                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                    selectable=False,
                    name="<vector_text_layer>",
                    textcolour=self._choose_text_colour(),
                    update=False,
                    fontsize=self.settings.fontsize,
                )

        self.stack_images()
        # if self.params.stack_images == 1:
        # self.show_filters()
        if self.settings.show_threshold_pix:
            image = self.pyslip.tiles.raw_image
            dispersion_debug_list = self._calculate_dispersion_debug(image)
            final_mask = [
                dispersion.final_mask() for dispersion in dispersion_debug_list
            ]
            value = []
            for pnl, mask in enumerate(final_mask):
                width = mask.all()[1]
                idx = mask.iselection()
                for i in idx:
                    y = i // width
                    x = i % width
                    y, x = self.pyslip.tiles.flex_image.tile_readout_to_picture(
                        pnl, y, x
                    )
                    value.append(
                        self.pyslip.tiles.picture_fast_slow_to_map_relative(x, y)
                    )

            base_color = self.prediction_colours[0][1:]
            # dim the color so it stands apart from the prediction
            r = base_color[0:2]
            g = base_color[2:4]
            b = base_color[4:6]
            r = max(int(r, 16) - int("50", 16), 0)
            g = max(int(g, 16) - int("50", 16), 0)
            b = max(int(b, 16) - int("50", 16), 0)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.dials_spotfinder_layers.append(
                self.pyslip.AddPointLayer(
                    value,
                    color=color,
                    name="<thresh_pix_layer_0>",
                    radius=2,
                    renderer=self.pyslip.LightweightDrawPointLayer2,
                    show_levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
                    update=False,
                )
            )

        if self.settings.show_resolution_rings:
            self.draw_resolution_rings()
        elif self.settings.show_ice_rings:
            unit_cell = self.settings.ice_rings.unit_cell
            space_group = self.settings.ice_rings.space_group.group()
            self.draw_resolution_rings(unit_cell=unit_cell, space_group=space_group)
        self.drawUntrustedPolygons()
        self.pyslip.Update()

    def get_mask(self, image):
        mask = image.get_mask()
        if self.mask_input is not None:
            for p1, p2 in zip(self.mask_input, mask):
                p2 &= p1
        if self.mask_image_viewer is not None:
            for p1, p2 in zip(self.mask_image_viewer, mask):
                p2 &= p1
        assert mask is not None, "Mask should never be None here"
        return mask

    def mask_image_data(self, image_data):
        mask = self.get_mask(self.pyslip.tiles.raw_image)
        for rd, m in zip(image_data, mask):
            rd.set_selected(~m, MASK_VAL)

    def __get_imageset_filter(
        self, reflections: flex.reflection_table, imageset: ImageSet
    ) -> Optional[flex.bool]:
        """Get a filter to ensure only reflections from an imageset.

        This is not a well-defined problem because of unindexed reflections
        - any unindexed reflections never get assigned an experiment. Using the
        imageset_id column you can disentangle this, but but at integration this
        data is currently not copied. This means that you can separate, but only if
        there is a single imageset.

        Args:
            reflections:
                The reflections table to filter
            imageset:
                The imageset to filter reflections to

        Returns:
            The selection, or None if there is nothing to select.
        """
        reflections_id = self.reflections.index(reflections)
        experimentlist = self.experiments[reflections_id]

        # If this imageset is not in this experiment, then skip
        if imageset not in experimentlist.imagesets():
            return None

        if "imageset_id" in reflections:
            # Only choose reflections that match this imageset
            imageset_id = experimentlist.imagesets().index(imageset)
            selection = reflections["imageset_id"] == imageset_id
        elif self.have_one_imageset:
            # If one imageset, no filtering is necessary
            selection = flex.bool(len(reflections), True)
        else:
            # Fallback:
            # Do filtering in a way that cannot handle complex unindexed reflections
            # Get the experiment IDs of every experiment with this imageset
            exp_ids = [
                i for i, exp in enumerate(experimentlist) if exp.imageset == imageset
            ]
            # No way to tell - don't show any unindexed
            selection = flex.bool(len(reflections), False)
            # OR together selections for all ids that have this imageset
            for eid in exp_ids:
                selection = selection | (reflections["id"] == eid)

        return selection

    def map_coords(self, x, y, p):
        """Convert coordinates in pixel, pixel, panel to picture coordinates
        required for correct positioning of overlays"""
        y, x = self.pyslip.tiles.flex_image.tile_readout_to_picture(p, y - 0.5, x - 0.5)
        return self.pyslip.tiles.picture_fast_slow_to_map_relative(x, y)

    def _rotation_axis_overlay_data(self):
        imageset = self.images.selected.image_set
        detector = self.pyslip.tiles.raw_image.get_detector()
        scan = imageset.get_scan()
        beam = imageset.get_beam()
        gonio = imageset.get_goniometer()
        still = scan is None or gonio is None
        if still:
            return
        axis = gonio.get_rotation_axis()
        try:
            panel, beam_centre = detector.get_ray_intersection(beam.get_s0())
        except RuntimeError as e:
            if "DXTBX_ASSERT(w_max > 0)" in str(e):
                # direct beam didn't hit a panel
                panel = 0
                beam_centre = detector[panel].get_ray_intersection(beam.get_s0())
            else:
                raise

        beam_x, beam_y = detector[panel].millimeter_to_pixel(beam_centre)
        beam_x, beam_y = self.map_coords(beam_x, beam_y, panel)

        # Find the plane containing the rotation axis and s0
        normal = matrix.col(beam.get_unit_s0()).cross(matrix.col(axis))

        # Find scattering angle at max inscribed resolution
        d_min = detector.get_max_inscribed_resolution(beam.get_s0())
        theta = math.asin(beam.get_wavelength() / (2.0 * d_min))

        # Rotate s0 in the plane so as to point to the inscribed circle
        # along the rotation axis
        a = matrix.col(beam.get_s0()).rotate(normal, 2.0 * theta)
        b = matrix.col(beam.get_s0()).rotate(normal, -2.0 * theta)

        panel_a = detector.get_panel_intersection(a)
        if panel_a < 0:
            return
        panel_b = detector.get_panel_intersection(b)
        if panel_b < 0:
            return
        x_a, y_a = detector[panel_a].get_ray_intersection_px(a)
        x_a, y_a = self.map_coords(x_a, y_a, panel_a)
        x_b, y_b = detector[panel_b].get_ray_intersection_px(b)
        x_b, y_b = self.map_coords(x_b, y_b, panel_b)

        result = []
        result.append(
            (
                ((x_b, y_b), (x_a, y_a)),
                {"width": 4, "color": "#1776f6", "closed": False},
            )
        )
        result.append(
            (
                x_a,
                y_a,
                "axis",
                {
                    "placement": "ne",
                    "fontsize": self.settings.fontsize,
                    "textcolor": "#1776f6",
                },
            )
        )
        return result

    def _reflection_overlay_data(self, i_frame):
        fg_code = MaskCode.Valid | MaskCode.Foreground
        strong_code = MaskCode.Valid | MaskCode.Strong
        shoebox_dict = {"width": 2, "color": "#0000FFA0", "closed": False}
        ctr_mass_dict = {"width": 2, "color": "#FF0000", "closed": False}

        shoebox_data = []
        all_pix_data = {}
        all_foreground_circles = {}
        overlapped_data = []
        ctr_mass_data = []
        max_pix_data = []
        predictions_data = []
        miller_indices_data = []

        for ref_list_id, ref_list in enumerate(self.reflections):
            if self.viewing_stills and ref_list_id != i_frame:
                continue

            # If we have more than one imageset, then we could be on the wrong one
            if not self.have_one_imageset:
                exp_filter = self.__get_imageset_filter(
                    ref_list, self.images.selected.image_set
                )
                if exp_filter is None:
                    continue
                ref_list = ref_list.select(exp_filter)

            if self.settings.show_indexed:
                indexed_sel = ref_list.get_flags(ref_list.flags.indexed, all=False)
                ref_list = ref_list.select(indexed_sel)

            if self.settings.show_integrated:
                integrated_sel = ref_list.get_flags(
                    ref_list.flags.integrated, all=False
                )
                ref_list = ref_list.select(integrated_sel)

            # Fast-fail if there's no reflections after filtering
            if len(ref_list) == 0:
                continue

            if "bbox" in ref_list:
                bbox = ref_list["bbox"]
                x0, x1, y0, y1, z0, z1 = bbox.parts()
                # ticket #107
                n = self.params.stack_images - 1
                if self.viewing_stills:
                    selected = ref_list
                elif self.viewing_still_scans:
                    identifiers = []
                    for elist in self.experiments:
                        for scan in elist.scans():
                            if scan.get_array_range()[0] == i_frame:
                                sel_expts = elist.where(scan=scan)
                                for i in sel_expts:
                                    identifiers.append(elist[i].identifier)
                    selected = ref_list.select_on_experiment_identifiers(identifiers)
                else:
                    bbox_sel = ~((i_frame >= z1) | ((i_frame + n) < z0))
                    selected = ref_list.select(bbox_sel)
                for reflection in selected.rows():
                    x0, x1, y0, y1, z0, z1 = reflection["bbox"]
                    panel = reflection["panel"]
                    nx = x1 - x0  # size of reflection box in x-direction
                    ny = y1 - y0  # size of reflection box in y-direction
                    # nz = z1 - z0 # number of frames this spot appears on
                    if (
                        self.settings.show_all_pix
                        and "shoebox" in reflection
                        and reflection["shoebox"].mask.size() > 0
                        and n == 0
                    ):
                        shoebox = reflection["shoebox"]
                        iz = i_frame - z0 if not self.viewing_stills else 0
                        if reflection["id"] not in all_pix_data:
                            all_pix_data[reflection["id"]] = []

                            all_foreground_circles[reflection["id"]] = []

                        this_spot_foreground_pixels = []
                        for ix in range(nx):
                            for iy in range(ny):
                                mask_value = shoebox.mask[iz, iy, ix]
                                if (mask_value == strong_code) or (
                                    mask_value == fg_code
                                ):
                                    x_, y_ = self.map_coords(
                                        ix + x0 + 0.5, iy + y0 + 0.5, panel
                                    )
                                    this_spot_foreground_pixels.append(
                                        matrix.col((x_, y_))
                                    )
                                    if len(all_pix_data) > 1:
                                        # look for overlapped pixels
                                        found_it = False
                                        for key, value in all_pix_data.items():
                                            if (x_, y_) in value:
                                                value.pop(value.index((x_, y_)))
                                                found_it = True
                                        if found_it:
                                            overlapped_data.append((x_, y_))
                                        else:
                                            all_pix_data[reflection["id"]].append(
                                                (x_, y_)
                                            )
                                    else:
                                        all_pix_data[reflection["id"]].append((x_, y_))
                        if (
                            self.display_foreground_circles_patch
                            and len(this_spot_foreground_pixels) > 1
                        ):
                            per_spot_mean = matrix.col((0.0, 0.0))
                            for pxl in this_spot_foreground_pixels:
                                per_spot_mean += pxl
                            per_spot_mean /= len(this_spot_foreground_pixels)
                            all_foreground_circles[reflection["id"]].append(
                                {
                                    "position": per_spot_mean,
                                    "radius": max(
                                        [
                                            (t - per_spot_mean).length()
                                            for t in this_spot_foreground_pixels
                                        ]
                                    ),
                                }
                            )

                    if self.settings.show_shoebox:
                        x0y0 = self.map_coords(x0, y0, panel)
                        x0y1 = self.map_coords(x0, y1, panel)
                        x1y0 = self.map_coords(x1, y0, panel)
                        x1y1 = self.map_coords(x1, y1, panel)
                        # Change shoebox colour depending on index id
                        my_attrs = dict(shoebox_dict)
                        # Reflections with *only* strong set should get default
                        if not (reflection["flags"] == ref_list.flags.strong):
                            my_attrs["color"] = self.prediction_colours[
                                reflection["id"]
                            ]
                        lines = [
                            ((x0y0, x0y1), my_attrs),
                            ((x0y1, x1y1), my_attrs),
                            ((x1y1, x1y0), my_attrs),
                            ((x1y0, x0y0), my_attrs),
                        ]
                        shoebox_data.extend(lines)

                    if (
                        self.settings.show_max_pix
                        and "shoebox" in reflection
                        and reflection["shoebox"].data.size() > 0
                    ):
                        shoebox = reflection["shoebox"].data
                        offset = flex.max_index(shoebox)
                        offset, k = divmod(offset, shoebox.all()[2])
                        offset, j = divmod(offset, shoebox.all()[1])
                        offset, i = divmod(offset, shoebox.all()[0])
                        max_index = (i, j, k)
                        if z0 + max_index[0] == i_frame or self.viewing_stills:
                            x, y = self.map_coords(
                                x0 + max_index[2] + 0.5,
                                y0 + max_index[1] + 0.5,
                                reflection["panel"],
                            )
                            max_pix_data.append((x, y))

                    if self.settings.show_ctr_mass and "xyzobs.px.value" in reflection:
                        centroid = reflection["xyzobs.px.value"]
                        # ticket #107
                        if self.viewing_stills or (
                            i_frame
                            <= centroid[2]
                            <= (i_frame + self.params.stack_images)
                        ):
                            x, y = self.map_coords(
                                centroid[0], centroid[1], reflection["panel"]
                            )
                            xm1, ym1 = self.map_coords(
                                centroid[0] - 1, centroid[1] - 1, reflection["panel"]
                            )
                            xp1, yp1 = self.map_coords(
                                centroid[0] + 1, centroid[1] + 1, reflection["panel"]
                            )
                            lines = [
                                (((x, ym1), (x, yp1)), ctr_mass_dict),
                                (((xm1, y), (xp1, y)), ctr_mass_dict),
                            ]
                            ctr_mass_data.extend(lines)

            if ("xyzcal.px" in ref_list or "xyzcal.mm" in ref_list) and (
                self.settings.show_predictions
                or (self.settings.show_miller_indices and "miller_index" in ref_list)
            ):
                if "xyzcal.px" in ref_list:
                    frame_numbers = ref_list["xyzcal.px"].parts()[2]
                else:
                    phi = ref_list["xyzcal.mm"].parts()[2]
                    scan = self.pyslip.tiles.raw_image.get_scan()
                    frame_numbers = scan.get_array_index_from_angle(math.degrees(phi))
                n = self.params.stack_images
                for i_expt in set(ref_list["id"]):
                    expt_sel = ref_list["id"] == i_expt
                    frame_predictions_sel = (frame_numbers >= (i_frame)) & (
                        frame_numbers < (i_frame + n)
                    )

                    sel = expt_sel
                    if not (self.viewing_stills or self.viewing_still_scans):
                        sel = frame_predictions_sel & expt_sel
                    elif self.viewing_still_scans:
                        if i_expt not in ref_list.experiment_identifiers():
                            sel = flex.bool(ref_list.size(), False)
                            break
                        identifier = ref_list.experiment_identifiers()[i_expt]
                        for expt_list in self.experiments:
                            if identifier in expt_list.identifiers():
                                idx = (
                                    expt_list.identifiers() == identifier
                                ).iselection()[0]
                                scan = expt_list[idx].scan.get_array_range()
                                if scan[0] == i_frame:
                                    # this scan corresponds to this frame: good!
                                    if not expt_list[idx].crystal:
                                        sel = flex.bool(ref_list.size(), False)
                                else:
                                    sel = flex.bool(ref_list.size(), False)
                                break
                    selected = ref_list.select(sel)
                    for reflection in selected.rows():
                        if (
                            self.settings.show_predictions
                            or self.settings.show_miller_indices
                        ):
                            x = None
                            if "xyzcal.px" in reflection:
                                x, y = self.map_coords(
                                    reflection["xyzcal.px"][0],
                                    reflection["xyzcal.px"][1],
                                    reflection["panel"],
                                )
                            elif "xyzcal.mm" in reflection:
                                detector = self.pyslip.tiles.raw_image.get_detector()
                                x, y = detector[
                                    reflection["panel"]
                                ].millimeter_to_pixel(reflection["xyzcal.mm"][:2])
                                x, y = self.map_coords(x, y, reflection["panel"])
                            if x is None:
                                next

                            if self.settings.show_predictions:
                                predictions_data.append(
                                    (x, y, {"colour": self.prediction_colours[i_expt]})
                                )

                            if (
                                self.settings.show_miller_indices
                                and "miller_index" in reflection
                                and reflection["miller_index"] != (0, 0, 0)
                            ):
                                miller_indices_data.append(
                                    (
                                        x,
                                        y,
                                        str(reflection["miller_index"]),
                                        {
                                            "placement": "ne",
                                            "radius": 0,
                                        },
                                    )
                                )

        if len(overlapped_data) > 0:
            # show overlapped pixels in a different color
            all_pix_data[max(all_pix_data.keys()) + 1] = overlapped_data

        return {
            "shoebox_data": shoebox_data,
            "all_pix_data": all_pix_data,
            "all_foreground_circles": all_foreground_circles,
            "overlapped_data": overlapped_data,
            "ctr_mass_data": ctr_mass_data,
            "max_pix_data": max_pix_data,
            "predictions_data": predictions_data,
            "miller_indices_data": miller_indices_data,
        }

    def get_spotfinder_data(self):
        self.prediction_colours = [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffff33",
            "#a65628",
            "#f781bf",
            "#999999",
        ] * 10

        if self.viewing_stills:
            i_frame = self.images.selected_index  # NOTE, the underbar is intentional
        else:
            i_frame = self.images.selected.index
        imageset = self.images.selected.image_set
        if imageset.get_scan() is not None:
            i_frame += imageset.get_scan().get_array_range()[0]

        refl_data = self._reflection_overlay_data(i_frame)

        vector_data = []
        vector_text_data = []
        if self.settings.show_rotation_axis:
            axis_data = self._rotation_axis_overlay_data()
            if axis_data:
                vector_data.append(axis_data[0])
                vector_text_data.append(axis_data[1])

        return SpotfinderData(
            all_pix_data=refl_data["all_pix_data"],
            all_foreground_circles=refl_data["all_foreground_circles"],
            shoebox_data=refl_data["shoebox_data"],
            ctr_mass_data=refl_data["ctr_mass_data"],
            max_pix_data=refl_data["max_pix_data"],
            predictions_data=refl_data["predictions_data"],
            miller_indices_data=refl_data["miller_indices_data"],
            vector_data=vector_data,
            vector_text_data=vector_text_data,
        )

    def get_detector(self):
        return self.imagesets[0].get_detector()

    def get_beam(self):
        return self.imagesets[0].get_beam()

    def predict(self):
        predicted_all = []
        for experiments in self.experiments:
            this_predicted = flex.reflection_table()
            for i_expt, expt in enumerate(experiments):
                # Populate the reflection table with predictions
                params = self.params.prediction
                predicted = flex.reflection_table.from_predictions(
                    expt, force_static=params.force_static, dmin=params.d_min
                )
                predicted["id"] = flex.int(len(predicted), i_expt)
                if expt.profile is not None:
                    expt.profile.params = self.params.profile
                try:
                    predicted.compute_bbox(ExperimentList([expt]))
                except Exception:
                    pass
                this_predicted.extend(predicted)
            predicted_all.append(this_predicted)

        return predicted_all

    def OnZeroMQEvent(self, event):
        message = event.message
        print("ZMQ Event received by gui:", message)
        try:
            if message["command"] == "load_image":
                filename = message["image"]
                self.load_image(filename)
        except Exception:
            print("Error parsing zeromq message")
            raise


class SpotSettingsFrame(wx.MiniFrame):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.settings = self.GetParent().settings
        self.params = self.GetParent().params
        szr = wx.BoxSizer(wx.VERTICAL)
        panel = SpotSettingsPanel(self, -1)
        self.SetSizer(szr)
        szr.Add(panel, 1, wx.EXPAND)
        szr.Fit(panel)
        self.panel = panel
        self.sizer = szr
        self.Fit()

        self.Bind(wx.EVT_WINDOW_DESTROY, self.OnDestroy)

    def OnDestroy(self, event):
        # Allow the panel to cleanup when destroying the Frame
        self.panel.OnDestroy(event)


class SpotSettingsPanel(wx.Panel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.settings = self.GetParent().settings
        self.params = self.GetParent().params

        # CONTROLS 4: additional settings for derived class
        self.settings.image_type = "corrected"
        self.settings.brightness = self.params.brightness
        self.settings.color_scheme = self.params.color_scheme
        self.settings.projection = self.params.projection
        self.settings.show_spotfinder_spots = False
        self.settings.show_dials_spotfinder_spots = True
        self.settings.show_resolution_rings = self.params.show_resolution_rings
        self.settings.untrusted = self.params.masking.untrusted
        self.settings.show_ice_rings = self.params.show_ice_rings
        self.settings.ice_rings = self.params.masking.ice_rings
        self.settings.show_ctr_mass = self.params.show_ctr_mass
        self.settings.show_max_pix = self.params.show_max_pix
        self.settings.show_all_pix = self.params.show_all_pix
        self.settings.show_threshold_pix = self.params.show_threshold_pix
        self.settings.show_shoebox = self.params.show_shoebox
        self.settings.show_indexed = self.params.show_indexed
        self.settings.show_integrated = self.params.show_integrated
        self.settings.show_predictions = self.params.show_predictions
        self.settings.show_miller_indices = self.params.show_miller_indices
        self.settings.fontsize = 10
        self.settings.show_mask = self.params.show_mask
        self.settings.show_rotation_axis = self.params.show_rotation_axis
        self.settings.display = self.params.display
        if self.settings.display == "global_threshold":
            self.settings.display = "global"
        self.settings.threshold_algorithm = "dispersion_extended"
        self.settings.nsigma_b = self.params.nsigma_b
        self.settings.nsigma_s = self.params.nsigma_s
        self.settings.global_threshold = self.params.global_threshold
        self.settings.kernel_size = self.params.kernel_size
        self.settings.min_local = self.params.min_local
        self.settings.gain = self.params.gain
        self.settings.n_iqr = self.params.n_iqr
        self.settings.blur = self.params.blur
        self.settings.n_bins = self.params.n_bins
        self.settings.find_spots_phil = "find_spots.phil"
        self.settings.show_beam_center = self.params.show_beam_center
        self._sizer = wx.BoxSizer(wx.VERTICAL)
        s = self._sizer
        self.SetSizer(self._sizer)

        grid = wx.FlexGridSizer(cols=2, rows=3, vgap=0, hgap=0)
        s.Add(grid)
        txt1 = wx.StaticText(self, -1, "Zoom level:")
        grid.Add(txt1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.levels = self.GetParent().GetParent().pyslip.tiles.levels
        # from scitbx.math import continued_fraction as cf
        # choices = ["%s" %(cf.from_real(2**l).as_rational()) for l in self.levels]
        choices = [f"{100 * 2 ** l:g}%" for l in self.levels]
        self.zoom_ctrl = wx.Choice(self, -1, choices=choices)
        self.zoom_ctrl.SetSelection(self.settings.zoom_level)
        grid.Add(self.zoom_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        txt11 = wx.StaticText(self, -1, "Color scheme:")
        grid.Add(txt11, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        color_schemes = ["grayscale", "rainbow", "heatmap", "invert"]
        self.color_ctrl = wx.Choice(self, -1, choices=color_schemes)
        self.color_ctrl.SetSelection(color_schemes.index(self.params.color_scheme))
        grid.Add(self.color_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self._sizer.Fit(self)

        txt12 = wx.StaticText(self, -1, "Projection:")
        projection_choices = ["lab", "image"]
        self.projection_ctrl = wx.Choice(self, -1, choices=projection_choices)
        if self.params.projection is None:
            self.projection_ctrl.SetSelection(1)
            self.projection_ctrl.Enable(False)
            txt12.Enable(False)
        else:
            self.projection_ctrl.SetSelection(
                projection_choices.index(self.params.projection)
            )
        grid.Add(txt12, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        grid.Add(self.projection_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self._sizer.Fit(self)

        box = wx.BoxSizer(wx.HORIZONTAL)
        s.Add(box)
        grid = wx.FlexGridSizer(cols=1, rows=2, vgap=0, hgap=0)
        box.Add(grid)
        txt2 = wx.StaticText(self, -1, "Brightness:")
        grid.Add(txt2, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        # Add a textual brightness control
        self.brightness_txt_ctrl = IntCtrl(
            self,
            value=self.settings.brightness,
            min=1,
            max=1000,
            name="brightness",
            style=wx.TE_PROCESS_ENTER,
        )
        grid.Add(self.brightness_txt_ctrl, 0, wx.ALL, 3)
        # Add a slider brightness control
        self.brightness_ctrl = wx.Slider(
            self, -1, size=(200, -1), style=wx.SL_AUTOTICKS | wx.SL_LABELS
        )
        self.brightness_ctrl.SetMin(1)
        self.brightness_ctrl.SetMax(1000)
        self.brightness_ctrl.SetValue(self.settings.brightness)
        self.brightness_ctrl.SetTickFreq(25)
        box.Add(self.brightness_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        grid = wx.FlexGridSizer(cols=2, rows=1, vgap=0, hgap=0)
        s.Add(grid)
        # Font size control
        txt = wx.StaticText(self, -1, "Font size:")
        grid.Add(txt, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        # Add a textual brightness control
        self.fontsize_ctrl = IntCtrl(
            self,
            value=self.settings.fontsize,
            min=8,
            max=32,
            name="Font size",
            style=wx.TE_PROCESS_ENTER,
        )
        grid.Add(self.fontsize_ctrl, 0, wx.ALL, 3)

        grid = wx.FlexGridSizer(cols=2, rows=8, vgap=0, hgap=0)
        s.Add(grid)

        # Resolution rings control
        self.resolution_rings_ctrl = wx.CheckBox(self, -1, "Show resolution rings")
        self.resolution_rings_ctrl.SetValue(self.settings.show_resolution_rings)
        grid.Add(self.resolution_rings_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Ice rings control
        self.ice_rings_ctrl = wx.CheckBox(self, -1, "Show ice rings")
        self.ice_rings_ctrl.SetValue(self.settings.show_ice_rings)
        grid.Add(self.ice_rings_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Center control
        self.center_ctrl = wx.CheckBox(self, -1, "Mark beam center")
        self.center_ctrl.SetValue(self.settings.show_beam_center)
        grid.Add(self.center_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Center of mass control
        self.ctr_mass = wx.CheckBox(self, -1, "Mark centers of mass")
        self.ctr_mass.SetValue(self.settings.show_ctr_mass)
        grid.Add(self.ctr_mass, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Max pixel control
        self.max_pix = wx.CheckBox(self, -1, "Spot max pixels")
        self.max_pix.SetValue(self.settings.show_max_pix)
        grid.Add(self.max_pix, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Spot pixels control
        self.all_pix = wx.CheckBox(self, -1, "Spot all pixels")
        self.all_pix.SetValue(self.settings.show_all_pix)
        grid.Add(self.all_pix, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Threshold control
        self.thresh_pix = wx.CheckBox(self, -1, "Threshold pixels")
        self.thresh_pix.SetValue(self.settings.show_threshold_pix)
        grid.Add(self.thresh_pix, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Spot shoebox control
        self.shoebox = wx.CheckBox(self, -1, "Draw reflection shoebox")
        self.shoebox.SetValue(self.settings.show_shoebox)
        grid.Add(self.shoebox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Spot predictions control
        self.predictions = wx.CheckBox(self, -1, "Show predictions")
        self.predictions.SetValue(self.settings.show_predictions)
        grid.Add(self.predictions, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Spot predictions control
        self.miller_indices = wx.CheckBox(self, -1, "Show hkl")
        self.miller_indices.SetValue(self.settings.show_miller_indices)
        grid.Add(self.miller_indices, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Spot predictions control
        self.show_mask = wx.CheckBox(self, -1, "Show mask")
        self.show_mask.SetValue(self.settings.show_mask)
        grid.Add(self.show_mask, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Integration shoeboxes only
        self.indexed = wx.CheckBox(self, -1, "Indexed only")
        self.indexed.SetValue(self.settings.show_indexed)
        grid.Add(self.indexed, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Integration shoeboxes only
        self.integrated = wx.CheckBox(self, -1, "Integrated only")
        self.integrated.SetValue(self.settings.show_integrated)
        grid.Add(self.integrated, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        # Toggle rotation axis display
        self.show_rotation_axis = wx.CheckBox(self, -1, "Rotation axis")
        self.show_rotation_axis.SetValue(self.settings.show_rotation_axis)
        grid.Add(self.show_rotation_axis, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        grid = wx.FlexGridSizer(cols=2, rows=1, vgap=0, hgap=0)
        self.clear_all_button = wx.Button(self, -1, "Clear all")
        grid.Add(self.clear_all_button, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.Bind(wx.EVT_BUTTON, self.OnClearAll, self.clear_all_button)
        s.Add(grid)

        # Minimum spot area control
        # box = wx.BoxSizer(wx.HORIZONTAL)
        # self.minspotarea_ctrl = PhilIntCtrl(self, -1, pos=(300,180), size=(80,-1),
        # value=self.GetParent().GetParent().horizons_phil.distl.minimum_spot_area,
        # name="Minimum spot area (pxls)")
        # self.minspotarea_ctrl.SetOptional(False)
        # box.Add(self.minspotarea_ctrl, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        # txtd = wx.StaticText(self, -1,  "Minimum spot area (pxls)",)
        # box.Add(txtd, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        # s.Add(box)

        # Stack type choice
        grid = wx.FlexGridSizer(cols=2, rows=1, vgap=0, hgap=0)
        txt1 = wx.StaticText(self, -1, "Stack type:")
        grid.Add(txt1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.stack_modes = ["max", "mean", "sum"]
        self.stack_mode_ctrl = wx.Choice(self, -1, choices=self.stack_modes)
        self.stack_mode_ctrl.SetSelection(
            self.stack_modes.index(self.params.stack_mode)
        )
        grid.Add(self.stack_mode_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        s.Add(grid)

        # Image type choice
        grid = wx.FlexGridSizer(cols=2, rows=1, vgap=0, hgap=0)
        txt1 = wx.StaticText(self, -1, "Image type:")
        grid.Add(txt1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.image_types = ["corrected", "raw"]
        self.image_type_ctrl = wx.Choice(self, -1, choices=self.image_types)
        self.image_type_ctrl.SetSelection(
            self.image_types.index(self.settings.image_type)
        )
        grid.Add(self.image_type_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        s.Add(grid)

        # Choice of thresholding algorithm
        grid = wx.FlexGridSizer(cols=2, rows=1, vgap=0, hgap=0)
        txt1 = wx.StaticText(self, -1, "Threshold algorithm:")
        grid.Add(txt1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.threshold_algorithm_types = [
            "dispersion",
            "dispersion_extended",
            "radial_profile",
        ]
        self.threshold_algorithm_ctrl = wx.Choice(
            self, -1, choices=self.threshold_algorithm_types
        )
        self.threshold_algorithm_ctrl.SetSelection(
            self.threshold_algorithm_types.index(self.settings.threshold_algorithm)
        )
        grid.Add(self.threshold_algorithm_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        s.Add(grid)

        # Spotfinding parameters relevant to dispersion algorithms
        self.dispersion_params_grid = wx.FlexGridSizer(cols=2, rows=6, vgap=0, hgap=0)
        s.Add(self.dispersion_params_grid)

        txt1 = wx.StaticText(self, -1, "Sigma background")
        self.dispersion_params_grid.Add(txt1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.nsigma_b_ctrl = FloatCtrl(
            self, value=self.settings.nsigma_b, name="sigma_background"
        )
        self.nsigma_b_ctrl.SetMin(0)
        self.dispersion_params_grid.Add(self.nsigma_b_ctrl, 0, wx.ALL, 3)

        txt2 = wx.StaticText(self, -1, "Sigma strong")
        self.dispersion_params_grid.Add(txt2, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.nsigma_s_ctrl = FloatCtrl(
            self, value=self.settings.nsigma_s, name="sigma_strong"
        )
        self.nsigma_s_ctrl.SetMin(0)
        self.dispersion_params_grid.Add(self.nsigma_s_ctrl, 0, wx.ALL, 3)

        txt1 = wx.StaticText(self, -1, "Global Threshold")
        self.dispersion_params_grid.Add(txt1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.global_threshold_ctrl = FloatCtrl(
            self, value=self.settings.global_threshold, name="global_threshold"
        )
        self.global_threshold_ctrl.SetMin(0)
        self.dispersion_params_grid.Add(self.global_threshold_ctrl, 0, wx.ALL, 3)

        txt4 = wx.StaticText(self, -1, "Min. local")
        self.dispersion_params_grid.Add(txt4, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.min_local_ctrl = PhilIntCtrl(
            self, value=self.settings.min_local, name="min_local"
        )
        self.min_local_ctrl.SetMin(0)
        self.dispersion_params_grid.Add(self.min_local_ctrl, 0, wx.ALL, 3)

        txt4 = wx.StaticText(self, -1, "Gain")
        self.dispersion_params_grid.Add(txt4, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.gain_ctrl = FloatCtrl(self, value=self.settings.gain, name="gain")
        self.gain_ctrl.SetMin(1e-6)
        self.dispersion_params_grid.Add(self.gain_ctrl, 0, wx.ALL, 3)

        txt3 = wx.StaticText(self, -1, "Kernel size")
        self.dispersion_params_grid.Add(txt3, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.kernel_size_ctrl = IntsCtrl(
            self, value=self.settings.kernel_size, name="kernel_size"
        )
        self.kernel_size_ctrl.SetSize(2)
        self.kernel_size_ctrl.SetMin(1)
        self.dispersion_params_grid.Add(self.kernel_size_ctrl, 0, wx.ALL, 3)

        self.Bind(
            EVT_PHIL_CONTROL, self.OnUpdateThresholdParameters, self.nsigma_b_ctrl
        )
        self.Bind(
            EVT_PHIL_CONTROL, self.OnUpdateThresholdParameters, self.nsigma_s_ctrl
        )
        self.Bind(
            EVT_PHIL_CONTROL,
            self.OnUpdateThresholdParameters,
            self.global_threshold_ctrl,
        )
        self.Bind(
            EVT_PHIL_CONTROL,
            self.OnUpdateThresholdParameters,
            self.kernel_size_ctrl,
        )
        self.Bind(
            EVT_PHIL_CONTROL, self.OnUpdateThresholdParameters, self.min_local_ctrl
        )
        self.Bind(EVT_PHIL_CONTROL, self.OnUpdateThresholdParameters, self.gain_ctrl)

        # Spotfinding parameters relevant to the radial_profile algorithm
        self.radial_profile_params_grid = wx.FlexGridSizer(
            cols=2, rows=3, vgap=0, hgap=0
        )
        s.Add(self.radial_profile_params_grid)

        txt1 = wx.StaticText(self, -1, "IQR multiplier")
        self.radial_profile_params_grid.Add(
            txt1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5
        )
        self.n_iqr_ctrl = FloatCtrl(
            self, value=self.settings.n_iqr, name="iqr_multiplier"
        )
        self.n_iqr_ctrl.SetMin(0)
        self.radial_profile_params_grid.Add(self.n_iqr_ctrl, 0, wx.ALL, 3)

        txt1 = wx.StaticText(self, -1, "Blur")
        self.radial_profile_params_grid.Add(
            txt1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5
        )
        self.blur_choices = [
            "None",
            "narrow",
            "wide",
        ]
        self.blur_ctrl = wx.Choice(self, -1, choices=self.blur_choices)
        self.blur_ctrl.SetSelection(self.blur_choices.index(str(self.settings.blur)))
        self.radial_profile_params_grid.Add(
            self.blur_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5
        )

        txt1 = wx.StaticText(self, -1, "N bins")
        self.radial_profile_params_grid.Add(
            txt1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5
        )
        self.n_bins_ctrl = PhilIntCtrl(self, value=self.settings.n_bins, name="n_bins")
        self.n_bins_ctrl.SetMin(10)
        self.radial_profile_params_grid.Add(self.n_bins_ctrl, 0, wx.ALL, 3)

        self.Bind(EVT_PHIL_CONTROL, self.OnUpdateThresholdParameters, self.n_iqr_ctrl)
        self.Bind(wx.EVT_CHOICE, self.OnUpdateThresholdParameters, self.blur_ctrl)
        self.Bind(
            EVT_PHIL_CONTROL,
            self.OnUpdateThresholdParameters,
            self.n_bins_ctrl,
        )

        # Save spotfinding PHIL control
        grid1 = wx.FlexGridSizer(cols=2, rows=1, vgap=0, hgap=0)
        s.Add(grid1)

        self.save_params_txt_ctrl = StrCtrl(
            self, value=self.settings.find_spots_phil, name="find_spots_phil"
        )

        grid1.Add(self.save_params_txt_ctrl, 0, wx.ALL, 3)
        self.Bind(EVT_PHIL_CONTROL, self.OnUpdate, self.save_params_txt_ctrl)

        self.save_params_button = wx.Button(self, -1, "Save")
        grid1.Add(self.save_params_button, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.Bind(wx.EVT_BUTTON, self.OnSaveFindSpotsParams, self.save_params_button)

        grid2 = wx.FlexGridSizer(cols=4, rows=2, vgap=0, hgap=0)
        s.Add(grid2)

        self.dispersion_buttons = []
        self.dispersion_labels = [
            "image",
            "mean",
            "variance",
            "dispersion",
            "sigma_b",
            "sigma_s",
            "global",
            "threshold",
        ]
        for label in self.dispersion_labels:
            btn = wx.ToggleButton(self, -1, label)
            self.dispersion_buttons.append(btn)
            grid2.Add(btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
            self.Bind(wx.EVT_TOGGLEBUTTON, self.OnDispersionThresholdDebug, btn)

        for label, button in zip(self.dispersion_labels, self.dispersion_buttons):
            if self.params.stack_images > 1:
                button.Disable()
        for button in self.dispersion_buttons:
            if button.GetLabelText() == self.settings.display:
                button.SetValue(True)
                break

        self.collect_values()

        # Hide parameters for deselected threshold algorithm
        self._toggle_params_grid(self.settings.threshold_algorithm)

        # CONTROLS 3:  Bind events to actions

        self.Bind(wx.EVT_TEXT_ENTER, self.OnUpdate, self.fontsize_ctrl)
        self.fontsize_ctrl.Bind(wx.EVT_KILL_FOCUS, self.OnUpdate)

        # Brightness-related events
        self.Bind(wx.EVT_SCROLL_CHANGED, self.OnUpdateBrightness, self.brightness_ctrl)
        self.Bind(wx.EVT_SLIDER, self.OnUpdateBrightness, self.brightness_ctrl)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnUpdateBrightness, self.brightness_txt_ctrl)
        self.brightness_txt_ctrl.Bind(wx.EVT_KILL_FOCUS, self.OnUpdateBrightness)

        self.Bind(wx.EVT_CHOICE, self.OnUpdateZoomLevel, self.zoom_ctrl)
        self.Bind(wx.EVT_CHOICE, self.OnUpdateImage, self.image_type_ctrl)
        self.Bind(
            wx.EVT_CHOICE,
            self.OnUpdateThresholdAlgorithm,
            self.threshold_algorithm_ctrl,
        )
        self.Bind(wx.EVT_CHOICE, self.OnUpdateImage, self.stack_mode_ctrl)
        self.Bind(wx.EVT_CHOICE, self.OnUpdate, self.color_ctrl)
        self.Bind(wx.EVT_CHOICE, self.OnUpdateProjection, self.projection_ctrl)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.resolution_rings_ctrl)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.ice_rings_ctrl)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.center_ctrl)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.ctr_mass)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.max_pix)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.all_pix)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.thresh_pix)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.shoebox)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.predictions)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.miller_indices)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.indexed)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.integrated)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdate, self.show_rotation_axis)
        self.Bind(wx.EVT_CHECKBOX, self.OnUpdateShowMask, self.show_mask)

        self.Bind(wx.EVT_UPDATE_UI, self.UpdateZoomCtrl)

    def OnDestroy(self, event):
        "Handle any cleanup when the windows is being destroyed. Manually Called."
        # If we don't remove this here, then we can get called after destroy
        self.brightness_txt_ctrl.Unbind(wx.EVT_KILL_FOCUS)

    # CONTROLS 2:  Fetch values from widgets
    def collect_values(self):
        if self.settings.enable_collect_values:
            self.settings.image_type = self.image_types[
                self.image_type_ctrl.GetSelection()
            ]
            self.params.stack_mode = self.stack_modes[
                self.stack_mode_ctrl.GetSelection()
            ]
            self.settings.show_resolution_rings = self.resolution_rings_ctrl.GetValue()
            self.settings.show_ice_rings = self.ice_rings_ctrl.GetValue()
            self.settings.zoom_level = self.levels[self.zoom_ctrl.GetSelection()]

            # Brightness has its own handler, so just make sure the controls are synced
            if self.brightness_txt_ctrl.GetValue() != self.settings.brightness:
                try:
                    self.brightness_txt_ctrl.ChangeValue(self.settings.brightness)
                except Exception:  # workaround for wxPython 2.8
                    self.brightness_txt_ctrl.ChangeValue(str(self.settings.brightness))
            if self.brightness_ctrl.GetValue() != self.settings.brightness:
                self.brightness_ctrl.SetValue(self.settings.brightness)

            self.settings.show_beam_center = self.center_ctrl.GetValue()
            self.settings.show_ctr_mass = self.ctr_mass.GetValue()
            self.settings.show_max_pix = self.max_pix.GetValue()
            self.settings.show_all_pix = self.all_pix.GetValue()
            self.settings.show_threshold_pix = self.thresh_pix.GetValue()
            self.settings.show_shoebox = self.shoebox.GetValue()
            self.settings.show_indexed = self.indexed.GetValue()
            self.settings.show_integrated = self.integrated.GetValue()
            self.settings.show_predictions = self.predictions.GetValue()
            self.settings.show_miller_indices = self.miller_indices.GetValue()
            self.settings.fontsize = self.fontsize_ctrl.GetValue()
            self.settings.show_mask = self.show_mask.GetValue()
            self.settings.show_rotation_axis = self.show_rotation_axis.GetValue()
            self.settings.threshold_algorithm = self.threshold_algorithm_types[
                self.threshold_algorithm_ctrl.GetSelection()
            ]
            self.settings.color_scheme = self.color_ctrl.GetSelection()
            self.settings.projection = self.projection_ctrl.GetSelection()
            self.settings.nsigma_b = self.nsigma_b_ctrl.GetPhilValue()
            self.settings.nsigma_s = self.nsigma_s_ctrl.GetPhilValue()
            self.settings.global_threshold = self.global_threshold_ctrl.GetPhilValue()
            self.settings.kernel_size = self.kernel_size_ctrl.GetPhilValue()
            self.settings.min_local = self.min_local_ctrl.GetPhilValue()
            self.settings.gain = self.gain_ctrl.GetPhilValue()
            self.settings.n_iqr = self.n_iqr_ctrl.GetPhilValue()
            self.settings.blur = self.blur_choices[self.blur_ctrl.GetSelection()]
            self.settings.n_bins = self.n_bins_ctrl.GetPhilValue()

            self.settings.find_spots_phil = self.save_params_txt_ctrl.GetPhilValue()

    def UpdateZoomCtrl(self, event):
        self.settings.zoom_level = self.levels.index(
            self.GetParent().GetParent().pyslip.level
        )
        self.zoom_ctrl.SetSelection(self.settings.zoom_level)

    def OnUpdate(self, event):
        """Collects all settings from the GUI and forwards to the viewer"""
        self.collect_values()
        self.GetParent().GetParent().update_settings()

    def OnUpdateImage(self, event):
        """Forces an update of the image"""
        self.OnUpdate(event)
        self.GetParent().GetParent().reload_image()

    def OnUpdateBrightness(self, event):
        """Handle updates from the brightness-related controls"""

        # Don't update whilst dragging the slider
        if event.GetEventType() == wx.EVT_SLIDER.typeId:
            if wx.GetMouseState().LeftIsDown():
                return

        # For e.g. IntCtrl check the value is valid
        if hasattr(event.EventObject, "IsInBounds"):
            if not event.EventObject.IsInBounds():
                return

        # Read the new value then update everything if we need to
        if self.settings.brightness != event.EventObject.GetValue():
            self.settings.brightness = event.EventObject.GetValue()
            self.OnUpdate(event)

    def OnUpdateShowMask(self, event):
        self.OnUpdate(event)
        self.params.show_mask = self.settings.show_mask
        self.GetParent().GetParent().reload_image()

    def OnClearAll(self, event):
        for btn in (
            self.center_ctrl,
            self.ctr_mass,
            self.max_pix,
            self.all_pix,
            self.thresh_pix,
            self.shoebox,
            self.predictions,
            self.miller_indices,
            self.show_mask,
            self.show_rotation_axis,
            self.ice_rings_ctrl,
            self.resolution_rings_ctrl,
        ):
            btn.SetValue(False)
        self.OnUpdate(event)

    def OnUpdateZoomLevel(self, event):
        self.collect_values()
        pyslip = self.GetParent().GetParent().pyslip

        # get center of view in map coords
        x, y = pyslip.view_width / 2, pyslip.view_height / 2
        center = pyslip.ConvertView2Geo((x, y))
        pyslip.ZoomToLevel(self.settings.zoom_level)
        pyslip.ZoomIn((x, y), update=False)
        pyslip.GotoPosition(center)

    def _toggle_params_grid(self, to_show):
        if to_show == "radial_profile":
            self.dispersion_params_grid.ShowItems(False)
            self.radial_profile_params_grid.ShowItems(True)
        else:
            self.dispersion_params_grid.ShowItems(True)
            self.radial_profile_params_grid.ShowItems(False)
        self._sizer.Layout()

    def OnUpdateThresholdAlgorithm(self, event):
        self._toggle_params_grid(event.GetString())
        self.OnUpdateThresholdParameters(event)

    def OnUpdateProjection(self, event):
        self.params.projection = event.GetString()
        self.OnUpdateImage(event)

    def OnSaveFindSpotsParams(self, event):
        params = find_spots_phil_scope.extract()
        threshold = params.spotfinder.threshold
        threshold.algorithm = self.settings.threshold_algorithm

        dispersion = threshold.dispersion
        dispersion.gain = self.settings.gain
        dispersion.global_threshold = self.settings.global_threshold
        dispersion.kernel_size = self.settings.kernel_size
        dispersion.min_local = self.settings.min_local
        dispersion.sigma_background = self.settings.nsigma_b
        dispersion.sigma_strong = self.settings.nsigma_s

        radial_profile = threshold.radial_profile
        radial_profile.n_iqr = self.settings.n_iqr
        if self.settings.blur == "None":
            radial_profile.blur = None
        else:
            radial_profile.blur = self.settings.blur
        radial_profile.n_bins = self.settings.n_bins

        print(f"Saving parameters to {self.settings.find_spots_phil}")
        with open(self.settings.find_spots_phil, "w") as f:
            find_spots_phil_scope.fetch_diff(find_spots_phil_scope.format(params)).show(
                f
            )

    def OnUpdateThresholdParameters(self, event):
        self.GetParent().GetParent().show_filters()
        self.OnUpdateImage(event)

    def OnDispersionThresholdDebug(self, event):
        button = event.GetEventObject()
        selected = button.GetLabelText()

        self.settings.display = selected

        # Disable corrected/raw selection when showing dispersion debug images
        if self.settings.display != "image":
            self.image_type_ctrl.SetSelection(0)
            self.image_type_ctrl.Disable()
        else:
            self.image_type_ctrl.Enable()

        # reset buttons
        for btn in self.dispersion_buttons:
            if btn.GetLabelText() == selected:
                btn.SetValue(True)
            else:
                btn.SetValue(False)

        self.GetParent().GetParent().show_filters()
