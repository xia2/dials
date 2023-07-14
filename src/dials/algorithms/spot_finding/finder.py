"""
Contains implementation interface for finding spots on one or many images
"""

from __future__ import annotations

import logging
import math
import pickle
from typing import Iterable, Tuple

import libtbx
from dxtbx.format.image import ImageBool
from dxtbx.imageset import ImageSequence, ImageSet
from dxtbx.model import ExperimentList

from dials.array_family import flex
from dials.model.data import PixelList, PixelListLabeller
from dials.util import Sorry, log
from dials.util.log import rehandle_cached_records
from dials.util.mp import available_cores, batch_multi_node_parallel_map

logger = logging.getLogger(__name__)


class ExtractPixelsFromImage:
    """
    A class to extract pixels from a single image
    """

    def __init__(
        self,
        imageset,
        threshold_function,
        mask,
        region_of_interest,
        max_strong_pixel_fraction,
        compute_mean_background,
    ):
        """
        Initialise the class

        :param imageset: The imageset to extract from
        :param threshold_function: The function to threshold with
        :param mask: The image mask
        :param region_of_interest: A region of interest to process
        :param max_strong_pixel_fraction: The maximum fraction of pixels allowed
        """
        self.threshold_function = threshold_function
        self.imageset = imageset
        self.mask = mask
        self.region_of_interest = region_of_interest
        self.max_strong_pixel_fraction = max_strong_pixel_fraction
        self.compute_mean_background = compute_mean_background
        if self.mask is not None:
            detector = self.imageset.get_detector()
            assert len(self.mask) == len(detector)

    def __call__(self, index):
        """
        Extract strong pixels from an image

        :param index: The index of the image
        """
        # Get the frame number
        if isinstance(self.imageset, ImageSequence):
            frame = self.imageset.get_array_range()[0] + index
        else:
            ind = self.imageset.indices()
            if len(ind) > 1:
                assert all(i1 + 1 == i2 for i1, i2 in zip(ind[0:-1], ind[1:-1]))
            frame = ind[index]

        # Create the list of pixel lists
        pixel_list = []

        # Get the image and mask
        image = self.imageset.get_corrected_data(index)
        mask = self.imageset.get_mask(index)

        # Set the mask
        if self.mask is not None:
            assert len(self.mask) == len(mask)
            mask = tuple(m1 & m2 for m1, m2 in zip(mask, self.mask))

        logger.debug(
            "Number of masked pixels for image %i: %i",
            index,
            sum(m.count(False) for m in mask),
        )

        # Add the images to the pixel lists
        num_strong = 0
        average_background = 0
        for i_panel, (im, mk) in enumerate(zip(image, mask)):
            if self.imageset.is_marked_for_rejection(index):
                threshold_mask = flex.bool(im.accessor(), False)
            elif self.region_of_interest is not None:
                x0, x1, y0, y1 = self.region_of_interest
                height, width = im.all()
                assert x0 < x1, "x0 < x1"
                assert y0 < y1, "y0 < y1"
                assert x0 >= 0, "x0 >= 0"
                assert y0 >= 0, "y0 >= 0"
                assert x1 <= width, "x1 <= width"
                assert y1 <= height, "y1 <= height"
                im_roi = im[y0:y1, x0:x1]
                mk_roi = mk[y0:y1, x0:x1]
                tm_roi = self.threshold_function.compute_threshold(
                    im_roi,
                    mk_roi,
                    imageset=self.imageset,
                    i_panel=i_panel,
                    region_of_interest=self.region_of_interest,
                )
                threshold_mask = flex.bool(im.accessor(), False)
                threshold_mask[y0:y1, x0:x1] = tm_roi
            else:
                threshold_mask = self.threshold_function.compute_threshold(
                    im, mk, imageset=self.imageset, i_panel=i_panel
                )

            # Add the pixel list
            plist = PixelList(frame, im, threshold_mask)
            pixel_list.append(plist)

            # Get average background
            if self.compute_mean_background:
                background = im.as_1d().select((mk & ~threshold_mask).as_1d())
                average_background += flex.mean(background)

            # Add to the spot count
            num_strong += len(plist)

        # Make average background
        average_background /= len(image)

        # Check total number of strong pixels
        if self.max_strong_pixel_fraction < 1:
            num_image = 0
            for im in image:
                num_image += len(im)
            max_strong = int(math.ceil(self.max_strong_pixel_fraction * num_image))
            if num_strong > max_strong:
                raise RuntimeError(
                    f"""
          The number of strong pixels found ({num_strong}) is greater than the
          maximum allowed ({max_strong}). Try changing spot finding parameters
        """
                )

        # Print some info
        if self.compute_mean_background:
            logger.info(
                "Found %d strong pixels on image %d with average background %f",
                num_strong,
                frame + 1,
                average_background,
            )
        else:
            logger.info("Found %d strong pixels on image %d", num_strong, frame + 1)

        # Return the result
        return pixel_list


class ExtractPixelsFromImage2DNoShoeboxes(ExtractPixelsFromImage):
    """
    A class to extract pixels from a single image
    """

    def __init__(
        self,
        imageset,
        threshold_function,
        mask,
        region_of_interest,
        max_strong_pixel_fraction,
        compute_mean_background,
        min_spot_size,
        max_spot_size,
        filter_spots,
    ):
        """
        Initialise the class

        :param imageset: The imageset to extract from
        :param threshold_function: The function to threshold with
        :param mask: The image mask
        :param region_of_interest: A region of interest to process
        :param max_strong_pixel_fraction: The maximum fraction of pixels allowed
        """
        super().__init__(
            imageset,
            threshold_function,
            mask,
            region_of_interest,
            max_strong_pixel_fraction,
            compute_mean_background,
        )

        # Save some stuff
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.filter_spots = filter_spots

    def __call__(self, index):
        """
        Extract strong pixels from an image

        :param index: The index of the image
        """
        # Initialise the pixel labeller
        num_panels = len(self.imageset.get_detector())
        pixel_labeller = [PixelListLabeller() for p in range(num_panels)]

        # Call the super function
        result = super().__call__(index)

        # Add pixel lists to the labeller
        assert len(pixel_labeller) == len(result), "Inconsistent size"
        for plabeller, plist in zip(pixel_labeller, result):
            plabeller.add(plist)

        # Create shoeboxes from pixel list
        reflections, _ = pixel_list_to_reflection_table(
            self.imageset,
            pixel_labeller,
            filter_spots=self.filter_spots,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            write_hot_pixel_mask=False,
        )

        # Delete the shoeboxes
        del reflections["shoeboxes"]

        # Return the reflections
        return [reflections]


class ExtractSpotsParallelTask:
    """
    Execute the spot finder task in parallel

    We need this external class so that we can pickle it for cluster jobs
    """

    def __init__(self, function):
        """
        Initialise with the function to call
        """
        self.function = function

    def __call__(self, task):
        """
        Call the function with th task and save the IO
        """
        log.config_simple_cached()
        result = self.function(task)
        handlers = logging.getLogger("dials").handlers
        assert len(handlers) == 1, "Invalid number of logging handlers"
        return result, handlers[0].records


def pixel_list_to_shoeboxes(
    imageset: ImageSet,
    pixel_labeller: Iterable[PixelListLabeller],
    min_spot_size: int,
    max_spot_size: int,
    write_hot_pixel_mask: bool,
) -> Tuple[flex.shoebox, Tuple[flex.size_t, ...]]:
    """Convert a pixel list to shoeboxes"""
    # Extract the pixel lists into a list of reflections
    shoeboxes = flex.shoebox()
    spotsizes = flex.size_t()
    hotpixels = tuple(flex.size_t() for i in range(len(imageset.get_detector())))
    if isinstance(imageset, ImageSequence):
        twod = imageset.get_scan().is_still()
    else:
        twod = True
    for i, (p, hp) in enumerate(zip(pixel_labeller, hotpixels)):
        if p.num_pixels() > 0:
            creator = flex.PixelListShoeboxCreator(
                p,
                i,  # panel
                0,  # zrange
                twod,  # twod
                min_spot_size,  # min_pixels
                max_spot_size,  # max_pixels
                write_hot_pixel_mask,
            )
            shoeboxes.extend(creator.result())
            spotsizes.extend(creator.spot_size())
            hp.extend(creator.hot_pixels())
    logger.info("\nExtracted %d spots", len(shoeboxes))

    # Get the unallocated spots and print some info
    selection = shoeboxes.is_allocated()
    shoeboxes = shoeboxes.select(selection)
    ntoosmall = (spotsizes < min_spot_size).count(True)
    ntoolarge = (spotsizes > max_spot_size).count(True)
    assert ntoosmall + ntoolarge == selection.count(False)
    logger.info("Removed %d spots with size < %d pixels", ntoosmall, min_spot_size)
    logger.info("Removed %d spots with size > %d pixels", ntoolarge, max_spot_size)

    # Return the shoeboxes
    return shoeboxes, hotpixels


def shoeboxes_to_reflection_table(
    imageset: ImageSet, shoeboxes: flex.shoebox, filter_spots
) -> flex.reflection_table:
    """Filter shoeboxes and create reflection table"""
    # Calculate the spot centroids
    centroid = shoeboxes.centroid_valid()
    logger.info("Calculated %d spot centroids", len(shoeboxes))

    # Calculate the spot intensities
    intensity = shoeboxes.summed_intensity()
    logger.info("Calculated %d spot intensities", len(shoeboxes))

    # Create the observations
    observed = flex.observation(shoeboxes.panels(), centroid, intensity)

    # Filter the reflections and select only the desired spots
    flags = filter_spots(
        None, sweep=imageset, observations=observed, shoeboxes=shoeboxes
    )
    observed = observed.select(flags)
    shoeboxes = shoeboxes.select(flags)

    # Return as a reflection list
    return flex.reflection_table(observed, shoeboxes)


def pixel_list_to_reflection_table(
    imageset: ImageSet,
    pixel_labeller: Iterable[PixelListLabeller],
    filter_spots,
    min_spot_size: int,
    max_spot_size: int,
    write_hot_pixel_mask: bool,
) -> Tuple[flex.shoebox, Tuple[flex.size_t, ...]]:
    """Convert pixel list to reflection table"""
    shoeboxes, hot_pixels = pixel_list_to_shoeboxes(
        imageset,
        pixel_labeller,
        min_spot_size=min_spot_size,
        max_spot_size=max_spot_size,
        write_hot_pixel_mask=write_hot_pixel_mask,
    )
    # Setup the reflection table converter
    return (
        shoeboxes_to_reflection_table(imageset, shoeboxes, filter_spots=filter_spots),
        hot_pixels,
    )


class ExtractSpots:
    """
    Class to find spots in an image and extract them into shoeboxes.
    """

    def __init__(
        self,
        threshold_function=None,
        mask=None,
        region_of_interest=None,
        max_strong_pixel_fraction=0.1,
        compute_mean_background=False,
        mp_method=None,
        mp_nproc=1,
        mp_njobs=1,
        mp_chunksize=1,
        min_spot_size=1,
        max_spot_size=20,
        filter_spots=None,
        no_shoeboxes_2d=False,
        min_chunksize=50,
        write_hot_pixel_mask=False,
    ):
        """
        Initialise the class with the strategy

        :param threshold_function: The image thresholding strategy
        :param mask: The mask to use
        :param mp_method: The multi processing method
        :param nproc: The number of processors
        :param max_strong_pixel_fraction: The maximum number of strong pixels
        """
        # Set the required strategies
        self.threshold_function = threshold_function
        self.mask = mask
        self.mp_method = mp_method
        self.mp_chunksize = mp_chunksize
        self.mp_nproc = mp_nproc
        self.mp_njobs = mp_njobs
        self.max_strong_pixel_fraction = max_strong_pixel_fraction
        self.compute_mean_background = compute_mean_background
        self.region_of_interest = region_of_interest
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.filter_spots = filter_spots
        self.no_shoeboxes_2d = no_shoeboxes_2d
        self.min_chunksize = min_chunksize
        self.write_hot_pixel_mask = write_hot_pixel_mask

    def __call__(self, imageset):
        """
        Find the spots in the imageset

        :param imageset: The imageset to process
        :return: The list of spot shoeboxes
        """
        if not self.no_shoeboxes_2d:
            return self._find_spots(imageset)
        else:
            return self._find_spots_2d_no_shoeboxes(imageset)

    def _compute_chunksize(self, nimg, nproc, min_chunksize):
        """
        Compute the chunk size for a given number of images and processes
        """
        chunksize = int(math.ceil(nimg / nproc))
        remainder = nimg % (chunksize * nproc)
        test_chunksize = chunksize - 1
        while test_chunksize >= min_chunksize:
            test_remainder = nimg % (test_chunksize * nproc)
            if test_remainder <= remainder:
                chunksize = test_chunksize
                remainder = test_remainder
            test_chunksize -= 1
        return chunksize

    def _find_spots(self, imageset):
        """
        Find the spots in the imageset

        :param imageset: The imageset to process
        :return: The list of spot shoeboxes
        """
        # Change the number of processors if necessary
        mp_nproc = self.mp_nproc
        mp_njobs = self.mp_njobs
        if mp_nproc is libtbx.Auto:
            mp_nproc = available_cores()
            logger.info(f"Setting nproc={mp_nproc}")
        if mp_nproc * mp_njobs > len(imageset):
            mp_nproc = min(mp_nproc, len(imageset))
            mp_njobs = int(math.ceil(len(imageset) / mp_nproc))

        mp_method = self.mp_method
        mp_chunksize = self.mp_chunksize

        if mp_chunksize is libtbx.Auto:
            mp_chunksize = self._compute_chunksize(
                len(imageset), mp_njobs * mp_nproc, self.min_chunksize
            )
            logger.info("Setting chunksize=%i", mp_chunksize)

        len_by_nproc = int(math.floor(len(imageset) / (mp_njobs * mp_nproc)))
        if mp_chunksize > len_by_nproc:
            mp_chunksize = len_by_nproc
        if mp_chunksize == 0:
            mp_chunksize = 1
        assert mp_nproc > 0, "Invalid number of processors"
        assert mp_njobs > 0, "Invalid number of jobs"
        assert mp_njobs == 1 or mp_method is not None, "Invalid cluster method"
        assert mp_chunksize > 0, "Invalid chunk size"

        # The extract pixels function
        function = ExtractPixelsFromImage(
            imageset=imageset,
            threshold_function=self.threshold_function,
            mask=self.mask,
            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
            compute_mean_background=self.compute_mean_background,
            region_of_interest=self.region_of_interest,
        )

        # The indices to iterate over
        indices = list(range(len(imageset)))

        # Initialise the pixel labeller
        num_panels = len(imageset.get_detector())
        pixel_labeller = [PixelListLabeller() for p in range(num_panels)]

        # Do the processing
        logger.info("Extracting strong pixels from images")
        if mp_njobs > 1:
            logger.info(
                " Using %s with %d parallel job(s) and %d processes per node\n",
                mp_method,
                mp_njobs,
                mp_nproc,
            )
        else:
            logger.info(" Using multiprocessing with %d parallel job(s)\n", mp_nproc)
        if mp_nproc > 1 or mp_njobs > 1:

            def process_output(result):
                rehandle_cached_records(result[1])
                assert len(pixel_labeller) == len(result[0]), "Inconsistent size"
                for plabeller, plist in zip(pixel_labeller, result[0]):
                    plabeller.add(plist)

            batch_multi_node_parallel_map(
                func=ExtractSpotsParallelTask(function),
                iterable=indices,
                nproc=mp_nproc,
                njobs=mp_njobs,
                cluster_method=mp_method,
                chunksize=mp_chunksize,
                callback=process_output,
            )
        else:
            for task in indices:
                result = function(task)
                assert len(pixel_labeller) == len(result), "Inconsistent size"
                for plabeller, plist in zip(pixel_labeller, result):
                    plabeller.add(plist)
                result.clear()

        # Create shoeboxes from pixel list
        return pixel_list_to_reflection_table(
            imageset,
            pixel_labeller,
            filter_spots=self.filter_spots,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            write_hot_pixel_mask=self.write_hot_pixel_mask,
        )

    def _find_spots_2d_no_shoeboxes(self, imageset):
        """
        Find the spots in the imageset

        :param imageset: The imageset to process
        :return: The list of spot shoeboxes
        """
        # Change the number of processors if necessary
        mp_nproc = self.mp_nproc
        mp_njobs = self.mp_njobs
        if mp_nproc * mp_njobs > len(imageset):
            mp_nproc = min(mp_nproc, len(imageset))
            mp_njobs = int(math.ceil(len(imageset) / mp_nproc))

        mp_method = self.mp_method
        mp_chunksize = self.mp_chunksize

        if mp_chunksize == libtbx.Auto:
            mp_chunksize = self._compute_chunksize(
                len(imageset), mp_njobs * mp_nproc, self.min_chunksize
            )
            logger.info("Setting chunksize=%i", mp_chunksize)

        len_by_nproc = int(math.floor(len(imageset) / (mp_njobs * mp_nproc)))
        if mp_chunksize > len_by_nproc:
            mp_chunksize = len_by_nproc
        assert mp_nproc > 0, "Invalid number of processors"
        assert mp_njobs > 0, "Invalid number of jobs"
        assert mp_njobs == 1 or mp_method is not None, "Invalid cluster method"
        assert mp_chunksize > 0, "Invalid chunk size"

        # The extract pixels function
        function = ExtractPixelsFromImage2DNoShoeboxes(
            imageset=imageset,
            threshold_function=self.threshold_function,
            mask=self.mask,
            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
            compute_mean_background=self.compute_mean_background,
            region_of_interest=self.region_of_interest,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            filter_spots=self.filter_spots,
        )

        # The indices to iterate over
        indices = list(range(len(imageset)))

        # The resulting reflections
        reflections = flex.reflection_table()

        # Do the processing
        logger.info("Extracting strong spots from images")
        if mp_njobs > 1:
            logger.info(
                " Using %s with %d parallel job(s) and %d processes per node\n",
                mp_method,
                mp_njobs,
                mp_nproc,
            )
        else:
            logger.info(" Using multiprocessing with %d parallel job(s)\n", mp_nproc)
        if mp_nproc > 1 or mp_njobs > 1:

            def process_output(result):
                for message in result[1]:
                    logger.log(message.levelno, message.msg)
                reflections.extend(result[0][0])
                result[0][0] = None

            batch_multi_node_parallel_map(
                func=ExtractSpotsParallelTask(function),
                iterable=indices,
                nproc=mp_nproc,
                njobs=mp_njobs,
                cluster_method=mp_method,
                chunksize=mp_chunksize,
                callback=process_output,
            )
        else:
            for task in indices:
                reflections.extend(function(task)[0])

        # Return the reflections
        return reflections, None


class SpotFinder:
    """
    A class to do spot finding and filtering.
    """

    def __init__(
        self,
        threshold_function=None,
        mask=None,
        region_of_interest=None,
        max_strong_pixel_fraction=0.1,
        compute_mean_background=False,
        mp_method=None,
        mp_nproc=1,
        mp_njobs=1,
        mp_chunksize=1,
        mask_generator=None,
        filter_spots=None,
        scan_range=None,
        write_hot_mask=True,
        hot_mask_prefix="hot_mask",
        min_spot_size=1,
        max_spot_size=20,
        no_shoeboxes_2d=False,
        min_chunksize=50,
        is_stills=False,
    ):
        """
        Initialise the class.

        :param find_spots: The spot finding algorithm
        :param filter_spots: The spot filtering algorithm
        :param scan_range: The scan range to find spots over
        :param is_stills:   [ADVANCED] Force still-handling of experiment
                            ID remapping for dials.stills_process.
        """

        # Set the filter and some other stuff
        self.threshold_function = threshold_function
        self.mask = mask
        self.region_of_interest = region_of_interest
        self.max_strong_pixel_fraction = max_strong_pixel_fraction
        self.compute_mean_background = compute_mean_background
        self.mask_generator = mask_generator
        self.filter_spots = filter_spots
        self.scan_range = scan_range
        self.write_hot_mask = write_hot_mask
        self.hot_mask_prefix = hot_mask_prefix
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.mp_method = mp_method
        self.mp_chunksize = mp_chunksize
        self.mp_nproc = mp_nproc
        self.mp_njobs = mp_njobs
        self.no_shoeboxes_2d = no_shoeboxes_2d
        self.min_chunksize = min_chunksize
        self.is_stills = is_stills

    def find_spots(self, experiments: ExperimentList) -> flex.reflection_table:
        """
        Do spotfinding for a set of experiments.

        Args:
            experiments: The experiment list to process

        Returns:
            A new reflection table of found reflections
        """
        # Loop through all the experiments and get the unique imagesets
        imagesets = []
        for experiment in experiments:
            if experiment.imageset not in imagesets:
                imagesets.append(experiment.imageset)

        # Loop through all the imagesets and find the strong spots
        reflections = flex.reflection_table()

        for j, imageset in enumerate(imagesets):

            # Find the strong spots in the sequence
            logger.info(
                "-" * 80 + "\nFinding strong spots in imageset %d\n" + "-" * 80, j
            )
            table, hot_mask = self._find_spots_in_imageset(imageset)

            # Fix up the experiment ID's now
            table["id"] = flex.int(table.nrows(), -1)
            for i, experiment in enumerate(experiments):
                if experiment.imageset is not imageset:
                    continue
                if not self.is_stills and experiment.scan:
                    z0, z1 = experiment.scan.get_array_range()
                    z = table["xyzobs.px.value"].parts()[2]
                    table["id"].set_selected((z > z0) & (z < z1), i)
                    if experiment.identifier:
                        table.experiment_identifiers()[i] = experiment.identifier
                else:
                    table["id"] = flex.int(table.nrows(), j)
                    if experiment.identifier:
                        table.experiment_identifiers()[j] = experiment.identifier
            missed = table["id"] == -1
            assert missed.count(True) == 0, "Failed to remap {} experiment IDs".format(
                missed.count(True)
            )

            reflections.extend(table)
            # Write a hot pixel mask
            if self.write_hot_mask:
                if not imageset.external_lookup.mask.data.empty():
                    for m1, m2 in zip(hot_mask, imageset.external_lookup.mask.data):
                        m1 &= m2.data()
                    imageset.external_lookup.mask.data = ImageBool(hot_mask)
                else:
                    imageset.external_lookup.mask.data = ImageBool(hot_mask)
                imageset.external_lookup.mask.filename = "%s_%d.pickle" % (
                    self.hot_mask_prefix,
                    i,
                )

                # Write the hot mask
                with open(imageset.external_lookup.mask.filename, "wb") as outfile:
                    pickle.dump(hot_mask, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        # Set the strong spot flag
        reflections.set_flags(
            flex.size_t_range(len(reflections)), reflections.flags.strong
        )

        # Check for overloads
        reflections.is_overloaded(experiments)

        # Return the reflections
        return reflections

    def _find_spots_in_imageset(self, imageset):
        """
        Do the spot finding.

        :param imageset: The imageset to process
        :return: The observed spots
        """
        # The input mask
        mask = self.mask_generator(imageset)
        if self.mask is not None:
            mask = tuple(m1 & m2 for m1, m2 in zip(mask, self.mask))

        # Set the spot finding algorithm
        extract_spots = ExtractSpots(
            threshold_function=self.threshold_function,
            mask=mask,
            region_of_interest=self.region_of_interest,
            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
            compute_mean_background=self.compute_mean_background,
            mp_method=self.mp_method,
            mp_nproc=self.mp_nproc,
            mp_njobs=self.mp_njobs,
            mp_chunksize=self.mp_chunksize,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            filter_spots=self.filter_spots,
            no_shoeboxes_2d=self.no_shoeboxes_2d,
            min_chunksize=self.min_chunksize,
            write_hot_pixel_mask=self.write_hot_mask,
        )

        # Get the max scan range
        if isinstance(imageset, ImageSequence):
            max_scan_range = imageset.get_array_range()
        else:
            max_scan_range = (0, len(imageset))

        # Get list of scan ranges
        if not self.scan_range or self.scan_range[0] is None:
            scan_range = [(max_scan_range[0] + 1, max_scan_range[1])]
        else:
            scan_range = self.scan_range

        # Get spots from bits of scan
        hot_pixels = tuple(flex.size_t() for i in range(len(imageset.get_detector())))
        reflections = flex.reflection_table()
        for j0, j1 in scan_range:
            # Make sure we were asked to do something sensible
            if j1 < j0:
                raise Sorry("Scan range must be in ascending order")
            elif j0 < max_scan_range[0] or j1 > max_scan_range[1]:
                raise Sorry(
                    "Scan range must be within image range {}..{}".format(
                        max_scan_range[0] + 1, max_scan_range[1]
                    )
                )

            logger.info("\nFinding spots in image %s to %s...", j0, j1)
            j0 -= 1
            if len(imageset) == 1:
                r, h = extract_spots(imageset)
            else:
                r, h = extract_spots(imageset[j0:j1])
            reflections.extend(r)
            if h is not None:
                for h1, h2 in zip(hot_pixels, h):
                    h1.extend(h2)

        # Find hot pixels
        hot_mask = self._create_hot_mask(imageset, hot_pixels)

        # Return as a reflection list
        return reflections, hot_mask

    def _create_hot_mask(self, imageset, hot_pixels):
        """
        Find hot pixels in images
        """
        # Write the hot mask
        if self.write_hot_mask:

            # Create the hot pixel mask
            hot_mask = tuple(
                flex.bool(flex.grid(p.get_image_size()[::-1]), True)
                for p in imageset.get_detector()
            )
            num_hot = 0
            if hot_pixels:
                for hp, hm in zip(hot_pixels, hot_mask):
                    for i in range(len(hp)):
                        hm[hp[i]] = False
                    num_hot += len(hp)
            logger.info("Found %d possible hot pixel(s)", num_hot)

        else:
            hot_mask = None

        # Return the hot mask
        return hot_mask
