from __future__ import absolute_import, division, print_function

import math
import random

from dials.algorithms.profile_model.gaussian_rs import BBoxCalculator3D
from dials.algorithms.profile_model.gaussian_rs import CoordinateSystem
from dials.algorithms.profile_model.gaussian_rs import transform
from dials.array_family import flex
from dials.model.serialize import load
from scitbx import matrix


def evaluate_gaussian(x, a, x0, sx):
    assert len(x) == len(x0)
    assert len(x) == len(sx)

    g = 0.0
    for xi, x0i, sxi in zip(x, x0, sx):
        g += (xi - x0i) ** 2 / (2.0 * sxi ** 2)
    return a * math.exp(-g)


def gaussian(size, a, x0, sx):
    result = flex.real(flex.grid(size))
    index = [0 for i in range(len(size))]
    while True:
        result[index[::-1]] = evaluate_gaussian(index[::-1], a, x0, sx)
        for j in range(len(size)):
            index[j] += 1
            if index[j] < size[::-1][j]:
                break
            index[j] = 0
            if j == len(size) - 1:
                return result


def test_forward(dials_data):
    sweep = load.sweep(dials_data("centroid_test_data").join("sweep.json").strpath)

    # Get the models
    beam = sweep.get_beam()
    detector = sweep.get_detector()
    gonio = sweep.get_goniometer()
    scan = sweep.get_scan()

    # Set some parameters
    sigma_divergence = beam.get_sigma_divergence(deg=False)
    mosaicity = 0.157 * math.pi / 180
    n_sigma = 3
    grid_size = 7
    delta_divergence = n_sigma * sigma_divergence

    step_size = delta_divergence / grid_size
    delta_divergence2 = delta_divergence + step_size * 0.5
    delta_mosaicity = n_sigma * mosaicity

    # Create the bounding box calculator
    calculate_bbox = BBoxCalculator3D(
        beam, detector, gonio, scan, delta_divergence2, delta_mosaicity
    )

    # Initialise the transform
    spec = transform.TransformSpec(
        beam, detector, gonio, scan, sigma_divergence, mosaicity, n_sigma + 1, grid_size
    )

    # tst_conservation_of_counts(self):

    assert len(detector) == 1

    s0 = beam.get_s0()
    m2 = gonio.get_rotation_axis()
    s0_length = matrix.col(beam.get_s0()).length()

    # Create an s1 map
    s1_map = transform.beam_vector_map(detector[0], beam, True)

    for i in range(100):

        # Get random x, y, z
        x = random.uniform(300, 1800)
        y = random.uniform(300, 1800)
        z = random.uniform(0, 9)

        # Get random s1, phi, panel
        s1 = matrix.col(detector[0].get_pixel_lab_coord((x, y))).normalize() * s0_length
        phi = scan.get_angle_from_array_index(z, deg=False)
        panel = 0

        # Calculate the bounding box
        bbox = calculate_bbox(s1, z, panel)
        x0, x1, y0, y1, z0, z1 = bbox

        # Create the coordinate system
        cs = CoordinateSystem(m2, s0, s1, phi)

        # The grid index generator
        step_size = delta_divergence / grid_size
        grid_index = transform.GridIndexGenerator(
            cs, x0, y0, (step_size, step_size), grid_size, s1_map
        )

        # Create the image
        # image = flex.double(flex.grid(z1 - z0, y1 - y0, x1 - x0), 1)
        image = gaussian(
            (z1 - z0, y1 - y0, x1 - x0), 10.0, (z - z0, y - y0, x - x0), (2.0, 2.0, 2.0)
        )
        mask = flex.bool(flex.grid(image.all()), False)
        for j in range(y1 - y0):
            for i in range(x1 - x0):
                inside = False
                gx00, gy00 = grid_index(j, i)
                gx01, gy01 = grid_index(j, i + 1)
                gx10, gy10 = grid_index(j + 1, i)
                gx11, gy11 = grid_index(j + 1, i + 1)
                mingx = min([gx00, gx01, gx10, gx11])
                maxgx = max([gx00, gx01, gx10, gx11])
                mingy = min([gy00, gy01, gy10, gy11])
                maxgy = max([gy00, gy01, gy10, gy11])
                if (
                    mingx >= 0
                    and maxgx < 2 * grid_size + 1
                    and mingy >= 0
                    and maxgy < 2 * grid_size + 1
                ):
                    inside = True
                for k in range(1, z1 - z0 - 1):
                    mask[k, j, i] = inside

        # Transform the image to the grid
        transformed = transform.TransformForward(
            spec, cs, bbox, 0, image.as_double(), mask
        )
        grid = transformed.profile()

        # Get the sums and ensure they're the same
        eps = 1e-7
        sum_grid = flex.sum(grid)
        sum_image = flex.sum(flex.double(flex.select(image, flags=mask)))
        assert abs(sum_grid - sum_image) <= eps

    # Test passed

    # tst_transform_with_background(self):

    assert len(detector) == 1
    s0 = beam.get_s0()
    m2 = gonio.get_rotation_axis()
    s0_length = matrix.col(beam.get_s0()).length()

    # Create an s1 map
    s1_map = transform.beam_vector_map(detector[0], beam, True)

    for i in range(100):

        # Get random x, y, z
        x = random.uniform(300, 1800)
        y = random.uniform(300, 1800)
        z = random.uniform(0, 9)

        # Get random s1, phi, panel
        s1 = matrix.col(detector[0].get_pixel_lab_coord((x, y))).normalize() * s0_length
        phi = scan.get_angle_from_array_index(z, deg=False)
        panel = 0

        # Calculate the bounding box
        bbox = calculate_bbox(s1, z, panel)
        x0, x1, y0, y1, z0, z1 = bbox

        # Create the coordinate system
        cs = CoordinateSystem(m2, s0, s1, phi)

        # The grid index generator
        step_size = delta_divergence / grid_size
        grid_index = transform.GridIndexGenerator(
            cs, x0, y0, (step_size, step_size), grid_size, s1_map
        )

        # Create the image
        # image = flex.double(flex.grid(z1 - z0, y1 - y0, x1 - x0), 1)
        image = gaussian(
            (z1 - z0, y1 - y0, x1 - x0), 10.0, (z - z0, y - y0, x - x0), (2.0, 2.0, 2.0)
        )
        background = flex.random_double(len(image))
        background.resize(image.accessor())
        mask = flex.bool(flex.grid(image.all()), False)
        for j in range(y1 - y0):
            for i in range(x1 - x0):
                inside = False
                gx00, gy00 = grid_index(j, i)
                gx01, gy01 = grid_index(j, i + 1)
                gx10, gy10 = grid_index(j + 1, i)
                gx11, gy11 = grid_index(j + 1, i + 1)
                mingx = min([gx00, gx01, gx10, gx11])
                maxgx = max([gx00, gx01, gx10, gx11])
                mingy = min([gy00, gy01, gy10, gy11])
                maxgy = max([gy00, gy01, gy10, gy11])
                if (
                    mingx >= 0
                    and maxgx <= 2 * grid_size + 1
                    and mingy >= 0
                    and maxgy <= 2 * grid_size + 1
                ):
                    inside = True
                for k in range(1, z1 - z0 - 1):
                    mask[k, j, i] = inside

        # Transform the image to the grid
        transformed = transform.TransformForward(
            spec, cs, bbox, 0, image.as_double(), background.as_double(), mask
        )
        igrid = transformed.profile()
        bgrid = transformed.background()

        # Get the sums and ensure they're the same
        eps = 1e-7
        sum_igrid = flex.sum(igrid)
        sum_bgrid = flex.sum(bgrid)
        sum_image = flex.sum(flex.double(flex.select(image, flags=mask)))
        sum_bkgrd = flex.sum(flex.double(flex.select(background, flags=mask)))
        try:
            assert abs(sum_igrid - sum_image) <= eps
            assert abs(sum_bgrid - sum_bkgrd) <= eps
        except Exception:
            print("Failed for: ", (x, y, z))
            raise


def test_forward_no_model(dials_data):
    sweep = load.sweep(dials_data("centroid_test_data").join("sweep.json").strpath)

    # Get the models
    beam = sweep.get_beam()
    detector = sweep.get_detector()
    gonio = sweep.get_goniometer()
    scan = sweep.get_scan()
    scan.set_image_range((0, 1000))

    # Set some parameters
    sigma_divergence = beam.get_sigma_divergence(deg=False)
    mosaicity = 0.157 * math.pi / 180
    n_sigma = 3
    grid_size = 20
    delta_divergence = n_sigma * sigma_divergence

    step_size = delta_divergence / grid_size
    delta_divergence2 = delta_divergence + step_size * 0.5
    delta_mosaicity = n_sigma * mosaicity

    # Create the bounding box calculator
    calculate_bbox = BBoxCalculator3D(
        beam, detector, gonio, scan, delta_divergence2, delta_mosaicity
    )

    # Initialise the transform
    spec = transform.TransformSpec(
        beam, detector, gonio, scan, sigma_divergence, mosaicity, n_sigma + 1, grid_size
    )

    # tst_conservation_of_counts(self):

    random.seed(0)

    assert len(detector) == 1

    s0 = beam.get_s0()
    m2 = gonio.get_rotation_axis()
    s0_length = matrix.col(beam.get_s0()).length()

    # Create an s1 map
    s1_map = transform.beam_vector_map(detector[0], beam, True)

    for i in range(100):

        # Get random x, y, z
        x = random.uniform(300, 1800)
        y = random.uniform(300, 1800)
        z = random.uniform(500, 600)

        # Get random s1, phi, panel
        s1 = matrix.col(detector[0].get_pixel_lab_coord((x, y))).normalize() * s0_length
        phi = scan.get_angle_from_array_index(z, deg=False)
        panel = 0

        # Calculate the bounding box
        bbox = calculate_bbox(s1, z, panel)
        x0, x1, y0, y1, z0, z1 = bbox

        # Create the coordinate system
        cs = CoordinateSystem(m2, s0, s1, phi)
        if abs(cs.zeta()) < 0.1:
            continue

        # The grid index generator
        step_size = delta_divergence / grid_size
        grid_index = transform.GridIndexGenerator(
            cs, x0, y0, (step_size, step_size), grid_size, s1_map
        )

        # Create the image
        # image = flex.double(flex.grid(z1 - z0, y1 - y0, x1 - x0), 1)
        image = gaussian(
            (z1 - z0, y1 - y0, x1 - x0), 10.0, (z - z0, y - y0, x - x0), (2.0, 2.0, 2.0)
        )
        mask = flex.bool(flex.grid(image.all()), False)
        for j in range(y1 - y0):
            for i in range(x1 - x0):
                inside = False
                gx00, gy00 = grid_index(j, i)
                gx01, gy01 = grid_index(j, i + 1)
                gx10, gy10 = grid_index(j + 1, i)
                gx11, gy11 = grid_index(j + 1, i + 1)
                mingx = min([gx00, gx01, gx10, gx11])
                maxgx = max([gx00, gx01, gx10, gx11])
                mingy = min([gy00, gy01, gy10, gy11])
                maxgy = max([gy00, gy01, gy10, gy11])
                if (
                    mingx >= 0
                    and maxgx < 2 * grid_size + 1
                    and mingy >= 0
                    and maxgy < 2 * grid_size + 1
                ):
                    inside = True
                for k in range(1, z1 - z0 - 1):
                    mask[k, j, i] = inside

        # Transform the image to the grid
        transformed = transform.TransformForwardNoModel(
            spec, cs, bbox, 0, image.as_double(), mask
        )
        grid = transformed.profile()

        # Get the sums and ensure they're the same
        eps = 1e-7
        sum_grid = flex.sum(grid)
        sum_image = flex.sum(flex.double(flex.select(image, flags=mask)))
        assert abs(sum_grid - sum_image) <= eps

        mask = flex.bool(flex.grid(image.all()), True)
        transformed = transform.TransformForwardNoModel(
            spec, cs, bbox, 0, image.as_double(), mask
        )
        grid = transformed.profile()

        # Boost the bbox to make sure all intensity is included
        x0, x1, y0, y1, z0, z1 = bbox
        bbox2 = (x0 - 10, x1 + 10, y0 - 10, y1 + 10, z0 - 10, z1 + 10)

        # Do the reverse transform
        transformed = transform.TransformReverseNoModel(spec, cs, bbox2, 0, grid)
        image2 = transformed.profile()

        # Check the sum of pixels are the same
        sum_grid = flex.sum(grid)
        sum_image = flex.sum(image2)
        assert abs(sum_grid - sum_image) <= eps

        # Do the reverse transform
        transformed = transform.TransformReverseNoModel(spec, cs, bbox, 0, grid)
        image2 = transformed.profile()

        from dials.algorithms.statistics import pearson_correlation_coefficient

        cc = pearson_correlation_coefficient(image.as_1d().as_double(), image2.as_1d())
        assert cc >= 0.99
        # if cc < 0.99:
        #   print cc, bbox
        #   from matplotlib import pylab
        # pylab.plot(image.as_numpy_array()[(z1-z0)/2,(y1-y0)/2,:])
        # pylab.show()
        # pylab.plot(image2.as_numpy_array()[(z1-z0)/2,(y1-y0)/2,:])
        # pylab.show()
        # pylab.plot((image.as_double()-image2).as_numpy_array()[(z1-z0)/2,(y1-y0)/2,:])
        # pylab.show()
