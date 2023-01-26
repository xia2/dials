/*
 * reciprocal_space_helpers.h
 *
 *  Copyright (C) 2014 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_SIMULATION_RECIPROCAL_SPACE_HELPERS_H
#define DIALS_ALGORITHMS_SIMULATION_RECIPROCAL_SPACE_HELPERS_H

#include <ctime>
#include <random>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <scitbx/array_family/tiny_types.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/goniometer.h>
#include <dxtbx/model/scan.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/profile_model/gaussian_rs/coordinate_system.h>
#include <dials/model/data/shoebox.h>

namespace dials { namespace algorithms {

  using dials::model::Foreground;
  using dxtbx::model::BeamBase;
  using dxtbx::model::Detector;
  using dxtbx::model::Goniometer;
  using dxtbx::model::Scan;
  using profile_model::gaussian_rs::CoordinateSystem;
  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::af::int6;

  /**
   * Simulate a gaussian in reciprocal space and transform back to detector
   * space.
   */
  int simulate_reciprocal_space_gaussian(
    const BeamBase &beam,
    const Detector &detector,
    const Goniometer &goniometer,
    const Scan &scan,
    double sigma_b,
    double sigma_m,
    const vec3<double> s1,
    double phi,
    const int6 &bbox,
    std::size_t I,
    af::ref<double, af::c_grid<3> > shoebox,
    const af::const_ref<int, af::c_grid<3> > &mask) {
    vec3<double> s0 = beam.get_s0();
    vec3<double> m2 = goniometer.get_rotation_axis();

    // Seed the random number generator
    std::mt19937 gen(time(0));
    std::normal_distribution<double> dist_x(0, sigma_b);
    std::normal_distribution<double> dist_y(0, sigma_b);
    std::normal_distribution<double> dist_z(0, sigma_m);

    // Do the simulation
    int counts = 0;
    CoordinateSystem cs(m2, s0, s1, phi);
    for (std::size_t i = 0; i < I; ++i) {
      // Get the random coordinates
      double e1 = dist_x(gen);
      double e2 = dist_y(gen);
      double e3 = dist_z(gen);

      // Get the beam vector and rotation angle
      vec3<double> s1_dash = cs.to_beam_vector(vec2<double>(e1, e2));
      double phi_dash = cs.to_rotation_angle_fast(e3);

      // Get the pixel coordinate
      vec2<double> mm = detector[0].get_ray_intersection(s1_dash);
      vec2<double> px = detector[0].millimeter_to_pixel(mm);

      // Get the frame
      double frame = scan.get_array_index_from_angle(phi_dash);

      // Make sure coordinate is within range
      if (px[0] < bbox[0] || px[0] >= bbox[1] || px[1] < bbox[2] || px[1] >= bbox[3]
          || frame < bbox[4] || frame >= bbox[5]) {
        continue;
      }

      // Get the pixel index
      int x = (int)(px[0] - bbox[0]);
      int y = (int)(px[1] - bbox[2]);
      int z = (int)(frame - bbox[4]);

      // Add the count
      shoebox(z, y, x) += 1;

      if (mask(z, y, x) & Foreground) {
        counts += 1;
      }
    }
    return counts;
  }

  /**
   * Simulate a gaussian in reciprocal space and transform back to detector
   * space. Estimate the expected intensity within the masked region.
   */
  int integrate_reciprocal_space_gaussian(
    const BeamBase &beam,
    const Detector &detector,
    const Goniometer &goniometer,
    const Scan &scan,
    double sigma_b,
    double sigma_m,
    const vec3<double> s1,
    double phi,
    const int6 &bbox,
    std::size_t I,
    const af::const_ref<int, af::c_grid<3> > &mask) {
    vec3<double> s0 = beam.get_s0();
    vec3<double> m2 = goniometer.get_rotation_axis();

    // Seed the random number generator
    std::mt19937 gen(time(0));
    std::normal_distribution<double> dist_x(0, sigma_b);
    std::normal_distribution<double> dist_y(0, sigma_b);
    std::normal_distribution<double> dist_z(0, sigma_m);

    // Do the simulation
    int counts = 0;
    CoordinateSystem cs(m2, s0, s1, phi);
    for (std::size_t i = 0; i < I; ++i) {
      // Get the random coordinates
      double e1 = dist_x(gen);
      double e2 = dist_y(gen);
      double e3 = dist_z(gen);

      // Get the beam vector and rotation angle
      vec3<double> s1_dash = cs.to_beam_vector(vec2<double>(e1, e2));
      double phi_dash = cs.to_rotation_angle_fast(e3);

      // Get the pixel coordinate
      vec2<double> mm = detector[0].get_ray_intersection(s1_dash);
      vec2<double> px = detector[0].millimeter_to_pixel(mm);

      // Get the frame
      double frame = scan.get_array_index_from_angle(phi_dash);

      // Make sure coordinate is within range
      if (px[0] < bbox[0] || px[0] >= bbox[1] || px[1] < bbox[2] || px[1] >= bbox[3]
          || frame < bbox[4] || frame >= bbox[5]) {
        continue;
      }

      // Get the pixel index
      int x = (int)(px[0] - bbox[0]);
      int y = (int)(px[1] - bbox[2]);
      int z = (int)(frame - bbox[4]);

      // Add the count
      if (mask(z, y, x) & Foreground) {
        counts += 1;
      }
    }
    return counts;
  }

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_SIMULATION_RECIPROCAL_SPACE_HELPERS_H
