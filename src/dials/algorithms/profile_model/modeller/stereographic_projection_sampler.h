/*
 * polar_resolution_sampler.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_STEREOGRAPHIC_PROJECTION_SAMPLER_H
#define DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_STEREOGRAPHIC_PROJECTION_SAMPLER_H

#include <memory>
#include <dials/algorithms/profile_model/modeller/sampler_interface.h>

namespace dials { namespace algorithms {

  using scitbx::af::double3;
  using scitbx::af::int3;

  /**
   * A sampler class that works in polar resolution coordinates and phi
   */
  class StereographicProjectionSampler : public SamplerIface {
  public:
    /**
     * Initialise the sampler
     */
    StereographicProjectionSampler(const std::shared_ptr<BeamBase> beam,
                                   const Detector &detector,
                                   const Goniometer &goniometer,
                                   const Scan &scan,
                                   double dmin,
                                   std::size_t num_scan_points)
        : num_profiles_(9), num_scan_points_(num_scan_points) {}

    /**
     * @returns The total number of grid points
     */
    std::size_t size() const {
      return num_profiles_ * num_scan_points_;
    }

    /**
     * Find the nearest reference profile to the given point.
     * @param xyz The coordinate
     * @returns The index of the reference profile
     */
    std::size_t nearest(std::size_t panel, double3 xyz) const {}

    /**
     * Find the nearest n reference profiles to the given point.
     * This is mostly used during learning to get the neighbouring profiles.
     * @param xyz The coordinate
     * @returns A list of reference profile indices
     */
    af::shared<std::size_t> nearest_n(std::size_t panel, double3 xyz) const {}

    /**
     * Get the weight for the given profile at the given coordinate.
     * @param index The profile index
     * @param xyz The coordinate
     * @returns The weight (between 1.0 and 0.0)
     */
    double weight(std::size_t index, std::size_t panel, double3 xyz) const {}

    /**
     * Get the x, y, z coordinate of the reference profile at the given index.
     * @param index The index of the reference profile.
     * @returns The x, y, z coordinate of the profile
     */
    double3 coord(std::size_t index) const {}

    /**
     * Return the neighbouring grid points.
     */
    af::shared<std::size_t> neighbours(std::size_t index) const {}

  private:
    std::size_t num_profiles_;
    std::size_t num_scan_points_;
  };

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_STEREOGRAPHIC_PROJECTION_SAMPLER_H \
        */
