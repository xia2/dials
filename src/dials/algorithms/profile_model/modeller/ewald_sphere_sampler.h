/*
 * ewald_sphere_sampler.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_EWALD_SPHERE_SAMPLER_H
#define DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_EWALD_SPHERE_SAMPLER_H

#include <cmath>
#include <memory>
#include <scitbx/array_family/tiny_types.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/goniometer.h>
#include <dxtbx/model/scan.h>
#include <dials/algorithms/profile_model/modeller/sampler_interface.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <scitbx/constants.h>
#include <dials/error.h>

namespace dials { namespace algorithms {

  using dxtbx::model::BeamBase;
  using dxtbx::model::Detector;
  using dxtbx::model::Goniometer;
  using dxtbx::model::Panel;
  using dxtbx::model::Scan;
  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::af::double3;
  using scitbx::af::int2;
  using scitbx::af::int3;
  using scitbx::constants::pi;
  using scitbx::constants::two_pi;

  /**
   * Class to sample reference profiles in a grid
   */
  class EwaldSphereSampler : public SamplerIface {
  public:
    /**
     * Initialise the sampler
     */
    EwaldSphereSampler(const std::shared_ptr<BeamBase> beam,
                       const Detector &detector,
                       const Goniometer &goniometer,
                       const Scan &scan,
                       std::size_t num_phi)
        : beam_(beam),
          detector_(detector),
          goniometer_(goniometer),
          scan_(scan),
          num_phi_(num_phi) {
      // Check input
      DIALS_ASSERT(num_phi > 0);

      // Compute the axes around s0
      vec3<double> s0 = beam->get_s0().normalize();
      vec3<double> m2 = goniometer.get_rotation_axis().normalize();
      zaxis_ = s0;
      yaxis_ = zaxis_.cross(m2);
      xaxis_ = zaxis_.cross(yaxis_);

      // Find the maximum angle between the corners of the detector and the beam
      // centre
      max_angle_ = 0;
      for (std::size_t i = 0; i < detector.size(); ++i) {
        const Panel &p = detector[i];
        std::size_t width = p.get_image_size()[0];
        std::size_t height = p.get_image_size()[1];
        vec3<double> s1 = p.get_pixel_lab_coord(vec2<double>(0, 0)).normalize();
        vec3<double> s2 = p.get_pixel_lab_coord(vec2<double>(width, 0)).normalize();
        vec3<double> s3 = p.get_pixel_lab_coord(vec2<double>(0, height)).normalize();
        vec3<double> s4 =
          p.get_pixel_lab_coord(vec2<double>(width, height)).normalize();
        double z1 = s1 * zaxis_;
        double z2 = s2 * zaxis_;
        double z3 = s3 * zaxis_;
        double z4 = s4 * zaxis_;
        double angle1 = std::acos(z1);
        double angle2 = std::acos(z2);
        double angle3 = std::acos(z3);
        double angle4 = std::acos(z4);
        if (angle1 > max_angle_) max_angle_ = angle1;
        if (angle2 > max_angle_) max_angle_ = angle2;
        if (angle3 > max_angle_) max_angle_ = angle3;
        if (angle4 > max_angle_) max_angle_ = angle4;
      }

      // Ensure max angle is > 0
      DIALS_ASSERT(max_angle_ > 0);

      // Compute the step size in phi
      int num_frames = scan.get_num_images();
      scan_range_ = scan.get_array_range();
      DIALS_ASSERT(num_frames > 0);
      step_phi_ = (double)num_frames / num_phi_;

      // Compute the intervals We want to have each profile cover equal
      // fractions of the surface of the Ewald sphere. There will be 4
      // concentric circles of profiles that cover the whole sphere. From there
      // we compute the area and work out the interval needed for all the
      // profiles to fit.
      num1_ = af::shared<std::size_t>(4);
      num1_[0] = 1;
      num1_[1] = 8;
      num1_[2] = 16;
      num1_[3] = 32;
      double tot_area = 2 * two_pi;
      double area_one = tot_area / af::sum(num1_.const_ref());
      std::size_t num_image = af::sum(num1_.const_ref());
      step1_ = af::shared<double>(4);
      step2_ = af::shared<double>(4);
      coord_ = af::shared<double3>(num_image * num_phi_);
      indx1_ = af::shared<std::size_t>(num_image * num_phi);
      step1_[0] = std::acos(1 - area_one / two_pi);
      step2_[0] = two_pi;
      for (std::size_t k = 0; k < num_phi_; ++k) {
        coord_[k * num_image][0] = 0;
        coord_[k * num_image][1] = 0;
        coord_[k * num_image][2] = scan_range_[0] + step_phi_ * 0.5;
        indx1_[k * num_image] = 0;
      }
      double sum = 0;
      for (std::size_t l = 1, i = 1; i < num1_.size(); ++i) {
        sum += num1_[i - 1];
        double mul = num1_[i] / sum;
        double x = (1 - (mul + 1) * (1 - std::cos(step1_[i - 1])));
        if (x < -1) x = -1;
        if (x > 1) x = 1;
        step1_[i] = std::acos(x);
        step2_[i] = two_pi / num1_[i];
        double c1 = (step1_[i] + step1_[i - 1]) * 0.5;
        for (std::size_t j = 0; j < num1_[i]; ++j, ++l) {
          double c2 = (j + 0.5) * step2_[i];
          for (std::size_t k = 0; k < num_phi_; ++k) {
            DIALS_ASSERT(l + k * num_image < coord_.size());
            coord_[l + k * num_image][0] = c1;
            coord_[l + k * num_image][1] = c2;
            coord_[l + k * num_image][2] = scan_range_[0] + step_phi_ * 0.5;
            indx1_[l + k * num_image] = i;
          }
        }
      }

      // Compute the nearest n profiles from a main index
      neighbours_ = af::shared<af::shared<std::size_t> >(num_image * num_phi);
      for (std::size_t iz = 0, l = 0; iz < num_phi_; ++iz) {
        for (std::size_t ix = 0; ix < num1_.size(); ++ix) {
          for (std::size_t iy = 0; iy < num1_[ix]; ++iy) {
            af::shared<std::size_t> temp;
            if (ix == 0) {
              for (std::size_t j = 0; j < num1_[1]; ++j) {
                temp.push_back(index(1, j, iz));
              }
            } else if (ix == 1) {
              temp.push_back(index(0, 0, iz));
              temp.push_back(index(1, (iy + 1) % num1_[1], iz));
              temp.push_back(index(1, (iy - 1) % num1_[1], iz));
              temp.push_back(index(2, 2 * iy, iz));
              temp.push_back(index(2, 2 * iy + 1, iz));
            } else if (ix == 2) {
              temp.push_back(index(1, (int)std::floor((double)iy / 2), iz));
              temp.push_back(index(2, (iy + 1) % num1_[2], iz));
              temp.push_back(index(2, (iy - 1) % num1_[2], iz));
              temp.push_back(index(3, 2 * iy, iz));
              temp.push_back(index(3, 2 * iy + 1, iz));
            } else if (ix == 3) {
              temp.push_back(index(2, (int)std::floor((double)iy / 2), iz));
              temp.push_back(index(3, (iy + 1) % num1_[3], iz));
              temp.push_back(index(3, (iy - 1) % num1_[3], iz));
            } else {
              throw DIALS_ERROR("Programmer Error!!!!");
            }
            neighbours_[l] = temp;
            l++;
          }
        }
      }
    }

    /**
     * Get number of profiles
     */
    std::size_t num_phi() const {
      return num_phi_;
    }

    /**
     * Get steps of profiles
     */
    double step_phi() const {
      return step_phi_;
    }

    /**
     * @returns The max angle
     */
    double max_angle() const {
      return max_angle_;
    }

    /**
     * @returns The total number of grid points
     */
    std::size_t size() const {
      return af::sum(num1_.const_ref()) * num_phi_;
    }

    /**
     * Find the nearest reference profile to the given point.
     * @param xyz The coordinate
     * @returns The index of the reference profile
     */
    std::size_t nearest(std::size_t panel, double3 xyz) const {
      vec3<double> s1 =
        detector_[panel].get_pixel_lab_coord(vec2<double>(xyz[0], xyz[1])).normalize();
      xyz[2] -= scan_range_[0];
      double z = s1 * zaxis_;
      double y = s1 * yaxis_;
      double x = s1 * xaxis_;
      double a = std::acos(z);
      double b = std::atan2(y, x);
      int iz = (int)floor(xyz[2] / step_phi_);
      if (iz < 0) iz = 0;
      if (iz >= num_phi_) iz = num_phi_ - 1;
      int ix = num1_.size() - 1;
      for (std::size_t i = 0; i < num1_.size(); ++i) {
        if (a < step1_[i]) {
          ix = i;
          break;
        }
      }
      if (ix < 0) ix = 0;
      if (ix >= num1_.size()) ix = num1_.size() - 1;
      if (b < 0) b += two_pi;
      int iy = (int)floor(b / step2_[ix]);
      if (iy < 0) iy = 0;
      if (iy >= num1_[ix]) iy = num1_[ix] - 1;
      return index(ix, iy, iz);
    }

    /**
     * Find the nearest n reference profiles to the given point.
     * This is mostly used during learning to get the neighbouring profiles.
     * @param xyz The coordinate
     * @returns A list of reference profile indices
     */
    af::shared<std::size_t> nearest_n(std::size_t panel, double3 xyz) const {
      std::size_t main_index = nearest(panel, xyz);
      return nearest_n_index(main_index);
    }

    af::shared<std::size_t> nearest_n_index(std::size_t index) const {
      af::shared<std::size_t> result(neighbours_[index].begin(),
                                     neighbours_[index].end());
      result.push_back(index);
      return result;
    }

    /**
     * Get the weight for the given profile at the given coordinate.
     * @param index The profile index
     * @param xyz The coordinate
     * @returns The weight (between 1.0 and 0.0)
     */
    double weight(std::size_t index, std::size_t panel, double3 xyz) const {
      vec3<double> s1 =
        detector_[panel].get_pixel_lab_coord(vec2<double>(xyz[0], xyz[1])).normalize();
      xyz[2] -= scan_range_[0];
      double z = s1 * zaxis_;
      double y = s1 * yaxis_;
      double x = s1 * xaxis_;
      double p1 = pi / 2 - std::acos(z);
      double l1 = std::atan2(y, x);
      double3 c = coord_[index];
      double p2 = pi / 2 - c[0];
      double l2 = c[1];
      double q = std::sin(p1) * std::sin(p2)
                 + std::cos(p1) * std::cos(p2) * std::cos(std::abs(l1 - l2));
      if (q > 1) q = 1;
      if (q < -1) q = -1;
      std::size_t idx = indx1_[index];
      double step = (idx == 0 ? 2 * step1_[idx] : step1_[idx] - step1_[idx - 1]);
      double d = std::acos(q) / step;
      return std::exp(-4.0 * d * d * std::log(2.0));
    }

    /**
     * Get the x, y, z coordinate of the reference profile at the given index.
     * @param index The index of the reference profile.
     * @returns The x, y, z coordinate of the profile
     */
    double3 coord(std::size_t index) const {
      throw DIALS_ERROR("Not implemented");
      return double3();
    }

    /**
     * Return the neighbouring grid points.
     */
    af::shared<std::size_t> neighbours(std::size_t index) const {
      af::shared<std::size_t> result(neighbours_[index].begin(),
                                     neighbours_[index].end());
      return result;
    }

    double3 profile_coord(std::size_t index) const {
      DIALS_ASSERT(index < coord_.size());
      return coord_[index];
    }

    std::shared_ptr<BeamBase> beam() const {
      return beam_;
    }

    Detector detector() const {
      return detector_;
    }

    Goniometer goniometer() const {
      return goniometer_;
    }

    Scan scan() const {
      return scan_;
    }

  private:
    std::size_t index(std::size_t ix, std::size_t iy, std::size_t iz) const {
      int tot_sum = af::sum(num1_.const_ref());
      int par_sum = 0;
      for (std::size_t i = 0; i < ix; ++i) {
        par_sum += num1_[i];
      }
      return par_sum + iy + iz * tot_sum;
    }

    std::shared_ptr<BeamBase> beam_;
    Detector detector_;
    Goniometer goniometer_;
    Scan scan_;
    std::size_t num_phi_;
    vec3<double> xaxis_;
    vec3<double> yaxis_;
    vec3<double> zaxis_;
    double max_angle_;
    af::shared<std::size_t> num1_;
    af::shared<double> step1_;
    af::shared<double> step2_;
    af::shared<double3> coord_;
    af::shared<std::size_t> indx1_;
    af::shared<af::shared<std::size_t> > neighbours_;
    double step_phi_;
    vec2<int> scan_range_;
  };

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_EWALD_SPHERE_SAMPLER_H */
