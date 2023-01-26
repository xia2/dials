/*
 * partiality_calculator.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_PARTIALITY_CALCULATOR_H
#define DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_PARTIALITY_CALCULATOR_H

#include <cmath>
#include <memory>
#include <boost/math/special_functions/erf.hpp>
#include <scitbx/constants.h>
#include <scitbx/vec3.h>
#include <scitbx/array_family/tiny_types.h>
#include <scitbx/array_family/ref_reductions.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/goniometer.h>
#include <dxtbx/model/scan.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/profile_model/gaussian_rs/coordinate_system.h>

namespace dials {
  namespace algorithms {
    namespace profile_model {
      namespace gaussian_rs {

  // Use a load of stuff from other namespaces
  using boost::math::erf;
  using dxtbx::model::BeamBase;
  using dxtbx::model::Detector;
  using dxtbx::model::Goniometer;
  using dxtbx::model::Scan;
  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::af::double2;
  using scitbx::af::double3;
  using scitbx::af::double4;
  using scitbx::af::int6;
  using scitbx::af::max;
  using scitbx::af::min;
  using std::ceil;
  using std::floor;

  /**
   * Interface for bounding box calculator.
   */
  class PartialityCalculatorIface {
  public:
    virtual ~PartialityCalculatorIface() {}

    virtual double single(vec3<double> s1, double frame, int6 bbox) const = 0;

    virtual af::shared<double> array(const af::const_ref<vec3<double> > &s1,
                                     const af::const_ref<double> &frame,
                                     const af::const_ref<int6> &bbox) const = 0;
  };

  /** Calculate the partiality for each reflection */
  class PartialityCalculator3D : public PartialityCalculatorIface {
  public:
    /**
     * Initialise the partiality calculation.
     * @param beam The beam parameters
     * @param goniometer The goniometer parameters
     * @param scan The scan parameters
     * @param sigma_mosaicity The xds sigma_mosaicity parameter
     */
    PartialityCalculator3D(const BeamBase &beam,
                           const Goniometer &gonio,
                           const Scan &scan,
                           double sigma_m)
        : s0_(beam.get_s0()),
          m2_(gonio.get_rotation_axis()),
          scan_(scan),
          sigma_m_(1, sigma_m) {
      DIALS_ASSERT(sigma_m > 0.0);
    }

    /**
     * Initialise the partiality calculation.
     * @param beam The beam parameters
     * @param goniometer The goniometer parameters
     * @param scan The scan parameters
     * @param sigma_mosaicity The xds sigma_mosaicity parameter
     */
    PartialityCalculator3D(const BeamBase &beam,
                           const Goniometer &gonio,
                           const Scan &scan,
                           const af::const_ref<double> &sigma_m)
        : s0_(beam.get_s0()),
          m2_(gonio.get_rotation_axis()),
          scan_(scan),
          sigma_m_(sigma_m.begin(), sigma_m.end()) {
      DIALS_ASSERT(sigma_m.all_gt(0.0));
      DIALS_ASSERT(sigma_m.size() == scan.get_num_images());
      DIALS_ASSERT(sigma_m.size() > 0);
    }

    /**
     * Calculate the Partiality of the reflection.
     *
     * @param s1 The diffracted beam vector
     * @param frame The frame number
     * @param bbox The bounding box
     * @returns The partiality as a fraction of the total phi extent
     */
    virtual double single(vec3<double> s1, double frame, int6 bbox) const {
      // Ensure our values are ok
      DIALS_ASSERT(s1.length_sq() > 0);
      DIALS_ASSERT(bbox[4] < bbox[5]);

      // Get the mosaicity at this point
      double sigma_m = 0.0;
      if (sigma_m_.size() == 1) {
        sigma_m = sigma_m_[0];
      } else {
        int frame0 = scan_.get_array_range()[0];
        int index = (int)std::floor(frame) - frame0;
        if (index < 0) {
          sigma_m = sigma_m_.front();
        } else if (index >= sigma_m_.size()) {
          sigma_m = sigma_m_.back();
        } else {
          sigma_m = sigma_m_[index];
        }
      }

      // Get the rotation angle
      double phi = scan_.get_angle_from_array_index(frame);
      double phia = scan_.get_angle_from_array_index(bbox[4]);
      double phib = scan_.get_angle_from_array_index(bbox[5]);

      // Compute the partiality
      double zeta = profile_model::gaussian_rs::zeta_factor(m2_.normalize(), s0_, s1);
      double c = std::abs(zeta) / (sqrt(2.0) * sigma_m);
      double p = 0.5 * (erf(c * (phib - phi)) - erf(c * (phia - phi)));
      DIALS_ASSERT(p >= 0.0 && p <= 1.0);
      return p;
    }

    /**
     * Calculate the partiality for an array of reflections
     * @param s1 The array of diffracted beam vectors
     * @param frame The array of frame numbers.
     * @param bbox The array of bboxes
     */
    virtual af::shared<double> array(const af::const_ref<vec3<double> > &s1,
                                     const af::const_ref<double> &frame,
                                     const af::const_ref<int6> &bbox) const {
      DIALS_ASSERT(s1.size() == frame.size());
      DIALS_ASSERT(s1.size() == bbox.size());
      af::shared<double> result(s1.size(), af::init_functor_null<double>());
      for (std::size_t i = 0; i < s1.size(); ++i) {
        result[i] = single(s1[i], frame[i], bbox[i]);
      }
      return result;
    }

  private:
    vec3<double> s0_;
    vec3<double> m2_;
    Scan scan_;
    af::shared<double> sigma_m_;
  };

  /** Calculate the partiality for each reflection */
  class PartialityCalculator2D : public PartialityCalculatorIface {
  public:
    /**
     * Initialise the partiality calculation.
     * @param beam The beam parameters
     * @param sigma_mosaicity The xds sigma_mosaicity parameter
     */
    PartialityCalculator2D(const BeamBase &beam, double sigma_m) : s0_(beam.get_s0()) {}

    /**
     * Calculate the Partiality of the reflection.
     *
     * @param s1 The diffracted beam vector
     * @param frame The frame number
     * @param bbox The bounding box
     * @returns The partiality as a fraction of the total phi extent
     */
    virtual double single(vec3<double> s1, double frame, int6 bbox) const {
      // Ensure our values are ok
      DIALS_ASSERT(s1.length_sq() > 0);
      DIALS_ASSERT(bbox[4] < bbox[5]);

      // FIXME This is a placeholder

      // Return the fraction recorded
      return 1.0;
    }

    /**
     * Calculate the partiality for an array of reflections
     * @param s1 The array of diffracted beam vectors
     * @param frame The array of frame numbers.
     * @param bbox The array of bboxes
     */
    virtual af::shared<double> array(const af::const_ref<vec3<double> > &s1,
                                     const af::const_ref<double> &frame,
                                     const af::const_ref<int6> &bbox) const {
      DIALS_ASSERT(s1.size() == frame.size());
      DIALS_ASSERT(s1.size() == bbox.size());
      af::shared<double> result(s1.size(), af::init_functor_null<double>());
      for (std::size_t i = 0; i < s1.size(); ++i) {
        result[i] = single(s1[i], frame[i], bbox[i]);
      }
      return result;
    }

  private:
    vec3<double> s0_;
    vec3<double> m2_;
    Scan scan_;
  };

  /**
   * Class to help compute partiality for multiple experiments.
   */
  class PartialityMultiCalculator {
  public:
    /**
     * Add a bbox calculator to the list.
     */
    void push_back(std::shared_ptr<PartialityCalculatorIface> obj) {
      compute_.push_back(obj);
    }

    /**
     * Get the number of calculators.
     */
    std::size_t size() const {
      return compute_.size();
    }

    /**
     * Calculate the rois for an array of reflections given by the array of
     * diffracted beam vectors and rotation angles.
     * @param id The experiment id
     * @param s1 The array of diffracted beam vectors
     * @param frame The array of frame numbers.
     * @param bbox The array of bounding boxes
     */
    af::shared<double> operator()(const af::const_ref<std::size_t> &id,
                                  const af::const_ref<vec3<double> > &s1,
                                  const af::const_ref<double> &frame,
                                  const af::const_ref<int6> &bbox) const {
      DIALS_ASSERT(s1.size() == id.size());
      DIALS_ASSERT(s1.size() == frame.size());
      DIALS_ASSERT(s1.size() == bbox.size());
      af::shared<double> result(s1.size(), af::init_functor_null<double>());
      for (std::size_t i = 0; i < s1.size(); ++i) {
        DIALS_ASSERT(id[i] < size());
        result[i] = compute_[id[i]]->single(s1[i], frame[i], bbox[i]);
      }
      return result;
    }

  private:
    std::vector<std::shared_ptr<PartialityCalculatorIface> > compute_;
  };

}}}}  // namespace dials::algorithms::profile_model::gaussian_rs

#endif  // DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_PARTIALITY_CALCULATOR_H
