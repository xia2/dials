/*
 * modeller.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */

#ifndef DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_MODELLER_H
#define DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_MODELLER_H

#include <memory>
#include <fstream>
#include <dials/algorithms/profile_model/gaussian_rs/transform/transform.h>
#include <dials/algorithms/profile_model/modeller/empirical_modeller.h>
#include <dials/algorithms/profile_model/modeller/single_sampler.h>
#include <dials/algorithms/profile_model/modeller/grid_sampler.h>
#include <dials/algorithms/profile_model/modeller/circle_sampler.h>
#include <dials/algorithms/profile_model/modeller/ewald_sphere_sampler.h>
#include <dials/algorithms/integration/fit/fitting.h>

namespace dials { namespace algorithms {

  using dials::algorithms::profile_model::gaussian_rs::CoordinateSystem;
  using dials::algorithms::profile_model::gaussian_rs::transform::TransformForward;
  using dials::algorithms::profile_model::gaussian_rs::transform::TransformReverse;
  using dials::algorithms::profile_model::gaussian_rs::transform::TransformSpec;
  using dials::model::Shoebox;
  using dxtbx::model::BeamBase;
  using dxtbx::model::Detector;
  using dxtbx::model::Goniometer;
  using dxtbx::model::Scan;

  /**
   * A base class to initialize the sampler
   */
  class GaussianRSProfileModellerBase {
  public:
    enum GridMethod {
      Single = 1,
      RegularGrid = 2,
      CircularGrid = 3,
      SphericalGrid = 4,
    };

    enum FitMethod { ReciprocalSpace = 1, DetectorSpace = 2 };

    GaussianRSProfileModellerBase(const std::shared_ptr<BeamBase> beam,
                                  const Detector &detector,
                                  const Goniometer &goniometer,
                                  const Scan &scan,
                                  double sigma_b,
                                  double sigma_m,
                                  double n_sigma,
                                  std::size_t grid_size,
                                  std::size_t num_scan_points,
                                  int grid_method,
                                  int fit_method)
        : beam_(beam),
          detector_(detector),
          goniometer_(goniometer),
          scan_(scan),
          sigma_b_(sigma_b),
          sigma_m_(sigma_m),
          n_sigma_(n_sigma),
          grid_size_(grid_size),
          num_scan_points_(num_scan_points),
          grid_method_(grid_method),
          fit_method_(fit_method),
          sampler_(init_sampler(beam,
                                detector,
                                goniometer,
                                scan,
                                num_scan_points,
                                grid_method)) {}

  protected:
    std::shared_ptr<SamplerIface> init_sampler(std::shared_ptr<BeamBase> beam,
                                               const Detector &detector,
                                               const Goniometer &goniometer,
                                               const Scan &scan,
                                               std::size_t num_scan_points,
                                               int grid_method) {
      int2 scan_range = scan.get_array_range();
      std::shared_ptr<SamplerIface> sampler;
      if (grid_method == RegularGrid || grid_method == CircularGrid) {
        if (detector.size() > 1) {
          grid_method = Single;
        }
      }
      switch (grid_method) {
      case Single:
        sampler = std::make_shared<SingleSampler>(scan_range, num_scan_points);
        break;
      case RegularGrid:
        DIALS_ASSERT(detector.size() == 1);
        sampler = std::make_shared<GridSampler>(
          detector[0].get_image_size(), scan_range, int3(3, 3, num_scan_points));
        break;
      case CircularGrid:
        DIALS_ASSERT(detector.size() == 1);
        sampler = std::make_shared<CircleSampler>(
          detector[0].get_image_size(), scan_range, num_scan_points);
        break;
      case SphericalGrid:
        sampler = std::make_shared<EwaldSphereSampler>(
          beam, detector, goniometer, scan, num_scan_points);
      default:
        throw DIALS_ERROR("Unknown grid method");
      };
      return sampler;
    }

    std::shared_ptr<BeamBase> beam_;
    Detector detector_;
    Goniometer goniometer_;
    Scan scan_;
    double sigma_b_;
    double sigma_m_;
    double n_sigma_;
    std::size_t grid_size_;
    std::size_t num_scan_points_;
    int grid_method_;
    int fit_method_;
    std::shared_ptr<SamplerIface> sampler_;
  };

  namespace detail {

    struct check_mask_code {
      int mask_code;
      check_mask_code(int code) : mask_code(code) {}
      bool operator()(int a) const {
        return ((a & mask_code) == mask_code);
      }
    };

    struct check_either_mask_code {
      int mask_code1;
      int mask_code2;
      check_either_mask_code(int code1, int code2)
          : mask_code1(code1), mask_code2(code2) {}
      bool operator()(int a) const {
        return ((a & mask_code1) == mask_code1) || ((a & mask_code2) == mask_code2);
      }
    };

  }  // namespace detail

  /**
   * The profile modeller for the gaussian rs profile model
   */
  class GaussianRSProfileModeller : public GaussianRSProfileModellerBase,
                                    public EmpiricalProfileModeller {
  public:
    /**
     * Initialize
     * @param beam The beam model
     * @param detector The detector model
     * @param goniometer The goniometer model
     * @param scan The scan model
     * @param sigma_b The beam divergence
     * @param sigma_m The mosaicity
     * @param n_sigma The extent
     * @param grid_size The size of the profile grid
     * @param num_scan_points The number of phi scan points
     * @param threshold The modelling threshold value
     * @param grid_method The gridding method
     */
    GaussianRSProfileModeller(std::shared_ptr<BeamBase> beam,
                              const Detector &detector,
                              const Goniometer &goniometer,
                              const Scan &scan,
                              double sigma_b,
                              double sigma_m,
                              double n_sigma,
                              std::size_t grid_size,
                              std::size_t num_scan_points,
                              double threshold,
                              int grid_method,
                              int fit_method)
        : GaussianRSProfileModellerBase(beam,
                                        detector,
                                        goniometer,
                                        scan,
                                        sigma_b,
                                        sigma_m,
                                        n_sigma,
                                        grid_size,
                                        num_scan_points,
                                        grid_method,
                                        fit_method),
          EmpiricalProfileModeller(
            sampler_->size(),
            int3(2 * grid_size + 1, 2 * grid_size + 1, 2 * grid_size + 1),
            threshold),
          spec_(beam,
                detector,
                goniometer,
                scan,
                sigma_b,
                sigma_m,
                n_sigma,
                grid_size) {
      DIALS_ASSERT(sampler_ != 0);
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

    double sigma_b() const {
      return sigma_b_;
    }

    double sigma_m() const {
      return sigma_m_;
    }

    double n_sigma() const {
      return n_sigma_;
    }

    std::size_t grid_size() const {
      return grid_size_;
    }

    std::size_t num_scan_points() const {
      return num_scan_points_;
    }

    double threshold() const {
      return threshold_;
    }

    int grid_method() const {
      return grid_method_;
    }

    int fit_method() const {
      return fit_method_;
    }

    vec3<double> coord(std::size_t index) const {
      return sampler_->coord(index);
    }

    /**
     * Model the profiles from the reflections
     * @param reflections The reflection list
     */
    void model(af::reflection_table reflections) {
      // Check input is OK
      DIALS_ASSERT(reflections.is_consistent());
      DIALS_ASSERT(reflections.contains("shoebox"));
      DIALS_ASSERT(reflections.contains("flags"));
      DIALS_ASSERT(reflections.contains("partiality"));
      DIALS_ASSERT(reflections.contains("s1"));
      DIALS_ASSERT(reflections.contains("xyzcal.px"));
      DIALS_ASSERT(reflections.contains("xyzcal.mm"));

      // Get some data
      af::const_ref<Shoebox<> > sbox = reflections["shoebox"];
      af::const_ref<double> partiality = reflections["partiality"];
      af::const_ref<vec3<double> > s1 = reflections["s1"];
      af::const_ref<vec3<double> > xyzpx = reflections["xyzcal.px"];
      af::const_ref<vec3<double> > xyzmm = reflections["xyzcal.mm"];
      af::ref<std::size_t> flags = reflections["flags"];

      // Loop through all the reflections and add them to the model
      for (std::size_t i = 0; i < reflections.size(); ++i) {
        DIALS_ASSERT(sbox[i].is_consistent());

        // Check if we want to use this reflection
        if (check1(flags[i], partiality[i], sbox[i])) {
          // Create the coordinate system
          vec3<double> m2 = spec_.goniometer().get_rotation_axis();
          vec3<double> s0 = spec_.beam()->get_s0();
          CoordinateSystem cs(m2, s0, s1[i], xyzmm[i][2]);

          // Create the data array
          af::versa<double, af::c_grid<3> > data(sbox[i].data.accessor());
          std::transform(sbox[i].data.begin(),
                         sbox[i].data.end(),
                         sbox[i].background.begin(),
                         data.begin(),
                         std::minus<double>());

          // Create the mask array
          af::versa<bool, af::c_grid<3> > mask(sbox[i].mask.accessor());
          std::transform(sbox[i].mask.begin(),
                         sbox[i].mask.end(),
                         mask.begin(),
                         detail::check_mask_code(Valid | Foreground));

          // Compute the transform
          TransformForward<double> transform(
            spec_, cs, sbox[i].bbox, sbox[i].panel, data.const_ref(), mask.const_ref());

          // Get the indices and weights of the profiles
          af::shared<std::size_t> indices =
            sampler_->nearest_n(sbox[i].panel, xyzpx[i]);
          af::shared<double> weights(indices.size());
          for (std::size_t j = 0; j < indices.size(); ++j) {
            weights[j] = sampler_->weight(indices[j], sbox[i].panel, xyzpx[i]);
          }

          // Add the profile
          add(
            indices.const_ref(), weights.const_ref(), transform.profile().const_ref());

          // Set the flags
          flags[i] |= af::UsedInModelling;
        }
      }
    }

    /**
     * Return a profile fitter
     * @return The profile fitter class
     */
    af::shared<bool> fit(af::reflection_table reflections) const {
      af::shared<bool> success;
      switch (fit_method_) {
      case ReciprocalSpace:
        success = fit_reciprocal_space(reflections);
        break;
      case DetectorSpace:
        success = fit_detector_space(reflections);
        break;
      default:
        throw DIALS_ERROR("Unknown fitting method");
      };
      return success;
    }

    /**
     * Return a profile fitter
     * @return The profile fitter class
     */
    void validate(af::reflection_table reflections) const {
      switch (fit_method_) {
      case ReciprocalSpace:
        fit_reciprocal_space(reflections);
        break;
      case DetectorSpace:
        fit_detector_space(reflections);
        break;
      default:
        throw DIALS_ERROR("Unknown fitting method");
      };
    }

    /**
     * Return a profile fitter
     * @return The profile fitter class
     */
    af::shared<bool> fit_reciprocal_space(af::reflection_table reflections) const {
      // Check input is OK
      DIALS_ASSERT(reflections.is_consistent());
      DIALS_ASSERT(reflections.contains("shoebox"));
      DIALS_ASSERT(reflections.contains("flags"));
      DIALS_ASSERT(reflections.contains("partiality"));
      DIALS_ASSERT(reflections.contains("s1"));
      DIALS_ASSERT(reflections.contains("xyzcal.px"));
      DIALS_ASSERT(reflections.contains("xyzcal.mm"));

      // Get some data
      af::const_ref<Shoebox<> > sbox = reflections["shoebox"];
      af::const_ref<vec3<double> > s1 = reflections["s1"];
      af::const_ref<vec3<double> > xyzpx = reflections["xyzcal.px"];
      af::const_ref<vec3<double> > xyzmm = reflections["xyzcal.mm"];
      af::ref<std::size_t> flags = reflections["flags"];
      af::ref<double> intensity_val = reflections["intensity.prf.value"];
      af::ref<double> intensity_var = reflections["intensity.prf.variance"];
      af::ref<double> reference_cor = reflections["profile.correlation"];
      // af::ref<double> reference_rmsd = reflections["profile.rmsd"];

      // Loop through all the reflections and process them
      af::shared<bool> success(reflections.size(), false);
      for (std::size_t i = 0; i < reflections.size(); ++i) {
        DIALS_ASSERT(sbox[i].is_consistent());

        // Set values to bad
        intensity_val[i] = 0.0;
        intensity_var[i] = -1.0;
        reference_cor[i] = 0.0;
        // reference_rmsd[i] = 0.0;
        flags[i] &= ~af::IntegratedPrf;
        bool integrate = !(flags[i] & af::DontIntegrate);

        // Check if we want to use this reflection
        if (integrate) {
          try {
            // Get the reference profiles
            std::size_t index = sampler_->nearest(sbox[i].panel, xyzpx[i]);
            data_const_reference p = data(index).const_ref();
            mask_const_reference mask1 = mask(index).const_ref();

            // Create the coordinate system
            vec3<double> m2 = spec_.goniometer().get_rotation_axis();
            vec3<double> s0 = spec_.beam()->get_s0();
            CoordinateSystem cs(m2, s0, s1[i], xyzmm[i][2]);

            // Create the data array
            af::versa<double, af::c_grid<3> > data(sbox[i].data.accessor());
            std::copy(sbox[i].data.begin(), sbox[i].data.end(), data.begin());

            // Create the background array
            af::versa<double, af::c_grid<3> > background(sbox[i].background.accessor());
            std::copy(
              sbox[i].background.begin(), sbox[i].background.end(), background.begin());

            // Create the mask array
            af::versa<bool, af::c_grid<3> > mask(sbox[i].mask.accessor());
            std::transform(sbox[i].mask.begin(),
                           sbox[i].mask.end(),
                           mask.begin(),
                           detail::check_mask_code(Valid | Foreground));

            // Compute the transform
            TransformForward<double> transform(spec_,
                                               cs,
                                               sbox[i].bbox,
                                               sbox[i].panel,
                                               data.const_ref(),
                                               background.const_ref(),
                                               mask.const_ref());

            // Get the transformed shoebox
            data_const_reference c = transform.profile().const_ref();
            data_const_reference b = transform.background().const_ref();
            mask_const_reference mask2 = transform.mask().const_ref();
            af::versa<bool, af::c_grid<3> > m(mask2.accessor());
            DIALS_ASSERT(mask1.size() == mask2.size());
            for (std::size_t j = 0; j < m.size(); ++j) {
              m[j] = mask1[j] && mask2[j];
            }

            // Do the profile fitting
            ProfileFitter<double> fit(c, b, m.const_ref(), p, 1e-3, 100);
            // DIALS_ASSERT(fit.niter() < 100);

            // Set the data in the reflection
            intensity_val[i] = fit.intensity()[0];
            intensity_var[i] = fit.variance()[0];
            reference_cor[i] = fit.correlation();
            // reference_rmsd[i] = fit.rmsd();

            // Set the integrated flag
            flags[i] |= af::IntegratedPrf;
            success[i] = true;

          } catch (dials::error const &e) {
            /* std::cout << e.what() << std::endl; */
            continue;
          }
        }
      }
      return success;
    }

    /**
     * Return a profile fitter
     * @return The profile fitter class
     */
    af::shared<bool> fit_detector_space(af::reflection_table reflections) const {
      // Check input is OK
      DIALS_ASSERT(reflections.is_consistent());
      DIALS_ASSERT(reflections.contains("shoebox"));
      DIALS_ASSERT(reflections.contains("flags"));
      DIALS_ASSERT(reflections.contains("partiality"));
      DIALS_ASSERT(reflections.contains("s1"));
      DIALS_ASSERT(reflections.contains("xyzcal.px"));
      DIALS_ASSERT(reflections.contains("xyzcal.mm"));

      // Get some data
      af::const_ref<Shoebox<> > sbox = reflections["shoebox"];
      af::const_ref<vec3<double> > s1 = reflections["s1"];
      af::const_ref<vec3<double> > xyzpx = reflections["xyzcal.px"];
      af::const_ref<vec3<double> > xyzmm = reflections["xyzcal.mm"];
      af::ref<std::size_t> flags = reflections["flags"];
      af::ref<double> intensity_val = reflections["intensity.prf.value"];
      af::ref<double> intensity_var = reflections["intensity.prf.variance"];
      af::ref<double> reference_cor = reflections["profile.correlation"];

      // Loop through all the reflections and process them
      af::shared<bool> success(reflections.size(), false);
      for (std::size_t i = 0; i < reflections.size(); ++i) {
        DIALS_ASSERT(sbox[i].is_consistent());

        // Set values to bad
        intensity_val[i] = 0.0;
        intensity_var[i] = -1.0;
        reference_cor[i] = 0.0;
        flags[i] &= ~af::IntegratedPrf;

        // Check if we want to use this reflection
        if (check2(flags[i], sbox[i])) {
          try {
            // Get the reference profiles
            std::size_t index = sampler_->nearest(sbox[i].panel, xyzpx[i]);
            data_const_reference d = data(index).const_ref();

            // Create the coordinate system
            vec3<double> m2 = spec_.goniometer().get_rotation_axis();
            vec3<double> s0 = spec_.beam()->get_s0();
            CoordinateSystem cs(m2, s0, s1[i], xyzmm[i][2]);

            // Compute the transform
            TransformReverse transform(spec_, cs, sbox[i].bbox, sbox[i].panel, d);

            // Get the transformed shoebox
            data_const_reference p = transform.profile().const_ref();

            // Create the data array
            af::versa<double, af::c_grid<3> > c(sbox[i].data.accessor());
            std::copy(sbox[i].data.begin(), sbox[i].data.end(), c.begin());

            // Create the background array
            af::versa<double, af::c_grid<3> > b(sbox[i].background.accessor());
            std::copy(sbox[i].background.begin(), sbox[i].background.end(), b.begin());

            // Create the mask array
            af::versa<bool, af::c_grid<3> > m(sbox[i].mask.accessor());

            std::transform(sbox[i].mask.begin(),
                           sbox[i].mask.end(),
                           m.begin(),
                           detail::check_mask_code(Valid | Foreground));

            // Do the profile fitting
            ProfileFitter<double> fit(
              c.const_ref(), b.const_ref(), m.const_ref(), p, 1e-3, 100);
            // DIALS_ASSERT(fit.niter() < 100);

            // Set the data in the reflection
            intensity_val[i] = fit.intensity()[0];
            intensity_var[i] = fit.variance()[0];
            reference_cor[i] = fit.correlation();

            // Set the integrated flag
            flags[i] |= af::IntegratedPrf;
            success[i] = true;

          } catch (dials::error const &e) {
            continue;
          }
        }
      }
      return success;
    }

    /**
     * @return a copy of the profile modller
     */
    pointer copy() const {
      GaussianRSProfileModeller result(beam_,
                                       detector_,
                                       goniometer_,
                                       scan_,
                                       sigma_b_,
                                       sigma_m_,
                                       n_sigma_,
                                       grid_size_,
                                       num_scan_points_,
                                       threshold_,
                                       grid_method_,
                                       fit_method_);
      result.finalized_ = finalized_;
      result.n_reflections_.assign(n_reflections_.begin(), n_reflections_.end());
      for (std::size_t i = 0; i < data_.size(); ++i) {
        if (data_[i].size() > 0) {
          result.data_[i] = data_type(accessor_, 0);
          result.mask_[i] = mask_type(accessor_, true);
          std::copy(data_[i].begin(), data_[i].end(), result.data_[i].begin());
          std::copy(mask_[i].begin(), mask_[i].end(), result.mask_[i].begin());
        }
      }
      return pointer(new GaussianRSProfileModeller(result));
    }

    void normalize_profiles() {
      finalize();
    }

  private:
    /**
     * Do we want to use the reflection in profile modelling
     * @param flags The reflection flags
     * @param partiality The reflection partiality
     * @param sbox The reflection shoebox
     * @return True/False
     */
    bool check1(std::size_t flags, double partiality, const Shoebox<> &sbox) const {
      // Check we're fully recorded
      bool full = partiality > 0.99;

      // Check reflection has been integrated
      bool integrated = flags & af::IntegratedSum;

      // Check if the bounding box is in the image
      bool bbox_valid = check_bbox_valid(flags, sbox);

      // Check if all pixels are valid
      bool pixels_valid = check_foreground_valid(flags, sbox);

      // Return whether to use or not
      return full && integrated && bbox_valid && pixels_valid;
    }

    /**
     * Do we want to use the reflection in profile fitting
     * @param flags The reflection flags
     * @param sbox The reflection shoebox
     * @return True/False
     */
    bool check2(std::size_t flags, const Shoebox<> &sbox) const {
      // Check if we want to integrate
      bool integrate = !(flags & af::DontIntegrate);

      // Check if the bounding box is in the image
      bool bbox_valid = check_bbox_valid(flags, sbox);

      // Check if all pixels are valid
      bool pixels_valid = check_foreground_valid(flags, sbox);

      // Return whether to use or not
      return integrate && bbox_valid && pixels_valid;
    }

    /**
     * Do we want to use the reflection in profile fitting
     * @param flags The reflection flags
     * @param sbox The reflection shoebox
     * @return True/False
     */
    bool check3(std::size_t flags, const Shoebox<> &sbox) const {
      // Check if we want to integrate
      bool integrate = !(flags & af::DontIntegrate);

      // Check if the bounding box is in the image
      bool bbox_valid = check_bbox_valid(flags, sbox);

      // Return whether to use or not
      return integrate && bbox_valid;
    }

    /**
     * Check if the bounding box is in entirely within the image
     * @param flags The reflection flags
     * @param sbox The reflection shoebox
     * @return True/False
     */
    bool check_bbox_valid(std::size_t flags, const Shoebox<> &sbox) const {
      return sbox.bbox[0] >= 0 && sbox.bbox[2] >= 0
             && sbox.bbox[1] <= spec_.detector()[sbox.panel].get_image_size()[0]
             && sbox.bbox[3] <= spec_.detector()[sbox.panel].get_image_size()[1];
    }

    /**
     * Check if all foreground pixels are valid
     * @param flags The reflection flags
     * @param sbox The reflection shoebox
     * @return True/False
     */
    bool check_foreground_valid(std::size_t flags, const Shoebox<> &sbox) const {
      bool pixels_valid = true;
      for (std::size_t i = 0; i < sbox.mask.size(); ++i) {
        if (sbox.mask[i] & Foreground && !(sbox.mask[i] & Valid)) {
          pixels_valid = false;
          break;
        }
      }
      return pixels_valid;
    }

    TransformSpec spec_;
  };

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_MODELLER_H
