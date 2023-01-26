/*
 * creator.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */

#ifndef DIALS_ALGORITHMS_BACKGROUND_GMODEL_CREATOR_H
#define DIALS_ALGORITHMS_BACKGROUND_GMODEL_CREATOR_H

#include <memory>
#include <dials/array_family/reflection_table.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/model/data/shoebox.h>
#include <dials/model/data/image_volume.h>
#include <dials/algorithms/background/gmodel/model.h>
#include <dials/algorithms/background/gmodel/robust_estimator.h>
#include <dials/algorithms/background/glm/robust_poisson_mean.h>
#include <dials/error.h>

namespace dials { namespace algorithms {

  using model::ImageVolume;
  using model::MultiPanelImageVolume;
  using model::Shoebox;

  /**
   * A class to create the background model
   */
  class GModelBackgroundCreator {
  public:
    /**
     * Initialise the creator
     * @param min_pixels The minimum number of pixels needed
     */
    GModelBackgroundCreator(std::shared_ptr<BackgroundModel> model,
                            bool robust,
                            std::size_t min_pixels)
        : model_(model), robust_(robust), min_pixels_(min_pixels) {
      DIALS_ASSERT(min_pixels > 0);
    }

    /**
     * Compute the background values
     * @param sbox The shoeboxes
     * @returns Success True/False
     */
    void single(Shoebox<> sbox) const {
      compute(sbox.panel,
              sbox.bbox,
              sbox.data.const_ref(),
              sbox.background.ref(),
              sbox.mask.ref());
    }

    /**
     * Compute the background values
     * @param sbox The shoeboxes
     * @returns Success True/False
     */
    af::shared<bool> shoebox(af::reflection_table reflections) const {
      DIALS_ASSERT(reflections.contains("shoebox"));
      af::ref<Shoebox<> > sbox = reflections["shoebox"];
      af::shared<bool> success(sbox.size(), true);
      af::shared<double> scale = reflections["background.scale"];
      for (std::size_t i = 0; i < sbox.size(); ++i) {
        try {
          DIALS_ASSERT(sbox[i].is_consistent());
          scale[i] = compute(sbox[i].panel,
                             sbox[i].bbox,
                             sbox[i].data.const_ref(),
                             sbox[i].background.ref(),
                             sbox[i].mask.ref());
        } catch (scitbx::error const &) {
          success[i] = false;
        } catch (dials::error const &) {
          success[i] = false;
        }
      }
      return success;
    }

    /**
     * Compute the background values
     * @param reflections The reflection table
     * @param volume The image volume
     * @returns Success True/False
     */
    af::shared<bool> volume(af::reflection_table reflections,
                            MultiPanelImageVolume<> volume) const {
      typedef MultiPanelImageVolume<>::float_type FloatType;

      DIALS_ASSERT(reflections.contains("bbox"));
      DIALS_ASSERT(reflections.contains("panel"));
      af::const_ref<int6> bbox = reflections["bbox"];
      af::const_ref<std::size_t> panel = reflections["panel"];
      af::shared<bool> success(bbox.size(), true);
      af::shared<double> scale = reflections["background.scale"];
      for (std::size_t i = 0; i < bbox.size(); ++i) {
        // Get the image volume
        ImageVolume<> v = volume.get(panel[i]);

        // Trim the bbox
        int6 b = v.trim_bbox(bbox[i]);
        std::size_t p = panel[i];

        // Extract from image volume
        af::versa<FloatType, af::c_grid<3> > data = v.extract_data(b);
        af::versa<FloatType, af::c_grid<3> > bgrd = v.extract_background(b);
        af::versa<int, af::c_grid<3> > mask = v.extract_mask(b, i);

        // Compute the background
        try {
          scale[i] = compute(p, b, data.const_ref(), bgrd.ref(), mask.ref());

          // Need to set the background in volume
          v.set_background(b, bgrd.const_ref());
        } catch (scitbx::error const &) {
          success[i] = false;
        } catch (dials::error const &) {
          success[i] = false;
        }
      }
      return success;
    }

  private:
    /**
     * Compute the background values for a single shoebox
     * @param sbox The shoebox
     */
    template <typename T>
    double compute(std::size_t panel,
                   int6 bbox,
                   const af::const_ref<T, af::c_grid<3> > &data,
                   af::ref<T, af::c_grid<3> > background,
                   af::ref<int, af::c_grid<3> > mask) const {
      if (robust_) {
        return compute_robust(panel, bbox, data, background, mask);
      } else {
        return compute_non_robust(panel, bbox, data, background, mask);
      }
    }

    /**
     * Compute the background values for a single shoebox
     * @param sbox The shoebox
     */
    template <typename T>
    double compute_non_robust(std::size_t panel,
                              int6 bbox,
                              const af::const_ref<T, af::c_grid<3> > &data,
                              af::ref<T, af::c_grid<3> > background,
                              af::ref<int, af::c_grid<3> > mask) const {
      af::versa<double, af::c_grid<3> > model = model_->extract(panel, bbox);
      double sum1 = 0;
      double sum2 = 0;
      double count = 0;
      int mask_code = Valid | Background;
      for (std::size_t i = 0; i < data.size(); ++i) {
        if ((mask[i] & mask_code) == mask_code) {
          sum1 += data[i];
          sum2 += model[i];
          count += 1;
        }
      }
      DIALS_ASSERT(count >= min_pixels_);
      DIALS_ASSERT(sum1 >= 0);
      DIALS_ASSERT(sum2 > 0);
      double scale = sum1 / sum2;
      for (std::size_t i = 0; i < data.size(); ++i) {
        background[i] = scale * model[i];
        if ((mask[i] & mask_code) == mask_code) {
          mask[i] |= BackgroundUsed;
        }
      }
      return scale;
    }

    /**
     * Compute the background values for a single shoebox
     * @param sbox The shoebox
     */
    template <typename T>
    double compute_robust(std::size_t panel,
                          int6 bbox,
                          const af::const_ref<T, af::c_grid<3> > &data,
                          af::ref<T, af::c_grid<3> > background,
                          af::ref<int, af::c_grid<3> > mask) const {
      af::versa<double, af::c_grid<3> > model = model_->extract(panel, bbox);

      // Compute number of background pixels
      std::size_t num_background = 0;
      double sum_model = 0.0;
      int mask_code = Valid | Background;
      for (std::size_t i = 0; i < data.size(); ++i) {
        if ((mask[i] & mask_code) == mask_code) {
          num_background++;
          sum_model += model[i];
        }
      }
      DIALS_ASSERT(sum_model > 0);
      DIALS_ASSERT(num_background >= min_pixels_);

      // Allocate some arrays
      af::shared<double> X(num_background, 0);
      af::shared<double> Y(num_background, 0);
      af::shared<double> W(num_background, 1.0);
      std::size_t l = 0;
      for (std::size_t i = 0; i < data.size(); ++i) {
        if ((mask[i] & mask_code) == mask_code) {
          DIALS_ASSERT(l < Y.size());
          DIALS_ASSERT(data[i] >= 0);
          X[l] = model[i];
          Y[l] = data[i];
          l++;
        }
      }
      DIALS_ASSERT(l == Y.size());

      // Estimate the weights for each pixel
      estimate_pixel_weights(X.const_ref(), Y.const_ref(), W.ref());

      // Estimate the scale parameter
      double scale =
        estimate_scale_parameter(X.const_ref(), Y.const_ref(), W.const_ref());

      // Fill in the background shoebox values
      for (std::size_t i = 0; i < data.size(); ++i) {
        background[i] = scale * model[i];
        if ((mask[i] & mask_code) == mask_code) {
          mask[i] |= BackgroundUsed;
        }
      }

      return scale;
    }

    void estimate_pixel_weights(const af::const_ref<double> &X,
                                const af::const_ref<double> &Y,
                                af::ref<double> W) const {
      af::shared<double> tX(X.size());
      af::shared<double> tY(X.size());
      for (std::size_t i = 0; i < X.size(); ++i) {
        tX[i] = 2.0 * std::sqrt(X[i] + 3.0 / 8.0);
        tY[i] = 2.0 * std::sqrt(Y[i] + 3.0 / 8.0);
      }

      double B = 1.0;
      for (std::size_t iter = 0; iter < 10; ++iter) {
        double XWX = 0.0;
        double XWY = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i) {
          double mu = B * tX[i];
          double r = std::abs(tY[i] - mu);
          W[i] = 1.0;
          if (r > 3.0) {
            W[i] = 3.0 / r;
          }
          XWX += tX[i] * W[i] * tX[i];
          XWY += tX[i] * W[i] * tY[i];
        }
        DIALS_ASSERT(XWX > 0);
        B = XWY / XWX;
      }
    }

    double estimate_scale_parameter(const af::const_ref<double> &X,
                                    const af::const_ref<double> &Y,
                                    const af::const_ref<double> &W) const {
      double XWX = 0.0;
      double XWY = 0.0;
      for (std::size_t i = 0; i < X.size(); ++i) {
        XWX += X[i] * W[i] * X[i];
        XWY += X[i] * W[i] * Y[i];
      }
      DIALS_ASSERT(XWX > 0);
      return XWY / XWX;
    }

    std::shared_ptr<BackgroundModel> model_;
    bool robust_;
    std::size_t min_pixels_;
  };

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_BACKGROUND_GMODEL_CREATOR_H
