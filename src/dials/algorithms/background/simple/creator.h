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
#ifndef DIALS_ALGORITHMS_BACKGROUND_CREATOR_H
#define DIALS_ALGORITHMS_BACKGROUND_CREATOR_H

#include <omptbx/omp_or_stubs.h>
#include <cmath>
#include <memory>
#include <dials/array_family/reflection_table.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/background/simple/outlier_rejector.h>
#include <dials/algorithms/background/simple/modeller.h>
#include <dials/model/data/shoebox.h>
#include <dials/model/data/image_volume.h>
#include <dials/error.h>

namespace dials { namespace algorithms { namespace background {

  using dials::model::Background;
  using dials::model::BackgroundUsed;
  using dials::model::Overlapped;
  using dials::model::Shoebox;
  using dials::model::Valid;
  using model::ImageVolume;
  using model::MultiPanelImageVolume;

  /**
   * Class to create background shoebox
   */
  class SimpleBackgroundCreator {
  public:
    /**
     * Initialise with the desired modeller.
     * @param modeller The background modeller
     */
    SimpleBackgroundCreator(std::shared_ptr<Modeller> modeller, std::size_t min_pixels)
        : modeller_(modeller), min_pixels_(min_pixels) {
      DIALS_ASSERT(modeller != NULL);
      DIALS_ASSERT(min_pixels > 0);
    }

    /**
     * Initialise with the desired modeller and outlier rejector.
     * @param modeller The background modeller
     * @param rejector The outlier rejector
     */
    SimpleBackgroundCreator(std::shared_ptr<Modeller> modeller,
                            std::shared_ptr<OutlierRejector> rejector,
                            std::size_t min_pixels)
        : modeller_(modeller), rejector_(rejector), min_pixels_(min_pixels) {
      DIALS_ASSERT(modeller != NULL);
      DIALS_ASSERT(min_pixels > 0);
    }

    /**
     * Create the background for the list of shoeboxes.
     * @param shoeboxes The list of shoeboxes
     * @return Success True/False per shoebox
     */
    template <typename FloatType>
    af::shared<bool> operator()(const af::const_ref<Shoebox<FloatType> > &shoeboxes,
                                af::ref<double> mse,
                                af::ref<double> dispersion) const {
      af::shared<bool> result(shoeboxes.size(), true);
      for (std::size_t i = 0; i < shoeboxes.size(); ++i) {
        try {
          af::tiny<FloatType, 2> r = this->operator()(shoeboxes[i]);
          mse[i] = r[0];
          dispersion[i] = r[1];
        } catch (dials::error const &) {
          result[i] = false;
          mse[i] = 0.0;
          dispersion[i] = 0.0;
        } catch (std::runtime_error const &) {
          result[i] = false;
          mse[i] = 0.0;
          dispersion[i] = 0.0;
        }
      }
      return result;
    }

    /**
     * Compute the background values
     * @param reflections The reflection table
     * @param volume The image volume
     * @returns Success True/False
     */
    template <typename FloatType>
    af::shared<bool> operator()(af::reflection_table reflections,
                                MultiPanelImageVolume<FloatType> volume) const {
      DIALS_ASSERT(reflections.contains("bbox"));
      DIALS_ASSERT(reflections.contains("panel"));
      af::const_ref<int6> bbox = reflections["bbox"];
      af::const_ref<std::size_t> panel = reflections["panel"];
      af::shared<bool> success(bbox.size(), true);
      for (std::size_t i = 0; i < bbox.size(); ++i) {
        // Get the image volume
        ImageVolume<FloatType> v = volume.get(panel[i]);

        // Trim the bbox
        int6 b = v.trim_bbox(bbox[i]);

        // Extract from image volume
        af::versa<FloatType, af::c_grid<3> > data = v.extract_data(b);
        af::versa<FloatType, af::c_grid<3> > bgrd = v.extract_background(b);
        af::versa<int, af::c_grid<3> > mask = v.extract_mask(b, i);

        // Compute the background
        try {
          this->operator()(data.const_ref(), mask.ref(), bgrd.ref());

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

    /**
     * Create the background for the shoebox
     * @param shoebox The shoebox
     */
    template <typename FloatType>
    af::tiny<FloatType, 2> operator()(Shoebox<FloatType> shoebox) const {
      return this->operator()(
        shoebox.data.const_ref(), shoebox.mask.ref(), shoebox.background.ref());
    }

    /**
     * Create the background for the given data
     * @param data The shoebox pixel values
     * @param mask The shoebox mask values
     * @param background The shoebox background
     */
    template <typename FloatType>
    af::tiny<FloatType, 2> operator()(
      const af::const_ref<FloatType, af::c_grid<3> > &data_in,
      af::ref<int, af::c_grid<3> > mask,
      af::ref<FloatType, af::c_grid<3> > background) const {
      // Copy the array to a double
      af::versa<double, af::c_grid<3> > data(data_in.accessor());
      std::copy(data_in.begin(), data_in.end(), data.begin());

      // Do outlier rejection on the pixels
      if (rejector_) {
        rejector_->mark(data.const_ref(), mask);
      } else {
        for (std::size_t k = 0; k < mask.accessor()[0]; ++k) {
          for (std::size_t j = 0; j < mask.accessor()[1]; ++j) {
            for (std::size_t i = 0; i < mask.accessor()[2]; ++i) {
              const int mask_code = Valid | Background;
              if ((mask(k, j, i) & mask_code) == mask_code
                  && ((mask(k, j, i) & Overlapped) == 0)) {
                mask(k, j, i) |= BackgroundUsed;
              }
            }
          }
        }
      }

      // Create a background boolean mask
      af::versa<bool, af::c_grid<3> > bgmask(mask.accessor(), false);
      for (std::size_t i = 0; i < mask.size(); ++i) {
        bgmask[i] = (mask[i] & BackgroundUsed) != 0;
      }

      // Create the background model
      std::shared_ptr<Model> model =
        modeller_->create(data.const_ref(), bgmask.const_ref());

      // Populate the background shoebox
      double mse = 0.0;
      double M = 0;
      double S = 0;
      std::size_t count = 0;
      for (std::size_t k = 0; k < background.accessor()[0]; ++k) {
        for (std::size_t j = 0; j < background.accessor()[1]; ++j) {
          for (std::size_t i = 0; i < background.accessor()[2]; ++i) {
            background(k, j, i) = model->value(k + 0.5, j + 0.5, i + 0.5);
            if (bgmask(k, j, i)) {
              // Accumulate mean and variance by Welford's one-pass algorithm
              count++;
              double x = data(k, j, i);
              double oldM = M;
              M = M + (x - M) / count;
              S = S + (x - M) * (x - oldM);

              // Also accumulate mean squared error of the background estimator
              double residual = (background(k, j, i) - data(k, j, i));
              mse += residual * residual;
            }
          }
        }
      }
      DIALS_ASSERT(count >= min_pixels_);
      double mean = M;
      double var = S / (count - 1);
      DIALS_ASSERT(mean >= 0);
      DIALS_ASSERT(var >= 0);
      double dispersion = mean > 0 ? var / mean : 0;
      mse /= count;
      return af::tiny<FloatType, 2>(mse, dispersion);
    }

  private:
    std::shared_ptr<Modeller> modeller_;
    std::shared_ptr<OutlierRejector> rejector_;
    std::size_t min_pixels_;
  };

}}}  // namespace dials::algorithms::background

#endif  // DIALS_ALGORITHMS_BACKGROUND_CREATOR_H
