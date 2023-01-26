/*
 * radial_average.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_BACKGROUND_RADIAL_AVERAGE_H
#define DIALS_ALGORITHMS_BACKGROUND_RADIAL_AVERAGE_H

#include <memory>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/detector.h>
#include <dials/error.h>

namespace dials { namespace algorithms {

  using dxtbx::model::BeamBase;
  using dxtbx::model::Detector;
  using dxtbx::model::Panel;

  class RadialAverage {
  public:
    RadialAverage(std::shared_ptr<BeamBase> beam,
                  const Detector &detector,
                  double vmin,
                  double vmax,
                  std::size_t num_bins)
        : beam_(beam),
          detector_(detector),
          sum_(num_bins, 0),
          weight_(num_bins, 0),
          inv_d2_(num_bins, 0),
          vmin_(vmin),
          vmax_(vmax),
          num_bins_(num_bins),
          current_(0) {
      DIALS_ASSERT(vmax > vmin);
      DIALS_ASSERT(num_bins > 0);
      for (std::size_t i = 0; i < inv_d2_.size(); ++i) {
        inv_d2_[i] = vmin + i * (vmax - vmin) / num_bins_;
      }
    }

    void add(const af::const_ref<double, af::c_grid<2> > &data,
             const af::const_ref<bool, af::c_grid<2> > &mask) {
      DIALS_ASSERT(data.accessor().all_eq(mask.accessor()));
      vec3<double> s0 = beam_->get_s0();
      const Panel &panel = detector_[current_++];
      std::size_t height = panel.get_image_size()[1];
      std::size_t width = panel.get_image_size()[0];
      DIALS_ASSERT(data.accessor()[0] == height);
      DIALS_ASSERT(data.accessor()[1] == width);
      for (std::size_t j = 0; j < height; ++j) {
        for (std::size_t i = 0; i < width; ++i) {
          if (mask(j, i)) {
            double d = panel.get_resolution_at_pixel(s0, vec2<double>(i, j));
            double d2 = (1.0 / (d * d));
            if (d2 >= vmin_ && d2 < vmax_) {
              double b = vmin_;
              double a = (vmax_ - vmin_) / num_bins_;
              int index = std::floor((d2 - b) / a);
              DIALS_ASSERT(index >= 0 && index < num_bins_);
              sum_[index] += data(j, i);
              weight_[index] += 1.0;
            }
          }
        }
      }
    }

    af::shared<double> mean() const {
      af::shared<double> result(sum_.size());
      for (std::size_t i = 0; i < sum_.size(); ++i) {
        if (weight_[i] > 0) {
          result[i] = sum_[i] / weight_[i];
        } else {
          result[i] = 0.0;
        }
      }
      return result;
    }

    af::shared<double> weight() const {
      return weight_;
    }

    af::shared<double> inv_d2() const {
      return inv_d2_;
    }

  private:
    std::shared_ptr<BeamBase> beam_;
    Detector detector_;
    af::shared<double> sum_;
    af::shared<double> weight_;
    af::shared<double> inv_d2_;
    double vmin_;
    double vmax_;
    std::size_t num_bins_;
    std::size_t current_;
  };

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_BACKGROUND_RADIAL_AVERAGE_H */
