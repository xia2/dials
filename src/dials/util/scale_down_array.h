/*
 * FIXME add a header
 */

#ifndef DIALS_UTIL_SCALED_DOWN_ARRAY_H
#define DIALS_UTIL_SCALED_DOWN_ARRAY_H

#include <ctime>
#include <random>
#include <dials/array_family/scitbx_shared_and_versa.h>

namespace dials { namespace util {
  /**
   * Calculate a randomly downscaled new image with the same dimensions
   * as the input array.
   */

  af::shared<int> scale_down_array(const af::const_ref<int> &image,
                                   const double scale_factor) {
    // create the RNG

    std::mt19937 gen(time(0));
    std::uniform_real_distribution<double> dist(0, 1);

    // create the target array

    af::shared<int> result(image.size(), 0);

    // now iterate through the array & do what we need to do...

    for (int j = 0; j < image.size(); ++j) {
      if (image[j] <= 0) {
        result[j] = image[j];
      } else {
        for (int k = 0; k < image[j]; k++) {
          if (dist(gen) < scale_factor) {
            result[j] += 1;
          }
        }
      }
    }

    return result;
  }
}}  // namespace dials::util

#endif
