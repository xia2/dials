/*
 * parallel_integrator.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */

#ifndef DIALS_ALGORITHMS_INTEGRATION_PARALLEL_INTEGRATOR_H
#define DIALS_ALGORITHMS_INTEGRATION_PARALLEL_INTEGRATOR_H

#include <atomic>
#include <boost/ptr_container/ptr_vector.hpp>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/scan.h>
#include <dxtbx/model/crystal.h>
#include <dxtbx/imageset.h>
#include <dials/array_family/reflection_table.h>
#include <dials/array_family/reflection.h>
#include <dials/error.h>
#include <dials/util/thread_pool.h>
#include <dials/algorithms/shoebox/find_overlapping.h>
#include <dials/algorithms/integration/sum/summation.h>
#include <dials/algorithms/centroid/centroid.h>
#include <dials/array_family/boost_python/flex_table_suite.h>
#include <map>

#include <dials/algorithms/integration/interfaces.h>

namespace dials { namespace algorithms {

  using dxtbx::model::Beam;
  using dxtbx::model::Detector;
  using dxtbx::model::Goniometer;
  using dxtbx::model::Panel;
  using dxtbx::model::Scan;

  using dxtbx::ImageSequence;
  using dxtbx::format::Image;
  using dxtbx::format::ImageTile;

  using dials::model::AdjacencyList;
  using dials::model::Shoebox;

  /**
   * Class to wrap logging
   */
  class Logger {
  public:
    Logger(boost::python::object obj) : obj_(obj) {}

    void info(const char *str) const {
      obj_.attr("info")(str);
    }

    void debug(const char *str) const {
      obj_.attr("debug")(str);
    }

  private:
    boost::python::object obj_;
  };

  /**
   * A class to store the image data buffer
   */
  class BufferBase {
  public:
    typedef Shoebox<>::float_type float_type;

    /**
     * Initialise the the size of the panels
     * @param detector The detector model
     * @param num_images The number of images
     * @param mask_value The value of masked pixels in the buffer
     * @param external_mask The external mask
     */
    BufferBase(const Detector &detector,
               std::size_t num_images,
               float_type mask_value,
               const Image<bool> &external_mask)
        : mask_value_(mask_value) {
      std::size_t zsize = num_images;
      DIALS_ASSERT(zsize > 0);
      for (std::size_t i = 0; i < detector.size(); ++i) {
        std::size_t xsize = detector[i].get_image_size()[0];
        std::size_t ysize = detector[i].get_image_size()[1];
        DIALS_ASSERT(xsize > 0);
        DIALS_ASSERT(ysize > 0);

        // Allocate all the data buffers
        data_.push_back(
          af::versa<float_type, af::c_grid<3> >(af::c_grid<3>(zsize, ysize, xsize)));

        // Allocate the static mask buffer
        static_mask_.push_back(
          af::versa<bool, af::c_grid<2> >(af::c_grid<2>(ysize, xsize), true));
      }

      // Set the external mask
      if (!external_mask.empty()) {
        DIALS_ASSERT(external_mask.n_tiles() == static_mask_.size());
        for (std::size_t i = 0; i < external_mask.n_tiles(); ++i) {
          set_external_mask_for_panel(external_mask.tile(i).data().const_ref(),
                                      static_mask_[i].ref());
        }
      }
    }

    /**
     * Copy an image to the buffer
     * @param data The image data
     * @param index The image index
     */
    void copy(const Image<double> &data, std::size_t index) {
      DIALS_ASSERT(data.n_tiles() == data_.size());
      for (std::size_t i = 0; i < data.n_tiles(); ++i) {
        copy(data.tile(i).data().const_ref(), data_[i].ref(), index);
        apply_mask(static_mask_[i].const_ref(), data_[i].ref(), index);
      }
    }

    /**
     * Copy an image to the buffer
     * @param data The image data
     * @param index The image index
     */
    void copy(const Image<double> &data, bool mask, std::size_t index) {
      DIALS_ASSERT(data.n_tiles() == data_.size());
      if (mask) {
        copy(data, index);
      } else {
        for (std::size_t i = 0; i < data.n_tiles(); ++i) {
          apply_mask_to_all_pixels(data_[i].ref(), index);
        }
      }
    }

    /**
     * Copy an image to the buffer
     * @param data The image data
     * @param mask The mask data
     * @param index The image index
     */
    void copy(const Image<double> &data, const Image<bool> &mask, std::size_t index) {
      DIALS_ASSERT(data.n_tiles() == mask.n_tiles());
      DIALS_ASSERT(data.n_tiles() == data_.size());
      for (std::size_t i = 0; i < data.n_tiles(); ++i) {
        copy(data.tile(i).data().const_ref(), data_[i].ref(), index);
        apply_mask(mask.tile(i).data().const_ref(), data_[i].ref(), index);
        apply_mask(static_mask_[i].const_ref(), data_[i].ref(), index);
      }
    }

    /**
     * @param The panel number
     * @returns The buffer for the panel
     */
    af::const_ref<float_type, af::c_grid<3> > data(std::size_t panel) const {
      DIALS_ASSERT(panel < data_.size());
      return data_[panel].const_ref();
    }

    /**
     * @param The panel number
     * @returns The buffer for the panel
     */
    af::const_ref<bool, af::c_grid<2> > static_mask(std::size_t panel) const {
      DIALS_ASSERT(panel < static_mask_.size());
      return static_mask_[panel].const_ref();
    }

  protected:
    /**
     * Copy the data from 1 panel
     * @param src The source
     * @param dst The destination
     * @param index The image index
     */
    template <typename InputType, typename OutputType>
    void copy(af::const_ref<InputType, af::c_grid<2> > src,
              af::ref<OutputType, af::c_grid<3> > dst,
              std::size_t index) {
      std::size_t ysize = src.accessor()[0];
      std::size_t xsize = src.accessor()[1];
      DIALS_ASSERT(index < dst.accessor()[0]);
      DIALS_ASSERT(src.accessor()[0] == dst.accessor()[1]);
      DIALS_ASSERT(src.accessor()[1] == dst.accessor()[2]);
      for (std::size_t j = 0; j < ysize * xsize; ++j) {
        dst[index * (xsize * ysize) + j] = src[j];
      }
    }

    /**
     * Mask all pixels
     * @param dst The destination
     * @param index The image index
     */
    template <typename OutputType>
    void apply_mask_to_all_pixels(af::ref<OutputType, af::c_grid<3> > dst,
                                  std::size_t index) {
      std::size_t ysize = dst.accessor()[1];
      std::size_t xsize = dst.accessor()[2];
      DIALS_ASSERT(index < dst.accessor()[0]);
      for (std::size_t j = 0; j < ysize * xsize; ++j) {
        dst[index * (xsize * ysize) + j] = mask_value_;
      }
    }

    /**
     * Apply the static mask to the dynamic mask
     * @param src The source
     * @param dst The destination
     * @param index The image index
     */
    template <typename OutputType>
    void apply_mask(af::const_ref<bool, af::c_grid<2> > src,
                    af::ref<OutputType, af::c_grid<3> > dst,
                    std::size_t index) {
      std::size_t ysize = src.accessor()[0];
      std::size_t xsize = src.accessor()[1];
      DIALS_ASSERT(index < dst.accessor()[0]);
      DIALS_ASSERT(src.accessor()[0] == dst.accessor()[1]);
      DIALS_ASSERT(src.accessor()[1] == dst.accessor()[2]);
      for (std::size_t j = 0; j < ysize * xsize; ++j) {
        if (!src[j]) {
          dst[index * (xsize * ysize) + j] = mask_value_;
        }
      }
    }

    /**
     * Set the static mask for a panel
     * @param panel The panel
     * @param panel_static_mask The static mask
     */
    void set_external_mask_for_panel(
      const af::const_ref<bool, af::c_grid<2> > &external_mask,
      af::ref<bool, af::c_grid<2> > panel_static_mask) {
      DIALS_ASSERT(external_mask.accessor().all_eq(panel_static_mask.accessor()));
      for (std::size_t j = 0; j < external_mask.size(); ++j) {
        panel_static_mask[j] = external_mask[j] && panel_static_mask[j];
      }
    }

    std::vector<af::versa<float_type, af::c_grid<3> > > data_;
    std::vector<af::versa<bool, af::c_grid<2> > > static_mask_;
    float_type mask_value_;
  };

  /**
   * A class to store the image data circular buffer
   */
  class Buffer {
  public:
    typedef Shoebox<>::float_type float_type;

    /**
     * Initialise the the size of the panels
     * @param detector The detector model
     * @param num_images The number of images
     * @param mask_value The value of masked pixels in the buffer
     * @param external_mask The external mask
     */
    Buffer(const Detector &detector,
           std::size_t num_images,
           std::size_t num_buffer,
           float_type mask_value,
           const Image<bool> &external_mask)
        : buffer_base_(detector, num_buffer, mask_value, external_mask),
          num_images_(num_images),
          num_buffer_(num_buffer),
          buffer_range_(0, num_buffer) {
      DIALS_ASSERT(num_buffer > 0);
      DIALS_ASSERT(num_images >= num_buffer);
    }

    /**
     * @returns The number of images
     */
    std::size_t num_images() const {
      return num_images_;
    }

    /**
     * @returns The number of images in the buffer
     */
    std::size_t num_buffer() const {
      return num_buffer_;
    }

    /**
     * @returns The current buffer image range
     */
    tiny<int, 2> buffer_range() const {
      return buffer_range_;
    }

    /**
     * Copy an image to the buffer
     * @param data The image data
     * @param index The image index
     */
    void copy(const Image<double> &data, std::size_t index) {
      DIALS_ASSERT(index < num_images_);
      DIALS_ASSERT(index >= buffer_range_[0]);
      DIALS_ASSERT(index <= buffer_range_[1]);
      DIALS_ASSERT(buffer_range_[0] >= 0);
      DIALS_ASSERT(buffer_range_[1] <= num_images_);
      DIALS_ASSERT(buffer_range_[1] > buffer_range_[0]);
      DIALS_ASSERT(buffer_range_[1] - buffer_range_[0] == num_buffer_);
      if (index == buffer_range_[1]) {
        buffer_range_[0]++;
        buffer_range_[1]++;
      }
      buffer_base_.copy(data, index % num_buffer_);
    }

    /**
     * Copy an image to the buffer
     * @param data The image data
     * @param mask The mask data
     * @param index The image index
     */
    void copy(const Image<double> &data, bool mask, std::size_t index) {
      DIALS_ASSERT(index < num_images_);
      DIALS_ASSERT(index >= buffer_range_[0]);
      DIALS_ASSERT(index <= buffer_range_[1]);
      DIALS_ASSERT(buffer_range_[0] >= 0);
      DIALS_ASSERT(buffer_range_[1] <= num_images_);
      DIALS_ASSERT(buffer_range_[1] > buffer_range_[0]);
      DIALS_ASSERT(buffer_range_[1] - buffer_range_[0] == num_buffer_);
      if (index == buffer_range_[1]) {
        buffer_range_[0]++;
        buffer_range_[1]++;
      }
      buffer_base_.copy(data, mask, index % num_buffer_);
    }

    /**
     * Copy an image to the buffer
     * @param data The image data
     * @param mask The mask data
     * @param index The image index
     */
    void copy(const Image<double> &data, const Image<bool> &mask, std::size_t index) {
      DIALS_ASSERT(index < num_images_);
      DIALS_ASSERT(index >= buffer_range_[0]);
      DIALS_ASSERT(index <= buffer_range_[1]);
      DIALS_ASSERT(buffer_range_[0] >= 0);
      DIALS_ASSERT(buffer_range_[1] <= num_images_);
      DIALS_ASSERT(buffer_range_[1] > buffer_range_[0]);
      DIALS_ASSERT(buffer_range_[1] - buffer_range_[0] == num_buffer_);
      if (index == buffer_range_[1]) {
        buffer_range_[0]++;
        buffer_range_[1]++;
      }
      buffer_base_.copy(data, mask, index % num_buffer_);
    }

    /**
     * @param The panel number
     * @returns The buffer for the panel
     */
    af::const_ref<float_type, af::c_grid<2> > data(std::size_t panel,
                                                   std::size_t index) const {
      DIALS_ASSERT(index < num_images_);
      DIALS_ASSERT(index >= buffer_range_[0]);
      DIALS_ASSERT(index < buffer_range_[1]);
      DIALS_ASSERT(buffer_range_[0] >= 0);
      DIALS_ASSERT(buffer_range_[1] <= num_images_);
      DIALS_ASSERT(buffer_range_[1] > buffer_range_[0]);
      DIALS_ASSERT(buffer_range_[1] - buffer_range_[0] == num_buffer_);
      af::const_ref<float_type, af::c_grid<3> > data_buffer = buffer_base_.data(panel);
      std::size_t ysize = data_buffer.accessor()[1];
      std::size_t xsize = data_buffer.accessor()[2];
      std::size_t offset = (index % num_buffer_) * (ysize * xsize);
      DIALS_ASSERT(offset < data_buffer.size());
      return af::const_ref<float_type, af::c_grid<2> >(&data_buffer[offset],
                                                       af::c_grid<2>(ysize, xsize));
    }

    /**
     * @param The panel number
     * @returns The buffer for the panel
     */
    af::const_ref<bool, af::c_grid<2> > static_mask(std::size_t panel) const {
      return buffer_base_.static_mask(panel);
    }

  protected:
    BufferBase buffer_base_;
    std::size_t num_images_;
    std::size_t num_buffer_;
    tiny<int, 2> buffer_range_;
  };

  /**
   * A class to integrate a single reflection
   */
  class ReflectionIntegrator {
  public:
    /**
     * Initialise the integrator
     * @param compute_mask The mask calculation function
     * @param compute_background The background calculation function
     * @param compute_intensity The intensity calculation function
     * @param buffer The image buffer array
     * @param zstart The first image index
     * @param min_trusted The minimum trusted value
     * @param max_trusted The maximum trusted value
     */
    ReflectionIntegrator(const MaskCalculatorIface &compute_mask,
                         const BackgroundCalculatorIface &compute_background,
                         const IntensityCalculatorIface &compute_intensity,
                         const Buffer &buffer,
                         int zstart,
                         double min_trusted,
                         double max_trusted,
                         bool debug)
        : compute_mask_(compute_mask),
          compute_background_(compute_background),
          compute_intensity_(compute_intensity),
          buffer_(buffer),
          zstart_(zstart),
          min_trusted_(min_trusted),
          max_trusted_(max_trusted),
          debug_(debug) {}

    /**
     * Integrate a reflection using the following procedure:
     *
     * 1. Extract the buffered image data into a shoebox
     * 2. Compute the mask for the reflection
     * 3. Compute the mask for adjacent reflections
     * 4. Compute the reflection background
     * 5. Compute the reflection centroid
     * 6. Compute the summed intensity
     * 7. Compute the profile fitted intensity
     * 8. Delete the shoebox unless debug has been set
     *
     * @param reflection The reflection object
     * @param adjacent_reflections The list of adjacent reflections
     */

    void operator()(std::size_t index,
                    af::ref<af::Reflection> reflection_list,
                    const AdjacencyList &adjacency_list) const {
      af::Reflection reflection;
      std::vector<af::Reflection> adjacent_reflections;

      // Get the reflection data
      get_reflection(
        index, reflection_list, adjacency_list, reflection, adjacent_reflections);

      // Extract the shoebox data
      extract_shoebox(buffer_, reflection, zstart_, min_trusted_, max_trusted_);

      // Compute the mask
      compute_mask_(reflection);

      // Set all the bounding boxes of adjacent reflections
      // And compute the mask for these reflections too.
      for (std::size_t i = 0; i < adjacent_reflections.size(); ++i) {
        adjacent_reflections[i]["bbox"] = reflection.get<int6>("bbox");
        adjacent_reflections[i]["shoebox"] = reflection.get<Shoebox<> >("shoebox");
        compute_mask_(adjacent_reflections[i], true);
      }

      // Compute the background
      try {
        compute_background_(reflection);
      } catch (dials::error const &) {
        finalize_shoebox(reflection, adjacent_reflections, min_trusted_, max_trusted_);
        return;
      }

      // Compute the centroid
      compute_centroid(reflection);

      // Compute the summed intensity
      compute_summed_intensity(reflection);

      // Compute the profile fitted intensity
      try {
        compute_intensity_(reflection, adjacent_reflections);
      } catch (dials::error const &) {
        std::size_t flags = reflection.get<std::size_t>("flags");
        flags |= af::FailedDuringProfileFitting;
        reflection["flags"] = flags;
      }

      // Erase the shoebox
      finalize_shoebox(reflection, adjacent_reflections, min_trusted_, max_trusted_);

      // Set the reflection data
      set_reflection(index, reflection_list, reflection);
    }

  protected:
    /**
     * Get the reflection data in a thread safe manner
     * @param index The reflection index
     * @param reflection_list The reflection list
     * @param adjacency_list The adjacency list
     * @param reflection The reflection data
     * @param adjacent_reflections The adjacent reflections
     */
    void get_reflection(std::size_t index,
                        const af::const_ref<af::Reflection> &reflection_list,
                        const AdjacencyList &adjacency_list,
                        af::Reflection &reflection,
                        std::vector<af::Reflection> &adjacent_reflections) const {
      DIALS_ASSERT(index < reflection_list.size());

      // Get the lock
      boost::lock_guard<boost::mutex> guard(mutex_);

      // Get the reflection
      reflection = reflection_list[index];

      // Get the adjacent reflections
      adjacent_reflections.reserve(adjacency_list.vertex_num_edges(index));
      AdjacencyList::edge_iterator_range edges = adjacency_list.edges(index);
      for (AdjacencyList::edge_iterator it = edges.first; it != edges.second; ++it) {
        DIALS_ASSERT(it->first == index);
        DIALS_ASSERT(it->second < reflection_list.size());
        adjacent_reflections.push_back(reflection_list[it->second]);
      }
    }

    /**
     * Set the reflection data in a thread safe way
     * @param index The reflection index
     * @param reflection_list The reflection list
     * @param reflection The reflection data
     */
    void set_reflection(std::size_t index,
                        af::ref<af::Reflection> reflection_list,
                        const af::Reflection &reflection) const {
      DIALS_ASSERT(index < reflection_list.size());
      boost::lock_guard<boost::mutex> guard(mutex_);
      reflection_list[index] = reflection;
    }

    /**
     * Before exiting do some stuff on the shoebox
     * @param reflection The reflection
     * @param adjacent_reflections The adjacent reflections
     */
    void finalize_shoebox(af::Reflection &reflection,
                          std::vector<af::Reflection> &adjacent_reflections,
                          double min_trusted,
                          double max_trusted) const {
      // Inspect the pixels
      inspect_pixels(reflection, min_trusted, max_trusted);

      // Delete the shoebox
      delete_shoebox(reflection, adjacent_reflections);
    }

    /**
     * Inspect the pixel and mask values
     * @param reflection The reflection
     */
    void inspect_pixels(af::Reflection &reflection,
                        double min_trusted,
                        double max_trusted) const {
      typedef Shoebox<>::float_type float_type;

      // Get the shoebox
      Shoebox<> &sbox = reflection.get<Shoebox<> >("shoebox");
      std::size_t flags = reflection.get<std::size_t>("flags");

      // Get the pixel data
      af::const_ref<float_type, af::c_grid<3> > data = sbox.data.const_ref();
      af::const_ref<int, af::c_grid<3> > mask = sbox.mask.const_ref();
      DIALS_ASSERT(data.accessor().all_eq(mask.accessor()));

      // Check pixel values
      std::size_t n_valid = 0;
      std::size_t n_background = 0;
      std::size_t n_background_used = 0;
      std::size_t n_foreground = 0;
      std::size_t mask_code1 = Valid;
      std::size_t mask_code2 = Valid | Background;
      std::size_t mask_code3 = Valid | Background | BackgroundUsed;
      std::size_t mask_code4 = Valid | Foreground;
      for (std::size_t i = 0; i < mask.size(); ++i) {
        double d = data[i];
        int m = mask[i];

        if (d > max_trusted) {
          flags |= af::Overloaded;
        }

        if ((m & Background) && !(m & Valid)) {
          flags |= af::BackgroundIncludesBadPixels;
        }

        if ((m & Foreground) && !(m & Valid)) {
          flags |= af::ForegroundIncludesBadPixels;
        }

        if ((m & Background) && (m & Overlapped)) {
          flags |= af::OverlappedBg;
        }

        if ((m & Foreground) && (m & Overlapped)) {
          flags |= af::OverlappedFg;
        }

        if ((m & mask_code1) == mask_code1) {
          n_valid++;
        }

        if ((m & mask_code2) == mask_code2) {
          n_background++;
        }

        if ((m & mask_code3) == mask_code3) {
          n_background_used++;
        }

        if ((m & mask_code4) == mask_code4) {
          n_foreground++;
        }
      }

      // Set some information in the reflection
      reflection["num_pixels.valid"] = (int)n_valid;
      reflection["num_pixels.background"] = (int)n_background;
      reflection["num_pixels.background_used"] = (int)n_background_used;
      reflection["num_pixels.foreground"] = (int)n_foreground;
      reflection["flags"] = flags;
    }

    /**
     * Delete the shoebox
     * @param reflection The reflection
     * @param adjacent_reflections The adjacent reflections
     */
    void delete_shoebox(af::Reflection &reflection,
                        std::vector<af::Reflection> &adjacent_reflections) const {
      // Erase the shoebox from the reflection
      if (!debug_) {
        reflection.erase("shoebox");
        for (std::size_t i = 0; i < adjacent_reflections.size(); ++i) {
          adjacent_reflections[i].erase("shoebox");
        }
      }
    }

    /**
     * Extract the shoebox data from the buffer
     */
    void extract_shoebox(const Buffer &buffer,
                         af::Reflection &reflection,
                         int zstart,
                         double min_trusted,
                         double max_trusted) const {
      typedef af::const_ref<Buffer::float_type, af::c_grid<2> > data_buffer_type;
      std::size_t panel = reflection.get<std::size_t>("panel");
      int6 bbox = reflection.get<int6>("bbox");
      Shoebox<> shoebox(panel, bbox);
      shoebox.allocate();
      af::ref<float, af::c_grid<3> > data = shoebox.data.ref();
      af::ref<int, af::c_grid<3> > mask = shoebox.mask.ref();
      int x0 = bbox[0];
      int x1 = bbox[1];
      int y0 = bbox[2];
      int y1 = bbox[3];
      int z0 = bbox[4];
      int z1 = bbox[5];
      DIALS_ASSERT(x1 > x0);
      DIALS_ASSERT(y1 > y0);
      DIALS_ASSERT(z1 > z0);
      std::size_t zsize = z1 - z0;
      std::size_t ysize = y1 - y0;
      std::size_t xsize = x1 - x0;
      DIALS_ASSERT(zsize == data.accessor()[0]);
      DIALS_ASSERT(ysize == data.accessor()[1]);
      DIALS_ASSERT(xsize == data.accessor()[2]);
      DIALS_ASSERT(shoebox.is_consistent());
      for (std::size_t k = 0; k < zsize; ++k) {
        int kk = z0 + k - zstart;
        if (kk < 0 || kk >= buffer.num_images()) {
          continue;
        }
        data_buffer_type data_buffer = buffer.data(panel, kk);
        for (std::size_t j = 0; j < ysize; ++j) {
          for (std::size_t i = 0; i < xsize; ++i) {
            int jj = y0 + j;
            int ii = x0 + i;
            if (jj >= 0 && ii >= 0 && jj < data_buffer.accessor()[0]
                && ii < data_buffer.accessor()[1]) {
              double d = data_buffer(jj, ii);
              int m = (d >= min_trusted && d <= max_trusted) ? Valid : 0;
              data(k, j, i) = d;
              mask(k, j, i) = m;
            } else {
              data(k, j, i) = 0;
              mask(k, j, i) = 0;
            }
          }
        }
      }
      reflection["shoebox"] = shoebox;
    }

    /**
     * Compute the centroid
     */
    void compute_centroid(af::Reflection &reflection) const {
      using dials::model::Centroid;

      // Get the shoebox and compute centroid
      Shoebox<> shoebox = reflection.get<Shoebox<> >("shoebox");
      Centroid centroid = shoebox.centroid_foreground_minus_background();

      // Set the centroid values
      reflection["xyzobs.px.value"] = centroid.px.position;
      reflection["xyzobs.px.variance"] = centroid.px.variance;
    }

    /**
     * Compute the summed intensity
     */
    void compute_summed_intensity(af::Reflection &reflection) const {
      using dials::model::Intensity;

      // Get flags and reset
      std::size_t flags = reflection.get<std::size_t>("flags");
      flags &= ~af::IntegratedSum;
      flags &= ~af::FailedDuringSummation;

      // Get the shoebox and compute the summed intensity
      Shoebox<> shoebox = reflection.get<Shoebox<> >("shoebox");
      Intensity intensity = shoebox.summed_intensity();

      // Set the intensities
      reflection["intensity.sum.value"] = intensity.observed.value;
      reflection["intensity.sum.variance"] = intensity.observed.variance;
      reflection["background.sum.value"] = intensity.background.value;
      reflection["background.sum.variance"] = intensity.background.variance;

      // Set the appropriate flag
      if (intensity.observed.success) {
        flags |= af::IntegratedSum;
      } else {
        flags |= af::FailedDuringSummation;
      }
      reflection["flags"] = flags;
    }

    const MaskCalculatorIface &compute_mask_;
    const BackgroundCalculatorIface &compute_background_;
    const IntensityCalculatorIface &compute_intensity_;
    const Buffer &buffer_;
    int zstart_;
    double min_trusted_;
    double max_trusted_;
    bool debug_;
    mutable boost::mutex mutex_;
  };

  /**
   * a class to sort the indices of all reflections that are fully recorded
   * after a particular image.
   */
  class Lookup {
  public:
    /**
     * @param bbox the bounding box
     * @param zstart the first frame number
     * @param n the number of frames
     */
    Lookup(af::const_ref<int6> bbox, int zstart, std::size_t n)
        : indices_(bbox.size()) {
      // fill the index array
      for (std::size_t i = 0; i < indices_.size(); ++i) {
        indices_[i] = i;
      }

      // sort the indices by the final frame number
      std::sort(indices_.begin(), indices_.end(), sort_by_frame(bbox));
      DIALS_ASSERT(bbox[indices_.front()][5] - zstart >= 1);
      DIALS_ASSERT(bbox[indices_.back()][5] - zstart <= n);

      // create an offset array that records the positions in the index array
      // where the frame increments such that
      // offset[i], offset[i+1] gives the range of indices for a frame.
      std::size_t i = 0;
      offset_.push_back(0);
      for (std::size_t j = 0; j < n; ++j) {
        while (i < indices_.size() && bbox[indices_[i]][5] - zstart <= j + 1) i++;
        offset_.push_back(i);
      }
      DIALS_ASSERT(offset_.size() == n + 1);
      DIALS_ASSERT(offset_.back() == indices_.size());
    }

    /**
     * @param z the frame number
     * @returns the indices for a given frame
     */
    af::const_ref<std::size_t> indices(std::size_t z) const {
      DIALS_ASSERT(z < offset_.size() - 1);
      DIALS_ASSERT(offset_[z + 1] >= offset_[z]);
      std::size_t i = offset_[z];
      std::size_t n = offset_[z + 1] - i;
      return af::const_ref<std::size_t>(&indices_[i], n);
    }

  private:
    /**
     * helper function to sort by final bbox frame
     */
    struct sort_by_frame {
      af::const_ref<int6> bbox_;
      sort_by_frame(af::const_ref<int6> bbox) : bbox_(bbox) {}
      bool operator()(std::size_t a, std::size_t b) const {
        return bbox_[a][5] < bbox_[b][5];
      }
    };

    std::vector<std::size_t> indices_;
    std::vector<std::size_t> offset_;
  };

  /**
   * A class to manage the image buffer. Reflections are processed and then notify
   * the manager when they are done. We maintain a counter of reflections which
   * require each image. After each reflection is processed, the appropriate
   * atomic counter is decremented. When the counter reaches zero, the image is no
   * longer needed.
   */
  class BufferManager {
  public:
    /**
     * Create the buffer manager
     * @param buffer The buffer to manage
     * @param bbox The bounding box
     * @param first_image The first image
     */
    BufferManager(Buffer &buffer,
                  const af::const_ref<int6> &bbox,
                  const af::const_ref<std::size_t> &flags,
                  int first_image)
        : buffer_(buffer),
          notifier_(bbox, flags, first_image, buffer.num_images(), buffer.num_buffer()),
          first_image_(first_image),
          max_images_(buffer.num_buffer()) {}

    /**
     * Copy the image to the buffer when we are able to accept more images
     * @param data The image data
     * @param index The image index
     */
    void copy_when_ready(const Image<double> &data, std::size_t index) {
      if (index >= max_images_) {
        while (!notifier_.complete(buffer_.buffer_range()[0]))
          ;
      }
      buffer_.copy(data, index);
    }

    /**
     * Copy the image to the buffer when we are able to accept more images
     * @param data The image data
     * @param mask A single value mask
     * @param index The image index
     */
    void copy_when_ready(const Image<double> &data, bool mask, std::size_t index) {
      if (index >= max_images_) {
        while (!notifier_.complete(buffer_.buffer_range()[0]))
          ;
      }
      buffer_.copy(data, mask, index);
    }

    /**
     * Copy the image to the buffer when we are able to accept more images
     * @param data The image data
     * @param mask The image mask
     * @param index The image index
     */
    void copy_when_ready(const Image<double> &data,
                         const Image<bool> &mask,
                         std::size_t index) {
      if (index >= max_images_) {
        while (!notifier_.complete(buffer_.buffer_range()[0]))
          ;
      }
      buffer_.copy(data, mask, index);
    }

    /**
     * Post the job to the pool
     * @param pool The thread pool
     * @param function The function to post
     * @param bbox_first_image The image index
     */
    template <typename ThreadPoolType, typename Function>
    void post(ThreadPoolType &pool, Function function, int bbox_first_image) {
      DIALS_ASSERT(bbox_first_image >= first_image_);
      pool.post(
        JobWrapper<Function>(function, notifier_, bbox_first_image - first_image_));
    }

    /**
     * Wait and check all are complete
     * @param pool The thread pool
     */
    template <typename ThreadPoolType>
    void wait(ThreadPoolType &pool) {
      pool.wait();
      DIALS_ASSERT(notifier_.all_complete());
    }

  protected:
    /**
     * A class to notify the buffer manager when all jobs that need to access an
     * image have completed so that the image can be deleted.
     */
    class Notifier {
    public:
      /**
       * Init the counters
       * @param bbox The reflection bbox
       * @param flags The reflection flags
       * @param first_image The first image
       * @param last_image The last image
       */
      Notifier(const af::const_ref<int6> &bbox,
               const af::const_ref<std::size_t> &flags,
               int first_image,
               std::size_t num_images,
               std::size_t max_images)
          : first_image_(first_image) {
        DIALS_ASSERT(bbox.size() == flags.size());
        DIALS_ASSERT(num_images > 0);

        // Initialise the vector of atomic ints
        for (std::size_t i = 0; i < num_images; ++i) {
          counter_.push_back(new std::atomic<int>(0));
        }

        // Increment the counter for each image
        int last_image = first_image + num_images;
        for (std::size_t j = 0; j < bbox.size(); ++j) {
          if ((flags[j] & af::DontIntegrate) == 0) {
            int z0 = bbox[j][4];
            int z1 = bbox[j][5];
            DIALS_ASSERT(z0 >= first_image);
            DIALS_ASSERT(z0 < last_image);
            DIALS_ASSERT(z1 - z0 <= max_images);
            int i = z0 - first_image_;
            DIALS_ASSERT(i >= 0);
            DIALS_ASSERT(i < counter_.size());
            counter_[i]++;
          }
        }
      }

      /**
       * Notify about a reflection using this image is finished.
       * Reduce the atomic counter for the image
       * @param image_index The image index
       */
      void notify(std::size_t index) {
        DIALS_ASSERT(index < counter_.size());
        counter_[index]--;
      }

      /**
       * @param image_index The image index
       * @returns The value of the counter
       */
      int counter(std::size_t index) {
        DIALS_ASSERT(index < counter_.size());
        return counter_[index];
      }

      /**
       * Check whether all reflections needing this image are finished
       * Check the atomic counter for the image
       * @param index The image index
       * @returns True/False complete
       */
      bool complete(std::size_t index) const {
        DIALS_ASSERT(index < counter_.size());
        return counter_[index] == 0;
      }

      /**
       * Check that all images are complete
       * @returns True/False complete
       */
      bool all_complete() const {
        for (std::size_t i = 0; i < counter_.size(); ++i) {
          if (counter_[i] != 0) {
            return false;
          }
        }
        return true;
      }

    protected:
      int first_image_;
      boost::ptr_vector<std::atomic<int> > counter_;
    };

    /**
     * A wrapper to call the job function and then call the notifier function
     */
    template <typename Function>
    class JobWrapper {
    public:
      /**
       * Construct
       * @param function The function to call
       * @param notifier The notifier function
       * @param index The image index
       */
      JobWrapper(Function function, Notifier &notifier, std::size_t index)
          : function_(function), notifier_(notifier), index_(index) {}

      /**
       * Call the function and notify
       */
      void operator()() {
        function_();
        notifier_.notify(index_);
      }

      Function function_;
      Notifier &notifier_;
      std::size_t index_;
    };

    Buffer &buffer_;
    Notifier notifier_;
    int first_image_;
    std::size_t max_images_;
  };

  /**
   * A class to perform parallel integration
   */
  class ParallelIntegrator {
  public:
    /**
     * Do the integration
     * @param reflections The reflection table
     * @param imageset The imageset
     * @param compute_mask The mask calulcation function
     * @param compute_background The background calculation function
     * @param compute_intensity The intensity calculation function
     * @param logger The logger class
     * @param nthreads The number of parallel threads
     * @param buffer_size The buffer_size
     * @param use_dynamic_mask Use the dynamic mask if present
     * @param debug Add debug output
     */
    ParallelIntegrator(af::reflection_table reflections,
                       ImageSequence imageset,
                       const MaskCalculatorIface &compute_mask,
                       const BackgroundCalculatorIface &compute_background,
                       const IntensityCalculatorIface &compute_intensity,
                       const Logger &logger,
                       std::size_t nthreads,
                       std::size_t buffer_size,
                       bool use_dynamic_mask,
                       bool debug) {
      using dials::algorithms::shoebox::find_overlapping_multi_panel;

      // Check the input
      DIALS_ASSERT(nthreads > 0);

      // Check the models
      DIALS_ASSERT(imageset.get_detector() != NULL);
      DIALS_ASSERT(imageset.get_scan() != NULL);
      Detector detector = *imageset.get_detector();
      Scan scan = *imageset.get_scan();

      // Get the size of the data buffer needed
      std::size_t zsize = imageset.size();
      DIALS_ASSERT(zsize > 0);
      if (buffer_size == 0 || buffer_size > zsize) {
        buffer_size = zsize;
      }

      // Get the starting frame and the trusted range values
      int zstart = scan.get_array_range()[0];
      double min_trusted = detector[0].get_trusted_range()[0];
      double max_trusted = detector[0].get_trusted_range()[1];
      DIALS_ASSERT(min_trusted < max_trusted);
      for (std::size_t i = 1; i < detector.size(); ++i) {
        DIALS_ASSERT(min_trusted == detector[i].get_trusted_range()[0]);
        DIALS_ASSERT(max_trusted == detector[i].get_trusted_range()[1]);
      }

      // Get the reflection flags and bbox
      af::const_ref<std::size_t> panel =
        reflections.get<std::size_t>("panel").const_ref();
      af::const_ref<int6> bbox = reflections.get<int6>("bbox").const_ref();
      af::ref<std::size_t> flags = reflections.get<std::size_t>("flags").ref();

      // Reset the flags
      reset_flags(flags);

      // Find the overlapping reflections
      AdjacencyList overlaps = find_overlapping_multi_panel(bbox, panel);

      // Allocate the array for the image data
      double mask_value = min_trusted - 1.0;
      Buffer buffer(
        detector, zsize, buffer_size, mask_value, imageset.get_static_mask());

      // If we have shoeboxes then delete
      if (reflections.contains("shoebox")) {
        reflections.erase("shoebox");
      }

      // Transform reflection data from column major to row major. The
      // reason for doing this is because we want to process each reflection
      // in parallel. Therefore passing a reflection object is better than
      // passing the whole table. Additionally, because the reflection table
      // uses a std::map, it may not be thread safe, so we want to avoid
      // accessing this across multiple threads.
      af::shared<af::Reflection> reflection_array =
        reflection_table_to_array(reflections);

      // The lookup class gives the indices of reflections whose bounding boxes
      // are complete at a given image. This is used to submit reflections for
      // integration after each image is processed.
      Lookup lookup(bbox, zstart, zsize);

      // Create the reflection integrator. This class is called for each
      // reflection to integrate the data
      ReflectionIntegrator integrator(compute_mask,
                                      compute_background,
                                      compute_intensity,
                                      buffer,
                                      zstart,
                                      min_trusted,
                                      max_trusted,
                                      debug);

      // Do the integration
      process(lookup,
              integrator,
              buffer,
              reflection_array.ref(),
              overlaps,
              imageset,
              bbox,
              flags,
              nthreads,
              use_dynamic_mask,
              logger);

      // Transform the row major reflection array to the reflection table
      reflections_ = reflection_table_from_array(reflection_array.const_ref());
    }

    /**
     * @returns The integrated reflections
     */
    af::reflection_table reflections() const {
      return reflections_;
    }

    /**
     * Static method to get the memory in bytes needed
     * @param imageset the imageset class
     */
    static std::size_t compute_required_memory(ImageSequence imageset,
                                               std::size_t block_size) {
      DIALS_ASSERT(imageset.get_detector() != NULL);
      DIALS_ASSERT(imageset.get_scan() != NULL);
      Detector detector = *imageset.get_detector();
      Scan scan = *imageset.get_scan();
      block_size = std::min(block_size, (std::size_t)scan.get_num_images());
      std::size_t nelements = 0;
      for (std::size_t i = 0; i < detector.size(); ++i) {
        std::size_t xsize = detector[i].get_image_size()[0];
        std::size_t ysize = detector[i].get_image_size()[1];
        nelements += xsize * ysize;
      }
      nelements *= block_size;
      std::size_t nbytes = nelements * sizeof(Buffer::float_type);
      return nbytes;
    }

    /**
     * Static method to get the memory in bytes needed
     * @param imageset the imageset class
     * @param max_memory_usage The maximum memory usage
     */
    static std::size_t compute_max_block_size(ImageSequence imageset,
                                              std::size_t max_memory_usage) {
      DIALS_ASSERT(max_memory_usage > 0);
      DIALS_ASSERT(imageset.get_detector() != NULL);
      Detector detector = *imageset.get_detector();
      std::size_t nelements = 0;
      for (std::size_t i = 0; i < detector.size(); ++i) {
        std::size_t xsize = detector[i].get_image_size()[0];
        std::size_t ysize = detector[i].get_image_size()[1];
        nelements += xsize * ysize;
      }
      std::size_t nbytes = nelements * sizeof(Buffer::float_type);
      DIALS_ASSERT(nbytes > 0);
      DIALS_ASSERT(max_memory_usage > nbytes);
      return (std::size_t)std::floor((float)max_memory_usage / (float)nbytes);
    }

  protected:
    /**
     * Reset the reflection flags
     */
    void reset_flags(af::ref<std::size_t> flags) const {
      for (std::size_t i = 0; i < flags.size(); ++i) {
        flags[i] &= ~af::IntegratedSum;
        flags[i] &= ~af::IntegratedPrf;
      }
    }

    /**
     * Do the processing by the following procedure.
     *
     * 1. For each reflection, construct a list of adjacent reflection
     * 2. Loop through all the images in the imageset
     * 3. Copy the images to a buffer
     * 4. Loop through all reflections complete on the image
     * 5. For each complete reflection post a reflection integration job to the
     *    thread pool.
     */
    void process(const Lookup &lookup,
                 const ReflectionIntegrator &integrator,
                 Buffer &buffer,
                 af::ref<af::Reflection> reflections,
                 const AdjacencyList &overlaps,
                 ImageSequence imageset,
                 af::const_ref<int6> bbox,
                 af::const_ref<std::size_t> flags,
                 std::size_t nthreads,
                 bool use_dynamic_mask,
                 const Logger &logger) const {
      using dials::util::ThreadPool;

      // Create the thread pool
      ThreadPool pool(nthreads);

      // Get the size of the array
      int zstart = imageset.get_scan()->get_array_range()[0];
      std::size_t zsize = imageset.size();

      // Create the buffer manager
      BufferManager bm(buffer, bbox, flags, zstart);

      // Loop through all the images
      for (std::size_t i = 0; i < zsize; ++i) {
        // Copy the image to the buffer. If the image number is greater than the
        // buffer size (i.e. we are now deleting old images) then wait for the
        // threads to finish so that we don't end up reading the wrong data
        if (imageset.is_marked_for_rejection(i)) {
          bm.copy_when_ready(imageset.get_corrected_data(i), false, i);
        } else if (use_dynamic_mask) {
          bm.copy_when_ready(
            imageset.get_corrected_data(i), imageset.get_dynamic_mask(i), i);
        } else {
          bm.copy_when_ready(imageset.get_corrected_data(i), i);
        }

        // Get the reflections recorded at this point
        af::const_ref<std::size_t> indices = lookup.indices(i);

        // Iterate through the reflection indices
        std::size_t count = 0;
        for (std::size_t j = 0; j < indices.size(); ++j) {
          // Get the reflection index
          std::size_t k = indices[j];

          // Check that the reflection bounding box is within the range of
          // images that have been read
          DIALS_ASSERT(bbox[k][5] <= zstart + i + 1);

          // Ignore if we're not integrating this reflection
          if (flags[k] & af::DontIntegrate) {
            continue;
          } else {
            count++;
          }

          // Post the integration job
          bm.post(
            pool,
            boost::bind(&ReflectionIntegrator::operator(),
                        boost::ref(integrator),
                        k,
                        af::ref<af::Reflection>(&reflections[0], reflections.size()),
                        boost::ref(overlaps)),
            bbox[k][4]);
        }

        // Print some output
        std::ostringstream ss;
        ss << "Integrating " << std::setw(5) << count << " reflections on image "
           << std::setw(6) << zstart + i;
        logger.info(ss.str().c_str());
      }

      // Wait for all the integration jobs to complete
      bm.wait(pool);
    }

    af::reflection_table reflections_;
  };

  /**
   * A class to manage jobs
   */
  class SimpleBlockList {
  public:
    /**
     * Compute the blocks
     * @param range The range of frames
     * @param block_size The size of the blocks
     */
    SimpleBlockList(tiny<int, 2> range, int block_size) {
      construct_block_list(range, block_size);
      construct_frame_to_block_lookup();
    }

    /**
     * Set the blocks
     * @params blocks The list of blocks
     */
    SimpleBlockList(const af::const_ref<tiny<int, 2> > &blocks) {
      DIALS_ASSERT(blocks.size() > 0);
      DIALS_ASSERT(blocks[0][1] > blocks[0][0]);
      blocks_.push_back(blocks[0]);
      for (std::size_t i = 1; i < blocks.size(); ++i) {
        DIALS_ASSERT(blocks[i][1] > blocks[i][0]);
        DIALS_ASSERT(blocks[i][0] > blocks[i - 1][0]);
        DIALS_ASSERT(blocks[i][1] > blocks[i - 1][1]);
        DIALS_ASSERT(blocks[i][0] <= blocks[i - 1][1]);
        blocks_.push_back(blocks[i]);
      }
      construct_frame_to_block_lookup();
    }

    /**
     * @returns The requested job
     */
    tiny<int, 2> operator[](std::size_t index) const {
      DIALS_ASSERT(index < blocks_.size());
      return blocks_[index];
    }

    /**
     * @returns The first frame
     */
    int first_frame() const {
      DIALS_ASSERT(blocks_.size() > 0);
      return blocks_[0][0];
    }

    /**
     * @returns The last frame
     */
    int last_frame() const {
      DIALS_ASSERT(blocks_.size() > 0);
      return blocks_[blocks_.size() - 1][1];
    }

    /**
     * @returns The number of blocks
     */
    std::size_t size() const {
      return blocks_.size();
    }

    /**
     * Get the index of the block closest to this frame
     * @param frame The frame number
     * @returns The block index
     */
    std::size_t block_index(int frame) const {
      int index = frame - blocks_.front()[0];
      if (index < 0) {
        index = 0;
      }
      if (index >= frame_to_block_lookup_.size()) {
        index = frame_to_block_lookup_.size() - 1;
      }
      DIALS_ASSERT(index >= 0);
      DIALS_ASSERT(index < frame_to_block_lookup_.size());
      return frame_to_block_lookup_[index];
    }

  private:
    /**
     * Construct the block list
     * @param range The range of frames
     * @param block_size The block size
     */
    void construct_block_list(tiny<int, 2> range, int block_size) {
      // Check some input
      int frame0 = range[0];
      int frame1 = range[1];
      DIALS_ASSERT(frame1 > frame0);
      int nframes = frame1 - frame0;
      DIALS_ASSERT(nframes > 0);

      // Block size is clamped to number of frames
      if (block_size > nframes) {
        block_size = nframes;
      }
      DIALS_ASSERT(block_size > 0);

      // If the block size is equal to 1, then add all frames as blocks
      // otherwise compute the blocks
      if (block_size == 1) {
        for (int f = frame0; f < frame1; ++f) {
          blocks_.push_back(tiny<int, 2>(f, f + 1));
        }
      } else {
        // Compute the half block size such that images are divided between
        // blocks more evenly spaced
        int nblocks = (int)std::ceil(2.0 * nframes / (double)block_size);
        DIALS_ASSERT(nblocks > 0 && nblocks <= nframes);
        int half_block_size = (int)std::ceil((double)nframes / (double)nblocks);

        // Construct a list of block indices
        af::shared<int> indices;
        indices.push_back(frame0);
        for (int i = 0; i < nblocks; ++i) {
          int frame = frame0 + (i + 1) * half_block_size;
          if (frame > frame1) {
            frame = frame1;
          }
          indices.push_back(frame);
          if (frame == frame1) {
            break;
          }
        }

        // Add all the blocks to the list
        DIALS_ASSERT(indices.front() == frame0);
        DIALS_ASSERT(indices.back() == frame1);
        DIALS_ASSERT(indices.size() > 2);
        for (std::size_t i = 0; i < indices.size() - 2; ++i) {
          int i1 = indices[i];
          int i2 = indices[i + 2];
          DIALS_ASSERT(i2 > i1);
          blocks_.push_back(tiny<int, 2>(i1, i2));
        }
        DIALS_ASSERT(blocks_.size() > 0);
      }
    }

    /**
     * Construct the frame to block lookup table
     */
    void construct_frame_to_block_lookup() {
      // Check all the jobs overlap and are in order
      for (std::size_t i = 0; i < blocks_.size() - 1; ++i) {
        DIALS_ASSERT(blocks_[i][0] < blocks_[i][1]);
        DIALS_ASSERT(blocks_[i + 1][0] < blocks_[i + 1][1]);
        DIALS_ASSERT(blocks_[i][0] < blocks_[i + 1][0]);
        DIALS_ASSERT(blocks_[i][1] >= blocks_[i + 1][0]);
        DIALS_ASSERT(blocks_[i][1] < blocks_[i + 1][1]);
      }

      // set the first and last frames
      int first_frame = blocks_.front()[0];
      int last_frame = blocks_.back()[1];
      DIALS_ASSERT(first_frame < last_frame);

      // Loop through all the frames and find the block to which the frame is
      // closest to the centre. Add the frame to that block
      std::size_t closest_index = 0;
      for (int frame = first_frame; frame < last_frame; ++frame) {
        int z0 = blocks_[closest_index][0];
        int z1 = blocks_[closest_index][1];
        double zc = (z0 + z1) / 2.0;
        double closest_distance = std::abs(zc - (frame + 0.5));
        for (std::size_t i = closest_index + 1; i < blocks_.size(); ++i) {
          int zz0 = blocks_[i][0];
          int zz1 = blocks_[i][1];
          double zzc = (zz0 + zz1) / 2.0;
          double distance = std::abs(zzc - (frame + 0.5));
          if (distance < closest_distance) {
            closest_distance = distance;
            closest_index = i;
          } else {
            break;
          }
        }
        frame_to_block_lookup_.push_back(closest_index);
      }
    }

    std::vector<tiny<int, 2> > blocks_;
    std::vector<std::size_t> frame_to_block_lookup_;
  };

  /**
   * A lookup class to split reflections along job boundaries and to gives
   * indices of reflections in each job
   */
  class SimpleReflectionLookup {
  public:
    /**
     * Construct the lookup
     * @param blocks The block list
     * @param data The reflections
     */
    SimpleReflectionLookup(const SimpleBlockList &blocks, af::reflection_table data)
        : blocks_(blocks), first_frame_(0), last_frame_(0) {
      // Split the reflections along block boundaries
      data_ = split_reflections(data);

      // Construct the lookup table
      construct_block_to_reflection_lookup();
    }

    /**
     * @returns The reflection data
     */
    af::reflection_table data() {
      return data_;
    }

    /**
     * Get the indices of reflections in this block
     * @param index The block index
     * @returns The list of reflection indices
     */
    af::const_ref<std::size_t> indices(std::size_t index) const {
      DIALS_ASSERT(index < block_to_reflection_lookup_.size());
      return af::const_ref<std::size_t>(&block_to_reflection_lookup_[index][0],
                                        block_to_reflection_lookup_[index].size());
    }

    /**
     * Get the index of the block closest to this frame
     * @param frame The frame number
     * @returns The block index
     */
    std::size_t block_index(int frame) const {
      return blocks_.block_index(frame);
    }

    /**
     * Get the frame range for the block
     * @param index The block index
     * @returns The frames in the block
     */
    tiny<int, 2> block_range(std::size_t index) const {
      return blocks_[index];
    }

  protected:
    /**
     * Create the lookup of reflections in each job
     */
    void construct_block_to_reflection_lookup() {
      DIALS_ASSERT(data_.is_consistent());
      DIALS_ASSERT(data_.size() > 0);
      DIALS_ASSERT(data_.contains("bbox"));
      DIALS_ASSERT(blocks_.size() > 0);

      // Get some arrays
      af::const_ref<int6> bbox = data_["bbox"];

      // For each reflection, get the frame range and then get the centre of
      // that range. Now find the block closest to that z point. Check that the
      // splitting has worked correctly and that the reflection is within the
      // block frame range. Then add the reflection to the list for the block.
      block_to_reflection_lookup_.resize(blocks_.size());
      for (std::size_t i = 0; i < bbox.size(); ++i) {
        int z0 = bbox[i][4];
        int z1 = bbox[i][5];
        int zc = (int)std::floor((z0 + z1) / 2.0);
        int index = block_index(zc);
        tiny<int, 2> block = block_range(index);
        DIALS_ASSERT(z0 >= block[0]);
        DIALS_ASSERT(z1 <= block[1]);
        DIALS_ASSERT(index < block_to_reflection_lookup_.size());
        block_to_reflection_lookup_[index].push_back(i);
      }
    }

    /**
     * Split the reflections overlapping job boundaries
     * @param data The input reflection table
     * @returns The split reflection table
     */
    af::reflection_table split_reflections(af::reflection_table data) const {
      // Check input
      DIALS_ASSERT(data.is_consistent());
      DIALS_ASSERT(data.size() > 0);
      DIALS_ASSERT(data.contains("bbox"));
      DIALS_ASSERT(blocks_.size() > 0);

      // Select reflections that are in range
      data = select_in_range_reflections(data);

      // Split the reflection
      af::const_ref<int6> bbox = data["bbox"];
      af::shared<int6> bbox_new;
      af::shared<std::size_t> indices;
      int f0 = blocks_.first_frame();
      int f1 = blocks_.last_frame();
      for (std::size_t i = 0; i < bbox.size(); ++i) {
        DIALS_ASSERT(bbox[i][4] < bbox[i][5]);
        int z0 = std::max(f0, bbox[i][4]);
        int z1 = std::min(f1, bbox[i][5]);
        DIALS_ASSERT(z0 < z1);
        std::vector<tiny<int, 2> > splits;
        split_at_boundaries(z0, z1, std::back_inserter(splits));
        DIALS_ASSERT(splits.size() > 0);
        for (std::size_t j = 0; j < splits.size(); ++j) {
          int6 b = bbox[i];
          b[4] = splits[j][0];
          b[5] = splits[j][1];
          indices.push_back(i);
          bbox_new.push_back(b);
        }
      }

      // Resize the reflection table
      DIALS_ASSERT(bbox_new.size() == indices.size());
      DIALS_ASSERT(bbox_new.size() >= bbox.size());
      data.resize(bbox_new.size());

      // Reorder the reflections
      af::boost_python::flex_table_suite::reorder(data, indices.const_ref());

      // Set the new bounding boxes
      af::boost_python::flex_table_suite::setitem_column(
        data, "bbox", bbox_new.const_ref());
      af::boost_python::flex_table_suite::setitem_column(
        data, "partial_id", indices.const_ref());

      // Return the data
      return data;
    }

    /**
     * Select reflections in range
     * @param data The input reflection table
     * @returns The split reflection table
     */
    af::reflection_table select_in_range_reflections(af::reflection_table data) const {
      using namespace af::boost_python::flex_table_suite;

      // Check if any need to be removed
      af::const_ref<int6> bbox = data["bbox"];
      af::shared<std::size_t> indices;
      int f0 = blocks_.first_frame();
      int f1 = blocks_.last_frame();
      for (std::size_t i = 0; i < bbox.size(); ++i) {
        DIALS_ASSERT(bbox[i][4] < bbox[i][5]);
        int z0 = std::max(f0, bbox[i][4]);
        int z1 = std::min(f1, bbox[i][5]);
        if (z0 < z1) {
          indices.push_back(i);
        }
      }

      // Select the reflections
      return select_rows_index(data, indices.const_ref());
    }

    /**
     * Split a reflection at block boundaries so that most of the reflection is
     * recorded within a single block. Works recursively.
     * @param z0 The first frame in the shoebox
     * @param z1 The last frame in the shoebox
     * @param out The output iterator
     */
    template <typename OutputIterator>
    void split_at_boundaries(int z0, int z1, OutputIterator out) const {
      DIALS_ASSERT(z0 < z1);

      // Compute the block closest to the centre
      int zc = (int)std::floor((z0 + z1) / 2.0);
      int index = block_index(zc);
      tiny<int, 2> block = block_range(index);
      DIALS_ASSERT(block[0] < block[1]);

      // Get the min and max frame range
      int zmin = std::max(z0, block[0]);
      int zmax = std::min(z1, block[1]);
      DIALS_ASSERT(zmin < zmax);

      // If the reflection extends below the block range
      // then split at the lower boundary
      if (z0 < zmin) {
        split_at_boundaries(z0, zmin, out);
      }

      // Append the new frame range
      *out++ = tiny<int, 2>(zmin, zmax);

      // If the reflection extends above the block range
      // then split at the upper boundary
      if (z1 > zmax) {
        split_at_boundaries(zmax, z1, out);
      }
    }

    SimpleBlockList blocks_;
    int first_frame_;
    int last_frame_;
    af::reflection_table data_;
    std::vector<std::vector<std::size_t> > block_to_reflection_lookup_;
  };

  /**
   * A class to manage the reflections
   */
  class SimpleReflectionManager {
  public:
    /**
     * Construct from the job list and reflection data
     * @param jobs The job list
     * @param data The reflection data
     */
    SimpleReflectionManager(const SimpleBlockList &blocks,
                            af::reflection_table data,
                            std::size_t njobs)
        : lookup_(blocks, data),
          njobs_(std::min(njobs, blocks.size())),
          finished_(njobs_, false),
          job_blocks_(njobs_) {
      DIALS_ASSERT(njobs_ > 0);
      std::size_t nblocks = blocks.size();
      DIALS_ASSERT(nblocks > 0);
      DIALS_ASSERT(nblocks >= njobs_);

      // Compute blocks per job and remaining blocks
      int blocks_per_job = (int)std::floor((double)nblocks / (double)njobs_);
      int remaining_blocks = nblocks % njobs_;
      DIALS_ASSERT(blocks_per_job >= 0);
      DIALS_ASSERT(remaining_blocks >= 0);

      // Populate the list of job blocks
      int total_blocks_in_job_list = 0;
      int last_block = 0;
      for (std::size_t i = 0; i < njobs_; ++i) {
        int num_blocks_in_job = blocks_per_job;
        if (remaining_blocks > 0) {
          num_blocks_in_job++;
          remaining_blocks--;
        }
        DIALS_ASSERT(num_blocks_in_job > 0);
        job_blocks_[i][0] = last_block;
        job_blocks_[i][1] = last_block + num_blocks_in_job;
        last_block += num_blocks_in_job;
        total_blocks_in_job_list += job_blocks_[i][1] - job_blocks_[i][0];
      }
      DIALS_ASSERT(total_blocks_in_job_list == nblocks);
    }

    /**
     * @returns The result data
     */
    af::reflection_table data() {
      DIALS_ASSERT(finished());
      return lookup_.data();
    }

    /**
     * @returns Is the process finished
     */
    bool finished() const {
      return finished_.all_eq(true);
    }

    /**
     * @returns The number of tasks
     */
    std::size_t size() const {
      return finished_.size();
    }

    /**
     * @returns The block
     */
    tiny<int, 2> block(std::size_t index) const {
      return lookup_.block_range(index);
    }

    /**
     * @returns The job
     */
    tiny<int, 2> job(std::size_t index) const {
      DIALS_ASSERT(index < job_blocks_.size());
      int i0 = job_blocks_[index][0];
      int i1 = job_blocks_[index][1] - 1;
      DIALS_ASSERT(i0 <= i1);
      int f0 = block(i0)[0];
      int f1 = block(i1)[1];
      DIALS_ASSERT(f0 < f1);
      return tiny<int, 2>(f0, f1);
    }

    /**
     * @returns The number of reflections in a job
     */
    std::size_t num_reflections(std::size_t index) const {
      DIALS_ASSERT(index < finished_.size());
      tiny<int, 2> frame = job(index);
      tiny<int, 2> blocks = job_blocks_[index];
      DIALS_ASSERT(frame[0] < frame[1]);
      DIALS_ASSERT(blocks[0] < blocks[1]);
      std::size_t n = 0;
      for (std::size_t block = blocks[0]; block < blocks[1]; ++block) {
        n += lookup_.indices(block).size();
      }
      return n;
    }

    /**
     * @returns The reflections for a particular block.
     */
    af::reflection_table split(std::size_t index) {
      using namespace af::boost_python::flex_table_suite;
      DIALS_ASSERT(index < finished_.size());

      // Get the job range
      tiny<int, 2> frame = job(index);
      tiny<int, 2> blocks = job_blocks_[index];
      DIALS_ASSERT(frame[0] < frame[1]);
      DIALS_ASSERT(blocks[0] < blocks[1]);

      // Select reflections to process in block
      af::shared<std::size_t> indices;
      for (std::size_t block = blocks[0]; block < blocks[1]; ++block) {
        af::const_ref<std::size_t> temp = lookup_.indices(block);
        indices.insert(indices.end(), temp.begin(), temp.end());
      }

      // Select reflections to process in block
      af::reflection_table data =
        select_rows_index(lookup_.data(), indices.const_ref());

      // Select other reflections from adjacent blocks. These reflections will
      // not be processed but will be used in finding adjacent reflections.
      // First select reflections from the block before.
      if (blocks[0] > 0) {
        af::reflection_table temp =
          select_rows_index(lookup_.data(), lookup_.indices(blocks[0] - 1));
        af::ref<int6> bbox = temp["bbox"];
        af::ref<std::size_t> flags = temp["flags"];
        af::shared<std::size_t> selection;
        for (std::size_t i = 0; i < flags.size(); ++i) {
          flags[i] |= af::DontIntegrate;
          if (bbox[i][5] > frame[0]) {
            if (bbox[i][4] < frame[0]) {
              bbox[i][4] = frame[0];
            }
            selection.push_back(i);
          }
        }
        temp = select_rows_index(temp, selection.const_ref());
        extend(data, temp);
      }

      // Now select reflections from the block after
      if (blocks[1] < size()) {
        af::reflection_table temp =
          select_rows_index(lookup_.data(), lookup_.indices(blocks[1]));
        af::ref<int6> bbox = temp["bbox"];
        af::ref<std::size_t> flags = temp["flags"];
        af::shared<std::size_t> selection;
        for (std::size_t i = 0; i < flags.size(); ++i) {
          flags[i] |= af::DontIntegrate;
          if (bbox[i][4] < frame[1]) {
            if (bbox[i][5] > frame[1]) {
              bbox[i][5] = frame[1];
            }
            selection.push_back(i);
          }
        }
        temp = select_rows_index(temp, selection.const_ref());
        extend(data, temp);
      }

      // Return the reflections
      return data;
    }

    /**
     * Accumulate the results.
     */
    void accumulate(std::size_t index, af::reflection_table result) {
      using namespace af::boost_python::flex_table_suite;
      DIALS_ASSERT(index < finished_.size());
      DIALS_ASSERT(finished_[index] == false);

      // Get the job range
      tiny<int, 2> frame = job(index);
      tiny<int, 2> blocks = job_blocks_[index];
      DIALS_ASSERT(frame[0] < frame[1]);
      DIALS_ASSERT(blocks[0] < blocks[1]);

      // Select reflections to process in block
      af::shared<std::size_t> indices;
      for (std::size_t block = blocks[0]; block < blocks[1]; ++block) {
        af::const_ref<std::size_t> temp = lookup_.indices(block);
        indices.insert(indices.end(), temp.begin(), temp.end());
      }

      // Get the total number of reflections to integrate
      std::size_t num_reflections = indices.size();

      // Check the number is less (because result includes adjacent reflections)
      DIALS_ASSERT(num_reflections <= result.size());

      // Resize the input reflections to just those that were processed
      result.resize(indices.size());

      // Get the data array
      af::reflection_table data = lookup_.data();

      // Set the result in the lookup data
      set_selected_rows_index(data, indices.const_ref(), result);
      finished_[index] = true;
    }

  private:
    SimpleReflectionLookup lookup_;
    std::size_t njobs_;
    af::shared<bool> finished_;
    af::shared<tiny<int, 2> > job_blocks_;
  };

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_INTEGRATION_PARALLEL_INTEGRATOR_H
