/*
 * transform.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_TRANSFORM_H
#define DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_TRANSFORM_H

#include <memory>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <scitbx/array_family/tiny_types.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/goniometer.h>
#include <dxtbx/model/scan.h>
#include <dials/algorithms/polygon/spatial_interpolation.h>
#include <dials/algorithms/profile_model/gaussian_rs/coordinate_system.h>
#include <dials/algorithms/profile_model/gaussian_rs/transform/map_frames.h>
#include <dials/algorithms/profile_model/gaussian_rs/transform/beam_vector_map.h>
#include <dials/model/data/shoebox.h>

namespace dials {
  namespace algorithms {
    namespace profile_model {
      namespace gaussian_rs {
  namespace transform {

    using dials::algorithms::polygon::simple_area;
    using dials::algorithms::polygon::clip::quad_with_convex_quad;
    using dials::algorithms::polygon::clip::vert4;
    using dials::algorithms::polygon::clip::vert8;
    using dials::algorithms::polygon::spatial_interpolation::Match;
    using dials::algorithms::polygon::spatial_interpolation::quad_to_grid;
    using dials::algorithms::polygon::spatial_interpolation::
      reverse_quad_inplace_if_backward;
    using dials::model::Foreground;
    using dials::model::Shoebox;
    using dials::model::Valid;
    using dxtbx::model::BeamBase;
    using dxtbx::model::Detector;
    using dxtbx::model::Goniometer;
    using dxtbx::model::Scan;
    using scitbx::vec2;
    using scitbx::vec3;
    using scitbx::af::double3;
    using scitbx::af::int2;
    using scitbx::af::int3;
    using scitbx::af::int6;

    template <typename T>
    T min4(T a, T b, T c, T d) {
      return std::min(std::min(a, b), std::min(c, d));
    }

    template <typename T>
    T max4(T a, T b, T c, T d) {
      return std::max(std::max(a, b), std::max(c, d));
    }

    /**
     * A class to construct the specification for the transform. Once instantiated
     * this object can be reused to transform lots of reflections.
     */
    class TransformSpec {
    public:
      /*
       * Initialise the class
       * @param beam The beam model
       * @param detector The detector model
       * @param gonio The goniometer model
       * @param scan The scan model
       * @param sigma_b The beam divergence
       * @param sigma_m The crystal mosaicity
       * @param n_sigma The number of standard deviations
       * @param grid_size The size of the reflection basis grid
       */
      TransformSpec(const std::shared_ptr<BeamBase> beam,
                    const Detector &detector,
                    const Goniometer &gonio,
                    const Scan &scan,
                    double sigma_b,
                    double sigma_m,
                    double n_sigma,
                    std::size_t grid_size)
          : beam_(beam),
            detector_(detector),
            goniometer_(gonio),
            scan_(scan),
            sigma_b_(sigma_b),
            sigma_m_(sigma_m),
            n_sigma_(n_sigma),
            half_grid_size_(grid_size),
            grid_size_(2 * grid_size + 1, 2 * grid_size + 1, 2 * grid_size + 1),
            step_size_(sigma_m_ * n_sigma_ / (grid_size + 0.5),
                       sigma_b_ * n_sigma_ / (grid_size + 0.5),
                       sigma_b_ * n_sigma_ / (grid_size + 0.5)),
            grid_centre_(grid_size + 0.5, grid_size + 0.5, grid_size + 0.5) {
        DIALS_ASSERT(sigma_m_ > 0);
        DIALS_ASSERT(sigma_b_ > 0);
        DIALS_ASSERT(n_sigma_ > 0);
        DIALS_ASSERT(detector.size() > 0);
        DIALS_ASSERT(step_size_.all_gt(0));
        DIALS_ASSERT(grid_size_.all_gt(0));
      }

      /** @returns the beam */
      const std::shared_ptr<BeamBase> beam() const {
        return beam_;
      }

      /** @returns the detector */
      const Detector &detector() const {
        return detector_;
      }

      /** @return the goniometer */
      const Goniometer &goniometer() const {
        return goniometer_;
      }

      /** @return the scan */
      const Scan &scan() const {
        return scan_;
      }

      /** @return sigma b */
      double sigma_b() const {
        return sigma_b_;
      }

      /** @return sigma m */
      double sigma_m() const {
        return sigma_m_;
      }

      /** @return n sigma */
      double n_sigma() const {
        return n_sigma_;
      }

      /** @returns The half grid size */
      std::size_t half_grid_size() const {
        return half_grid_size_;
      }

      /** @returns the grid size */
      int3 grid_size() const {
        return grid_size_;
      }

      /** @returns the grid step size */
      double3 step_size() const {
        return step_size_;
      }

      /** @returns the grid centre */
      double3 grid_centre() const {
        return grid_centre_;
      }

    private:
      std::shared_ptr<BeamBase> beam_;
      Detector detector_;
      Goniometer goniometer_;
      Scan scan_;
      double sigma_b_;
      double sigma_m_;
      double n_sigma_;
      std::size_t half_grid_size_;
      int3 grid_size_;
      double3 step_size_;
      double3 grid_centre_;
    };

    /**
     * A class to perform the local coordinate transform for a single reflection.
     * The class has a number of different constructors to allow the transform
     * to be done with lots of different inputs.
     *
     * Example:
     *
     *  from dials.algorithms.profile_model::gaussian_rs import transform
     *  forward = transform.Forward(spec, reflection)
     *  print forward.profile()
     *  print forward.background()
     */
    template <typename FloatType = double>
    class TransformForward {
    public:
      typedef FloatType float_type;
      typedef TransformSpec transform_spec_type;

      TransformForward() {}

      TransformForward(const TransformSpec &spec,
                       const CoordinateSystem &cs,
                       int6 bbox,
                       std::size_t panel,
                       const af::const_ref<FloatType, af::c_grid<3> > &image,
                       const af::const_ref<bool, af::c_grid<3> > &mask) {
        init(spec, cs, bbox, panel);
        call(spec.detector()[panel], image, mask);
      }

      TransformForward(const TransformSpec &spec,
                       const CoordinateSystem &cs,
                       int6 bbox,
                       std::size_t panel,
                       const af::const_ref<FloatType, af::c_grid<3> > &image,
                       const af::const_ref<FloatType, af::c_grid<3> > &bkgrd,
                       const af::const_ref<bool, af::c_grid<3> > &mask) {
        init(spec, cs, bbox, panel);
        call(spec.detector()[panel], image, bkgrd, mask);
      }

      /** @returns The transformed profile */
      af::versa<FloatType, af::c_grid<3> > profile() const {
        return profile_;
      }

      /** @returns The transformed background (if set) */
      af::versa<FloatType, af::c_grid<3> > background() const {
        return background_;
      }

      af::versa<bool, af::c_grid<3> > mask() const {
        return mask_;
      }

    private:
      /** Initialise using a coordinate system struct */
      void init(const TransformSpec &spec,
                const CoordinateSystem &cs,
                int6 bbox,
                std::size_t panel) {
        // Initialise some stuff
        x0_ = bbox[0];
        y0_ = bbox[2];
        shoebox_size_ = int3(bbox[5] - bbox[4], bbox[3] - bbox[2], bbox[1] - bbox[0]);
        DIALS_ASSERT(shoebox_size_.all_gt(0));
        step_size_ = spec.step_size();
        grid_size_ = spec.grid_size();
        grid_cent_ = spec.grid_centre();
        s1_ = cs.s1();
        DIALS_ASSERT(s1_.length() > 0);
        e1_ = cs.e1_axis() / s1_.length();
        e2_ = cs.e2_axis() / s1_.length();

        // Calculate the fraction of intensity contributed from each data
        // frame to each grid coordinate
        vec2<int> zrange(bbox[4], bbox[5]);

        // Create the frame mapper
        MapFramesForward<FloatType> map_frames_forward(spec.scan().get_array_range()[0],
                                                       spec.scan().get_oscillation()[0],
                                                       spec.scan().get_oscillation()[1],
                                                       spec.sigma_m(),
                                                       spec.n_sigma(),
                                                       spec.grid_size()[2] / 2);
        zfraction_arr_ = map_frames_forward(zrange, cs.phi(), cs.zeta());

        MapFramesReverse<FloatType> map_frames_backward(
          spec.scan().get_array_range()[0],
          spec.scan().get_oscillation()[0],
          spec.scan().get_oscillation()[1],
          spec.sigma_m(),
          spec.n_sigma(),
          spec.grid_size()[2] / 2);
        efraction_arr_ = map_frames_backward(zrange, cs.phi(), cs.zeta());
      }

      /**
       * Map the pixel values from the input image to the output grid.
       * @param image The image to transform
       * @param mask The mask accompanying the image
       */
      void call(const Panel &panel,
                const af::const_ref<FloatType, af::c_grid<3> > &image,
                const af::const_ref<bool, af::c_grid<3> > &mask) {
        // Check the input
        DIALS_ASSERT(image.accessor().all_eq(shoebox_size_));
        DIALS_ASSERT(image.accessor().all_eq(mask.accessor()));

        af::const_ref<FloatType, af::c_grid<2> > zfraction = zfraction_arr_.const_ref();
        af::const_ref<FloatType, af::c_grid<2> > efraction = efraction_arr_.const_ref();

        // Initialise the profile arrays
        af::c_grid<3> accessor(grid_size_);
        profile_ = af::versa<FloatType, af::c_grid<3> >(accessor, 0.0);
        mask_ = af::versa<bool, af::c_grid<3> >(accessor, 0.0);

        // Compute the mask
        for (std::size_t kk = 0; kk < grid_size_[0]; ++kk) {
          double sumk = 0;
          for (std::size_t k = 0; k < shoebox_size_[0]; ++k) {
            sumk += efraction(kk, k);
          }
          if (sumk > 0.99) {
            for (std::size_t j = 0; j < grid_size_[1]; ++j) {
              for (std::size_t i = 0; i < grid_size_[2]; ++i) {
                mask_(kk, j, i) = true;
              }
            }
          }
        }

        vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1_);
        double attenuation_length = panel.attenuation_length(shoebox_centroid_px);

        af::versa<vec2<double>, af::c_grid<2> > gc_array(
          af::c_grid<2>(shoebox_size_[1] + 1, shoebox_size_[2] + 1));
        for (int j = 0; j <= shoebox_size_[1]; ++j) {
          for (int i = 0; i <= shoebox_size_[2]; ++i) {
            gc_array(j, i) = gc(panel, j, i, attenuation_length);
          }
        }

        // Loop through all the points in the shoebox. Calculate the polygon
        // formed by the pixel in the local coordinate system. Find the points
        // on the grid which intersect with the polygon and the fraction of the
        // pixel area shared with each grid point. For each intersection, loop
        // through the frames, mapping the fraction of the pixel value in each
        // frame to the grid point.
        af::c_grid<2> grid_size2(grid_size_[1], grid_size_[2]);
        for (std::size_t j = 0; j < shoebox_size_[1]; ++j) {
          if (y0_ + j < 0 | y0_ + j >= panel.get_image_size()[1]) {
            // This y-coordinate is outside the bounds of the panel
            continue;
          }
          for (std::size_t i = 0; i < shoebox_size_[2]; ++i) {
            if (x0_ + i < 0 | x0_ + i >= panel.get_image_size()[0]) {
              // This x-coordinate is outside the bounds of the panel
              continue;
            }
            vert4 input(gc_array(j, i),
                        gc_array(j, i + 1),
                        gc_array(j + 1, i + 1),
                        gc_array(j + 1, i));
            af::shared<Match> matches = quad_to_grid(input, grid_size2, 0);
            for (int m = 0; m < matches.size(); ++m) {
              FloatType fraction = matches[m].fraction;
              int index = matches[m].out;
              int ii = index % grid_size_[2];
              int jj = index / grid_size_[2];
              for (int k = 0; k < shoebox_size_[0]; ++k) {
                if (mask(k, j, i)) {
                  FloatType value = image(k, j, i) * fraction;
                  for (int kk = 0; kk < grid_size_[0]; ++kk) {
                    profile_(kk, jj, ii) += value * zfraction(k, kk);
                  }
                }
              }
            }
          }
        }
      }

      /**
       * Map the pixel values from the input image to the output grid.
       * @param image The image to transform
       * @param bkgrd The background image to transform
       * @param mask The mask accompanying the image
       */
      void call(const Panel &panel,
                const af::const_ref<FloatType, af::c_grid<3> > &image,
                const af::const_ref<FloatType, af::c_grid<3> > &bkgrd,
                const af::const_ref<bool, af::c_grid<3> > &mask) {
        // Check the input
        DIALS_ASSERT(image.accessor().all_eq(shoebox_size_));
        DIALS_ASSERT(image.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(image.accessor().all_eq(bkgrd.accessor()));

        af::const_ref<FloatType, af::c_grid<2> > zfraction = zfraction_arr_.const_ref();
        af::const_ref<FloatType, af::c_grid<2> > efraction = efraction_arr_.const_ref();

        // Initialise the profile arrays
        af::c_grid<3> accessor(grid_size_);
        profile_ = af::versa<FloatType, af::c_grid<3> >(accessor, 0.0);
        background_ = af::versa<FloatType, af::c_grid<3> >(accessor, 0.0);
        mask_ = af::versa<bool, af::c_grid<3> >(accessor, 0.0);

        // Compute the mask
        for (std::size_t kk = 0; kk < grid_size_[0]; ++kk) {
          double sumk = 0;
          for (std::size_t k = 0; k < shoebox_size_[0]; ++k) {
            sumk += efraction(kk, k);
          }
          if (sumk > 0.99) {
            for (std::size_t j = 0; j < grid_size_[1]; ++j) {
              for (std::size_t i = 0; i < grid_size_[2]; ++i) {
                mask_(kk, j, i) = true;
              }
            }
          }
        }

        vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1_);
        double attenuation_length = panel.attenuation_length(shoebox_centroid_px);

        // Mapping from shoebox image coordinate to transformed grid coordinate
        af::versa<vec2<double>, af::c_grid<2> > gc_array(
          af::c_grid<2>(shoebox_size_[1] + 1, shoebox_size_[2] + 1));
        for (int j = 0; j <= shoebox_size_[1]; ++j) {
          for (int i = 0; i <= shoebox_size_[2]; ++i) {
            gc_array(j, i) = gc(panel, j, i, attenuation_length);
          }
        }

        // Loop through all the points in the shoebox. Calculate the polygon
        // formed by the pixel in the local coordinate system. Find the points
        // on the grid which intersect with the polygon and the fraction of the
        // pixel area shared with each grid point. For each intersection, loop
        // through the frames, mapping the fraction of the pixel value in each
        // frame to the grid point.
        af::c_grid<2> grid_size2(grid_size_[1], grid_size_[2]);
        for (std::size_t j = 0; j < shoebox_size_[1]; ++j) {
          if (y0_ + j < 0 | y0_ + j >= panel.get_image_size()[1]) {
            // This y-coordinate is outside the bounds of the panel
            continue;
          }
          for (std::size_t i = 0; i < shoebox_size_[2]; ++i) {
            if (x0_ + i < 0 | x0_ + i >= panel.get_image_size()[0]) {
              // This x-coordinate is outside the bounds of the panel
              continue;
            }
            // The corners of the image pixel mapped to transformed grid coordinates
            vert4 input(gc_array(j, i),
                        gc_array(j, i + 1),
                        gc_array(j + 1, i + 1),
                        gc_array(j + 1, i));
            af::shared<Match> matches = quad_to_grid(input, grid_size2, 0);
            for (int m = 0; m < matches.size(); ++m) {
              FloatType fraction = matches[m].fraction;
              int index = matches[m].out;
              int jj = index / grid_size_[2];
              int ii = index % grid_size_[2];
              for (int k = 0; k < shoebox_size_[0]; ++k) {
                if (mask(k, j, i)) {
                  FloatType ivalue = image(k, j, i) * fraction;
                  FloatType bvalue = bkgrd(k, j, i) * fraction;
                  for (int kk = 0; kk < grid_size_[0]; ++kk) {
                    FloatType zf = zfraction(k, kk);
                    profile_(kk, jj, ii) += ivalue * zf;
                    background_(kk, jj, ii) += bvalue * zf;
                  }
                }
              }
            }
          }
        }
      }

      /**
       * Get a grid coordinate from an image coordinate
       * @param j The y index
       * @param i The x index
       * @returns The grid (c1, c2) index
       */
      vec2<double> gc(const Panel &panel, std::size_t j, std::size_t i) const {
        vec3<double> sp = panel.get_pixel_lab_coord(vec2<double>(x0_ + i, y0_ + j));
        vec3<double> ds = sp.normalize() * s1_.length() - s1_;
        return vec2<double>(grid_cent_[2] + (e1_ * ds) / step_size_[2],
                            grid_cent_[1] + (e2_ * ds) / step_size_[1]);
      }

      vec2<double> gc(const Panel &panel,
                      std::size_t j,
                      std::size_t i,
                      double attenuation_length) const {
        vec3<double> sp =
          panel.get_pixel_lab_coord(vec2<double>(x0_ + i, y0_ + j), attenuation_length);
        vec3<double> ds = sp.normalize() * s1_.length() - s1_;
        return vec2<double>(grid_cent_[2] + (e1_ * ds) / step_size_[2],
                            grid_cent_[1] + (e2_ * ds) / step_size_[1]);
      }

      int x0_, y0_;
      int3 shoebox_size_;
      int3 grid_size_;
      double3 step_size_;
      double3 grid_cent_;
      vec3<double> s1_, e1_, e2_;
      af::versa<bool, af::c_grid<3> > mask_;
      af::versa<FloatType, af::c_grid<3> > profile_;
      af::versa<FloatType, af::c_grid<3> > background_;
      af::versa<FloatType, af::c_grid<2> > zfraction_arr_;
      af::versa<FloatType, af::c_grid<2> > efraction_arr_;
    };

    /**
     * A class to do a forward transform with no phi model
     */
    class TransformForwardNoModel {
    public:
      TransformForwardNoModel() {}

      TransformForwardNoModel(const TransformSpec &spec,
                              const CoordinateSystem &cs,
                              int6 bbox,
                              std::size_t panel,
                              const af::const_ref<double, af::c_grid<3> > &image,
                              const af::const_ref<bool, af::c_grid<3> > &mask) {
        af::versa<double, af::c_grid<3> > bkgrd;
        init(spec, cs, bbox, panel, image, bkgrd.const_ref(), mask);
      }

      TransformForwardNoModel(const TransformSpec &spec,
                              const CoordinateSystem &cs,
                              int6 bbox,
                              std::size_t panel,
                              const af::const_ref<double, af::c_grid<3> > &image,
                              const af::const_ref<double, af::c_grid<3> > &bkgrd,
                              const af::const_ref<bool, af::c_grid<3> > &mask) {
        init(spec, cs, bbox, panel, image, bkgrd, mask);
      }

      /** @returns The transformed profile */
      af::versa<double, af::c_grid<3> > profile() const {
        return data_;
      }

      /** @returns The transformed background (if set) */
      af::versa<double, af::c_grid<3> > background() const {
        return background_;
      }

    private:
      void init(const TransformSpec &spec,
                const CoordinateSystem &cs,
                int6 bbox,
                std::size_t panel,
                const af::const_ref<double, af::c_grid<3> > &image,
                const af::const_ref<double, af::c_grid<3> > &bkgrd,
                const af::const_ref<bool, af::c_grid<3> > &mask) {
        // Check if we're using background
        bool use_background = bkgrd.size() > 0;

        // Check the input
        if (use_background) {
          DIALS_ASSERT(image.accessor().all_eq(bkgrd.accessor()));
        }
        DIALS_ASSERT(image.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(bbox[1] > bbox[0]);
        DIALS_ASSERT(bbox[3] > bbox[2]);
        DIALS_ASSERT(bbox[5] > bbox[4]);

        // Init the arrays
        af::c_grid<3> accessor(spec.grid_size());
        data_ = af::versa<double, af::c_grid<3> >(accessor, 0);
        if (use_background) {
          background_ = af::versa<double, af::c_grid<3> >(accessor, 0);
        }

        // Get the bbox size
        std::size_t xs = bbox[1] - bbox[0];
        std::size_t ys = bbox[3] - bbox[2];
        std::size_t zs = bbox[5] - bbox[4];

        // Compute the deltas
        double delta_b = spec.sigma_b() * spec.n_sigma();
        double delta_m = spec.sigma_m() * spec.n_sigma();

        // Compute the grid step and offset
        double xoff = -delta_b;
        double yoff = -delta_b;
        double zoff = -delta_m;
        double xstep = (2.0 * delta_b) / data_.accessor()[2];
        double ystep = (2.0 * delta_b) / data_.accessor()[1];
        double zstep = (2.0 * delta_m) / data_.accessor()[0];

        // Get the panel
        const Panel &dp = spec.detector()[panel];

        // Compute the detector coordinates of each point on the grid
        af::versa<vec2<double>, af::c_grid<2> > xy(
          af::c_grid<2>(data_.accessor()[1] + 1, data_.accessor()[2] + 1));
        for (std::size_t j = 0; j <= data_.accessor()[1]; ++j) {
          for (std::size_t i = 0; i <= data_.accessor()[2]; ++i) {
            double c1 = xoff + i * xstep;
            double c2 = yoff + j * ystep;
            vec3<double> s1p = cs.to_beam_vector(vec2<double>(c1, c2));
            vec2<double> xyp = dp.get_ray_intersection_px(s1p);
            xyp[0] -= bbox[0];
            xyp[1] -= bbox[2];
            xy(j, i) = xyp;
          }
        }

        // Compute the frame numbers of each slice on the grid
        af::shared<double> z(data_.accessor()[0] + 1);
        for (std::size_t k = 0; k <= data_.accessor()[0]; ++k) {
          double c3 = zoff + k * zstep;
          double phip = cs.to_rotation_angle_fast(c3);
          z[k] = spec.scan().get_array_index_from_angle(phip) - bbox[4];
        }

        // Get a list of pairs of overlapping polygons
        for (std::size_t j = 0; j < data_.accessor()[1]; ++j) {
          for (std::size_t i = 0; i < data_.accessor()[2]; ++i) {
            vec2<double> xy00 = xy(j, i);
            vec2<double> xy01 = xy(j, i + 1);
            vec2<double> xy11 = xy(j + 1, i + 1);
            vec2<double> xy10 = xy(j + 1, i);
            int x0 = (int)std::floor(min4(xy00[0], xy01[0], xy11[0], xy10[0]));
            int x1 = (int)std::ceil(max4(xy00[0], xy01[0], xy11[0], xy10[0]));
            int y0 = (int)std::floor(min4(xy00[1], xy01[1], xy11[1], xy10[1]));
            int y1 = (int)std::ceil(max4(xy00[1], xy01[1], xy11[1], xy10[1]));
            DIALS_ASSERT(x0 < x1);
            DIALS_ASSERT(y0 < y1);
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 > xs) x1 = xs;
            if (y1 > ys) y1 = ys;
            vert4 p1(xy00, xy01, xy11, xy10);
            reverse_quad_inplace_if_backward(p1);
            for (std::size_t jj = y0; jj < y1; ++jj) {
              for (std::size_t ii = x0; ii < x1; ++ii) {
                vec2<double> xy200(ii, jj);
                vec2<double> xy201(ii, jj + 1);
                vec2<double> xy211(ii + 1, jj + 1);
                vec2<double> xy210(ii + 1, jj);
                vert4 p2(xy200, xy201, xy211, xy210);
                reverse_quad_inplace_if_backward(p2);
                vert8 p3 = quad_with_convex_quad(p1, p2);
                double area = simple_area(p3);
                const double EPS = 1e-7;
                if (area < 0.0) {
                  DIALS_ASSERT(area > -EPS);
                  area = 0.0;
                }
                if (area > 1.0) {
                  DIALS_ASSERT(area <= (1.0 + EPS));
                  area = 1.0;
                }
                DIALS_ASSERT(0.0 <= area && area <= 1.0);
                if (area > 0) {
                  for (std::size_t k = 0; k < data_.accessor()[0]; ++k) {
                    double f00 = std::min(z[k], z[k + 1]);
                    double f01 = std::max(z[k], z[k + 1]);
                    DIALS_ASSERT(f01 > f00);
                    int z0 = std::max((int)0, (int)std::floor(f00));
                    int z1 = std::min((int)zs, (int)std::ceil(f01));
                    DIALS_ASSERT(z0 >= 0 && z1 <= (int)zs);
                    for (int kk = z0; kk < z1; ++kk) {
                      if (mask(kk, jj, ii)) {
                        std::size_t f10 = kk;
                        std::size_t f11 = kk + 1;
                        double f0 = std::max(f00, (double)f10);
                        double f1 = std::min(f01, (double)f11);
                        double fraction = f1 > f0 ? (f1 - f0) / 1.0 : 0.0;
                        DIALS_ASSERT(fraction <= 1.0);
                        DIALS_ASSERT(fraction >= 0.0);
                        data_(k, j, i) += fraction * area * image(kk, jj, ii);
                        if (use_background) {
                          background_(k, j, i) += fraction * area * bkgrd(kk, jj, ii);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      af::versa<double, af::c_grid<3> > data_;
      af::versa<double, af::c_grid<3> > background_;
    };

    /**
     * A class to do a reverse transform with no phi model
     */
    class TransformReverseNoModel {
    public:
      TransformReverseNoModel() {}

      TransformReverseNoModel(const TransformSpec &spec,
                              const CoordinateSystem &cs,
                              int6 bbox,
                              std::size_t panel,
                              const af::const_ref<double, af::c_grid<3> > &data) {
        init(spec, cs, bbox, panel, data);
      }

      /** @returns The transformed profile */
      af::versa<double, af::c_grid<3> > profile() const {
        return profile_;
      }

    private:
      void init(const TransformSpec &spec,
                const CoordinateSystem &cs,
                int6 bbox,
                std::size_t panel,
                const af::const_ref<double, af::c_grid<3> > &data) {
        DIALS_ASSERT(data.accessor().all_eq(spec.grid_size()));
        DIALS_ASSERT(bbox[1] > bbox[0]);
        DIALS_ASSERT(bbox[3] > bbox[2]);
        DIALS_ASSERT(bbox[5] > bbox[4]);

        // Create the profile array
        std::size_t xs = bbox[1] - bbox[0];
        std::size_t ys = bbox[3] - bbox[2];
        std::size_t zs = bbox[5] - bbox[4];
        profile_ = af::versa<double, af::c_grid<3> >(af::c_grid<3>(zs, ys, xs), 0);

        // Compute the deltas
        double delta_b = spec.sigma_b() * spec.n_sigma();
        double delta_m = spec.sigma_m() * spec.n_sigma();

        // Compute the grid step and offset
        double xoff = -delta_b;
        double yoff = -delta_b;
        double zoff = -delta_m;
        double xstep = (2.0 * delta_b) / data.accessor()[2];
        double ystep = (2.0 * delta_b) / data.accessor()[1];
        double zstep = (2.0 * delta_m) / data.accessor()[0];

        // Get the panel
        const Panel &dp = spec.detector()[panel];

        // Compute the detector coordinates of each point on the grid
        af::versa<vec2<double>, af::c_grid<2> > xy(
          af::c_grid<2>(data.accessor()[1] + 1, data.accessor()[2] + 1));
        for (std::size_t j = 0; j <= data.accessor()[1]; ++j) {
          for (std::size_t i = 0; i <= data.accessor()[2]; ++i) {
            double c1 = xoff + i * xstep;
            double c2 = yoff + j * ystep;
            vec3<double> s1p = cs.to_beam_vector(vec2<double>(c1, c2));
            vec2<double> xyp = dp.get_ray_intersection_px(s1p);
            xyp[0] -= bbox[0];
            xyp[1] -= bbox[2];
            xy(j, i) = xyp;
          }
        }

        // Compute the frame numbers of each slice on the grid
        af::shared<double> z(data.accessor()[0] + 1);
        for (std::size_t k = 0; k <= data.accessor()[0]; ++k) {
          double c3 = zoff + k * zstep;
          double phip = cs.to_rotation_angle_fast(c3);
          z[k] = spec.scan().get_array_index_from_angle(phip) - bbox[4];
        }

        // Get a list of pairs of overlapping polygons
        for (std::size_t j = 0; j < data.accessor()[1]; ++j) {
          for (std::size_t i = 0; i < data.accessor()[2]; ++i) {
            vec2<double> xy00 = xy(j, i);
            vec2<double> xy01 = xy(j, i + 1);
            vec2<double> xy11 = xy(j + 1, i + 1);
            vec2<double> xy10 = xy(j + 1, i);
            int x0 = (int)std::floor(min4(xy00[0], xy01[0], xy11[0], xy10[0]));
            int x1 = (int)std::ceil(max4(xy00[0], xy01[0], xy11[0], xy10[0]));
            int y0 = (int)std::floor(min4(xy00[1], xy01[1], xy11[1], xy10[1]));
            int y1 = (int)std::ceil(max4(xy00[1], xy01[1], xy11[1], xy10[1]));
            DIALS_ASSERT(x0 < x1);
            DIALS_ASSERT(y0 < y1);
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 > xs) x1 = xs;
            if (y1 > ys) y1 = ys;
            vert4 p1(xy00, xy01, xy11, xy10);
            double p1_area = simple_area(p1);
            DIALS_ASSERT(p1_area > 0);
            reverse_quad_inplace_if_backward(p1);
            for (std::size_t jj = y0; jj < y1; ++jj) {
              for (std::size_t ii = x0; ii < x1; ++ii) {
                vec2<double> xy200(ii, jj);
                vec2<double> xy201(ii, jj + 1);
                vec2<double> xy211(ii + 1, jj + 1);
                vec2<double> xy210(ii + 1, jj);
                vert4 p2(xy200, xy201, xy211, xy210);
                reverse_quad_inplace_if_backward(p2);
                vert8 p3 = quad_with_convex_quad(p1, p2);
                double area = simple_area(p3);
                area /= p1_area;
                const double EPS = 1e-7;
                if (area < 0.0) {
                  DIALS_ASSERT(area > -EPS);
                  area = 0.0;
                }
                if (area > 1.0) {
                  DIALS_ASSERT(area <= (1.0 + EPS));
                  area = 1.0;
                }
                DIALS_ASSERT(0.0 <= area && area <= 1.0);
                if (area > 0) {
                  for (std::size_t k = 0; k < data.accessor()[0]; ++k) {
                    double f00 = std::min(z[k], z[k + 1]);
                    double f01 = std::max(z[k], z[k + 1]);
                    DIALS_ASSERT(f01 > f00);
                    double fr = f01 - f00;
                    int z0 = std::max((int)0, (int)std::floor(f00));
                    int z1 = std::min((int)zs, (int)std::ceil(f01));
                    DIALS_ASSERT(z0 >= 0 && z1 <= (int)zs);
                    for (int kk = z0; kk < z1; ++kk) {
                      std::size_t f10 = kk;
                      std::size_t f11 = kk + 1;
                      double f0 = std::max(f00, (double)f10);
                      double f1 = std::min(f01, (double)f11);
                      double fraction = f1 > f0 ? (f1 - f0) / fr : 0.0;
                      DIALS_ASSERT(fraction <= 1.0);
                      DIALS_ASSERT(fraction >= 0.0);
                      double value = fraction * area * data(k, j, i);
                      profile_(kk, jj, ii) += value;
                    }
                  }
                }
              }
            }
          }
        }
      }

      af::versa<double, af::c_grid<3> > profile_;
    };

    /**
     * A class to do a reverse transform with no phi model
     */
    class TransformReverse {
    public:
      TransformReverse() {}

      TransformReverse(const TransformSpec &spec,
                       const CoordinateSystem &cs,
                       int6 bbox,
                       std::size_t panel,
                       const af::const_ref<double, af::c_grid<3> > &data) {
        init(spec, cs, bbox, panel, data);
      }

      /** @returns The transformed profile */
      af::versa<double, af::c_grid<3> > profile() const {
        return profile_;
      }

    private:
      void init(const TransformSpec &spec,
                const CoordinateSystem &cs,
                int6 bbox,
                std::size_t panel,
                const af::const_ref<double, af::c_grid<3> > &data) {
        DIALS_ASSERT(data.accessor().all_eq(spec.grid_size()));
        DIALS_ASSERT(bbox[1] > bbox[0]);
        DIALS_ASSERT(bbox[3] > bbox[2]);
        DIALS_ASSERT(bbox[5] > bbox[4]);

        // Create the profile array
        std::size_t xs = bbox[1] - bbox[0];
        std::size_t ys = bbox[3] - bbox[2];
        std::size_t zs = bbox[5] - bbox[4];
        profile_ = af::versa<double, af::c_grid<3> >(af::c_grid<3>(zs, ys, xs), 0);

        // Compute the deltas
        double delta_b = spec.sigma_b() * spec.n_sigma();
        double delta_m = spec.sigma_m() * spec.n_sigma();

        // Compute the grid step and offset
        double xoff = -delta_b;
        double yoff = -delta_b;
        double zoff = -delta_m;
        double xstep = (2.0 * delta_b) / data.accessor()[2];
        double ystep = (2.0 * delta_b) / data.accessor()[1];
        double zstep = (2.0 * delta_m) / data.accessor()[0];

        // Get the panel
        const Panel &dp = spec.detector()[panel];

        // Compute the detector coordinates of each point on the grid
        af::versa<vec2<double>, af::c_grid<2> > xy(
          af::c_grid<2>(data.accessor()[1] + 1, data.accessor()[2] + 1));
        for (std::size_t j = 0; j <= data.accessor()[1]; ++j) {
          for (std::size_t i = 0; i <= data.accessor()[2]; ++i) {
            double c1 = xoff + i * xstep;
            double c2 = yoff + j * ystep;
            vec3<double> s1p = cs.to_beam_vector(vec2<double>(c1, c2));
            vec2<double> xyp = dp.get_ray_intersection_px(s1p);
            xyp[0] -= bbox[0];
            xyp[1] -= bbox[2];
            xy(j, i) = xyp;
          }
        }

        // Compute the frame numbers of each slice on the grid
        af::shared<double> z(data.accessor()[0] + 1);
        for (std::size_t k = 0; k <= data.accessor()[0]; ++k) {
          double c3 = zoff + k * zstep;
          double phip = cs.to_rotation_angle_fast(c3);
          z[k] = spec.scan().get_array_index_from_angle(phip) - bbox[4];
        }

        // Create the frame mapper
        vec2<int> zrange(bbox[4], bbox[5]);
        MapFramesReverse<double> map_frames(spec.scan().get_array_range()[0],
                                            spec.scan().get_oscillation()[0],
                                            spec.scan().get_oscillation()[1],
                                            spec.sigma_m(),
                                            spec.n_sigma(),
                                            spec.grid_size()[2] / 2);
        af::versa<double, af::c_grid<2> > zfraction_arr =
          map_frames(zrange, cs.phi(), cs.zeta());
        af::const_ref<double, af::c_grid<2> > zfraction = zfraction_arr.const_ref();

        // Get a list of pairs of overlapping polygons
        for (std::size_t j = 0; j < data.accessor()[1]; ++j) {
          for (std::size_t i = 0; i < data.accessor()[2]; ++i) {
            vec2<double> xy00 = xy(j, i);
            vec2<double> xy01 = xy(j, i + 1);
            vec2<double> xy11 = xy(j + 1, i + 1);
            vec2<double> xy10 = xy(j + 1, i);
            int x0 = (int)std::floor(min4(xy00[0], xy01[0], xy11[0], xy10[0]));
            int x1 = (int)std::ceil(max4(xy00[0], xy01[0], xy11[0], xy10[0]));
            int y0 = (int)std::floor(min4(xy00[1], xy01[1], xy11[1], xy10[1]));
            int y1 = (int)std::ceil(max4(xy00[1], xy01[1], xy11[1], xy10[1]));
            DIALS_ASSERT(x0 < x1);
            DIALS_ASSERT(y0 < y1);
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 > xs) x1 = xs;
            if (y1 > ys) y1 = ys;
            vert4 p1(xy00, xy01, xy11, xy10);
            double p1_area = simple_area(p1);
            DIALS_ASSERT(p1_area > 0);
            reverse_quad_inplace_if_backward(p1);
            for (std::size_t jj = y0; jj < y1; ++jj) {
              for (std::size_t ii = x0; ii < x1; ++ii) {
                vec2<double> xy200(ii, jj);
                vec2<double> xy201(ii, jj + 1);
                vec2<double> xy211(ii + 1, jj + 1);
                vec2<double> xy210(ii + 1, jj);
                vert4 p2(xy200, xy201, xy211, xy210);
                reverse_quad_inplace_if_backward(p2);
                vert8 p3 = quad_with_convex_quad(p1, p2);
                double area = simple_area(p3);
                area /= p1_area;
                const double EPS = 1e-7;
                if (area < 0.0) {
                  DIALS_ASSERT(area > -EPS);
                  area = 0.0;
                }
                if (area > 1.0) {
                  DIALS_ASSERT(area <= (1.0 + EPS));
                  area = 1.0;
                }
                DIALS_ASSERT(0.0 <= area && area <= 1.0);
                if (area > 0) {
                  for (std::size_t k = 0; k < data.accessor()[0]; ++k) {
                    double f00 = std::min(z[k], z[k + 1]);
                    double f01 = std::max(z[k], z[k + 1]);
                    DIALS_ASSERT(f01 > f00);
                    int z0 = std::max((int)0, (int)std::floor(f00));
                    int z1 = std::min((int)zs, (int)std::ceil(f01));
                    DIALS_ASSERT(z0 >= 0 && z1 <= (int)zs);
                    for (int kk = z0; kk < z1; ++kk) {
                      DIALS_ASSERT(kk < zfraction.accessor()[1]);
                      DIALS_ASSERT(k < zfraction.accessor()[0]);
                      double fraction = zfraction(k, kk);
                      DIALS_ASSERT(fraction <= 1.0 + EPS);
                      DIALS_ASSERT(fraction >= 0.0 - EPS);
                      double value = fraction * area * data(k, j, i);
                      profile_(kk, jj, ii) += value;
                    }
                  }
                }
              }
            }
          }
        }
      }

      af::versa<double, af::c_grid<3> > profile_;
    };

}}}}}  // namespace dials::algorithms::profile_model::gaussian_rs::transform

#endif /* DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_TRANSFORM */
