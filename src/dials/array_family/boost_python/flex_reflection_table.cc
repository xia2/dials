/*
 * flex_reflection_table.cc
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <memory>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <dials/util/python_streambuf.h>
#include <numeric>
#include <dials/array_family/boost_python/flex_table_suite.h>
#include <dials/array_family/reflection_table.h>
#include <dials/array_family/reflection.h>
#include <dials/array_family/reflection_table_msgpack_adapter.h>
#include <dials/model/data/shoebox.h>
#include <dials/model/data/observation.h>
#include <dials/algorithms/profile_model/gaussian_rs/coordinate_system.h>
#include <scitbx/array_family/tiny_types.h>
#include <scitbx/array_family/ref_reductions.h>
#include <scitbx/vec3.h>
#include <scitbx/vec2.h>
#include <cctbx/miller.h>

namespace dials { namespace af { namespace boost_python {

  using namespace boost::python;
  using dials::algorithms::profile_model::gaussian_rs::CoordinateSystem;
  using dials::model::Observation;
  using dials::model::Shoebox;
  using dials::util::streambuf;
  using flex_table_suite::column_to_object_visitor;
  using flex_table_suite::flex_table_wrapper;
  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::af::int6;

  /**
   * Construct a reflection table from a list of observations and shoeboxes
   * @param o The observation
   * @param s The shoeboxes
   * @returns A reflection table
   */
  template <typename T>
  T *make_from_observation_and_shoebox(const af::const_ref<Observation> &o,
                                       const af::const_ref<Shoebox<> > &s) {
    // FIXME Should remove as we don't know whether intensity is summed or
    // profile fitted. In any case observation model should be deprecated.
    DIALS_ASSERT(o.size() == s.size());

    // The reflection table
    T result(o.size());
    af::shared<std::size_t> panel = result["panel"];
    af::shared<vec3<double> > xyzval = result["xyzobs.px.value"];
    af::shared<vec3<double> > xyzvar = result["xyzobs.px.variance"];
    af::shared<double> iraw = result["intensity.sum.value"];
    af::shared<double> irawv = result["intensity.sum.variance"];
    af::shared<int6> bbox = result["bbox"];
    af::shared<Shoebox<> > sbox = result["shoebox"];

    // Copy all the values
    for (std::size_t i = 0; i < result.nrows(); ++i) {
      // Check panel numbers
      DIALS_ASSERT(o[i].panel == s[i].panel);
      panel[i] = o[i].panel;

      // Copy observation info
      xyzval[i] = o[i].centroid.px.position;
      xyzvar[i] = o[i].centroid.px.std_err_sq;
      iraw[i] = o[i].intensity.observed.value;
      irawv[i] = o[i].intensity.observed.variance;

      // Copy shoebox info
      bbox[i] = s[i].bbox;
      sbox[i].panel = s[i].panel;
      sbox[i].bbox = s[i].bbox;
      sbox[i].data = s[i].data;
      sbox[i].mask = s[i].mask;
      sbox[i].background = s[i].background;
    }

    // Return the new reflection table
    return new T(result);
  }

  /**
   * A function to print some help about keys
   */
  template <typename T>
  std::string help_keys(const T &self) {
    std::string result =
      "Standard column names:\n"
      "======================\n"
      "\n"
      " Columns in the reflection table can have any name and type;\n"
      " however, it is helpful to have a set of standard data columns\n"
      " which can be used by different algorithms. These are shown below.\n"
      "\n"
      " General properties\n"
      " ------------------\n"
      "\n"
      "  flags:                         bit mask status flags\n"
      "  id:                            experiment id\n"
      "  panel:                         the detector panel index\n"
      "\n"
      " Predicted properties\n"
      " --------------------\n"
      "\n"
      "  miller_index:                  miller indices\n"
      "  entering:                      reflection entering/exiting\n"
      "  s1:                            the diffracted beam vector\n"
      "  xyzcal.mm:                     the predicted location (mm, mm, rad)\n"
      "  xyzcal.px:                     the predicted location (px, px, frame)\n"
      "  ub_matrix:                     predicted crystal setting\n"
      "\n"
      " Observed properties\n"
      " -------------------\n"
      "\n"
      "  xyzobs.px.value:               centroid pixel position (px, px, frame)\n"
      "  xyzobs.px.variance:            centroid pixel variance\n"
      "  xyzobs.mm.value:               centroid millimetre position (mm, mm, rad)\n"
      "  xyzobs.mm.variance:            centroid millimetre variance\n"
      "  rlp:                           reciprocal lattice point\n"
      "  intensity.sum.value:           raw intensity value\n"
      "  intensity.sum.variance:        raw intensity variance\n"
      "  intensity.prf.value:           profile fitted intensity value\n"
      "  intensity.prf.variance:        profile fitted intensity variance\n"
      "  intensity.scale.value:         intensity value used for scaling (without "
      "inverse scale factor applied)\n"
      "  intensity.scale.variance:      variance of intensity value used for scaling\n"
      "  inverse_scale_factor:          scale factor determined by scaling (divisory)\n"
      "  inverse_scale_factor_variance: variance of inverse scale factor\n"
      "  lp:                            LP correction (multiplicative)\n"
      "  qe:                            detector quantum efficiency correction "
      "(divisory)\n"
      "  profile.correlation:           correlation in profile fitting\n"
      "  partiality:                    fraction of reflection measured (i.e. Ifull = "
      "Isum/partiality)\n"
      "\n"
      " Shoebox properties\n"
      " ------------------\n"
      "\n"
      "  bbox:                          bounding box\n"
      "  shoebox:                       shoebox data/mask/background struct\n"
      "\n";
    return result;
  }

  /**
   * Do ray intersections for all items
   */
  template <typename T>
  af::shared<vec2<double> > compute_ray_intersections(const T &self,
                                                      const model::Detector &detector) {
    af::shared<vec2<double> > result(self.nrows());
    af::const_ref<vec3<double> > s1 =
      self.template get<vec3<double> >("s1").const_ref();
    af::const_ref<std::size_t> panel =
      self.template get<std::size_t>("panel").const_ref();
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = detector[panel[i]].get_ray_intersection(s1[i]);
    }
    return result;
  }

  /**
   * Get where the flag value is set
   */
  template <typename T>
  af::shared<bool> get_flags(const T &self, std::size_t value, bool all) {
    af::shared<bool> result(self.nrows());
    af::shared<std::size_t> flags = self.template get<std::size_t>("flags");
    DIALS_ASSERT(flags.size() == result.size());
    if (all) {
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = (flags[i] & value) == value;
      }
    } else {
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = (flags[i] & value) != 0;
      }
    }
    return result;
  }

  /**
   * Set the flags.
   */
  template <typename T>
  void set_flags_by_mask(T self, const af::const_ref<bool> mask, std::size_t value) {
    DIALS_ASSERT(mask.size() == self.nrows());
    af::shared<std::size_t> flags = self.template get<std::size_t>("flags");
    for (std::size_t i = 0; i < mask.size(); ++i) {
      if (mask[i]) {
        flags[i] |= value;
      }
    }
  }

  /**
   * Set the flags.
   */
  template <typename T>
  void set_flags_by_index(T self,
                          const af::const_ref<std::size_t> index,
                          std::size_t value) {
    af::shared<std::size_t> flags = self.template get<std::size_t>("flags");
    for (std::size_t i = 0; i < index.size(); ++i) {
      DIALS_ASSERT(index[i] < flags.size());
      flags[index[i]] |= value;
    }
  }

  /**
   * Unset the flags.
   */
  template <typename T>
  void unset_flags_by_mask(T self, const af::const_ref<bool> mask, std::size_t value) {
    DIALS_ASSERT(mask.size() == self.nrows());
    af::shared<std::size_t> flags = self.template get<std::size_t>("flags");
    for (std::size_t i = 0; i < mask.size(); ++i) {
      if (mask[i]) {
        flags[i] &= ~value;
      }
    }
  }

  /**
   * Unset the flags.
   */
  template <typename T>
  void unset_flags_by_index(T self,
                            const af::const_ref<std::size_t> index,
                            std::size_t value) {
    af::shared<std::size_t> flags = self.template get<std::size_t>("flags");
    for (std::size_t i = 0; i < index.size(); ++i) {
      DIALS_ASSERT(index[i] < flags.size());
      flags[index[i]] &= ~value;
    }
  }

  /**
   * Split the reflection table in partials.
   */
  template <typename T>
  void split_partials(T self) {
    // Check the input
    DIALS_ASSERT(self.is_consistent());
    DIALS_ASSERT(self.contains("bbox"));
    DIALS_ASSERT(self.contains("shoebox") == false);

    // Compute the number of partials
    af::const_ref<int6> bbox = self["bbox"];
    std::size_t num_full = bbox.size();
    std::size_t num_partial = 0;
    for (std::size_t i = 0; i < bbox.size(); ++i) {
      DIALS_ASSERT(bbox[i][1] > bbox[i][0]);
      DIALS_ASSERT(bbox[i][3] > bbox[i][2]);
      DIALS_ASSERT(bbox[i][5] > bbox[i][4]);
      num_partial += (bbox[i][5] - bbox[i][4]);
    }
    DIALS_ASSERT(num_partial >= num_full);

    // If num partial is the same as num full exit early
    if (num_partial == num_full) {
      return;
    }

    // Create the new bounding boxes and indices
    af::shared<int6> bbox_new(num_partial);
    af::shared<std::size_t> indices(num_partial);
    std::size_t j = 0;
    for (std::size_t i = 0; i < num_full; ++i) {
      int6 b = bbox[i];
      for (int z = bbox[i][4]; z < bbox[i][5]; ++z) {
        DIALS_ASSERT(j < num_partial);
        bbox_new[j] = b;
        bbox_new[j][4] = z;
        bbox_new[j][5] = z + 1;
        indices[j] = i;
        j++;
      }
    }
    DIALS_ASSERT(j == num_partial);

    // Resize the reflection table
    self.resize(num_partial);

    // Reorder the reflections
    flex_table_suite::reorder(self, indices.const_ref());

    // Set the new bounding boxes
    flex_table_suite::setitem_column(self, "bbox", bbox_new.const_ref());
    flex_table_suite::setitem_column(self, "partial_id", indices.const_ref());
  }

  /**
   * Split the reflection table in partials.
   */
  template <typename T>
  void split_partials_with_shoebox(T self) {
    // Check the input
    DIALS_ASSERT(self.is_consistent());
    DIALS_ASSERT(self.contains("bbox"));
    DIALS_ASSERT(self.contains("shoebox"));
    DIALS_ASSERT(self.contains("panel"));

    // Compute the number of partials
    af::const_ref<Shoebox<> > sbox = self["shoebox"];
    af::const_ref<int6> bbox = self["bbox"];
    af::const_ref<std::size_t> panel = self["panel"];
    std::size_t num_full = bbox.size();
    std::size_t num_partial = 0;
    for (std::size_t i = 0; i < bbox.size(); ++i) {
      DIALS_ASSERT(bbox[i][1] > bbox[i][0]);
      DIALS_ASSERT(bbox[i][3] > bbox[i][2]);
      DIALS_ASSERT(bbox[i][5] > bbox[i][4]);
      num_partial += (bbox[i][5] - bbox[i][4]);
    }
    DIALS_ASSERT(num_partial >= num_full);

    // If num partial is the same as num full exit early
    if (num_partial == num_full) {
      return;
    }

    // Create the new bounding boxes and indices
    af::shared<Shoebox<> > sbox_new(num_partial);
    af::shared<int6> bbox_new(num_partial);
    af::shared<std::size_t> indices(num_partial);
    std::size_t j = 0;
    for (std::size_t i = 0; i < num_full; ++i) {
      const Shoebox<> &s1 = sbox[i];
      int6 b = bbox[i];
      DIALS_ASSERT(s1.is_consistent());
      DIALS_ASSERT(s1.bbox[0] == b[0] && s1.bbox[1] == b[1]);
      DIALS_ASSERT(s1.bbox[2] == b[2] && s1.bbox[3] == b[3]);
      DIALS_ASSERT(s1.bbox[4] == b[4] && s1.bbox[5] == b[5]);
      std::size_t first = 0;
      for (int z = bbox[i][4]; z < bbox[i][5]; ++z) {
        DIALS_ASSERT(j < num_partial);
        bbox_new[j] = b;
        bbox_new[j][4] = z;
        bbox_new[j][5] = z + 1;
        indices[j] = i;
        Shoebox<> s2(panel[i], bbox_new[j]);
        s2.allocate();
        DIALS_ASSERT(s2.is_consistent());
        std::size_t last = first + s2.data.size();
        std::copy(s1.data.begin() + first, s1.data.begin() + last, s2.data.begin());
        std::copy(s1.mask.begin() + first, s1.mask.begin() + last, s2.mask.begin());
        std::copy(s1.background.begin() + first,
                  s1.background.begin() + last,
                  s2.background.begin());
        sbox_new[j] = s2;
        j++;
        first = last;
      }
    }
    DIALS_ASSERT(j == num_partial);

    // Resize the reflection table
    self.resize(num_partial);

    // Reorder the reflections
    flex_table_suite::reorder(self, indices.const_ref());

    // Set the new bounding boxes
    flex_table_suite::setitem_column(self, "bbox", bbox_new.const_ref());
    flex_table_suite::setitem_column(self, "shoebox", sbox_new.const_ref());
    flex_table_suite::setitem_column(self, "partial_id", indices.const_ref());
  }

  /**
   * Split the reflection table in partials and return indices.
   */
  template <typename T>
  af::shared<std::size_t> split_partial_indices(T self) {
    // Check the input
    DIALS_ASSERT(self.is_consistent());
    DIALS_ASSERT(self.contains("bbox"));

    // Compute the number of partials
    af::const_ref<int6> bbox = self["bbox"];
    std::size_t num_full = bbox.size();
    std::size_t num_partial = 0;
    for (std::size_t i = 0; i < bbox.size(); ++i) {
      DIALS_ASSERT(bbox[i][1] > bbox[i][0]);
      DIALS_ASSERT(bbox[i][3] > bbox[i][2]);
      DIALS_ASSERT(bbox[i][5] > bbox[i][4]);
      num_partial += (bbox[i][5] - bbox[i][4]);
    }
    DIALS_ASSERT(num_partial >= num_full);

    // If num partial is the same as num full exit early
    af::shared<std::size_t> indices(num_partial);
    if (num_partial == num_full) {
      for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
      }
      return indices;
    }

    // Create the new bounding boxes and indices
    std::size_t j = 0;
    for (std::size_t i = 0; i < num_full; ++i) {
      for (int z = bbox[i][4]; z < bbox[i][5]; ++z) {
        DIALS_ASSERT(j < num_partial);
        indices[j] = i;
        j++;
      }
    }
    DIALS_ASSERT(j == num_partial);
    return indices;
  }

  /**
   * Split reflection table by experiment id
   */
  template <typename T>
  boost::python::list split_by_experiment_id(T self) {
    DIALS_ASSERT(self.contains("id"));

    // Get the id array
    af::const_ref<int> id = self["id"];

    // Get the number of experiments
    std::size_t num_expr = 0;
    for (std::size_t i = 0; i < id.size(); ++i) {
      DIALS_ASSERT(id[i] >= 0);
      if (id[i] >= num_expr) num_expr = id[i] + 1;
    }

    // Get the number of each
    std::vector<std::size_t> num(num_expr, 0);
    for (std::size_t i = 0; i < id.size(); ++i) {
      num[id[i]]++;
    }

    // Compute the indices
    std::vector<std::size_t> indices(id.size());
    std::vector<std::size_t> offset(1, 0);
    std::partial_sum(num.begin(), num.end(), std::back_inserter(offset));
    num.assign(num.size(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i) {
      std::size_t j = id[i];
      DIALS_ASSERT(j < offset.size() - 1);
      std::size_t off1 = offset[j];
      std::size_t off2 = offset[j + 1];
      DIALS_ASSERT(off2 > off1);
      DIALS_ASSERT(off2 <= indices.size());
      std::size_t k = off1 + num[j];
      num[j]++;
      indices[k] = i;
    }

    // For each experiment if select the reflections in the list
    boost::python::list result;
    for (std::size_t i = 0; i < offset.size() - 1; ++i) {
      std::size_t off1 = offset[i];
      std::size_t off2 = offset[i + 1];
      DIALS_ASSERT(off2 >= off1);
      std::size_t off = off1;
      std::size_t num = off2 - off1;
      if (num > 0) {
        DIALS_ASSERT(off + num <= indices.size());
        reflection_table table = flex_table_suite::select_rows_index(
          self, const_ref<std::size_t>(&indices[off], num));
        typedef reflection_table::experiment_map_type::const_iterator const_iterator;
        for (const_iterator it = self.experiment_identifiers()->begin();
             it != self.experiment_identifiers()->end();
             ++it) {
          int first_elem = 0;
          af::const_ref<int> id = table["id"];
          const_iterator found = self.experiment_identifiers()->find(id[first_elem]);
          if (found != self.experiment_identifiers()->end()) {
            (*table.experiment_identifiers())[found->first] = found->second;
          }
        }
        result.append(table);
      }
    }

    // Return the result
    return result;
  }

  /**
   * Split reflection table by experiment id
   */
  template <typename T>
  boost::python::list split_indices_by_experiment_id(T self, std::size_t num_expr) {
    DIALS_ASSERT(self.size() > 0);
    DIALS_ASSERT(num_expr > 0);
    DIALS_ASSERT(self.contains("id"));

    // Get the id array
    af::const_ref<int> id = self["id"];

    // Get the number of each
    std::vector<std::size_t> num(num_expr, 0);
    for (std::size_t i = 0; i < id.size(); ++i) {
      DIALS_ASSERT(id[i] >= 0);
      DIALS_ASSERT(id[i] < num_expr);
      num[id[i]]++;
    }

    // Compute the indices
    std::vector<std::size_t> indices(id.size());
    std::vector<std::size_t> offset(1, 0);
    std::partial_sum(num.begin(), num.end(), std::back_inserter(offset));
    num.assign(num.size(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i) {
      std::size_t exp_id = id[i];
      DIALS_ASSERT(exp_id < offset.size() - 1);
      std::size_t off1 = offset[exp_id];
      std::size_t off2 = offset[exp_id + 1];
      DIALS_ASSERT(off2 > off1);
      DIALS_ASSERT(off2 <= indices.size());
      std::size_t k = off1 + num[exp_id];
      num[exp_id]++;
      indices[k] = i;
    }

    // For each experiment if select the reflections in the list
    boost::python::list result;
    for (std::size_t i = 0; i < offset.size() - 1; ++i) {
      std::size_t off1 = offset[i];
      std::size_t off2 = offset[i + 1];
      DIALS_ASSERT(off2 >= off1);
      std::size_t off = off1;
      std::size_t num = off2 - off1;
      if (num > 0) {
        DIALS_ASSERT(off + num <= indices.size());
        result.append(af::shared<std::size_t>(&indices[off], &indices[off] + num));
      } else {
        result.append(af::shared<std::size_t>());
      }
    }

    // Return the result
    return result;
  }

  /**
   * Compute phi range of reflection
   */
  template <typename T>
  af::shared<vec2<double> > compute_phi_range(T self,
                                              vec3<double> m2,
                                              vec3<double> s0,
                                              double sigma_m,
                                              double n_sigma) {
    // Check contents
    DIALS_ASSERT(sigma_m > 0);
    DIALS_ASSERT(n_sigma > 0);
    DIALS_ASSERT(self.contains("s1"));
    DIALS_ASSERT(self.contains("xyzcal.mm"));

    af::const_ref<vec3<double> > s1 = self["s1"];
    af::const_ref<vec3<double> > xyz = self["xyzcal.mm"];
    af::shared<vec2<double> > result(self.size());
    double delta_m = sigma_m * n_sigma;

    for (std::size_t i = 0; i < self.size(); ++i) {
      // Create the coordinate system for the reflection
      CoordinateSystem xcs(m2, s0, s1[i], xyz[i][2]);

      /// Calculate the rotation angles at the following XDS
      // e3 coordinates: -delta_m, +delta_m
      double phi1 = xcs.to_rotation_angle_fast(-delta_m);
      double phi2 = xcs.to_rotation_angle_fast(+delta_m);
      if (phi2 < phi1) {
        std::swap(phi1, phi2);
      }
      result[i] = vec2<double>(phi1, phi2);
    }
    return result;
  }

  /**
   * Select a number of rows from the table via an index array
   * @param self The current table
   * @param index The index array
   * @returns The new table with the requested rows
   */
  template <typename T>
  T reflection_table_select_rows_index(const T &self,
                                       const af::const_ref<std::size_t> &index) {
    T result = flex_table_suite::select_rows_index<T>(self, index);
    return result;
  }

  /**
   * Select a number of rows from the table via an index array
   * @param self The current table
   * @param flags The flag array
   * @returns The new table with the requested rows
   */
  template <typename T>
  T reflection_table_select_rows_flags(const T &self,
                                       const af::const_ref<bool> &flags) {
    T result = flex_table_suite::select_rows_flags<T>(self, flags);
    return result;
  }

  /**
   * Select a number of columns from the table via an key array
   * @param self The current table
   * @param keys The key array
   * @returns The new table with the requested columns
   */
  template <typename T>
  T reflection_table_select_cols_keys(const T &self,
                                      const af::const_ref<std::string> &keys) {
    T result = flex_table_suite::select_cols_keys<T>(self, keys);
    flex_table_suite::reflection_table_extend_identifiers(result, self);
    return result;
  }

  /**
   * Select a number of columns from the table via an key array
   * @param self The current table
   * @param keys The key array
   * @returns The new table with the requested columns
   */
  template <typename T>
  T reflection_table_select_cols_tuple(const T &self, boost::python::tuple keys) {
    T result = flex_table_suite::select_cols_tuple<T>(self, keys);
    flex_table_suite::reflection_table_extend_identifiers(result, self);
    return result;
  }

  /**
   * Extend the reflection table
   */
  void reflection_table_extend(reflection_table &self, const reflection_table &other) {
    flex_table_suite::reflection_table_extend_identifiers(self, other);
    flex_table_suite::extend(self, other);
  }

  /**
   * Update the reflection table
   */
  void reflection_table_update(reflection_table &self, const reflection_table &other) {
    flex_table_suite::reflection_table_extend_identifiers(self, other);
    flex_table_suite::update(self, other);
  }

  /**
   * A visitor to convert an item to an object
   */
  struct item_to_object_visitor : public boost::static_visitor<object> {
    template <typename T>
    object operator()(T &data) {
      return object(data);
    }
  };

  /**
   * Get an item from the reflection
   * @param self The reflection
   * @param name The item name
   */
  boost::python::object Reflection_get(const Reflection &self, std::string name) {
    Reflection::mapped_type item = self[name];
    item_to_object_visitor visitor;
    return item.apply_visitor(visitor);
  }

  /**
   * Set a bool item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_bool(Reflection &self, std::string name, bool item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a int item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_int(Reflection &self, std::string name, int item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a std::size_t item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_size_t(Reflection &self, std::string name, std::size_t item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a double item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_double(Reflection &self, std::string name, double item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a string item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_string(Reflection &self, std::string name, std::string item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a vec2<double> item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_vec2_double(Reflection &self,
                                  std::string name,
                                  vec2<double> item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a vec3<double> item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_vec3_double(Reflection &self,
                                  std::string name,
                                  vec3<double> item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a mat3<double> item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_mat3_double(Reflection &self,
                                  std::string name,
                                  mat3<double> item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a int6 item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_int6(Reflection &self, std::string name, int6 item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a miller_index item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_miller_index(Reflection &self,
                                   std::string name,
                                   cctbx::miller::index<> item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Set a Shoebox<> item in the reflection
   * @param self The reflection
   * @param name The name of the item
   * @param item The item
   */
  void Reflection_set_shoebox(Reflection &self, std::string name, Shoebox<> item) {
    self[name] = Reflection::data_type(item);
  }

  /**
   * Copy the reflection
   * @param self The reflection
   */
  Reflection Reflection_copy(const Reflection &self) {
    return Reflection(self);
  }

  /**
   * Convert reflection table to list of reflections
   * @param self The reflection table
   * @returns The list of reflections
   */
  boost::python::list reflection_table_to_list_of_reflections(reflection_table self) {
    af::shared<Reflection> array = reflection_table_to_array(self);
    boost::python::list result;
    for (std::size_t i = 0; i < array.size(); ++i) {
      result.append(array[i]);
    }
    return result;
  }

  /**
   * Pack the reflection table in msgpack format
   * @param self The reflection table
   * @returns The msgpack string
   */
  boost::python::object reflection_table_as_msgpack(reflection_table self) {
    std::stringstream buffer;
    msgpack::pack(buffer, self);
    // Convert to a python bytes object
    std::string data = buffer.str();
    boost::python::object data_bytes(
      boost::python::handle<>(PyBytes_FromStringAndSize(data.c_str(), data.size())));
    return data_bytes;
  }

  /**
   * Pack the reflection table in msgpack format into a streambuf object
   * @param self The reflection table
   * @param output A streambuf object encapsulating a Python file-like object
   */
  void reflection_table_as_msgpack_to_file(reflection_table self, streambuf &output) {
    streambuf::ostream os(output);
    msgpack::pack(os, self);
  }

  /**
   * Override default reference func to avoid copying
   */
  bool reflection_table_reference_func(msgpack::type::object_type type,
                                       std::size_t length,
                                       void *user_data) {
    return true;
  }

  /**
   * Unpack the reflection table from msgpack format
   * @param the msgpack string
   * @returns The reflection table
   */
  reflection_table reflection_table_from_msgpack(boost::python::object packed) {
    const char *data = PyBytes_AsString(packed.ptr());
    std::size_t size = PyBytes_Size(packed.ptr());
    msgpack::unpacked result;
    std::size_t off = 0;
    msgpack::unpack(result, data, size, off, reflection_table_reference_func);
    reflection_table r = result.get().as<reflection_table>();
    return r;
  }

  /*
   * Class to pickle and unpickle the table
   */
  struct flex_reflection_table_pickle_suite : boost::python::pickle_suite {
    typedef reflection_table flex_table_type;
    typedef reflection_table::const_iterator const_iterator;

    static boost::python::tuple getstate(const flex_table_type &self) {
      DIALS_ASSERT(self.is_consistent());
      unsigned int version = 2;

      // Get the identifiers as a dictionary
      dict identifiers;
      for (reflection_table::experiment_map_type::const_iterator it =
             self.experiment_identifiers()->begin();
           it != self.experiment_identifiers()->end();
           ++it) {
        identifiers[it->first] = it->second;
      }

      // Get the columns as a dictionary
      dict columns;
      column_to_object_visitor visitor;
      for (const_iterator it = self.begin(); it != self.end(); ++it) {
        columns[it->first] = it->second.apply_visitor(visitor);
      }

      // Make the tuple
      return boost::python::make_tuple(
        version, identifiers, self.nrows(), self.ncols(), columns);
    }

    static void setstate(flex_table_type &self, boost::python::tuple state) {
      DIALS_ASSERT(boost::python::len(state) > 0);
      std::size_t version = extract<unsigned int>(state[0]);
      if (version == 1) {
        setstate_version_1(self, state);
      } else if (version == 2) {
        setstate_version_2(self, state);
      } else {
        throw DIALS_ERROR("Unknown pickle version");
      }
    }

    static void setstate_version_1(flex_table_type &self, boost::python::tuple state) {
      DIALS_ASSERT(boost::python::len(state) == 4);
      DIALS_ASSERT(extract<unsigned int>(state[0]) == 1);
      std::size_t nrows = extract<std::size_t>(state[1]);
      std::size_t ncols = extract<std::size_t>(state[2]);
      self.resize(nrows);

      // Extract the columns
      dict columns = extract<dict>(state[3]);
      DIALS_ASSERT(len(columns) == ncols);
      object items = list(columns.items());
      object self_obj(self);
      DIALS_ASSERT(len(items) == ncols);
      for (std::size_t i = 0; i < ncols; ++i) {
        object item = items[i];
        DIALS_ASSERT(len(item[1]) == nrows);
        std::string name = extract<std::string>(item[0]);
        self_obj[name] = item[1];
      }
      DIALS_ASSERT(self.is_consistent());
    }

    static void setstate_version_2(flex_table_type &self, boost::python::tuple state) {
      DIALS_ASSERT(boost::python::len(state) == 5);
      DIALS_ASSERT(extract<unsigned int>(state[0]) == 2);

      // Extract the identifiers
      dict identifiers = extract<dict>(state[1]);
      object identifier_items = list(identifiers.items());
      DIALS_ASSERT(len(identifier_items) == len(identifiers));
      for (std::size_t i = 0; i < len(identifiers); ++i) {
        object item = identifier_items[i];
        std::size_t index = extract<std::size_t>(item[0]);
        std::string ident = extract<std::string>(item[1]);
        (*self.experiment_identifiers())[index] = ident;
      }

      // Extract nrows and cols
      std::size_t nrows = extract<std::size_t>(state[2]);
      std::size_t ncols = extract<std::size_t>(state[3]);
      self.resize(nrows);

      // Extract the columns
      dict columns = extract<dict>(state[4]);
      DIALS_ASSERT(len(columns) == ncols);
      object items = list(columns.items());
      object self_obj(self);
      DIALS_ASSERT(len(items) == ncols);
      for (std::size_t i = 0; i < ncols; ++i) {
        object item = items[i];
        DIALS_ASSERT(len(item[1]) == nrows);
        std::string name = extract<std::string>(item[0]);
        self_obj[name] = item[1];
      }
      DIALS_ASSERT(self.is_consistent());
    }
  };

  /**
   * Struct to facilitate wrapping reflection table type
   */
  template <typename T>
  struct flex_reflection_table_wrapper : public flex_table_wrapper<T> {
    typedef flex_table_wrapper<T> base_type;
    typedef typename base_type::flex_table_type flex_table_type;
    typedef typename base_type::class_type class_type;

    /**
     * Wrap the reflection table class
     */
    static class_type wrap(const char *name) {
      // Wrap with flex table bindings
      class_type result = base_type::wrap(name);

      // Add functions
      result
        .def("__init__",
             make_constructor(&make_from_observation_and_shoebox<flex_table_type>))
        .def("help_keys", &help_keys<flex_table_type>)
        .def("compute_ray_intersections", &compute_ray_intersections<flex_table_type>)
        .def("get_flags",
             &get_flags<flex_table_type>,
             (boost::python::arg("value"), boost::python::arg("all") = true))
        .def("set_flags", &set_flags_by_mask<flex_table_type>)
        .def("set_flags", &set_flags_by_index<flex_table_type>)
        .def("unset_flags", &unset_flags_by_mask<flex_table_type>)
        .def("unset_flags", &unset_flags_by_index<flex_table_type>)
        .def("split_partials", &split_partials<flex_table_type>)
        .def("split_partials_with_shoebox",
             &split_partials_with_shoebox<flex_table_type>)
        .def("split_partial_indices", &split_partial_indices<flex_table_type>)
        .def("split_by_experiment_id", &split_by_experiment_id<flex_table_type>)
        .def("split_indices_by_experiment_id",
             &split_indices_by_experiment_id<flex_table_type>)
        .def("compute_phi_range", &compute_phi_range<flex_table_type>)
        .def("as_msgpack", &reflection_table_as_msgpack)
        .def("as_msgpack_to_file", &reflection_table_as_msgpack_to_file)
        .def("from_msgpack", &reflection_table_from_msgpack)
        .staticmethod("from_msgpack")
        .def("experiment_identifiers", &T::experiment_identifiers)
        .def("select", &reflection_table_select_rows_index<flex_table_type>)
        .def("select", &reflection_table_select_rows_flags<flex_table_type>)
        .def("select", &reflection_table_select_cols_keys<flex_table_type>)
        .def("select", &reflection_table_select_cols_tuple<flex_table_type>)
        .def("extend", reflection_table_extend)
        .def("update", reflection_table_update)
        .def_pickle(flex_reflection_table_pickle_suite());

      // Create the flags enum in the reflection table scope
      scope in_table = result;
      enum_<Flags>("flags")
        .value("predicted", Predicted)
        .value("observed", Observed)
        .value("indexed", Indexed)
        .value("used_in_refinement", UsedInRefinement)
        .value("strong", Strong)
        .value("reference_spot", ReferenceSpot)
        .value("dont_integrate", DontIntegrate)
        .value("integrated_sum", IntegratedSum)
        .value("integrated_prf", IntegratedPrf)
        .value("integrated", Integrated)
        .value("overloaded", Overloaded)
        .value("overlapped_bg", OverlappedBg)
        .value("overlapped_fg", OverlappedFg)
        .value("in_powder_ring", InPowderRing)
        .value("foreground_includes_bad_pixels", ForegroundIncludesBadPixels)
        .value("background_includes_bad_pixels", BackgroundIncludesBadPixels)
        .value("includes_bad_pixels", IncludesBadPixels)
        .value("bad_shoebox", BadShoebox)
        .value("bad_spot", BadSpot)
        .value("used_in_modelling", UsedInModelling)
        .value("centroid_outlier", CentroidOutlier)
        .value("failed_during_background_modelling", FailedDuringBackgroundModelling)
        .value("failed_during_summation", FailedDuringSummation)
        .value("failed_during_profile_fitting", FailedDuringProfileFitting)
        .value("bad_reference", BadReference)
        .value("user_excluded_in_scaling", UserExcludedInScaling)
        .value("outlier_in_scaling", OutlierInScaling)
        .value("excluded_for_scaling", ExcludedForScaling)
        .value("bad_for_scaling", BadForScaling)
        .value("scaled", Scaled)
        .value("excluded_for_refinement", ExcludedForRefinement)
        .value("bad_for_refinement", BadForRefinement);

      // return the wrapped class
      return result;
    }
  };

  /**
   * Functions for experiment identifier map
   */
  namespace experiment_map_type_detail {

    /**
     * Get an item
     */
    std::string getitem(const reflection_table::experiment_map_type &self,
                        std::size_t index) {
      typedef reflection_table::experiment_map_type::const_iterator iterator;
      iterator it = self.find(index);
      DIALS_ASSERT(it != self.end());
      return it->second;
    }

    /**
     * Set an item
     */
    void setitem(reflection_table::experiment_map_type &self,
                 std::size_t index,
                 std::string value) {
      self[index] = value;
    }

    /**
     * Del an item
     */
    void delitem(reflection_table::experiment_map_type &self, std::size_t index) {
      self.erase(index);
    }

    /**
     * Check if the map contains an item
     */
    bool contains(const reflection_table::experiment_map_type &self,
                  std::size_t index) {
      return self.find(index) != self.end();
    }

    /**
     * Get the keys
     */
    af::shared<std::size_t> keys(const reflection_table::experiment_map_type &self) {
      typedef reflection_table::experiment_map_type::const_iterator iterator;
      af::shared<std::size_t> k;
      for (iterator it = self.begin(); it != self.end(); ++it) {
        k.push_back(it->first);
      }
      return k;
    }

    /**
     * Get the values
     */
    af::shared<std::string> values(const reflection_table::experiment_map_type &self) {
      typedef reflection_table::experiment_map_type::const_iterator iterator;
      af::shared<std::string> v;
      for (iterator it = self.begin(); it != self.end(); ++it) {
        v.push_back(it->second);
      }
      return v;
    }

    /**
     * A proxy iterator
     */
    class iterator {
    public:
      typedef reflection_table::experiment_map_type map_type;
      typedef ptrdiff_t difference_type;
      typedef std::forward_iterator_tag iterator_category;
      typedef boost::python::tuple value_type;
      typedef const value_type *pointer;
      typedef const value_type reference;

      iterator(const map_type::const_iterator &it) : it_(it) {}

      reference operator*() {
        boost::python::tuple result;
        return boost::python::make_tuple(it_->first, it_->second);
      }

      iterator &operator++() {
        ++it_;
        return *this;
      }

      iterator operator++(int) {
        iterator result(*this);
        ++(*this);
        return result;
      }

      bool operator==(const iterator &rhs) const {
        return it_ == rhs.it_;
      }

      bool operator!=(const iterator &rhs) const {
        return !(*this == rhs);
      }

    private:
      map_type::const_iterator it_;
    };

    /**
     * Map the iterator range
     */
    struct make_iterator {
      static iterator begin(const reflection_table::experiment_map_type &self) {
        return iterator(self.begin());
      }

      static iterator end(const reflection_table::experiment_map_type &self) {
        return iterator(self.end());
      }

      static object range() {
        return boost::python::range(&make_iterator::begin, &make_iterator::end);
      }
    };
  }  // namespace experiment_map_type_detail

  void export_flex_reflection_table() {
    // Set the do
    docstring_options local_docstring_options;
    local_docstring_options.enable_user_defined();
    local_docstring_options.enable_py_signatures();
    local_docstring_options.disable_cpp_signatures();

    // Export the experiment id map
    class_<reflection_table::experiment_map_type,
           std::shared_ptr<reflection_table::experiment_map_type> >("experiment_id_map")
      .def("__len__", &reflection_table::experiment_map_type::size)
      .def("__getitem__", &experiment_map_type_detail::getitem)
      .def("__setitem__", &experiment_map_type_detail::setitem)
      .def("__delitem__", &experiment_map_type_detail::delitem)
      .def("__contains__", &experiment_map_type_detail::contains)
      .def("keys", &experiment_map_type_detail::keys)
      .def("values", &experiment_map_type_detail::values)
      .def("__iter__", experiment_map_type_detail::make_iterator::range());
    ;
    ;

    // Export the reflection table
    flex_reflection_table_wrapper<reflection_table>::wrap("reflection_table");

    // Export the reflection object
    class_<Reflection>("Reflection")
      .def("get", &Reflection_get)
      .def("set_bool", &Reflection_set_bool)
      .def("set_int", &Reflection_set_int)
      .def("set_size_t", &Reflection_set_size_t)
      .def("set_double", &Reflection_set_double)
      .def("set_string", &Reflection_set_string)
      .def("set_vec2_double", &Reflection_set_vec2_double)
      .def("set_vec3_double", &Reflection_set_vec3_double)
      .def("set_mat3_double", &Reflection_set_mat3_double)
      .def("set_int6", &Reflection_set_int6)
      .def("set_miller_index", &Reflection_set_miller_index)
      .def("set_shoebox", &Reflection_set_shoebox)
      .def("copy", &Reflection_copy);

    // Helper function
    def("reflection_table_to_list_of_reflections",
        &reflection_table_to_list_of_reflections);
  }

}}}  // namespace dials::af::boost_python
