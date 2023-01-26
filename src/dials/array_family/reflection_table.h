/*
 * reflection_table.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ARRAY_FAMILY_REFLECTION_TABLE_H
#define DIALS_ARRAY_FAMILY_REFLECTION_TABLE_H

#include <memory>
#include <dials/array_family/flex_table.h>
#include <dials/model/data/shoebox.h>
#include <scitbx/array_family/tiny_types.h>
#include <scitbx/mat3.h>
#include <scitbx/vec3.h>
#include <scitbx/vec2.h>
#include <cctbx/miller.h>

namespace dials { namespace af {

  using model::Shoebox;
  using scitbx::mat3;
  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::af::int6;

  template <>
  inline vec2<double> init_zero<vec2<double> >() {
    return vec2<double>(0.0, 0.0);
  }

  template <>
  inline vec3<double> init_zero<vec3<double> >() {
    return vec3<double>(0.0, 0.0, 0.0);
  }

  template <>
  inline mat3<double> init_zero<mat3<double> >() {
    return mat3<double>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }

  template <>
  inline int6 init_zero<int6>() {
    return int6(0, 0, 0, 0, 0, 0);
  }

  template <>
  inline cctbx::miller::index<> init_zero<cctbx::miller::index<> >() {
    return cctbx::miller::index<>(0, 0, 0);
  }

  typedef flex_type_generator<bool,
                              int,
                              std::size_t,
                              double,
                              std::string,
                              vec2<double>,
                              vec3<double>,
                              mat3<double>,
                              int6,
                              cctbx::miller::index<>,
                              Shoebox<> >
    reflection_table_type_generator;

  typedef reflection_table_type_generator::type reflection_table_types;

  class reflection_table : public flex_table<reflection_table_types> {
  public:
    typedef flex_table<reflection_table_types>::map_type map_type;
    typedef flex_table<reflection_table_types>::key_type key_type;
    typedef flex_table<reflection_table_types>::mapped_type mapped_type;
    typedef flex_table<reflection_table_types>::map_value_type map_value_type;
    typedef flex_table<reflection_table_types>::iterator iterator;
    typedef flex_table<reflection_table_types>::const_iterator const_iterator;
    typedef flex_table<reflection_table_types>::size_type size_type;
    typedef std::map<std::size_t, std::string> experiment_map_type;

    reflection_table()
        : flex_table<reflection_table_types>(),
          experiment_identifiers_(std::make_shared<experiment_map_type>()) {}

    reflection_table(size_type n)
        : flex_table<reflection_table_types>(n),
          experiment_identifiers_(std::make_shared<experiment_map_type>()) {}

    std::shared_ptr<experiment_map_type> experiment_identifiers() const {
      return experiment_identifiers_;
    }

  protected:
    std::shared_ptr<experiment_map_type> experiment_identifiers_;
  };

  enum Flags {

    // Predicted/Observed
    Predicted = (1 << 0),
    Observed = (1 << 1),

    // Use in indexing/refinement
    Indexed = (1 << 2),
    UsedInRefinement = (1 << 3),
    Strong = (1 << 5),

    // Role in integration
    ReferenceSpot = (1 << 6),
    DontIntegrate = (1 << 7),

    // Integated
    IntegratedSum = (1 << 8),
    IntegratedPrf = (1 << 9),
    Integrated = IntegratedSum | IntegratedPrf,

    // Bad shoebox
    Overloaded = (1 << 10),
    OverlappedBg = (1 << 11),
    OverlappedFg = (1 << 12),
    InPowderRing = (1 << 13),
    ForegroundIncludesBadPixels = (1 << 14),
    BackgroundIncludesBadPixels = (1 << 15),
    IncludesBadPixels = ForegroundIncludesBadPixels | BackgroundIncludesBadPixels,
    BadShoebox =
      Overloaded | OverlappedBg | OverlappedFg | InPowderRing | IncludesBadPixels,

    // Bad spot
    BadSpot = BadShoebox,

    // Profile Modelling
    UsedInModelling = (1 << 16),

    // Centroid outlier
    CentroidOutlier = (1 << 17),

    // Some Error Codes
    FailedDuringBackgroundModelling = (1 << 18),
    FailedDuringSummation = (1 << 19),
    FailedDuringProfileFitting = (1 << 20),

    // Bad reference
    BadReference = (1 << 21),

    // Used in scaling
    UserExcludedInScaling = (1 << 22),
    OutlierInScaling = (1 << 23),
    ExcludedForScaling = (1 << 24),
    BadForScaling = UserExcludedInScaling | OutlierInScaling | ExcludedForScaling,

    // Excluded by refinement
    ExcludedForRefinement = (1 << 25),
    BadForRefinement = ExcludedForRefinement | CentroidOutlier,

    // Scaled flag indicates good reflections after scaling
    Scaled = (1 << 26),

  };

}}  // namespace dials::af

#endif  // DIALS_ARRAY_FAMILY_REFLECTION_TABLE_H
