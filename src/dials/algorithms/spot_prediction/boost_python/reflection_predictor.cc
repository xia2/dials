/*
 * reflection_predictor.cc
 *
 *  Copyright (C) 2014 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <memory>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/spot_prediction/reflection_predictor.h>

namespace dials { namespace algorithms { namespace boost_python {

  using namespace boost::python;

  void export_scan_static_reflection_predictor() {
    typedef ScanStaticReflectionPredictor Predictor;

    class_<Predictor>("ScanStaticReflectionPredictor", no_init)
      .def(init<const std::shared_ptr<BeamBase>,
                const Detector&,
                const Goniometer&,
                const Scan&,
                const cctbx::uctbx::unit_cell&,
                const cctbx::sgtbx::space_group_type&,
                double,
                double,
                double>())
      .def("for_ub_old_index_generator", &Predictor::for_ub_old_index_generator)
      .def("for_ub", &Predictor::for_ub)
      .def("for_hkl", &Predictor::for_hkl)
      .def("for_hkl", &Predictor::for_hkl_with_individual_ub)
      .def("for_reflection_table", &Predictor::for_reflection_table)
      .def("for_reflection_table", &Predictor::for_reflection_table_with_individual_ub);
  }

  void export_scan_varying_reflection_predictor() {
    typedef ScanVaryingReflectionPredictor Predictor;

    class_<Predictor>("ScanVaryingReflectionPredictor", no_init)
      .def(init<const std::shared_ptr<BeamBase>,
                const Detector&,
                const Goniometer&,
                const Scan&,
                const cctbx::sgtbx::space_group_type&,
                double,
                std::size_t,
                double>())
      .def("for_ub", &Predictor::for_ub)
      .def("for_ub_on_single_image", &Predictor::for_ub_on_single_image)
      .def("for_varying_models", &Predictor::for_varying_models)
      .def("for_varying_models_on_single_image",
           &Predictor::for_varying_models_on_single_image)
      .def("for_reflection_table", &Predictor::for_reflection_table);
  }

  void export_stills_delta_psi_reflection_predictor() {
    typedef StillsDeltaPsiReflectionPredictor Predictor;

    af::reflection_table (Predictor::*predict_all)() const = &Predictor::operator();

    af::reflection_table (Predictor::*predict_observed)(
      const af::const_ref<cctbx::miller::index<> >&) = &Predictor::operator();

    af::reflection_table (Predictor::*predict_observed_with_panel)(
      const af::const_ref<cctbx::miller::index<> >&, std::size_t) =
      &Predictor::operator();

    af::reflection_table (Predictor::*predict_observed_with_panel_list)(
      const af::const_ref<cctbx::miller::index<> >&,
      const af::const_ref<std::size_t>&) = &Predictor::operator();

    class_<Predictor>("StillsDeltaPsiReflectionPredictor", no_init)
      .def(init<const std::shared_ptr<BeamBase>,
                const Detector&,
                mat3<double>,
                const cctbx::uctbx::unit_cell&,
                const cctbx::sgtbx::space_group_type&,
                const double&>())
      .def("__call__", predict_all)
      .def("for_ub", &Predictor::for_ub)
      .def("__call__", predict_observed)
      .def("__call__", predict_observed_with_panel)
      .def("__call__", predict_observed_with_panel_list)
      .def("for_reflection_table", &Predictor::for_reflection_table)
      .def("for_reflection_table", &Predictor::for_reflection_table_with_individual_ub);
  }

  void export_nave_stills_reflection_predictor() {
    typedef NaveStillsReflectionPredictor Predictor;

    af::reflection_table (Predictor::*predict_all)() const = &Predictor::operator();

    af::reflection_table (Predictor::*predict_observed)(
      const af::const_ref<cctbx::miller::index<> >&) = &Predictor::operator();

    af::reflection_table (Predictor::*predict_observed_with_panel)(
      const af::const_ref<cctbx::miller::index<> >&, std::size_t) =
      &Predictor::operator();

    af::reflection_table (Predictor::*predict_observed_with_panel_list)(
      const af::const_ref<cctbx::miller::index<> >&,
      const af::const_ref<std::size_t>&) = &Predictor::operator();

    class_<Predictor>("NaveStillsReflectionPredictor", no_init)
      .def(init<const std::shared_ptr<BeamBase>,
                const Detector&,
                mat3<double>,
                const cctbx::uctbx::unit_cell&,
                const cctbx::sgtbx::space_group_type&,
                const double&,
                const double&,
                const double&>())
      .def("__call__", predict_all)
      .def("for_ub", &Predictor::for_ub)
      .def("__call__", predict_observed)
      .def("__call__", predict_observed_with_panel)
      .def("__call__", predict_observed_with_panel_list)
      .def("for_reflection_table", &Predictor::for_reflection_table)
      .def("for_reflection_table", &Predictor::for_reflection_table_with_individual_ub);
  }

  void export_spherical_relp_stills_reflection_predictor() {
    typedef SphericalRelpStillsReflectionPredictor Predictor;

    af::reflection_table (Predictor::*predict_all)() const = &Predictor::operator();

    af::reflection_table (Predictor::*predict_observed)(
      const af::const_ref<cctbx::miller::index<> >&) = &Predictor::operator();

    af::reflection_table (Predictor::*predict_observed_with_panel)(
      const af::const_ref<cctbx::miller::index<> >&, std::size_t) =
      &Predictor::operator();

    af::reflection_table (Predictor::*predict_observed_with_panel_list)(
      const af::const_ref<cctbx::miller::index<> >&,
      const af::const_ref<std::size_t>&) = &Predictor::operator();

    class_<Predictor>("SphericalRelpStillsReflectionPredictor", no_init)
      .def(init<const std::shared_ptr<BeamBase>,
                const Detector&,
                mat3<double>,
                const cctbx::uctbx::unit_cell&,
                const cctbx::sgtbx::space_group_type&,
                const double&>())
      .def("__call__", predict_all)
      .def("for_ub", &Predictor::for_ub)
      .def("__call__", predict_observed)
      .def("__call__", predict_observed_with_panel)
      .def("__call__", predict_observed_with_panel_list)
      .def("for_reflection_table", &Predictor::for_reflection_table)
      .def("for_reflection_table", &Predictor::for_reflection_table_with_individual_ub);
  }

  void export_reflection_predictor() {
    export_scan_static_reflection_predictor();
    export_scan_varying_reflection_predictor();
    export_stills_delta_psi_reflection_predictor();
    export_nave_stills_reflection_predictor();
    export_spherical_relp_stills_reflection_predictor();
  }

}}}  // namespace dials::algorithms::boost_python
