/*
 * transform_ext.cc
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
#include <dials/algorithms/profile_model/gaussian_rs/transform/beam_vector_map.h>
#include <dials/algorithms/profile_model/gaussian_rs/transform/index_generator.h>
#include <dials/algorithms/profile_model/gaussian_rs/transform/map_frames.h>
#include <dials/algorithms/profile_model/gaussian_rs/transform/transform.h>

namespace dials {
  namespace algorithms {
    namespace profile_model {
      namespace gaussian_rs {
  namespace transform { namespace boost_python {

    using namespace boost::python;

    CoordinateGenerator *make_coordinate_generator(
      const CoordinateSystem &cs,
      int x0,
      int y0,
      af::versa<vec3<double>, scitbx::af::flex_grid<> > s1_map) {
      return new CoordinateGenerator(
        cs,
        x0,
        y0,
        af::versa<vec3<double>, af::c_grid<2> >(s1_map.handle(),
                                                af::c_grid<2>(s1_map.accessor())));
    }

    GridIndexGenerator *make_grid_index_generator(
      const CoordinateSystem &cs,
      int x0,
      int y0,
      vec2<double> step_size,
      std::size_t grid_half_size,
      af::versa<vec3<double>, scitbx::af::flex_grid<> > s1_map) {
      return new GridIndexGenerator(
        cs,
        x0,
        y0,
        step_size,
        grid_half_size,
        af::versa<vec3<double>, af::c_grid<2> >(s1_map.handle(),
                                                af::c_grid<2>(s1_map.accessor())));
    }

    std::shared_ptr<TransformSpec> make_transform_spec_from_experiment(
      object experiment,
      double sigma_b,
      double sigma_m,
      double nsigma,
      std::size_t grid_size) {
      return std::shared_ptr<TransformSpec>(
        new TransformSpec(extract<std::shared_ptr<BeamBase> >(experiment.attr("beam")),
                          extract<Detector>(experiment.attr("detector")),
                          extract<Goniometer>(experiment.attr("goniometer")),
                          extract<Scan>(experiment.attr("scan")),
                          sigma_b,
                          sigma_m,
                          nsigma,
                          grid_size));
    }

    /**
     * A class to pickle the transform spec
     */
    struct TransformSpecPickleSuite : boost::python::pickle_suite {
      static boost::python::tuple getinitargs(const TransformSpec &obj) {
        return boost::python::make_tuple(obj.beam(),
                                         obj.detector(),
                                         obj.goniometer(),
                                         obj.scan(),
                                         obj.sigma_b(),
                                         obj.sigma_m(),
                                         obj.n_sigma(),
                                         obj.half_grid_size());
      }
    };

    /* template <typename FloatType> */
    /* af::versa< vec3<double>, af::c_grid<2> > transform_spec_s1_map( */
    /*     const TransformSpec &self, */
    /*     std::size_t panel) { */
    /*   af::const_ref< vec3<double>, af::c_grid<2> > array = self.s1_map(panel); */
    /*   af::versa< vec3<double>, af::c_grid<2> > result(array.accessor()); */
    /*   std::copy(array.begin(), array.end(), result.begin()); */
    /*   return result; */
    /* } */

    template <typename FloatType>
    void transform_forward_wrapper(const char *name) {
      typedef TransformForward<FloatType> TransformForwardType;

      class_<TransformForwardType>(name, no_init)
        .def(init<const TransformSpec &,
                  const CoordinateSystem &,
                  int6,
                  std::size_t,
                  const af::const_ref<FloatType, af::c_grid<3> > &,
                  const af::const_ref<bool, af::c_grid<3> > &>())
        .def(init<const TransformSpec &,
                  const CoordinateSystem &,
                  int6,
                  std::size_t,
                  const af::const_ref<FloatType, af::c_grid<3> > &,
                  const af::const_ref<FloatType, af::c_grid<3> > &,
                  const af::const_ref<bool, af::c_grid<3> > &>())
        .def("profile", &TransformForwardType::profile)
        .def("background", &TransformForwardType::background)
        .def("mask", &TransformForwardType::mask);
    }

    template <typename FloatType>
    void transform_reverse_wrapper(const char *name) {
      class_<TransformReverse>(name, no_init)
        .def(init<const TransformSpec &,
                  const CoordinateSystem &,
                  int6,
                  std::size_t,
                  const af::const_ref<double, af::c_grid<3> > &>())
        .def("profile", &TransformReverse::profile);
    }

    void transform_forward_no_model_wrapper(const char *name) {
      class_<TransformForwardNoModel>(name, no_init)
        .def(init<const TransformSpec &,
                  const CoordinateSystem &,
                  int6,
                  std::size_t,
                  const af::const_ref<double, af::c_grid<3> > &,
                  const af::const_ref<bool, af::c_grid<3> > &>())
        .def(init<const TransformSpec &,
                  const CoordinateSystem &,
                  int6,
                  std::size_t,
                  const af::const_ref<double, af::c_grid<3> > &,
                  const af::const_ref<double, af::c_grid<3> > &,
                  const af::const_ref<bool, af::c_grid<3> > &>())
        .def("profile", &TransformForwardNoModel::profile)
        .def("background", &TransformForwardNoModel::background);
    }

    void transform_reverse_no_model_wrapper(const char *name) {
      class_<TransformReverseNoModel>(name, no_init)
        .def(init<const TransformSpec &,
                  const CoordinateSystem &,
                  int6,
                  std::size_t,
                  const af::const_ref<double, af::c_grid<3> > &>())
        .def("profile", &TransformReverseNoModel::profile);
    }

    BOOST_PYTHON_MODULE(dials_algorithms_profile_model_gaussian_rs_transform_ext) {
      af::versa<vec3<double>, af::c_grid<2> > (*overload1)(
        const Panel &, const BeamBase &, std::size_t, bool) = &beam_vector_map;
      af::versa<vec3<double>, af::c_grid<2> > (*overload2)(
        const Panel &, const BeamBase &, bool) = &beam_vector_map;
      af::versa<vec3<double>, af::c_grid<2> > (*overload3)(
        const Panel &, const BeamBase &) = &beam_vector_map;

      def("beam_vector_map",
          overload1,
          (arg("detector"), arg("beam"), arg("n_div"), arg("corners")));
      def("beam_vector_map", overload2, (arg("detector"), arg("beam"), arg("corners")));
      def("beam_vector_map", overload3, (arg("detector"), arg("beam")));

      class_<CoordinateGenerator>("CoordinateGenerator", no_init)
        .def("__init__",
             make_constructor(&make_coordinate_generator,
                              default_call_policies(),
                              (arg("cs"), arg("x0"), arg("y0"), arg("s1_map"))))
        .def("__call__", &CoordinateGenerator::operator());

      class_<GridIndexGenerator>("GridIndexGenerator", no_init)
        .def("__init__",
             make_constructor(&make_grid_index_generator,
                              default_call_policies(),
                              (arg("cs"),
                               arg("x0"),
                               arg("y0"),
                               arg("step_size"),
                               arg("grid_half_size"),
                               arg("s1_map"))))
        .def("__call__", &GridIndexGenerator::operator());

      typedef MapFramesForward<> MapFramesForwardType;
      typedef MapFramesReverse<> MapFramesReverseType;

      class_<MapFramesForwardType>("MapFramesForward", no_init)
        .def(init<int, double, double, double, double, int>((arg("starting_frame"),
                                                             arg("starting_angle"),
                                                             arg("oscillation"),
                                                             arg("mosaicity"),
                                                             arg("n_sigma"),
                                                             arg("grid_size_e3"))))
        .def("__call__",
             &MapFramesForwardType::operator(),
             (arg("frames"), arg("phi"), arg("zeta")));

      class_<MapFramesReverseType>("MapFramesReverse", no_init)
        .def(init<int, double, double, double, double, int>((arg("starting_frame"),
                                                             arg("starting_angle"),
                                                             arg("oscillation"),
                                                             arg("mosaicity"),
                                                             arg("n_sigma"),
                                                             arg("grid_size_e3"))))
        .def("__call__",
             &MapFramesReverseType::operator(),
             (arg("frames"), arg("phi"), arg("zeta")));

      class_<TransformSpec>("TransformSpec", no_init)
        .def(init<std::shared_ptr<BeamBase>,
                  const Detector &,
                  const Goniometer &,
                  const Scan &,
                  double,
                  double,
                  double,
                  std::size_t>())
        .def("__init__", make_constructor(&make_transform_spec_from_experiment))
        .def("sigma_b", &TransformSpec::sigma_b)
        .def("sigma_m", &TransformSpec::sigma_m)
        .def("n_sigma", &TransformSpec::n_sigma)
        .def("grid_size", &TransformSpec::grid_size)
        .def("step_size", &TransformSpec::step_size)
        .def("grid_centre", &TransformSpec::grid_centre)
        .def_pickle(TransformSpecPickleSuite());

      transform_forward_wrapper<double>("TransformForward");
      transform_reverse_wrapper<double>("TransformReverse");
      transform_forward_no_model_wrapper("TransformForwardNoModel");
      transform_reverse_no_model_wrapper("TransformReverseNoModel");
    }

}}}}}
}  // namespace dials::algorithms::profile_model::gaussian_rs::transform::boost_python
