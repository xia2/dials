/*
 * helpers.cc
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
#include <dials/algorithms/background/helpers.h>
#include <dials/algorithms/background/radial_average.h>

namespace dials { namespace algorithms { namespace background { namespace boost_python {

  using namespace boost::python;

  void export_helpers() {
    def("set_shoebox_background_value",
        &set_shoebox_background_value<float>,
        (arg("reflections"), arg("value")));

    class_<RadialAverage>("RadialAverage", no_init)
      .def(
        init<std::shared_ptr<BeamBase>, const Detector&, double, double, std::size_t>())
      .def("add", &RadialAverage::add)
      .def("mean", &RadialAverage::mean)
      .def("weight", &RadialAverage::weight)
      .def("inv_d2", &RadialAverage::inv_d2);
  }

}}}}  // namespace dials::algorithms::background::boost_python
