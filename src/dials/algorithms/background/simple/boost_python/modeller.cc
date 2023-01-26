/*
 * modeller.cc
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
#include <dials/algorithms/background/simple/modeller.h>

namespace dials { namespace algorithms { namespace background { namespace boost_python {

  using namespace boost::python;

  void export_modeller() {
    // An abstract class for background model
    class_<Model, std::shared_ptr<Model>, boost::noncopyable>("Model", no_init)
      .def("value", &Model::value, (arg("z"), arg("y"), arg("x")))
      .def("params", &Model::params)
      .def("variances", &Model::variances);

    // An abstract class for background modeller
    class_<Modeller, boost::noncopyable>("Modeller", no_init)
      .def("create", &Modeller::create, (arg("data"), arg("mask")));

    // A constant 2d model
    class_<Constant2dModel, bases<Model> >("Constant2dModel", no_init)
      .def(init<af::shared<double>, af::shared<double> >((arg("a"), arg("va"))));

    // A constant 3d model
    class_<Constant3dModel, bases<Model> >("Constant3dModel", no_init)
      .def(init<double, double>((arg("a"), arg("va"))));

    // A linear 2d model
    class_<Linear2dModel, bases<Model> >("Linear2dModel", no_init)
      .def(init<af::shared<double>,
                af::shared<double>,
                af::shared<double>,
                af::shared<double>,
                af::shared<double>,
                af::shared<double> >(
        (arg("a"), arg("b"), arg("c"), arg("va"), arg("vb"), arg("vc"))));

    // A linear 3d model
    class_<Linear3dModel, bases<Model> >("Linear3dModel", no_init)
      .def(init<double, double, double, double, double, double, double, double>(
        (arg("a"),
         arg("b"),
         arg("c"),
         arg("d"),
         arg("va"),
         arg("vb"),
         arg("vc"),
         arg("vd"))));

    // The modeller classes
    class_<Constant2dModeller, bases<Modeller> >("Constant2dModeller");
    class_<Constant3dModeller, bases<Modeller> >("Constant3dModeller");
    class_<Linear2dModeller, bases<Modeller> >("Linear2dModeller");
    class_<Linear3dModeller, bases<Modeller> >("Linear3dModeller");
  }

}}}}  // namespace dials::algorithms::background::boost_python
