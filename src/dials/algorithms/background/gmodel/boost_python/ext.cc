/*
 * ext.cc
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
#include <dials/algorithms/background/gmodel/creator.h>
#include <dials/algorithms/background/gmodel/model.h>
#include <dials/algorithms/background/gmodel/polar_transform.h>

namespace dials { namespace algorithms { namespace background { namespace boost_python {

  using namespace boost::python;

  struct StaticBackgroundModelPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getstate(const StaticBackgroundModel &obj) {
      boost::python::list data;
      for (std::size_t i = 0; i < obj.size(); ++i) {
        data.append(obj.data(i));
      }
      return boost::python::make_tuple(data);
    }

    static void setstate(StaticBackgroundModel &obj, boost::python::tuple state) {
      DIALS_ASSERT(boost::python::len(state) == 1);
      boost::python::list data = boost::python::extract<boost::python::list>(state[0]);
      for (std::size_t i = 0; i < boost::python::len(data); ++i) {
        af::const_ref<double, af::c_grid<2> > arr =
          boost::python::extract<af::const_ref<double, af::c_grid<2> > >(data[i]);
        obj.add(arr);
      }
    }
  };

  BOOST_PYTHON_MODULE(dials_algorithms_background_gmodel_ext) {
    class_<PolarTransformResult>("PolarTransformResult", no_init)
      .def("data", &PolarTransformResult::data)
      .def("mask", &PolarTransformResult::mask);

    class_<PolarTransform>("PolarTransform", no_init)
      .def(init<const BeamBase &, const Panel &, const Goniometer &>())
      .def("image_xmap", &PolarTransform::image_xmap)
      .def("image_ymap", &PolarTransform::image_ymap)
      .def("discontinuity", &PolarTransform::discontinuity)
      .def("to_polar", &PolarTransform::to_polar)
      .def("from_polar", &PolarTransform::from_polar);

    class_<BackgroundModel, boost::noncopyable, std::shared_ptr<BackgroundModel> >(
      "BackgroundModel", no_init)
      .def("extract", pure_virtual(&BackgroundModel::extract));

    class_<StaticBackgroundModel, bases<BackgroundModel> >("StaticBackgroundModel")
      .def("add", &StaticBackgroundModel::add)
      .def("__len__", &StaticBackgroundModel::size)
      .def("data", &StaticBackgroundModel::data)
      .def_pickle(StaticBackgroundModelPickleSuite());

    class_<GModelBackgroundCreator> creator("Creator", no_init);
    creator
      .def(init<std::shared_ptr<BackgroundModel>, bool, std::size_t>(
        (arg("model"), arg("robust"), arg("min_pixels") = 10)))
      .def("__call__", &GModelBackgroundCreator::shoebox)
      .def("__call__", &GModelBackgroundCreator::volume);
  }

}}}}  // namespace dials::algorithms::background::boost_python
