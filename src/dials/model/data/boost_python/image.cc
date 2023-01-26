/*
 * observation.cc
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
#include <scitbx/array_family/flex_types.h>
#include <dials/model/data/image.h>
#include <dials/error.h>

namespace dials { namespace model { namespace boost_python {

  using namespace boost::python;

  template <typename T>
  std::shared_ptr<Image<T> > make_from_single(typename af::flex<T>::type data,
                                              typename af::flex<bool>::type mask) {
    DIALS_ASSERT(data.accessor().all().size() == 2);
    DIALS_ASSERT(mask.accessor().all().size() == 2);
    return std::make_shared<Image<T> >(
      af::versa<T, af::c_grid<2> >(data.handle(), af::c_grid<2>(data.accessor())),
      af::versa<bool, af::c_grid<2> >(mask.handle(), af::c_grid<2>(data.accessor())));
  }

  template <typename T>
  std::shared_ptr<Image<T> > make_from_tuple(boost::python::tuple data,
                                             boost::python::tuple mask) {
    DIALS_ASSERT(len(data) == len(mask));
    af::shared<typename Image<T>::data_type> d(boost::python::len(data));
    af::shared<typename Image<T>::bool_type> m(boost::python::len(mask));
    for (std::size_t i = 0; i < d.size(); ++i) {
      typename af::flex<T>::type dd =
        boost::python::extract<typename af::flex<T>::type>(data[i]);
      DIALS_ASSERT(dd.accessor().all().size() == 2);
      d[i] = af::versa<T, af::c_grid<2> >(dd.handle(), af::c_grid<2>(dd.accessor()));
      typename af::flex_bool mm =
        boost::python::extract<typename af::flex<bool>::type>(mask[i]);
      DIALS_ASSERT(mm.accessor().all().size() == 2);
      m[i] = af::versa<bool, af::c_grid<2> >(mm.handle(), af::c_grid<2>(mm.accessor()));
    }
    return std::make_shared<Image<T> >(d.const_ref(), m.const_ref());
  }

  object make_from_tuple2(boost::python::tuple data, boost::python::tuple mask) {
    DIALS_ASSERT(len(data) > 0);
    extract<af::flex<int>::type> get_int(data[0]);
    extract<af::flex<double>::type> get_double(data[0]);
    object result;
    if (get_int.check()) {
      result = object(make_from_tuple<int>(data, mask));
    } else if (get_double.check()) {
      result = object(make_from_tuple<double>(data, mask));
    } else {
      throw DIALS_ERROR("Unknown Image Data Type");
    }
    return result;
  }

  template <typename T>
  void wrap_image(const char *name) {
    class_<Image<T>, std::shared_ptr<Image<T> > >(name, no_init)
      .def("__init__", make_constructor(make_from_single<T>))
      .def("__init__", make_constructor(make_from_tuple<T>))
      .def("__len__", &Image<T>::npanels);
  }

  void export_image() {
    wrap_image<int>("ImageInt");
    wrap_image<double>("ImageDouble");

    def("make_image", &make_from_single<int>);
    def("make_image", &make_from_single<double>);
    def("make_image", &make_from_tuple2);
  }

}}}  // namespace dials::model::boost_python
