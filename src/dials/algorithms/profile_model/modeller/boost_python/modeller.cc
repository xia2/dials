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
#include <iostream>
#include <dials/algorithms/profile_model/modeller/modeller_interface.h>
#include <dials/algorithms/profile_model/modeller/empirical_modeller.h>
#include <dials/algorithms/profile_model/modeller/multi_experiment_modeller.h>

namespace dials { namespace algorithms { namespace boost_python {

  using namespace boost::python;

  struct ProfileModellerIfaceWrapper : ProfileModellerIface,
                                       wrapper<ProfileModellerIface> {
    void model(af::reflection_table reflections) {
      this->get_override("model")(reflections);
    }

    void fit(af::reflection_table reflections) {
      this->get_override("fit")(reflections);
    }

    void validate(af::reflection_table reflections) const {
      this->get_override("validate")(reflections);
    }

    void accumulate(std::shared_ptr<ProfileModellerIface> other) {
      this->get_override("accumulate")(other);
    }

    void finalize() {
      this->get_override("finalize")();
    }

    bool finalized() const {
      return this->get_override("finalized")();
    }

    data_type data(std::size_t index) const {
      return this->get_override("data")(index);
    }

    mask_type mask(std::size_t index) const {
      return this->get_override("mask")(index);
    }

    std::size_t size() const {
      return this->get_override("size")();
    }

    af::shared<bool> fit(af::reflection_table reflections) const {
      return this->get_override("fit")(reflections);
    }

    pointer copy() const {
      return this->get_override("copy")();
    }
  };

  struct EmpiricalProfileModellerWrapper : EmpiricalProfileModeller,
                                           wrapper<EmpiricalProfileModeller> {
    EmpiricalProfileModellerWrapper(std::size_t n, int3 accessor, double threshold)
        : EmpiricalProfileModeller(n, accessor, threshold) {}

    void model(af::reflection_table reflections) {
      this->get_override("model")(reflections);
    }

    af::shared<bool> fit(af::reflection_table reflections) const {
      return this->get_override("fit")(reflections);
    }

    void validate(af::reflection_table reflections) const {
      this->get_override("validate")(reflections);
    }

    pointer copy() const {
      return this->get_override("copy")();
    }
  };

  struct MultiExpProfileModellerPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getstate(const MultiExpProfileModeller &obj) {
      boost::python::list result;
      for (std::size_t i = 0; i < obj.size(); ++i) {
        result.append(obj[i]);
      }
      return boost::python::make_tuple(result);
    }

    static void setstate(MultiExpProfileModeller &obj, boost::python::tuple state) {
      DIALS_ASSERT(boost::python::len(state) == 1);
      boost::python::list result = extract<boost::python::list>(state[0]);
      for (std::size_t i = 0; i < boost::python::len(result); ++i) {
        obj.add(extract<std::shared_ptr<ProfileModellerIface> >(result[i]));
      }
    }
  };

  void export_modeller() {
    class_<ProfileModellerIfaceWrapper,
           std::shared_ptr<ProfileModellerIfaceWrapper>,
           boost::noncopyable>("ProfileModellerIface")
      .def("model", pure_virtual(&ProfileModellerIface::model))
      .def("fit", pure_virtual(&ProfileModellerIface::fit))
      .def("validate", pure_virtual(&ProfileModellerIface::validate))
      .def("accumulate", pure_virtual(&ProfileModellerIface::accumulate))
      .def("finalize", pure_virtual(&ProfileModellerIface::finalize))
      .def("finalized", pure_virtual(&ProfileModellerIface::finalized))
      .def("data", pure_virtual(&ProfileModellerIface::data))
      .def("mask", pure_virtual(&ProfileModellerIface::mask))
      .def("size", pure_virtual(&ProfileModellerIface::size))
      .def("copy", pure_virtual(&ProfileModellerIface::copy))
      .def("__len__", pure_virtual(&ProfileModellerIface::size));

    register_ptr_to_python<std::shared_ptr<ProfileModellerIface> >();

    class_<EmpiricalProfileModellerWrapper,
           boost::noncopyable,
           bases<ProfileModellerIface> >("EmpiricalProfileModeller", no_init)
      .def(init<std::size_t, int3, double>())
      .def("add", &EmpiricalProfileModeller::add)
      .def("valid", &EmpiricalProfileModeller::valid)
      .def("n_reflections", &EmpiricalProfileModeller::n_reflections);

    class_<MultiExpProfileModeller>("MultiExpProfileModeller")
      .def("add", &MultiExpProfileModeller::add)
      .def("__getitem__", &MultiExpProfileModeller::operator[])
      .def("model", &MultiExpProfileModeller::model)
      .def("accumulate", &MultiExpProfileModeller::accumulate)
      .def("finalize", &MultiExpProfileModeller::finalize)
      .def("finalized", &MultiExpProfileModeller::finalized)
      .def("fit", &MultiExpProfileModeller::fit)
      .def("validate", &MultiExpProfileModeller::validate)
      .def("__len__", &MultiExpProfileModeller::size)
      .def("copy", &MultiExpProfileModeller::copy)
      .def_pickle(MultiExpProfileModellerPickleSuite());
    ;
  }

}}}  // namespace dials::algorithms::boost_python
