/*
 * creator.cc
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
#include <dials/algorithms/background/simple/creator.h>

namespace dials { namespace algorithms { namespace background { namespace boost_python {

  using namespace boost::python;

  template <typename FloatType>
  af::shared<bool> call_1(const SimpleBackgroundCreator &self,
                          const af::const_ref<Shoebox<FloatType> > &sbox) {
    af::shared<double> mse(sbox.size());
    af::shared<double> dispersion(sbox.size());
    return self(sbox, mse.ref(), dispersion.ref());
  }

  template <typename FloatType>
  af::shared<bool> call_2(const SimpleBackgroundCreator &self,
                          const af::const_ref<Shoebox<FloatType> > &sbox,
                          af::ref<double> mse,
                          af::ref<double> dispersion) {
    return self(sbox, mse, dispersion);
  }

  template <typename FloatType>
  af::tiny<FloatType, 2> call_3(const SimpleBackgroundCreator &self,
                                Shoebox<FloatType> shoebox) {
    return self(shoebox);
  }

  template <typename FloatType>
  af::tiny<FloatType, 2> call_4(const SimpleBackgroundCreator &self,
                                const af::const_ref<FloatType, af::c_grid<3> > &data,
                                af::ref<int, af::c_grid<3> > mask,
                                af::ref<FloatType, af::c_grid<3> > background) {
    return self(data, mask, background);
  }

  template <typename FloatType>
  af::shared<bool> call_5(const SimpleBackgroundCreator &self,
                          af::reflection_table reflections,
                          MultiPanelImageVolume<FloatType> image_volume) {
    return self(reflections, image_volume);
  }

  void creator_wrapper() {
    class_<SimpleBackgroundCreator>("Creator", no_init)
      .def(init<std::shared_ptr<Modeller>, std::size_t>(
        (arg("modeller"), arg("min_pixels") = 10)))
      .def(
        init<std::shared_ptr<Modeller>, std::shared_ptr<OutlierRejector>, std::size_t>(
          (arg("modeller"), arg("rejector"), arg("min_pixels") = 10)))
      .def("__call__", &call_1<float>)
      .def("__call__", &call_2<float>)
      .def("__call__", &call_3<float>)
      .def("__call__", &call_4<float>)
      .def("__call__", &call_5<float>);
  }

  void export_creator() {
    creator_wrapper();
  }

}}}}  // namespace dials::algorithms::background::boost_python
