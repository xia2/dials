/*
 * sampler.cc
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
#include <boost/python/iterator.hpp>
#include <dials/algorithms/profile_model/modeller/sampler_interface.h>
#include <dials/algorithms/profile_model/modeller/single_sampler.h>
#include <dials/algorithms/profile_model/modeller/grid_sampler.h>
#include <dials/algorithms/profile_model/modeller/circle_sampler.h>
#include <dials/algorithms/profile_model/modeller/ewald_sphere_sampler.h>

namespace dials { namespace algorithms { namespace boost_python {

  using namespace boost::python;

  struct SingleSamplerPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const SingleSampler &obj) {
      return boost::python::make_tuple(obj.scan_range(), obj.grid_size());
    }
  };

  struct GridSamplerPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const GridSampler &obj) {
      return boost::python::make_tuple(
        obj.image_size(), obj.scan_range(), obj.grid_size());
    }
  };

  struct CircleSamplerPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const CircleSampler &obj) {
      return boost::python::make_tuple(obj.image_size(), obj.scan_range(), obj.num_z());
    }
  };

  struct EwaldSphereSamplerPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const EwaldSphereSampler &obj) {
      return boost::python::make_tuple(
        obj.beam(), obj.detector(), obj.goniometer(), obj.scan(), obj.num_phi());
    }
  };

  struct SamplerIfaceWrapper : SamplerIface, wrapper<SamplerIface> {
    std::size_t size() const {
      return this->get_override("size")();
    }

    std::size_t nearest(std::size_t panel, double3 xyz) const {
      return this->get_override("nearest")(panel, xyz);
    }

    af::shared<std::size_t> nearest_n(std::size_t panel, double3 xyz) const {
      return this->get_override("nearest_n")(panel, xyz);
    }

    double weight(std::size_t index, std::size_t panel, double3 xyz) const {
      return this->get_override("weight")(index, panel, xyz);
    }

    double3 coord(std::size_t index) const {
      return this->get_override("coord")(index);
    }

    af::shared<std::size_t> neighbours(std::size_t index) const {
      return this->get_override("neighbours")(index);
    }
  };

  void export_sampler() {
    class_<SamplerIfaceWrapper, boost::noncopyable>("SamplerIface")
      .def("size", pure_virtual(&SamplerIface::size))
      .def("nearest", pure_virtual(&SamplerIface::nearest))
      .def("nearest_n", pure_virtual(&SamplerIface::nearest_n))
      .def("weight", pure_virtual(&SamplerIface::weight))
      .def("coord", pure_virtual(&SamplerIface::coord))
      .def("neighbours", pure_virtual(&SamplerIface::neighbours))
      .def("__len__", &SamplerIface::size);

    class_<SingleSampler, bases<SamplerIface> >("SingleSampler", no_init)
      .def(init<int2, std::size_t>((arg("scan_range"), arg("grid_size"))))
      .def("scan_range", &SingleSampler::scan_range)
      .def("grid_size", &SingleSampler::grid_size)
      .def("step_size", &SingleSampler::step_size)
      .def_pickle(SingleSamplerPickleSuite());

    class_<GridSampler, bases<SamplerIface> >("GridSampler", no_init)
      .def(init<int2, int2, int3>(
        (arg("image_size"), arg("scan_range"), arg("grid_size"))))
      .def("image_size", &GridSampler::image_size)
      .def("scan_range", &GridSampler::scan_range)
      .def("grid_size", &GridSampler::grid_size)
      .def("step_size", &GridSampler::step_size)
      .def_pickle(GridSamplerPickleSuite());

    class_<CircleSampler, bases<SamplerIface> >("CircleSampler", no_init)
      .def(init<int2, int2, std::size_t>(
        (arg("image_size"), arg("scan_range"), arg("num_z"))))
      .def("image_size", &CircleSampler::image_size)
      .def("scan_range", &CircleSampler::scan_range)
      .def("num_z", &CircleSampler::num_z)
      .def("image_centre", &CircleSampler::image_centre)
      .def("r0", &CircleSampler::r0)
      .def("r1", &CircleSampler::r1)
      .def("r2", &CircleSampler::r2)
      .def_pickle(CircleSamplerPickleSuite());

    class_<EwaldSphereSampler, bases<SamplerIface> >("EwaldSphereSampler", no_init)
      .def(init<const std::shared_ptr<BeamBase>,
                const Detector &,
                const Goniometer &,
                const Scan &,
                std::size_t>())
      .def("beam", &EwaldSphereSampler::beam)
      .def("detector", &EwaldSphereSampler::detector)
      .def("goniometer", &EwaldSphereSampler::goniometer)
      .def("scan", &EwaldSphereSampler::scan)
      .def("max_angle", &EwaldSphereSampler::max_angle)
      .def("num_phi", &EwaldSphereSampler::num_phi)
      .def("step_phi", &EwaldSphereSampler::step_phi)
      .def("profile_coord", &EwaldSphereSampler::profile_coord)
      .def("nearest_n", &EwaldSphereSampler::nearest_n_index)
      .def_pickle(EwaldSphereSamplerPickleSuite());
  }

}}}  // namespace dials::algorithms::boost_python
