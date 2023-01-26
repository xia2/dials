/*
 * nexus_ext.cc
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost_adaptbx/optional_conversions.h>
#include <scitbx/array_family/boost_python/flex_wrapper.h>
#include <H5Cpp.h>
#include <dials/nexus/nxmx.h>
#include <dials/error.h>

namespace dials { namespace nexus { namespace boost_python {

  using namespace boost::python;

  namespace detail {

    template <typename Type, typename Class>
    struct getter {
      typedef Type value_type;
      typedef Class class_type;
      typedef value_type class_type::*pointer_type;

      const pointer_type variable;

      getter(pointer_type p) : variable(p) {}

      value_type operator()(const class_type &obj) const {
        return obj.*variable;
      }
    };

    template <typename Type, typename Class>
    struct setter {
      typedef Type value_type;
      typedef Class class_type;
      typedef value_type class_type::*pointer_type;

      pointer_type variable;

      setter(pointer_type p) : variable(p) {}

      void operator()(class_type &obj, value_type value) const {
        obj.*variable = value;
      }
    };

    template <typename Type, typename Class>
    object make_getter(Type Class::*ptr) {
      typedef boost::mpl::vector<Type, const Class &> signature;
      return make_function(
        getter<Type, Class>(ptr), default_call_policies(), signature());
    }

    template <typename Type, typename Class>
    object make_setter(Type Class::*ptr) {
      typedef boost::mpl::vector<void, Class &, Type> signature;
      return make_function(
        setter<Type, Class>(ptr), default_call_policies(), signature());
    }

  }  // namespace detail

  af::shared<NXmx> load(const char *filename) {
    // Turn off printing errors
    H5::Exception::dontPrint();

    // The result
    af::shared<NXmx> result;

    // Open the hdf file
    H5::H5File handle(filename, H5F_ACC_RDONLY);

    // Initialize the entries list
    af::shared<H5::Group> entries;

    // Find and load the nxmx entries
    find_nx_entries(handle, std::back_inserter(entries));

    // Check the entries for recognized classes
    for (std::size_t i = 0; i < entries.size(); ++i) {
      if (is_nxmx_entry(entries[i])) {
        try {
          result.push_back(serialize<NXmx>::load(entries[i]));
        } catch (H5::Exception e) {
          throw std::runtime_error(e.getDetailMsg());
        }
      }
    }

    // Return the result
    return result;
  };

  void dump(const NXmx &obj, const char *filename) {}

  BOOST_PYTHON_MODULE(dials_nexus_ext) {
    using boost_adaptbx::optional_conversions::to_and_from_python;
    using scitbx::af::boost_python::flex_wrapper;

    class_<NXattenuator>("NXattenuator")
      .add_property("attenuator_transmission",
                    detail::make_getter(&NXattenuator::attenuator_transmission),
                    detail::make_setter(&NXattenuator::attenuator_transmission));

    class_<NXdetector_module>("NXdetector_module")
      .add_property("data_origin",
                    detail::make_getter(&NXdetector_module::data_origin),
                    detail::make_setter(&NXdetector_module::data_origin))
      .add_property("data_size",
                    detail::make_getter(&NXdetector_module::data_size),
                    detail::make_setter(&NXdetector_module::data_size))
      .add_property("module_offset",
                    detail::make_getter(&NXdetector_module::module_offset),
                    detail::make_setter(&NXdetector_module::module_offset))
      .add_property("fast_pixel_direction",
                    detail::make_getter(&NXdetector_module::fast_pixel_direction),
                    detail::make_setter(&NXdetector_module::fast_pixel_direction))
      .add_property("slow_pixel_direction",
                    detail::make_getter(&NXdetector_module::slow_pixel_direction),
                    detail::make_setter(&NXdetector_module::slow_pixel_direction));

    class_<NXdetector>("NXdetector")
      .add_property("description",
                    detail::make_getter(&NXdetector::description),
                    detail::make_setter(&NXdetector::description))
      .add_property("time_per_channel",
                    detail::make_getter(&NXdetector::time_per_channel),
                    detail::make_setter(&NXdetector::time_per_channel))
      .add_property("distance",
                    detail::make_getter(&NXdetector::distance),
                    detail::make_setter(&NXdetector::distance))
      .add_property("beam_centre_x",
                    detail::make_getter(&NXdetector::beam_centre_x),
                    detail::make_setter(&NXdetector::beam_centre_x))
      .add_property("beam_centre_y",
                    detail::make_getter(&NXdetector::beam_centre_y),
                    detail::make_setter(&NXdetector::beam_centre_y))
      .add_property("dead_time",
                    detail::make_getter(&NXdetector::dead_time),
                    detail::make_setter(&NXdetector::dead_time))
      .add_property("count_time",
                    detail::make_getter(&NXdetector::count_time),
                    detail::make_setter(&NXdetector::count_time))
      .add_property("frame_time",
                    detail::make_getter(&NXdetector::frame_time),
                    detail::make_setter(&NXdetector::frame_time))
      .add_property("detector_readout_time",
                    detail::make_getter(&NXdetector::detector_readout_time),
                    detail::make_setter(&NXdetector::detector_readout_time))
      .add_property("bit_depth_readout",
                    detail::make_getter(&NXdetector::bit_depth_readout),
                    detail::make_setter(&NXdetector::bit_depth_readout))
      .add_property("saturation_value",
                    detail::make_getter(&NXdetector::saturation_value),
                    detail::make_setter(&NXdetector::saturation_value))
      .add_property("sensor_material",
                    detail::make_getter(&NXdetector::sensor_material),
                    detail::make_setter(&NXdetector::sensor_material))
      .add_property("sensor_thickness",
                    detail::make_getter(&NXdetector::sensor_thickness),
                    detail::make_setter(&NXdetector::sensor_thickness))
      .add_property("threshold_energy",
                    detail::make_getter(&NXdetector::threshold_energy),
                    detail::make_setter(&NXdetector::threshold_energy))
      .add_property("type",
                    detail::make_getter(&NXdetector::type),
                    detail::make_setter(&NXdetector::type))
      .add_property("gain_setting",
                    detail::make_getter(&NXdetector::gain_setting),
                    detail::make_setter(&NXdetector::gain_setting))
      .add_property("angular_calibration_applied",
                    detail::make_getter(&NXdetector::angular_calibration_applied),
                    detail::make_setter(&NXdetector::angular_calibration_applied))
      .add_property("flatfield_applied",
                    detail::make_getter(&NXdetector::flatfield_applied),
                    detail::make_setter(&NXdetector::flatfield_applied))
      .add_property("pixel_mask_applied",
                    detail::make_getter(&NXdetector::pixel_mask_applied),
                    detail::make_setter(&NXdetector::pixel_mask_applied))
      .add_property(
        "countrate_correction_applied_applied",
        detail::make_getter(&NXdetector::countrate_correction_applied_applied),
        detail::make_setter(&NXdetector::countrate_correction_applied_applied))
      .add_property("angular_calibration",
                    detail::make_getter(&NXdetector::angular_calibration),
                    detail::make_setter(&NXdetector::angular_calibration))
      .add_property("flatfield",
                    detail::make_getter(&NXdetector::flatfield),
                    detail::make_setter(&NXdetector::flatfield))
      .add_property("flatfield_error",
                    detail::make_getter(&NXdetector::flatfield_error),
                    detail::make_setter(&NXdetector::flatfield_error))
      .add_property("pixel_mask",
                    detail::make_getter(&NXdetector::pixel_mask),
                    detail::make_setter(&NXdetector::pixel_mask))
      .add_property("module",
                    detail::make_getter(&NXdetector::module),
                    detail::make_setter(&NXdetector::module));

    class_<NXinstrument>("NXinstrument")
      .add_property("attenuator",
                    detail::make_getter(&NXinstrument::attenuator),
                    detail::make_setter(&NXinstrument::attenuator))
      .add_property("detector",
                    detail::make_getter(&NXinstrument::detector),
                    detail::make_setter(&NXinstrument::detector));

    class_<NXbeam>("NXbeam")
      .add_property("incident_wavelength",
                    detail::make_getter(&NXbeam::incident_wavelength),
                    detail::make_setter(&NXbeam::incident_wavelength))
      .add_property(
        "flux", detail::make_getter(&NXbeam::flux), detail::make_setter(&NXbeam::flux))
      .add_property("incident_polarization_stokes",
                    detail::make_getter(&NXbeam::incident_polarization_stokes),
                    detail::make_setter(&NXbeam::incident_polarization_stokes));

    class_<NXsample>("NXsample")
      .add_property("name",
                    detail::make_getter(&NXsample::name),
                    detail::make_setter(&NXsample::name))
      .add_property("chemical_formula",
                    detail::make_getter(&NXsample::chemical_formula),
                    detail::make_setter(&NXsample::chemical_formula))
      .add_property("temperature",
                    detail::make_getter(&NXsample::temperature),
                    detail::make_setter(&NXsample::temperature))
      .add_property("unit_cell_class",
                    detail::make_getter(&NXsample::unit_cell_class),
                    detail::make_setter(&NXsample::unit_cell_class))
      .add_property("unit_cell_group",
                    detail::make_getter(&NXsample::unit_cell_group),
                    detail::make_setter(&NXsample::unit_cell_group))
      .add_property("sample_orientation",
                    detail::make_getter(&NXsample::sample_orientation),
                    detail::make_setter(&NXsample::sample_orientation))
      .add_property("orientation_matrix",
                    detail::make_getter(&NXsample::orientation_matrix),
                    detail::make_setter(&NXsample::orientation_matrix))
      .add_property("unit_cell",
                    detail::make_getter(&NXsample::unit_cell),
                    detail::make_setter(&NXsample::unit_cell))
      .add_property("beam",
                    detail::make_getter(&NXsample::beam),
                    detail::make_setter(&NXsample::beam));

    class_<NXdata>("NXdata");

    class_<NXmx>("NXmx")
      .add_property(
        "title", detail::make_getter(&NXmx::title), detail::make_setter(&NXmx::title))
      .add_property("start_time",
                    detail::make_getter(&NXmx::start_time),
                    detail::make_setter(&NXmx::start_time))
      .add_property("end_time",
                    detail::make_getter(&NXmx::end_time),
                    detail::make_setter(&NXmx::end_time))
      .add_property("instrument",
                    detail::make_getter(&NXmx::instrument),
                    detail::make_setter(&NXmx::instrument))
      .add_property("sample",
                    detail::make_getter(&NXmx::sample),
                    detail::make_setter(&NXmx::sample));

    flex_wrapper<NXmx>::plain("flex_nxmx");
    flex_wrapper<NXinstrument>::plain("flex_nxinstrument");
    flex_wrapper<NXsample>::plain("flex_nxsample");
    flex_wrapper<NXattenuator>::plain("flex_nxdetector");
    flex_wrapper<NXdetector>::plain("flex_nxdetector");
    flex_wrapper<NXdetector_module>::plain("flex_nxdetector_module");

    // Expose the optional values
    to_and_from_python<boost::optional<mat3<double> > >();
    to_and_from_python<boost::optional<af::tiny<double, 4> > >();
    to_and_from_python<boost::optional<NXattenuator> >();
    to_and_from_python<boost::optional<NXdetector> >();
    to_and_from_python<boost::optional<NXbeam> >();
    to_and_from_python<boost::optional<NXdata> >();

    // Methods to load and dump NXmx
    def("load", load);
    def("dump", dump);
  }

}}}  // namespace dials::nexus::boost_python
