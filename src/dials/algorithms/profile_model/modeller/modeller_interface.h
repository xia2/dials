/*
 * modeller_interface.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */

#ifndef DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_MODELLER_INTERFACE_H
#define DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_MODELLER_INTERFACE_H

#include <memory>
#include <dials/array_family/reflection_table.h>

namespace dials { namespace algorithms {

  /**
   * The interface for profile modellers
   */
  class ProfileModellerIface {
  public:
    typedef af::versa<double, af::c_grid<3> > data_type;
    typedef af::versa<bool, af::c_grid<3> > mask_type;
    typedef af::ref<double, af::c_grid<3> > data_reference;
    typedef af::ref<bool, af::c_grid<3> > mask_reference;
    typedef af::const_ref<double, af::c_grid<3> > data_const_reference;
    typedef af::const_ref<bool, af::c_grid<3> > mask_const_reference;
    typedef std::shared_ptr<ProfileModellerIface> pointer;

    virtual ~ProfileModellerIface() {}

    virtual void model(af::reflection_table) = 0;

    virtual void accumulate(std::shared_ptr<ProfileModellerIface>) = 0;

    virtual void finalize() = 0;

    virtual bool finalized() const = 0;

    virtual data_type data(std::size_t) const = 0;

    virtual mask_type mask(std::size_t) const = 0;

    virtual std::size_t size() const = 0;

    virtual af::shared<bool> fit(af::reflection_table) const = 0;

    virtual void validate(af::reflection_table) const = 0;

    virtual pointer copy() const = 0;
  };

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_MODELLER_INTERFACE_H
