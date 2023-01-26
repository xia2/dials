/*
 * empirical_modeller.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */

#ifndef DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_EMPIRICAL_MODELLER_H
#define DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_EMPIRICAL_MODELLER_H

#include <memory>
#include <vector>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/profile_model/modeller/modeller_interface.h>

namespace dials { namespace algorithms {

  /**
   * A class to do empirical profile modelling
   */
  class EmpiricalProfileModeller : public ProfileModellerIface {
  public:
    /**
     * Initialise the modeller
     * @param n The number of profiles
     * @param accessor The size of the profiles
     * @param threshold The threshold for counts
     */
    EmpiricalProfileModeller(std::size_t n, int3 datasize, double threshold)
        : data_(n),
          mask_(n),
          n_reflections_(n, 0),
          accessor_(af::c_grid<3>(datasize[0], datasize[1], datasize[2])),
          threshold_(threshold),
          finalized_(false) {
      DIALS_ASSERT(n > 0);
      DIALS_ASSERT(datasize.all_gt(0));
      DIALS_ASSERT(threshold_ >= 0);
    }

    virtual ~EmpiricalProfileModeller() {}

    /**
     * Add a profile with indices and weights
     * @param indices The indices of the profiles to add to
     * @param weights The weight to give the profile
     * @param profile The profile data
     */
    void add(const af::const_ref<std::size_t> &indices,
             const af::const_ref<double> &weights,
             data_const_reference profile) {
      DIALS_ASSERT(finalized_ == false);
      DIALS_ASSERT(indices.size() == weights.size());
      DIALS_ASSERT(indices.size() > 0);
      DIALS_ASSERT(profile.accessor().all_eq(accessor_));
      double sum_data = sum(profile);
      if (sum_data > 0) {
        for (std::size_t j = 0; j < indices.size(); ++j) {
          std::size_t index = indices[j];
          double weight = weights[j];
          DIALS_ASSERT(index < data_.size());
          if (data_[index].size() == 0) {
            data_[index] = data_type(accessor_, 0);
            mask_[index] = mask_type(accessor_, true);
          } else {
            DIALS_ASSERT(data_[index].accessor().all_eq(accessor_));
            DIALS_ASSERT(mask_[index].accessor().all_eq(accessor_));
          }
          data_reference data = data_[index].ref();
          for (std::size_t i = 0; i < data.size(); ++i) {
            data[i] += weight * profile[i] / sum_data;
          }
          n_reflections_[index]++;
        }
      }
    }

    /**
     * Add a profile with indices and weights
     * @param index The index of the profile to add to
     * @param weight The weight to give the profile
     * @param profile The profile data
     */
    void add_single(std::size_t index, double weight, data_const_reference profile) {
      DIALS_ASSERT(finalized_ == false);
      DIALS_ASSERT(profile.accessor().all_eq(accessor_));
      DIALS_ASSERT(index < data_.size());
      double sum_data = sum(profile);
      if (sum_data > 0) {
        if (data_[index].size() == 0) {
          data_[index] = data_type(accessor_, 0);
          mask_[index] = mask_type(accessor_, true);
        } else {
          DIALS_ASSERT(data_[index].accessor().all_eq(accessor_));
          DIALS_ASSERT(mask_[index].accessor().all_eq(accessor_));
        }
        data_reference data = data_[index].ref();
        for (std::size_t i = 0; i < data.size(); ++i) {
          data[i] += weight * profile[i] / sum_data;
        }
        n_reflections_[index]++;
      }
    }

    /**
     * Accumulate the results of another modeller
     * @param other The other modeller
     */
    void accumulate(std::shared_ptr<ProfileModellerIface> other) {
      std::shared_ptr<EmpiricalProfileModeller> obj =
        std::dynamic_pointer_cast<EmpiricalProfileModeller>(other);
      DIALS_ASSERT(obj != 0);
      accumulate_raw_pointer(obj.get());
    }

    /**
     * Accumulate the results of another modeller
     * @param other The other modeller
     */
    void accumulate_raw_pointer(const EmpiricalProfileModeller *other) {
      DIALS_ASSERT(other != NULL);
      DIALS_ASSERT(finalized_ == false);

      // Check sizes are the same
      DIALS_ASSERT(data_.size() == other->data_.size());
      DIALS_ASSERT(accessor_.all_eq(other->accessor_));

      // Loop through all the profiles. If needed, allocate them, then
      // add the pixel values from the other modeller to this
      for (std::size_t i = 0; i < data_.size(); ++i) {
        n_reflections_[i] += other->n_reflections_[i];
        if (other->data_[i].size() != 0) {
          if (data_[i].size() == 0) {
            data_[i] = data_type(accessor_, 0);
            mask_[i] = mask_type(accessor_, true);
          }
          data_reference d1 = data_[i].ref();
          mask_reference m1 = mask_[i].ref();
          data_const_reference d2 = other->data_[i].const_ref();
          mask_const_reference m2 = other->mask_[i].const_ref();
          DIALS_ASSERT(d1.accessor().all_eq(d2.accessor()));
          DIALS_ASSERT(m1.accessor().all_eq(m2.accessor()));
          for (std::size_t j = 0; j < d1.size(); ++j) {
            d1[j] += d2[j];
            m1[j] = m1[j] && m2[j];
          }
        }
      }
    }

    /**
     * Finalize the modeller
     */
    void finalize() {
      DIALS_ASSERT(finalized_ == false);
      for (std::size_t i = 0; i < data_.size(); ++i) {
        if (data_[i].size() != 0) {
          finalize(i);
        }
      }
      finalized_ = true;
    }

    /**
     * @return True/False model is finalized
     */
    bool finalized() const {
      return finalized_;
    }

    /**
     * Set whether this is finalized.
     */
    void set_finalized(bool finalized) {
      finalized_ = finalized;
    }

    /**
     * Set the data
     */
    void set_data(std::size_t index, data_type value) {
      DIALS_ASSERT(index < data_.size());
      DIALS_ASSERT(value.size() == 0 || value.accessor().all_eq(accessor_));
      data_[index] = value;
    }

    /**
     * Set the mask
     */
    void set_mask(std::size_t index, mask_type value) {
      DIALS_ASSERT(index < mask_.size());
      DIALS_ASSERT(value.size() == 0 || value.accessor().all_eq(accessor_));
      mask_[index] = value;
    }

    void set_n_reflections(std::size_t index, std::size_t value) {
      DIALS_ASSERT(index < n_reflections_.size());
      n_reflections_[index] = value;
    }

    /**
     * @return The profile at the index
     */
    data_type data(std::size_t index) const {
      DIALS_ASSERT(index < data_.size());
      DIALS_ASSERT(data_[index].size() != 0);
      return data_[index];
    }

    /**
     * @return The mask at the index
     */
    mask_type mask(std::size_t index) const {
      DIALS_ASSERT(index < mask_.size());
      DIALS_ASSERT(mask_[index].size() != 0);
      return mask_[index];
    }

    /**
     * @return The number of profiles
     */
    std::size_t size() const {
      return data_.size();
    }

    /**
     * @return The accessor
     */
    int3 datasize() const {
      return int3(accessor_[0], accessor_[1], accessor_[2]);
    }

    /**
     * @return The threshold
     */
    double threshold() const {
      return threshold_;
    }

    /**
     * @return Is the profile valid
     */
    bool valid(std::size_t index) const {
      DIALS_ASSERT(index < mask_.size());
      return mask_[index].size() != 0;
    }

    /**
     * @return The number of reflections used
     */
    std::size_t n_reflections(std::size_t index) const {
      DIALS_ASSERT(index < n_reflections_.size());
      return n_reflections_[index];
    }

    /**
     * The model method
     */
    virtual void model(af::reflection_table) {
      throw DIALS_ERROR("No implemented");
    }

    /**
     * The model copy method
     */
    virtual pointer copy() const {
      throw DIALS_ERROR("No implemented");
    }

    /**
     * The fit method
     */
    virtual af::shared<bool> fit(af::reflection_table) const {
      throw DIALS_ERROR("No implemented");
    }

    /**
     * The validate method
     */
    virtual void validate(af::reflection_table) const {
      throw DIALS_ERROR("No implemented");
    }

  protected:
    /**
     * Finalize a single profile
     * @param index The index of the profile to finalize
     */
    void finalize(std::size_t index) {
      // Check data
      DIALS_ASSERT(data_[index].accessor().all_eq(accessor_));
      DIALS_ASSERT(mask_[index].accessor().all_eq(accessor_));

      // Get the reference profile at the index
      data_reference data = data_[index].ref();
      // mask_reference mask = mask_[index].ref();

      // Calculate the profile maximum and signal threshold
      // double threshold = threshold_ * max(data);

      // Get the sum of signal pixels
      double signal_sum = 0.0;
      for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] >= 0.0) {  // threshold) {
          signal_sum += data[i];
        } else {
          data[i] = 0.0;
          // mask[i] = false;
        }
      }

      // Normalize the profile such that sum of signal pixels == 1
      DIALS_ASSERT(signal_sum > 0);
      for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] /= signal_sum;
      }
    }

    af::shared<data_type> data_;
    af::shared<mask_type> mask_;
    af::shared<std::size_t> n_reflections_;
    af::c_grid<3> accessor_;
    double threshold_;
    bool finalized_;
  };

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_PROFILE_MODEL_MODELLER_EMPIRICAL_MODELLER_H
