/*
 * flex_table_suite.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */

#ifndef DIALS_FRAMEWORK_TABLE_BOOST_PYTHON_FLEX_TABLE_SUITE_H
#define DIALS_FRAMEWORK_TABLE_BOOST_PYTHON_FLEX_TABLE_SUITE_H

#include <string>
#include <iterator>
#include <iostream>
#include <sstream>
#include <set>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/slice.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/mpl/for_each.hpp>
#include <scitbx/array_family/flex_types.h>
#include <scitbx/array_family/boost_python/ref_pickle_double_buffered.h>
#include <scitbx/boost_python/slice.h>
#include <scitbx/boost_python/utils.h>
#include <dials/array_family/flex_table.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/error.h>
#include <dxtbx/model/experiment.h>
#include <dxtbx/model/experiment_list.h>

namespace dials { namespace af { namespace boost_python { namespace flex_table_suite {

  using namespace boost::python;

  /**
   * A visitor to convert the column to a boost python object
   */
  struct column_to_object_visitor : public boost::static_visitor<object> {
    template <typename T>
    object operator()(T &col) {
      return object(col);
    }
  };

  /**
   * A visitor to extract a column element as a python object
   */
  struct element_to_object_visitor : public boost::static_visitor<object> {
    std::size_t n_;
    element_to_object_visitor(std::size_t n) : n_(n) {}
    template <typename T>
    object operator()(T &col) {
      return object(col[n_]);
    }
  };

  /**
   * A visitor to set the row item from a python object
   */
  struct setitem_row_visitor : public boost::static_visitor<void> {
    std::size_t index;
    object item;

    setitem_row_visitor(std::size_t index_, object item_)
        : index(index_), item(item_) {}

    template <typename T>
    void operator()(T &column) {
      DIALS_ASSERT(index < column.size());
      column[index] = extract<typename T::value_type>(item);
    }
  };

  /**
   * A visitor to append column data from 1 table to another
   */
  template <typename T>
  struct extend_column_visitor : public boost::static_visitor<void> {
    T &self;
    typename T::key_type key;
    typename T::size_type na, nb;

    extend_column_visitor(T &self_,
                          typename T::key_type key_,
                          typename T::size_type na_,
                          typename T::size_type nb_)
        : self(self_), key(key_), na(na_), nb(nb_) {}

    template <typename U>
    void operator()(const U &other_column) {
      U self_column = self[key];
      DIALS_ASSERT(na + nb == self_column.size());
      for (typename T::size_type i = 0; i < nb; ++i) {
        self_column[na + i] = other_column[i];
      }
    }
  };

  /**
   * A visitor to add new columns (and over-write old columns) in the table.
   */
  template <typename T>
  struct update_column_visitor : public boost::static_visitor<void> {
    T &self;
    typename T::key_type key;

    update_column_visitor(T &self_, typename T::key_type key_)
        : self(self_), key(key_) {}

    template <typename U>
    void operator()(const U &other_column) {
      self.erase(key);
      U self_column = self[key];
      DIALS_ASSERT(self_column.size() == other_column.size());
      for (std::size_t i = 0; i < other_column.size(); ++i) {
        self_column[i] = other_column[i];
      }
    }
  };

  /**
   * Read a slice from the input column and write it to a column in the
   * output table.
   */
  template <typename T>
  struct copy_to_slice_visitor : public boost::static_visitor<void> {
    T &self;
    typename T::key_type key;
    scitbx::boost_python::adapted_slice slice;

    copy_to_slice_visitor(T &self_,
                          typename T::key_type key_,
                          scitbx::boost_python::adapted_slice slice_)
        : self(self_), key(key_), slice(slice_) {}

    template <typename U>
    void operator()(const U &other_column) {
      U self_column = self[key];
      for (std::size_t i = 0, j = slice.start; i < self.nrows(); ++i, j += slice.step) {
        DIALS_ASSERT(i < self_column.size());
        DIALS_ASSERT(j < other_column.size());
        self_column[i] = other_column[j];
      }
    }
  };

  /**
   * Copy from a slice of column data into the column table.
   */
  template <typename T>
  struct copy_from_slice_visitor : public boost::static_visitor<void> {
    T &self;
    typename T::key_type key;
    scitbx::boost_python::adapted_slice slice;
    typename T::size_type num;

    copy_from_slice_visitor(T &self_,
                            typename T::key_type key_,
                            scitbx::boost_python::adapted_slice slice_,
                            std::size_t num_)
        : self(self_), key(key_), slice(slice_), num(num_) {}

    template <typename U>
    void operator()(const U &other_column) {
      U self_column = self[key];
      for (std::size_t i = 0, j = slice.start; i < num; ++i, j += slice.step) {
        DIALS_ASSERT(j < self_column.size());
        DIALS_ASSERT(i < other_column.size());
        self_column[j] = other_column[i];
      }
    }
  };

  /**
   * Copy all the values in a column
   */
  template <typename T>
  struct copy_column_visitor : public boost::static_visitor<void> {
    T &result;
    typename T::key_type key;

    copy_column_visitor(T &result_, typename T::key_type key_)
        : result(result_), key(key_) {}

    template <typename U>
    void operator()(const U &other_column) {
      U result_column = result[key];
      DIALS_ASSERT(result_column.size() == other_column.size());
      for (std::size_t i = 0; i < other_column.size(); ++i) {
        result_column[i] = other_column[i];
      }
    }
  };

  /**
   * Copy the selected rows from the input column to a output column.
   */
  template <typename T>
  struct copy_from_indices_visitor : public boost::static_visitor<void> {
    T &result;
    typename T::key_type key;
    af::const_ref<std::size_t> index;

    copy_from_indices_visitor(T &result_,
                              typename T::key_type key_,
                              af::const_ref<std::size_t> index_)
        : result(result_), key(key_), index(index_) {}

    template <typename U>
    void operator()(const U &other_column) {
      U result_column = result[key];
      DIALS_ASSERT(result_column.size() == index.size());
      for (std::size_t i = 0; i < index.size(); ++i) {
        result_column[i] = other_column[index[i]];
      }
    }
  };

  /**
   * Copy the selected rows from the input column to a output column.
   */
  template <typename T>
  struct copy_to_indices_visitor : public boost::static_visitor<void> {
    T &result;
    typename T::key_type key;
    af::const_ref<std::size_t> index;

    copy_to_indices_visitor(T &result_,
                            typename T::key_type key_,
                            af::const_ref<std::size_t> index_)
        : result(result_), key(key_), index(index_) {}

    template <typename U>
    void operator()(const U &other_column) {
      U result_column = result[key];
      DIALS_ASSERT(other_column.size() == index.size());
      for (std::size_t i = 0; i < index.size(); ++i) {
        result_column[index[i]] = other_column[i];
      }
    }
  };

  /**
   * Copy the selected rows from the input column to a output column.
   */
  template <typename T>
  struct copy_to_indices_with_mask_visitor : public boost::static_visitor<void> {
    T &result;
    typename T::key_type key;
    af::const_ref<std::size_t> index;
    af::const_ref<bool> mask;

    copy_to_indices_with_mask_visitor(T &result_,
                                      typename T::key_type key_,
                                      af::const_ref<std::size_t> index_,
                                      af::const_ref<bool> mask_)
        : result(result_), key(key_), index(index_), mask(mask_) {}

    template <typename U>
    void operator()(const U &other_column) {
      U result_column = result[key];
      DIALS_ASSERT(other_column.size() == index.size());
      for (std::size_t i = 0; i < index.size(); ++i) {
        if (mask[i]) {
          result_column[index[i]] = other_column[i];
        }
      }
    }
  };

  /**
   * A visitor to reorder the elements of a column
   */
  struct reorder_visitor : public boost::static_visitor<void> {
    af::const_ref<std::size_t> index;

    reorder_visitor(const af::const_ref<std::size_t> &index_) : index(index_) {}

    template <typename T>
    void operator()(T &column) {
      std::vector<typename T::value_type> temp(column.begin(), column.end());
      DIALS_ASSERT(index.size() == column.size());
      for (std::size_t i = 0; i < index.size(); ++i) {
        column[i] = temp[index[i]];
      }
    }
  };

  /**
   * Functor to compare elements by index
   */
  template <typename T>
  struct compare_index {
    const T &v_;

    compare_index(const T &v) : v_(v) {}

    template <typename U>
    bool operator()(U a, U b) {
      return v_[a] < v_[b];
    }
  };

  /**
   * A visitor to sort the table by columns
   */
  struct sort_visitor : public boost::static_visitor<void> {
    af::ref<std::size_t> index;

    sort_visitor(af::ref<std::size_t> index_) : index(index_) {
      for (std::size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
      }
    }

    template <typename T>
    void operator()(const T &col) {
      std::sort(index.begin(), index.end(), compare_index<T>(col));
    }
  };

  /**
   * A visitor to remove elements by flag
   */
  struct remove_if_flag_visitor : public boost::static_visitor<void> {
    af::const_ref<bool> flags;

    remove_if_flag_visitor(const af::const_ref<bool> &flags_) : flags(flags_) {}

    template <typename T>
    void operator()(T &col) {
      for (std::size_t i = 0, j = 0; i < col.size(); ++i) {
        if (!flags[i]) {
          col[j++] = col[i];
        }
      }
    }
  };

  /**
   * Initialise the column table from a list of (key, column) pairs
   * @param columns The list of columns
   * @returns The column table
   */
  template <typename T>
  T *make_flex_table(list columns) {
    T self;
    object obj(self);
    for (std::size_t i = 0; i < len(columns); ++i) {
      obj[columns[i][0]] = columns[i][1];
    }
    self = extract<T>(obj);
    return new T(self);
  }

  /**
   * Get a column of data
   * @param self The column table
   * @param key The column name
   * @returns The boost python object with the column data
   */
  template <typename T>
  object getitem_column(T &self, const typename T::key_type &key) {
    typename T::mapped_type column = self[key].variant();
    column_to_object_visitor visitor;
    return column.apply_visitor(visitor);
  }

  /**
   * Delete a column of data
   * @param self The column table
   * @param key The name of the column
   */
  template <typename T>
  void delitem_column(T &self, const typename T::key_type &key) {
    self.erase(key);
  }

  /**
   * Set a column of data
   * @param self The table to populate
   * @param key The column name
   * @param data The column data
   */
  template <typename T, typename U>
  void setitem_column(T &self,
                      const typename T::key_type &key,
                      const af::const_ref<U> &data) {
    self.erase(key);
    DIALS_ASSERT(self.ncols() == 0 || data.size() == self.nrows());
    self.resize(data.size());
    af::shared<U> column = self[key];
    std::copy(data.begin(), data.end(), column.begin());
  }

  /**
   * Get a row of data from the table.
   * @param self The table
   * @param n The position of the row
   */
  template <typename T>
  dict getitem_row(const T &self, typename T::size_type n) {
    typedef typename T::const_iterator iterator;
    if (n >= self.nrows()) {
      scitbx::boost_python::raise_index_error();
    }
    dict result;
    element_to_object_visitor visitor(n);
    for (iterator it = self.begin(); it != self.end(); ++it) {
      result[it->first] = it->second.apply_visitor(visitor);
    }
    return result;
  }

  /**
   * Delete a row of data
   * @param self The column table
   * @param n The index of the row
   */
  template <typename T>
  void delitem_row(T &self, typename T::size_type n) {
    if (n >= self.nrows()) {
      scitbx::boost_python::raise_index_error();
    }
    self.erase(n);
  }

  /**
   * Set a row of data in the table
   * @param self The table to modify
   * @param n The position of the row
   * @param row The row data to set
   */
  template <typename T>
  void setitem_row(T &self, typename T::size_type n, dict row) {
    if (n >= self.nrows()) {
      scitbx::boost_python::raise_index_error();
    }
    typedef typename T::iterator iterator;
    object items = list(row.items());
    DIALS_ASSERT(len(items) == len(row));
    for (std::size_t i = 0; i < len(row); ++i) {
      object item = items[i];
      setitem_row_visitor visitor(n, item[1]);
      iterator it = self.find(extract<std::string>(item[0]));
      DIALS_ASSERT(it != self.end());
      it->second.apply_visitor(visitor);
    }
  }

  /**
   * Get a slice of the table and return a new table
   * @param self The current table
   * @param slice The slice
   * @returns A new table with the chosen elements
   */
  template <typename T>
  T getitem_slice(const T &self, slice s) {
    typedef typename T::const_iterator iterator;
    scitbx::boost_python::adapted_slice as(s, self.nrows());
    T result(as.size);
    for (iterator it = self.begin(); it != self.end(); ++it) {
      copy_to_slice_visitor<T> visitor(result, it->first, as);
      it->second.apply_visitor(visitor);
    }
    return result;
  }

  /**
   * Remove elements if the flag is true
   * @param self The table
   * @param flags The list of flags
   */
  template <typename T>
  void remove_if_flag(T &self, const af::const_ref<bool> &flags) {
    DIALS_ASSERT(flags.size() == self.nrows());
    std::size_t n = std::count(flags.begin(), flags.end(), false);
    typedef typename T::iterator iterator;
    remove_if_flag_visitor visitor(flags);
    for (iterator it = self.begin(); it != self.end(); ++it) {
      it->second.apply_visitor(visitor);
    }
    self.resize(n);
  }

  /**
   * Delete a slice from the table
   * @param self The table
   * @param s The slice
   */
  template <typename T>
  void delitem_slice(T &self, slice s) {
    scitbx::boost_python::adapted_slice as(s, self.nrows());
    if (as.step == 1) {
      self.erase(as.start, as.size);
    } else if (as.step == -1) {
      self.erase(as.stop, as.size);
    } else {
      af::shared<bool> flags(self.nrows(), false);
      for (std::size_t j = as.start; j < flags.size(); j += as.step) {
        flags[j] = true;
      }
      remove_if_flag<T>(self, flags.const_ref());
    }
  }

  /**
   * Set a slice of data from one table into the current table.
   * @param self The current table
   * @param slice The slice to set
   * @param other The other table whose elements to set
   */
  template <typename T>
  void setitem_slice(T &self, slice s, const T &other) {
    typedef typename T::const_iterator iterator;
    DIALS_ASSERT(self.is_consistent());
    DIALS_ASSERT(other.is_consistent());
    scitbx::boost_python::adapted_slice as(s, self.nrows());
    for (iterator it = other.begin(); it != other.end(); ++it) {
      copy_from_slice_visitor<T> visitor(self, it->first, as, other.nrows());
      it->second.apply_visitor(visitor);
    }
  }

  /**
   * Check if the table has the given key
   * @param self The table
   * @param key The key to check for
   */
  template <typename T>
  bool has_key(const T &self, const typename T::key_type &key) {
    return self.count(key) == 1;
  }

  /**
   * Append a row onto the end of the columns
   * @param self The column table
   * @param row The row to add
   */
  template <typename T>
  void append(T &self, dict row) {
    self.resize(self.nrows() + 1);
    setitem_row(self, self.nrows() - 1, row);
  }

  /**
   * Insert a row into the table at the given position
   * @param self The column table
   * @param n The position to insert the row at
   * @param row The row to insert
   */
  template <typename T>
  void insert(T &self, typename T::size_type n, dict row) {
    self.insert(n);
    setitem_row(self, n, row);
  }

  /**
   * Extend the identifiers
   */
  template <typename T>
  void reflection_table_extend_identifiers(T &self, const T &other) {
    typedef typename T::experiment_map_type::const_iterator const_iterator;
    typedef typename T::experiment_map_type::iterator iterator;
    for (const_iterator it = other.experiment_identifiers()->begin();
         it != other.experiment_identifiers()->end();
         ++it) {
      iterator found = self.experiment_identifiers()->find(it->first);
      if (found == self.experiment_identifiers()->end()) {
        (*self.experiment_identifiers())[it->first] = it->second;
      } else if (it->second != found->second) {
        throw DIALS_ERROR("Experiment identifiers do not match");
      }
    }
  }

  /**
   * Extend the table with column data from another table. This will add
   * all the columns from the other table onto the end of the current table.
   * @param self The current table
   * @param other The other table
   */
  template <typename T>
  void extend(T &self, const T &other) {
    typedef typename T::const_iterator iterator;
    typename T::size_type ns = self.nrows();
    typename T::size_type no = other.nrows();
    self.resize(ns + no);
    for (iterator it = other.begin(); it != other.end(); ++it) {
      extend_column_visitor<T> visitor(self, it->first, ns, no);
      it->second.apply_visitor(visitor);
    }
    // now extend identifiers
    reflection_table_extend_identifiers(self, other);
  }

  /**
   * Update the table with column data from another table. New columns are added
   * to the table and existing columns are over-written by columns from the
   * other table.
   * @param self The current table
   * @param other The other table
   */
  template <typename T>
  void update(T &self, const T &other) {
    typedef typename T::const_iterator iterator;
    if (self.ncols() == 0) {
      self.resize(other.nrows());
    }
    DIALS_ASSERT(self.nrows() == other.nrows());
    for (iterator it = other.begin(); it != other.end(); ++it) {
      update_column_visitor<T> visitor(self, it->first);
      it->second.apply_visitor(visitor);
    }
  }

  /**
   * Select a number of rows from the table via an index array
   * @param self The current table
   * @param index The index array
   * @returns The new table with the requested rows
   */
  template <typename T>
  T select_rows_index(const T &self, const af::const_ref<std::size_t> &index) {
    // Check that indices are valid
    std::size_t nrows = self.nrows();
    for (std::size_t i = 0; i < index.size(); ++i) {
      DIALS_ASSERT(index[i] < nrows);
    }

    // Get the indices from the table
    T result(index.size());
    for (typename T::const_iterator it = self.begin(); it != self.end(); ++it) {
      copy_from_indices_visitor<T> visitor(result, it->first, index);
      it->second.apply_visitor(visitor);
    }

    // Get the id column (if it exists) and make a set of unique values
    if (self.contains("id")) {
      af::shared<int> col = result["id"];
      std::set<int> new_ids(col.begin(), col.end());

      // Copy across identifiers for ids in new table
      typedef typename T::experiment_map_type::const_iterator const_iterator;
      typedef typename T::experiment_map_type::iterator iterator;
      for (std::set<int>::iterator i = new_ids.begin(); i != new_ids.end(); ++i) {
        iterator found = self.experiment_identifiers()->find(*i);
        if (found != self.experiment_identifiers()->end()) {
          (*result.experiment_identifiers())[found->first] = found->second;
        }
      }
    }
    // Return new table
    return result;
  }

  /**
   * Select a number of rows from the table via an index array
   * @param self The current table
   * @param flags The flag array
   * @returns The new table with the requested rows
   */
  template <typename T>
  T select_rows_flags(const T &self, const af::const_ref<bool> &flags) {
    DIALS_ASSERT(self.nrows() == flags.size());
    af::shared<std::size_t> index;
    for (std::size_t i = 0; i < flags.size(); ++i) {
      if (flags[i]) index.push_back(i);
    }
    return select_rows_index(self, index.const_ref());
  }

  /**
   * Select a number of columns from the table via an key array
   * @param self The current table
   * @param keys The key array
   * @returns The new table with the requested columns
   */
  template <typename T>
  T select_cols_keys(const T &self, const af::const_ref<std::string> &keys) {
    T result(self.nrows());
    for (std::size_t i = 0; i < keys.size(); ++i) {
      copy_column_visitor<T> visitor(result, keys[i]);
      typename T::const_iterator it = self.find(keys[i]);
      DIALS_ASSERT(it != self.end());
      it->second.apply_visitor(visitor);
    }
    return result;
  }

  /**
   * Select a number of columns from the table via an key array
   * @param self The current table
   * @param keys The key array
   * @returns The new table with the requested columns
   */
  template <typename T>
  T select_cols_tuple(const T &self, boost::python::tuple keys) {
    af::shared<std::string> keys_array;
    for (std::size_t i = 0; i < len(keys); ++i) {
      keys_array.push_back(extract<std::string>(keys[i]));
    }
    return select_cols_keys(self, keys_array.const_ref());
  }

  template <typename T>
  T select_using_experiment(T &self, dxtbx::model::Experiment expt) {
    typedef typename T::experiment_map_type::const_iterator const_iterator;

    std::string identifier = expt.get_identifier();
    int id_value = -1;
    for (const_iterator it = self.experiment_identifiers()->begin();
         it != self.experiment_identifiers()->end();
         ++it) {
      if (identifier == it->second) {
        id_value = it->first;
        break;
      }
    }

    T result;
    if (self.contains("id") && id_value != -1) {
      af::shared<int> col1 = self["id"];
      af::shared<std::size_t> sel;
      for (int i = 0; i < col1.size(); ++i) {
        if (col1[i] == id_value) {
          sel.push_back(i);
        }
      }
      af::const_ref<std::size_t> idx = sel.const_ref();

      result = select_rows_index(self, idx);
    }

    return result;
  }

  template <typename T>
  T select_using_experiments(T &self, dxtbx::model::ExperimentList expts) {
    typedef typename T::experiment_map_type::const_iterator const_iterator;
    typedef dxtbx::model::ExperimentList::shared_type::const_iterator
      expt_const_iterator;
    T result;
    for (expt_const_iterator expt = expts.begin(); expt != expts.end(); ++expt) {
      std::string identifier = expt->get_identifier();
      int id_value = -1;
      for (const_iterator it = self.experiment_identifiers()->begin();
           it != self.experiment_identifiers()->end();
           ++it) {
        if (identifier == it->second) {
          id_value = it->first;
          break;
        }
      }

      if (self.contains("id") && id_value != -1) {
        af::shared<int> col1 = self["id"];
        af::shared<std::size_t> sel;
        for (int i = 0; i < col1.size(); ++i) {
          if (col1[i] == id_value) {
            sel.push_back(i);
          }
        }
        af::const_ref<std::size_t> idx = sel.const_ref();

        T sel_refl = select_rows_index(self, idx);
        extend(result, sel_refl);
      }
    }

    return result;
  }

  /**
   * Set the selected number of rows from the table via an index array
   * @param self The current table
   * @param index The index array
   * @param other The other table
   */
  template <typename T>
  void set_selected_rows_index(T &self,
                               const af::const_ref<std::size_t> &index,
                               const T &other) {
    typedef typename T::const_iterator iterator;
    DIALS_ASSERT(index.size() == other.nrows());
    for (iterator it = other.begin(); it != other.end(); ++it) {
      copy_to_indices_visitor<T> visitor(self, it->first, index);
      it->second.apply_visitor(visitor);
    }
  }

  /**
   * Set the selected number of rows from the table via an index array
   * @param self The current table
   * @param index The index array
   * @param mask The indices in other to use
   * @param other The other table
   */
  template <typename T>
  void set_selected_rows_index_mask(T &self,
                                    const af::const_ref<std::size_t> &index,
                                    const af::const_ref<bool> &mask,
                                    const T &other) {
    typedef typename T::const_iterator iterator;
    DIALS_ASSERT(index.size() == other.nrows());
    DIALS_ASSERT(index.size() == mask.size());
    for (iterator it = other.begin(); it != other.end(); ++it) {
      copy_to_indices_with_mask_visitor<T> visitor(self, it->first, index, mask);
      it->second.apply_visitor(visitor);
    }
  }

  /**
   * Set the selected number of rows from the table via an index array
   * @param self The current table
   * @param flags The flag array
   * @param other The other table
   */
  template <typename T>
  void set_selected_rows_flags(T &self,
                               const af::const_ref<bool> &flags,
                               const T &other) {
    DIALS_ASSERT(self.nrows() == flags.size());
    af::shared<std::size_t> index;
    for (std::size_t i = 0; i < flags.size(); ++i) {
      if (flags[i]) index.push_back(i);
    }
    set_selected_rows_index(self, index.const_ref(), other);
  }

  /**
   * Set the selected number of columns from the table via an key array
   * @param self The current table
   * @param keys The key array
   * @param other The other table
   */
  template <typename T>
  void set_selected_cols_keys(T &self,
                              const af::const_ref<std::string> &keys,
                              const T &other) {
    DIALS_ASSERT(self.nrows() == other.nrows());
    for (std::size_t i = 0; i < keys.size(); ++i) {
      copy_column_visitor<T> visitor(self, keys[i]);
      typename T::const_iterator it = other.find(keys[i]);
      DIALS_ASSERT(it != other.end());
      it->second.apply_visitor(visitor);
    }
  }

  /**
   * Set the selected number of columns from the table via an key array
   * @param self The current table
   * @param keys The key array
   * @param other The other table
   */
  template <typename T>
  void set_selected_cols_tuple(T &self, boost::python::tuple keys, const T &other) {
    af::shared<std::string> keys_array;
    for (std::size_t i = 0; i < len(keys); ++i) {
      keys_array.push_back(extract<std::string>(keys[i]));
    }
    set_selected_cols_keys(self, keys_array.const_ref(), other);
  }

  /**
   * Delete selected items by flags
   * @param self The table
   * @param flags The array of boolean flags
   */
  template <typename T>
  void del_selected_rows_flags(T &self, const af::const_ref<bool> &flags) {
    remove_if_flag(self, flags);
  }

  /**
   * Delete the selected rows from the table
   * @param self The table
   * @param index The index array
   */
  template <typename T>
  void del_selected_rows_index(T &self, const af::const_ref<std::size_t> &index) {
    af::shared<bool> flags(self.nrows(), false);
    for (std::size_t i = 0; i < index.size(); ++i) {
      DIALS_ASSERT(index[i] < flags.size());
      flags[index[i]] = true;
    }
    del_selected_rows_flags(self, flags.const_ref());
  }

  /**
   * Delete the selected columns
   * @param self The table
   * @param keys The columns to delete
   */
  template <typename T>
  void del_selected_cols_keys(T &self, const af::const_ref<std::string> &keys) {
    for (std::size_t i = 0; i < keys.size(); ++i) {
      self.erase(keys[i]);
    }
  }

  /**
   * Delete the selected number of columns from the table via an key array
   * @param self The current table
   * @param keys The key array
   */
  template <typename T>
  void del_selected_cols_tuple(T &self, boost::python::tuple keys) {
    af::shared<std::string> keys_array;
    for (std::size_t i = 0; i < len(keys); ++i) {
      keys_array.push_back(extract<std::string>(keys[i]));
    }
    del_selected_cols_keys(self, keys_array.const_ref());
  }

  /**
   * An MPL function to populate the list of types
   */
  struct type_appender {
    list type_list;
    type_appender(list type_list_) : type_list(type_list_) {}
    template <typename U>
    void operator()(U x) {
      typename U::value_type a = typename U::value_type();
      type_list.append(object(handle<>(PyObject_Type(object(a).ptr()))));
    }
  };

  /**
   * Get a list of valid column types.
   * @param self The table
   * @returns A list of python types.
   */
  template <typename T>
  list types(const T &self) {
    list result;
    boost::mpl::for_each<typename T::mapped_type::types>(type_appender(result));
    return result;
  }

  /**
   * Reorder all the columns according to the input indices
   * @param self The table object
   * @param indices The array of indices
   */
  template <typename T>
  void reorder(T &self, const af::const_ref<std::size_t> &index) {
    typedef typename T::iterator iterator;
    DIALS_ASSERT(self.is_consistent());
    reorder_visitor visitor(index);
    for (iterator it = self.begin(); it != self.end(); ++it) {
      it->second.apply_visitor(visitor);
    }
  }

  /**
   * Sort the table with respect to a given column
   * @param self The table object
   * @param key The column key
   * @param reverse True/False reverse the sort
   */
  template <typename T>
  void sort(T &self, typename T::key_type key, bool reverse) {
    af::shared<std::size_t> index(self.nrows());
    sort_visitor visitor(index.ref());
    typename T::mapped_type col = self[key];
    col.apply_visitor(visitor);
    if (reverse) {
      std::reverse(index.begin(), index.end());
    }
    reorder(self, index.const_ref());
  }

  /**
   * Perform a shallow copy
   */
  template <typename T>
  T copy(const T &self) {
    DIALS_ASSERT(self.is_consistent());
    return T(self);
  }

  /**
   * Perform a deep copy
   */
  template <typename T>
  T deepcopy(const T &self, dict obj) {
    typedef typename T::const_iterator iterator;
    T result(self.nrows());
    for (iterator it = self.begin(); it != self.end(); ++it) {
      copy_column_visitor<T> visitor(result, it->first);
      it->second.apply_visitor(visitor);
    }
    typedef typename T::experiment_map_type::const_iterator const_iterator;
    for (const_iterator it = self.experiment_identifiers()->begin();
         it != self.experiment_identifiers()->end();
         ++it) {
      (*result.experiment_identifiers())[it->first] = it->second;
    }
    return result;
  }

  /**
   * A proxy iterator to iterate over the table keys
   */
  template <typename T>
  class key_iterator {
  public:
    typedef std::forward_iterator_tag iterator_category;
    typedef T table_type;
    typedef typename T::const_iterator base_iterator;
    typedef ptrdiff_t difference_type;
    typedef typename T::key_type value_type;
    typedef const value_type *pointer;
    typedef const value_type &reference;

    key_iterator(base_iterator it) : it_(it) {}

    reference operator*() {
      return it_->first;
    }

    key_iterator &operator++() {
      ++it_;
      return *this;
    }

    key_iterator operator++(int) {
      key_iterator result(*this);
      ++(*this);
      return result;
    }

    bool operator==(const key_iterator &rhs) const {
      return it_ == rhs.it_;
    }

    bool operator!=(const key_iterator &rhs) const {
      return !(*this == rhs);
    }

  private:
    base_iterator it_;
  };

  /**
   * A proxy iterator to iterate over the table column
   */
  template <typename T>
  class column_iterator {
  public:
    typedef T table_type;
    typedef typename T::const_iterator base_iterator;
    typedef ptrdiff_t difference_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef object value_type;
    typedef const value_type *pointer;
    typedef const value_type reference;

    column_iterator(base_iterator it) : it_(it) {}

    reference operator*() {
      column_to_object_visitor visitor;
      return boost::python::make_tuple(it_->first, it_->second.apply_visitor(visitor));
    }

    column_iterator &operator++() {
      ++it_;
      return *this;
    }

    column_iterator operator++(int) {
      column_iterator result(*this);
      ++(*this);
      return result;
    }

    bool operator==(const column_iterator &rhs) const {
      return it_ == rhs.it_;
    }

    bool operator!=(const column_iterator &rhs) const {
      return !(*this == rhs);
    }

  private:
    base_iterator it_;
  };

  /**
   * Proxy iterator to iterate through the table rows
   */
  template <typename T>
  class row_iterator {
  public:
    typedef T table_type;
    typedef ptrdiff_t difference_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef dict value_type;
    typedef const value_type *pointer;
    typedef const value_type reference;
    typedef typename T::mapped_type mapped_type;

    row_iterator(const T &self, std::size_t index) : index_(index) {
      typedef typename T::const_iterator iterator;
      for (iterator it = self.begin(); it != self.end(); ++it) {
        keys.push_back(it->first);
        cols.push_back(it->second);
      }
    }

    reference operator*() {
      dict result;
      element_to_object_visitor visitor(index_);
      for (std::size_t i = 0; i < keys.size(); ++i) {
        result[keys[i]] = cols[i].apply_visitor(visitor);
      }
      return result;
    }

    row_iterator &operator++() {
      ++index_;
      return *this;
    }

    row_iterator operator++(int) {
      row_iterator result(*this);
      ++(*this);
      return result;
    }

    bool operator==(const row_iterator &rhs) const {
      return index_ == rhs.index_;
    }

    bool operator!=(const row_iterator &rhs) const {
      return !(*this == rhs);
    }

  private:
    std::vector<mapped_type> cols;
    std::vector<std::string> keys;
    std::size_t index_;
  };

  /**
   * Struct to help in creation of table proxy iterators
   */
  template <typename Iterator>
  struct make_iterator {
    static Iterator begin(const typename Iterator::table_type &self) {
      DIALS_ASSERT(self.is_consistent());
      return Iterator(self.begin());
    }

    static Iterator end(const typename Iterator::table_type &self) {
      return Iterator(self.end());
    }

    static object range() {
      return boost::python::range(&make_iterator<Iterator>::begin,
                                  &make_iterator<Iterator>::end);
    }
  };

  /**
   * Specialization for row iterator
   */
  template <typename T>
  struct make_iterator<row_iterator<T> > {
    static row_iterator<T> begin(const T &self) {
      DIALS_ASSERT(self.is_consistent());
      return row_iterator<T>(self, 0);
    }

    static row_iterator<T> end(const T &self) {
      return row_iterator<T>(self, self.nrows());
    }

    static object range() {
      return boost::python::range(&make_iterator<row_iterator<T> >::begin,
                                  &make_iterator<row_iterator<T> >::end);
    }
  };

  /**
   * Class to pickle and unpickle the table
   */
  template <typename T>
  struct flex_table_pickle_suite : boost::python::pickle_suite {
    typedef T flex_table_type;
    typedef typename T::const_iterator const_iterator;

    static boost::python::tuple getstate(const flex_table_type &self) {
      DIALS_ASSERT(self.is_consistent());
      unsigned int version = 1;

      // Get the columns as a dictionary
      dict columns;
      column_to_object_visitor visitor;
      for (const_iterator it = self.begin(); it != self.end(); ++it) {
        columns[it->first] = it->second.apply_visitor(visitor);
      }

      // Make the tuple
      return boost::python::make_tuple(version, self.nrows(), self.ncols(), columns);
    }

    static void setstate(flex_table_type &self, boost::python::tuple state) {
      DIALS_ASSERT(boost::python::len(state) == 4);
      DIALS_ASSERT(extract<unsigned int>(state[0]) == 1);
      std::size_t nrows = extract<std::size_t>(state[1]);
      std::size_t ncols = extract<std::size_t>(state[2]);
      self.resize(nrows);

      // Extract the columns
      dict columns = extract<dict>(state[3]);
      DIALS_ASSERT(len(columns) == ncols);
      object items = list(columns.items());
      DIALS_ASSERT(len(items) == ncols);
      object self_obj(self);
      for (std::size_t i = 0; i < ncols; ++i) {
        object item = items[i];
        DIALS_ASSERT(len(item[1]) == nrows);
        std::string name = extract<std::string>(item[0]);
        self_obj[name] = item[1];
      }
      DIALS_ASSERT(self.is_consistent());
    }
  };

  /**
   * An MPL generator to create setitem functions for each type
   */
  template <typename T>
  struct setitem_column_generator {
    class_<T> table_class;
    setitem_column_generator(class_<T> table_class_) : table_class(table_class_) {}
    template <typename U>
    void operator()(const U &x) {
      table_class.def("__setitem__",
                      &setitem_column<T, typename U::value_type>,
                      return_internal_reference<>());
    }
  };

  /**
   * Export the wrapped column table class to python
   */
  template <typename T>
  struct flex_table_wrapper {
    typedef T flex_table_type;
    typedef class_<flex_table_type> class_type;
    typedef typename flex_table_type::mapped_type flex_types;

    static class_type wrap(const char *name) {
      class_type flex_table_class(name);
      flex_table_class.def(init<std::size_t>())
        .def("__init__", make_constructor(&make_flex_table<flex_table_type>))
        .def("types", &types<flex_table_type>)
        .def("has_key", &has_key<flex_table_type>)
        .def("clear", &flex_table_type::clear)
        .def("empty", &flex_table_type::empty)
        .def("resize", &flex_table_type::resize)
        .def("append", &append<flex_table_type>)
        .def("insert", &insert<flex_table_type>)
        .def("extend", &extend<flex_table_type>)
        .def("update", &update<flex_table_type>)
        .def("nrows", &flex_table_type::nrows)
        .def("ncols", &flex_table_type::ncols)
        .def("is_consistent", &flex_table_type::is_consistent)
        .def("size", &flex_table_type::size)
        .def("__len__", &flex_table_type::size)
        .def("__contains__", &has_key<flex_table_type>)
        .def("__getitem__", &getitem_column<flex_table_type>)
        .def("__getitem__", &getitem_row<flex_table_type>)
        .def("__getitem__", &getitem_slice<flex_table_type>)
        .def("__setitem__", &setitem_row<flex_table_type>)
        .def("__setitem__", &setitem_slice<flex_table_type>)
        .def("__delitem__", &delitem_column<flex_table_type>)
        .def("__delitem__", &delitem_row<flex_table_type>)
        .def("__delitem__", &delitem_slice<flex_table_type>)
        .def("__iter__", make_iterator<key_iterator<flex_table_type> >::range())
        .def("cols", make_iterator<column_iterator<flex_table_type> >::range())
        .def("items", make_iterator<column_iterator<flex_table_type> >::range())
        .def("rows", make_iterator<row_iterator<flex_table_type> >::range())
        .def("keys", make_iterator<key_iterator<flex_table_type> >::range())
        .def("select", &select_rows_index<flex_table_type>)
        .def("select", &select_rows_flags<flex_table_type>)
        .def("select", &select_cols_keys<flex_table_type>)
        .def("select", &select_cols_tuple<flex_table_type>)
        .def("select", &select_using_experiment<flex_table_type>)
        .def("select", &select_using_experiments<flex_table_type>)
        .def("set_selected", &set_selected_rows_index<flex_table_type>)
        .def("set_selected", &set_selected_rows_flags<flex_table_type>)
        .def("set_selected", &set_selected_cols_keys<flex_table_type>)
        .def("set_selected", &set_selected_cols_tuple<flex_table_type>)
        .def("del_selected", &del_selected_rows_index<flex_table_type>)
        .def("del_selected", &del_selected_rows_flags<flex_table_type>)
        .def("del_selected", &del_selected_cols_keys<flex_table_type>)
        .def("del_selected", &del_selected_cols_tuple<flex_table_type>)
        .def("reorder", &reorder<flex_table_type>)
        .def("__copy__", &copy<flex_table_type>)
        .def("__deepcopy__", &deepcopy<flex_table_type>)
        //.def("sort", &sort<flex_table_type>, (
        // arg("column"),
        // arg("reverse")=false))
        ;

      // For each column type, create a __setitem__ method to set column data
      boost::mpl::for_each<typename flex_types::types>(
        setitem_column_generator<flex_table_type>(flex_table_class));

      // Return the class
      return flex_table_class;
    }
  };

}}}}  // namespace dials::af::boost_python::flex_table_suite

#endif  // DIALS_FRAMEWORK_TABLE_BOOST_PYTHON_FLEX_TABLE_SUITE_H
