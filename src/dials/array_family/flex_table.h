/*
 * flex_table.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ARRAY_FAMILY_FLEX_TABLE_H
#define DIALS_ARRAY_FAMILY_FLEX_TABLE_H

#include <algorithm>
#include <map>
#include <memory>
#include <vector>
#include <boost/python.hpp>
#include <boost/variant.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/remove_if.hpp>
#include <boost/mpl/transform.hpp>
#include <dials/error.h>
#include <dials/array_family/scitbx_shared_and_versa.h>

namespace dials { namespace af {

  using namespace boost::python;

  /**
   * Init to zero
   */
  template <typename ElementType>
  ElementType init_zero() {
    return ElementType();
  }

  /**
   * A class to represent a column-centric table. I.e. a table in which the
   * data is represented as a list of columns. It is created with a variant
   * type of column data. It can be instantiated as follows:
   *
   * typedef flex_type_generator<int, double>::type flex_types;
   * flex_table<flex_types> table;
   *
   * The columns can be accessed as values in a std::map as follows:
   *
   * af::shared<int> col = table["column"];
   */
  template <typename VarientType>
  class flex_table {
  public:
    typedef std::map<std::string, VarientType> map_type;
    typedef typename map_type::key_type key_type;
    typedef typename map_type::mapped_type mapped_type;
    typedef typename map_type::value_type map_value_type;
    typedef typename map_type::iterator iterator;
    typedef typename map_type::const_iterator const_iterator;
    typedef typename map_type::size_type size_type;

  private:
    /**
     * Visitor to copy a column from mapped type
     */
    struct copy_column_visitor : public boost::static_visitor<void> {
      flex_table *t_;
      key_type k_;
      copy_column_visitor(flex_table *t, key_type &k) : t_(t), k_(k) {}
      template <typename T>
      void operator()(const af::shared<T> &other_column) const {
        size_type n = t_->nrows();
        std::shared_ptr<map_type> table = t_->table_;
        iterator it = table->lower_bound(k_);
        if (it == table->end() || table->key_comp()(k_, it->first)) {
          it = table->insert(
            it, map_value_type(k_, mapped_type(af::shared<T>(n, init_zero<T>()))));
        }
        af::shared<T> this_column = boost::get<af::shared<T> >(it->second);
        DIALS_ASSERT(this_column.size() == other_column.size());
        for (std::size_t i = 0; i < this_column.size(); ++i) {
          this_column[i] = other_column[i];
        }
      }
    };

    /**
     * operator[] proxy to aid in returning and casting elements.
     */
    struct proxy {
      flex_table *t_;
      key_type k_;

      proxy(flex_table *t, key_type k) : t_(t), k_(k) {}

      /**
       * Assign a column.
       */
      template <typename T>
      void operator=(const af::shared<T> other_column) {
        DIALS_ASSERT(other_column.size() == t_->nrows());
        af::shared<T> this_column = (af::shared<T>)(*this);
        for (std::size_t i = 0; i < this_column.size(); ++i) {
          this_column[i] = other_column[i];
        }
      }

      /**
       * Assign a column
       */
      void operator=(const mapped_type &item) {
        copy_column_visitor visitor(t_, k_);
        item.apply_visitor(visitor);
      }

      /**
       * Assign from another proxy
       */
      void operator=(const proxy &p) {
        (*this) = p.variant();
      }

      /**
       * Cast the element to the desired column data type. If no element is
       * present, a new element with the desired type is created and returned.
       * Otherwise the current element is returned. If the types don't match,
       * an exception is raised.
       */
      template <typename T>
      operator af::shared<T>() const {
        size_type n = t_->nrows();
        std::shared_ptr<map_type> table = t_->table_;
        iterator it = table->lower_bound(k_);
        if (it == table->end() || table->key_comp()(k_, it->first)) {
          it = table->insert(
            it, map_value_type(k_, mapped_type(af::shared<T>(n, init_zero<T>()))));
        }
        return boost::get<af::shared<T> >(it->second);
      }

      /**
       * Helper to convert shared to ref
       */
      template <typename T>
      operator af::ref<T>() const {
        af::shared<T> result = (af::shared<T>)(*this);
        return result.ref();
      }

      /**
       * Helper to convert shared to const ref
       */
      template <typename T>
      operator af::const_ref<T>() const {
        af::shared<T> result = (af::shared<T>)(*this);
        return result.const_ref();
      }

      /**
       * Return the mapped variant type at the given element directly.
       */
      mapped_type variant() const {
        std::shared_ptr<map_type> table = t_->table_;
        iterator it = table->find(k_);
        if (it == table->end()) {
          PyErr_Format(PyExc_KeyError, "Unknown column '%s'", k_.c_str());
          boost::python::throw_error_already_set();
        }
        return it->second;
      }
    };

    /** Get the size of each column */
    struct size_visitor : boost::static_visitor<size_type> {
      template <typename T>
      size_type operator()(const T &v) const {
        return v.size();
      }
    };

    /** Resize each column */
    struct resize_visitor : boost::static_visitor<void> {
      size_type n_;
      resize_visitor(size_type n) : n_(n) {}
      template <typename T>
      void operator()(T &v) const {
        v.resize(n_);
      }
    };

    /** Insert an element into each column */
    struct insert_visitor : boost::static_visitor<void> {
      size_type pos, n;
      insert_visitor(size_type pos_, size_type n_) : pos(pos_), n(n_) {}
      template <typename T>
      void operator()(T &v) const {
        v.insert(v.begin() + pos, n, typename T::value_type());
      }
    };

    /** Erase an element from each column */
    struct erase_visitor : boost::static_visitor<void> {
      size_type pos, n;
      erase_visitor(size_type pos_, size_type n_) : pos(pos_), n(n_) {}
      template <typename T>
      void operator()(T &v) const {
        typename T::iterator first = v.begin() + pos;
        typename T::iterator last = first + n;
        v.erase(first, last);
      }
    };

  public:
    /** Initialise the table */
    flex_table() : table_(std::make_shared<map_type>()), default_nrows_(0) {}

    /**
     * Initialise the table to a certain size
     * @param n The size to initialise to
     */
    flex_table(size_type n) : table_(std::make_shared<map_type>()), default_nrows_(n) {}

    /**
     * Virtual destructor
     */
    virtual ~flex_table() {}

    /**
     * Access a column by key
     * @param key The column name
     * @returns The proxy object to access the column data
     */
    proxy operator[](const key_type &key) {
      return proxy(this, key);
    }

    /**
     * Access a column by key
     * @param key The column name
     * @returns The column.
     */
    template <typename T>
    af::shared<T> get(const key_type &key) {
      af::shared<T> result = proxy(this, key);
      return result;
    }

    /**
     * Access a column by key
     * @param key The column name
     * @returns The column.
     */
    template <typename T>
    af::shared<T> get(const key_type &key) const {
      const_iterator it = find(key);
      DIALS_ASSERT(it != end());
      return boost::get<af::shared<T> >(it->second);
    }

    /** @returns An iterator to the beginning of the column map */
    iterator begin() {
      return table_->begin();
    }

    /** @returns An iterator to the end of the column map */
    iterator end() {
      return table_->end();
    }

    /** @returns A const iterator to the beginning of the column map */
    const_iterator begin() const {
      return table_->begin();
    }

    /** @returns A const iterator to the end of the column map */
    const_iterator end() const {
      return table_->end();
    }

    /** @returns The number of rows in the table */
    size_type nrows() const {
      size_type size = default_nrows_;
      if (!empty()) {
        size_visitor visitor;
        const_iterator it = begin();
        size = it->second.apply_visitor(visitor);
        for (++it; it != end(); ++it) {
          if (it->second.apply_visitor(visitor) != size) {
            throw DIALS_ERROR("Column sizes are inconsistent");
          }
        }
      }
      return size;
    }

    /** @returns The number of columns in the table */
    size_type ncols() const {
      return table_->size();
    }

    /** @returns The number of columns in the table */
    size_type size() const {
      return nrows();
    }

    /** @returns Is the table empty */
    bool empty() const {
      return table_->empty();
    }

    /** @returns Are the column sizes consistent */
    bool is_consistent() const {
      if (!empty()) {
        size_visitor visitor;
        const_iterator it = begin();
        size_type size = it->second.apply_visitor(visitor);
        for (++it; it != end(); ++it) {
          if (it->second.apply_visitor(visitor) != size) {
            return false;
          }
        }
      }
      return true;
    }

    /** @returns The number of columns matching the key (0 or 1) */
    size_type count(const key_type &key) const {
      return table_->count(key);
    }

    /**
     * Find a column matching the key
     * @param key The column name
     * @returns An iterator to the column
     */
    iterator find(const key_type &key) {
      return table_->find(key);
    }

    /**
     * Find a column matching the key
     * @param key The column name
     * @returns A const iterator to the column
     */
    const_iterator find(const key_type &key) const {
      return table_->find(key);
    }

    /**
     * Resize the columns to the given size
     * @param n The size to resize to
     */
    void resize(size_type n) {
      DIALS_ASSERT(is_consistent());
      resize_visitor visitor(n);
      for (iterator it = begin(); it != end(); ++it) {
        it->second.apply_visitor(visitor);
      }
      DIALS_ASSERT(is_consistent());
      default_nrows_ = n;
    }

    /**
     * Insert an element at the given position in each column
     * @param pos The position to insert at
     */
    void insert(size_type pos) {
      insert(pos, 1);
    }

    /**
     * Insert a number of elements at the given position in each column
     * @param pos The position to insert at
     * @param n The number of elements to insert
     */
    void insert(size_type pos, size_type n) {
      size_type nr = nrows();
      DIALS_ASSERT(pos <= nr);
      insert_visitor visitor(pos, n);
      for (iterator it = begin(); it != end(); ++it) {
        it->second.apply_visitor(visitor);
      }
      DIALS_ASSERT(is_consistent());
      default_nrows_ = nr + n;
    }

    /**
     * Erase an element from all the columns
     * @param pos The position of the element
     */
    void erase(size_type pos) {
      erase(pos, 1);
    }

    /**
     * Erase a number of elements from all the columns
     * @param pos The position of the element
     * @param n The number of elements to erase
     */
    void erase(size_type pos, size_type n) {
      size_type nr = nrows();
      DIALS_ASSERT(pos + n <= nr);
      erase_visitor visitor(pos, n);
      for (iterator it = begin(); it != end(); ++it) {
        it->second.apply_visitor(visitor);
      }
      DIALS_ASSERT(is_consistent());
      default_nrows_ = nr - n;
    }

    /**
     * Erase a column from the table.
     * @param key The column name
     * @returns The number of columns removed
     */
    size_type erase(const key_type &key) {
      return table_->erase(key);
    }

    /** Clear the table */
    void clear() {
      table_->clear();
      resize(0);
    }

    /** @returns Does the table contain the key. */
    bool contains(const key_type &key) const {
      const_iterator it = find(key);
      return it != end();
    }

  private:
    std::shared_ptr<map_type> table_;
    size_type default_nrows_;
  };

  struct null_type {};

  template <typename T>
  struct is_null_type : public boost::mpl::bool_<false> {};

  template <>
  struct is_null_type<null_type> : public boost::mpl::bool_<true> {};

  /**
   * A class to help in generating a variant type for use in the flex_table
   * class. Given a variadic template list of types it will convert the list
   * into list of af::shared<T> types and then export a variant.
   *
   * For example:
   *  flex_type_generator<int, double, std::string>::type
   *
   * Becomes:
   *
   *  variant<
   *    af::shared<int>,
   *    af::shared<double>,
   *    af::shared<std::string>
   *  >
   */
  template <typename T0,
            typename T1 = null_type,
            typename T2 = null_type,
            typename T3 = null_type,
            typename T4 = null_type,
            typename T5 = null_type,
            typename T6 = null_type,
            typename T7 = null_type,
            typename T8 = null_type,
            typename T9 = null_type,
            typename T10 = null_type,
            typename T11 = null_type,
            typename T12 = null_type>
  class flex_type_generator {
  private:
    template <typename T>
    struct create_flex_type {
      typedef af::shared<T> type;
    };

    // MPL List of all input types
    typedef boost::mpl::list<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>
      all_types;

    // Remove any types if they are null
    typedef typename boost::mpl::remove_if<
      all_types,
      typename boost::mpl::lambda<is_null_type<boost::mpl::_1> >::type>::type
      valid_types;

    // Create a list of af::shared<T> types
    typedef typename boost::mpl::transform<
      valid_types,
      typename boost::mpl::lambda<create_flex_type<boost::mpl::_1> >::type>::type
      flex_types;

  public:
    // Expose the variant type
    typedef typename boost::make_variant_over<flex_types>::type type;

    // Expose the individual types
    typedef typename boost::make_variant_over<valid_types>::type data_type;
  };

}}  // namespace dials::af

#endif  // DIALS_ARRAY_FAMILY_FLEX_TABLE_H
