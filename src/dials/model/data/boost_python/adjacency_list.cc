/*
 * adjacency_list.cc
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
#include <boost/python/iterator.hpp>
#include <boost_adaptbx/std_pair_conversion.h>
#include <dials/model/data/adjacency_list.h>

namespace dials { namespace model { namespace boost_python {

  using namespace boost::python;

  struct adjacent_vertices_iterator {
    AdjacencyList::edge_iterator first_;
    AdjacencyList::edge_iterator last_;

    adjacent_vertices_iterator(AdjacencyList::edge_iterator first,
                               AdjacencyList::edge_iterator last)
        : first_(first), last_(last) {}

    std::size_t next() {
      if (first_ == last_) {
        PyErr_SetString(PyExc_StopIteration, "No more data.");
        boost::python::throw_error_already_set();
      }
      std::size_t result = first_->second;
      first_++;
      return result;
    }

    static object iter(object const &o) {
      return o;
    }
  };

  template <typename ClassIter>
  struct iterator_wrapper {
    static void wrap(const char *name) {
      class_<ClassIter>(name, no_init)
        .def("next", &ClassIter::next)
        .def("__next__", &ClassIter::next)
        .def("__iter__", &ClassIter::iter);
    }
  };

  static AdjacencyList::edge_iterator edges_begin(const AdjacencyList &list) {
    return list.edges().first;
  }

  static AdjacencyList::edge_iterator edges_end(const AdjacencyList &list) {
    return list.edges().second;
  }

  adjacent_vertices_iterator make_adjacent_vertices_iterator(const AdjacencyList &self,
                                                             std::size_t index) {
    return adjacent_vertices_iterator(self.edges(index).first,
                                      self.edges(index).second);
  }

  void export_adjacency_list() {
    class_<AdjacencyList::edge_descriptor>("EdgeDescriptor", no_init);

    iterator_wrapper<adjacent_vertices_iterator>::wrap("AdjacentVerticesIter");

    class_<AdjacencyList>("AdjacencyList", no_init)
      .def("source", &AdjacencyList::source)
      .def("target", &AdjacencyList::target)
      .def("adjacent_vertices", &make_adjacent_vertices_iterator)
      .def("edges", boost::python::range(edges_begin, edges_end))
      .def("add_edge", &AdjacencyList::add_edge)
      .def("num_vertices", &AdjacencyList::num_vertices)
      .def("num_edges", &AdjacencyList::num_edges);
  }

}}}  // namespace dials::model::boost_python
