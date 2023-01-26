#include <scitbx/array_family/boost_python/flex_fwd.h>

#include <scitbx/array_family/boost_python/flex_wrapper.h>
#include <scitbx/serialization/single_buffered.h>
#include <scitbx/matrix/transpose_multiply.h>
#include <scitbx/math/utils.h>
#include <scitbx/array_family/tiny_types.h>
#include <boost/python/make_constructor.hpp>
#include <boost/python/args.hpp>
#include <boost/python/return_arg.hpp>
#include <scitbx/array_family/boost_python/flex_helpers.h>

namespace scitbx { namespace serialization { namespace single_buffered {

  inline char* to_string(char* start, scitbx::af::int6 const& value) {
    return to_string(
      to_string(
        to_string(to_string(to_string(to_string(start, value[0]), value[1]), value[2]),
                  value[3]),
        value[4]),
      value[5]);
  }

  template <>
  struct from_string<scitbx::af::int6> {
    from_string(const char* start) {
      end = start;
      for (std::size_t i = 0; i < 6; i++) {
        from_string<int> proxy(end);
        value[i] = proxy.value;
        end = proxy.end;
      }
    }

    scitbx::af::int6 value;
    const char* end;
  };

}}}  // namespace scitbx::serialization::single_buffered

#include <scitbx/array_family/boost_python/flex_pickle_single_buffered.h>

namespace scitbx { namespace af { namespace boost_python {

  template <>
  struct flex_default_element<scitbx::af::int6> {
    static scitbx::af::int6 get() {
      return scitbx::af::int6(0, 0, 0, 0, 0, 0);
    }
  };

}}}  // namespace scitbx::af::boost_python

namespace dials { namespace af {
  namespace {

    boost::python::tuple parts(
      scitbx::af::versa<scitbx::af::int6, scitbx::af::flex_grid<> > const& O) {
      scitbx::af::tiny<scitbx::af::versa<int, scitbx::af::flex_grid<> >, 6> result;
      std::size_t n = O.size();
      for (std::size_t i = 0; i < 6; i++) {
        result[i].resize(O.accessor());
        for (std::size_t j = 0; j < n; j++) {
          result[i][j] = O[j][i];
        }
      }
      return boost::python::make_tuple(
        result[0], result[1], result[2], result[3], result[4], result[5]);
    }

    scitbx::af::flex<scitbx::af::int6>::type* join(
      scitbx::af::const_ref<int> const& a,
      scitbx::af::const_ref<int> const& b,
      scitbx::af::const_ref<int> const& c,
      scitbx::af::const_ref<int> const& d,
      scitbx::af::const_ref<int> const& e,
      scitbx::af::const_ref<int> const& f) {
      SCITBX_ASSERT(a.size() == b.size());
      SCITBX_ASSERT(a.size() == c.size());
      SCITBX_ASSERT(a.size() == d.size());
      SCITBX_ASSERT(a.size() == e.size());
      SCITBX_ASSERT(a.size() == f.size());
      scitbx::af::shared<scitbx::af::int6> result((scitbx::af::reserve(a.size())));
      for (std::size_t i = 0; i < a.size(); i++) {
        result.push_back(scitbx::af::int6(a[i], b[i], c[i], d[i], e[i], f[i]));
      }
      return new scitbx::af::flex<scitbx::af::int6>::type(result, result.size());
    }

    scitbx::af::flex<scitbx::af::int6>::type* from_int(
      scitbx::af::const_ref<int> const& x) {
      SCITBX_ASSERT(x.size() % 6 == 0);
      std::size_t result_size = x.size() / 6;
      scitbx::af::shared<scitbx::af::int6> result(result_size);
      const int* d = x.begin();
      for (std::size_t i = 0; i < result_size; i++) {
        for (std::size_t j = 0; j < 6; ++j) {
          result[i][j] = *d++;
        }
      }
      return new scitbx::af::flex<scitbx::af::int6>::type(result, result.size());
    }

    scitbx::af::flex_int as_int(scitbx::af::flex<scitbx::af::int6>::type const& a) {
      SCITBX_ASSERT(a.accessor().is_trivial_1d());
      scitbx::af::flex_int result(a.size() * 6, scitbx::af::init_functor_null<int>());
      int* r = result.begin();
      scitbx::af::const_ref<scitbx::af::int6> a_ref = a.const_ref().as_1d();
      for (std::size_t i = 0; i < a_ref.size(); i++) {
        for (std::size_t j = 0; j < 6; j++) {
          *r++ = a_ref[i][j];
        }
      }
      return result;
    }

  }  // namespace

  namespace boost_python {

    void export_flex_int6() {
      using namespace boost::python;
      using boost::python::arg;
      typedef scitbx::af::boost_python::flex_wrapper<scitbx::af::int6> f_w;
      f_w::plain("int6")
        .def_pickle(
          scitbx::af::boost_python::flex_pickle_single_buffered<
            scitbx::af::int6,
            6 * scitbx::af::boost_python::pickle_size_per_element<int>::value>())
        .def("__init__", make_constructor(join))
        .def("__init__", make_constructor(from_int))
        .def("parts", parts)
        .def("as_int", as_int);
      ;
    }

  }  // namespace boost_python
}}   // namespace dials::af
