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
#include <cctbx/uctbx.h>
#include <cctbx/miller.h>
#include <dials/error.h>

namespace dials { namespace af { namespace boost_python {

  using cctbx::uctbx::unit_cell;

  scitbx::af::shared<double> d(
    const scitbx::af::const_ref<unit_cell> &self,
    const scitbx::af::const_ref<cctbx::miller::index<> > &hkl,
    const scitbx::af::const_ref<std::size_t> index) {
    DIALS_ASSERT(index.size() == hkl.size());
    scitbx::af::shared<double> result(hkl.size());
    for (std::size_t i = 0; i < hkl.size(); ++i) {
      std::size_t j = index[i];
      DIALS_ASSERT(j < self.size());
      result[i] = self[j].d(hkl[i]);
    }
    return result;
  }

  void export_flex_unit_cell() {
    using namespace boost::python;
    using boost::python::arg;
    typedef scitbx::af::boost_python::flex_wrapper<unit_cell> f_w;
    f_w::plain("unit_cell").def("d", &d, (arg("hkl"), arg("id")));
    ;
  }

}}}  // namespace dials::af::boost_python
