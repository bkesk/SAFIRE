#pragma once

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"
#include "numerics/operations/split_singular_vals_impl.hpp"

namespace kernels::device
{

//FIX : add version for 1 D matrix, but ovlp is a vector of size nbatch

namespace detail
{
  template<typename V1, typename V2, typename V3, typename V4, typename V5>
  void splitDmatrix_impl(V1 const& A, V2& B, V3& C, V4& res, V5 const& scl);

}

template<nda::MemoryMatrix A_t, nda::MemoryMatrix B_t, nda::MemoryMatrix C_t, nda::MemoryVector V_t,
                     nda::MemoryVector S_t>
requires(std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C() and
         std::decay_t<C_t>::is_stride_order_C() and nda::mem::have_compatible_addr_space<A_t,B_t,C_t,V_t,S_t>)
void splitDmatrix(A_t const& A, B_t && B, C_t && C, V_t && log_det, S_t const& scl) {
  using T = nda::get_value_t<V_t>;
  //static_assert(nda::is_complex_v<nda::get_value_t<V>>,
  //              "log_determinant_from_getrf expects complex numbers.");
  sfqmc::utils::check(A.shape() == B.shape(), "Size mismatch");
  sfqmc::utils::check(A.shape() == C.shape(), "Size mismatch");
  sfqmc::utils::check(A.extent(0) == log_det.extent(0), "Size mismatch");
  sfqmc::utils::check(A.extent(0) == scl.extent(0), "Size mismatch");
  sfqmc::utils::check(A.extent(1) == A.extent(2), "Size mismatch");

  auto A_b = to_basic_layout(A());
  auto B_b = to_basic_layout(B());
  auto C_b = to_basic_layout(C());
  auto log_det_b = to_basic_layout(log_det());
  auto scl_b = to_basic_layout(scl());

  detail::splitDmatrix_impl(A_b,B_b,C_b,log_det_b,scl_b);
}

} //kernels::device
