#pragma once

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"
#include "numerics/operations/determinants_impl.hpp"

namespace kernels::device
{

namespace detail
{
  template<typename V1, typename V2, typename V3>
  void log_determinant_from_getrf_impl(V1 const& a, V2 const& ipiv, V3& res);

  template<typename V1, typename V2, typename V3>
  void log_determinant_from_geqrf_impl(V1 const& a, V2& ipiv, V3& res);
}

template<nda::MemoryArrayOfRank<3> A, nda::MemoryMatrix IPIV, nda::MemoryVector V>
requires(std::decay_t<A>::is_stride_order_C() and std::decay_t<IPIV>::is_stride_order_C() and
         nda::mem::have_compatible_addr_space<A,IPIV,V>)
void log_determinant_from_getrf(A const& a, IPIV const& ipiv, V && log_det) {
  using T = nda::get_value_t<V>;
  static_assert(nda::is_complex_v<nda::get_value_t<V>>,
                "log_determinant_from_getrf expects complex numbers.");
  sfqmc::utils::check(a.extent(0) == ipiv.extent(0), "Size mismatch");
  sfqmc::utils::check(a.extent(0) == log_det.extent(0), "Size mismatch");
  sfqmc::utils::check(a.extent(1) == ipiv.extent(1), "Size mismatch");
  sfqmc::utils::check(a.extent(1) == a.extent(2), "Size mismatch");

  auto a_b = to_basic_layout(a());
  auto ipiv_b = to_basic_layout(ipiv());
  auto log_det_b = to_basic_layout(log_det());

  detail::log_determinant_from_getrf_impl(a_b,ipiv_b,log_det_b);
}

template<nda::MemoryArrayOfRank<3> A, nda::MemoryMatrix S, nda::MemoryVector V>
requires(std::decay_t<A>::is_stride_order_C() and std::decay_t<S>::is_stride_order_C() and
         nda::mem::have_compatible_addr_space<A,S,V>)
void log_determinant_from_geqrf(A const& a, S && scl, V && log_det) {
  using T = nda::get_value_t<V>;
  static_assert(nda::is_complex_v<nda::get_value_t<V>>,
                "log_determinant_from_getrf expects complex numbers.");
  sfqmc::utils::check(a.extent(0) == log_det.extent(0), "Size mismatch");
  sfqmc::utils::check(a.extent(0) == scl.extent(0), "Size mismatch");
  sfqmc::utils::check(scl.extent(1) >= std::min(a.extent(1),a.extent(2)), "Size mismatch");

  auto a_b = to_basic_layout(a());
  auto scl_b = to_basic_layout(scl());
  auto log_det_b = to_basic_layout(log_det());

  detail::log_determinant_from_geqrf_impl(a_b,scl_b,log_det_b);
}

} //kernels::device
