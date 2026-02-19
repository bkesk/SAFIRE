#pragma once

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"
#include "numerics/operations/add_diagonal_impl.hpp"

namespace kernels::device
{

namespace detail
{
  template<typename V1, typename V2>
  void add_diagonal_impl(V1 alpha, V2& a);
}

template<typename V, nda::MemoryArrayOfRank<3> A>
requires(std::decay_t<A>::is_stride_order_C())
void add_diagonal(V alpha, A&& a) {
  using T = nda::get_value_t<V>;
  sfqmc::utils::check(a.extent(1) == a.extent(2), "add_diagonal requires square matrices");

  auto a_b = to_basic_layout(a());

  detail::add_diagonal_impl(alpha,a_b);
}

} // namespace kernels::device