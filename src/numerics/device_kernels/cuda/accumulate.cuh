
#pragma once

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"

namespace kernels::device
{

namespace detail
{

template<typename A, typename B> 
void accumulate_impl(nda::get_value_t<A> alpha, A const& a, B & b);

template<typename A> 
void scale_impl(nda::get_value_t<A> alpha, A& a);

template<typename A, typename B> 
void copy_impl(A const& a, B & b);

}

/*
 *   B(...) += alpha * A(...)
 */
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
void accumulate(nda::get_value_t<A_t> alpha, A_t const& A, B_t && B)
requires( nda::get_rank<A_t> == nda::get_rank<B_t> ) 
{
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  if(A.is_contiguous() and B.is_contiguous() and nda::get_rank<A_t> > 1) {
    kernels::device::accumulate(alpha,nda::flatten(A),nda::flatten(B));
    return;
  }
  // careful with ranks here 
  auto A_b = to_basic_layout(A());
  auto B_b = to_basic_layout(B());
  if constexpr (nda::get_rank<A_t> < 5)
    detail::accumulate_impl(alpha,A_b,B_b);
  else
    sfqmc::utils::check(false, "Calling accumulate_cast with rank > 4 array. Finish");
} 

/*
 *   A(...) *= alpha
 */
template<nda::MemoryArray A_t>
void scale(nda::get_value_t<A_t> alpha, A_t && A)
{
  if(A.is_contiguous() and nda::get_rank<A_t> > 1) { 
    kernels::device::scale(alpha,nda::flatten(A));
    return;
  }
  // careful with ranks here 
  auto A_b = to_basic_layout(A());
  if constexpr (nda::get_rank<A_t> < 5)
    detail::scale_impl(alpha,A_b);
  else
    sfqmc::utils::check(false, "Calling accumulate_cast with rank > 2 array. Finish");
}

/*
 *   B(...) = A(...) 
 */
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
void copy(A_t const& A, B_t && B)
requires( nda::get_rank<A_t> == nda::get_rank<B_t> ) 
{
  if(A.is_contiguous() and B.is_contiguous() and nda::get_rank<A_t> > 1) {
    kernels::device::copy(nda::flatten(A),nda::flatten(B));
    return;
  }
  // careful with ranks here 
  auto A_b = to_basic_layout(A());
  auto B_b = to_basic_layout(B());
  detail::copy_impl(A_b,B_b);
}


} // namespace kernels::device

