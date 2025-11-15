
#pragma once

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"

namespace kernels::device
{

namespace detail
{

template<typename A, typename B> 
void copy_cast_impl(A const& a, B & b);

}

/*
 *   B(...) = T(A(...)) 
 */
template<nda::MemoryArray A_t, nda::MemoryArray B_t> 
void copy_cast(A_t const& A, B_t && B)
requires( nda::get_rank<A_t> == nda::get_rank<B_t> and
       ((std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()) or
        (std::decay_t<A_t>::is_stride_order_Fortran() and std::decay_t<B_t>::is_stride_order_Fortran())))
{
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  if(A.is_contiguous() and B.is_contiguous()) {
    copy_cast(flatten(A),flatten(B));
    return; 
  }
  // careful with ranks here 
  auto A_b = to_basic_layout(A()); 
  auto B_b = to_basic_layout(B()); 
  if constexpr (nda::get_rank<A> < 3)
    detail::copy_copy_impl(A_b,B_b);  
  else
    utils::check(false, "Calling copy_cast with rank > 2 array. Finish");
}

} // namespace kernels::device

#endif
