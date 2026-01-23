#pragma once

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace math
{

// B(...) = T(A(...)), where T = nda::get_value_t<decltype(B)>
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
requires( nda::get_rank<A_t> == nda::get_rank<B_t> ) 
void copy(A_t const& A, B_t && B) {
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  if constexpr (std::is_same_v<nda::get_value_t<A_t>,nda::get_value_t<B_t>>) {
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t> and nda::get_rank<A_t> < 5) {
      kernels::device::copy(A,B);
    } else
#endif
      B() = A();
  } else {
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t>) {
      kernels::device::copy_cast(A,B);
    } else 
#endif
      B() = A();
  }
}

// B(...) += T(alpha*A(...)), where T = nda::get_value_t<decltype(B)>
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
requires( nda::get_rank<A_t> == nda::get_rank<B_t> )
void accumulate(nda::get_value_t<A_t> alpha, A_t const& A, B_t && B) {
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  if constexpr (std::is_same_v<nda::get_value_t<A_t>,nda::get_value_t<B_t>>) {
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t> and nda::get_rank<A_t> < 5) {
      kernels::device::accumulate(alpha,A,B);
    } else
#endif
      nda::tensor::add(alpha,A,nda::get_value_t<B_t>{1.0},B);
  } else {
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t>) {
      kernels::device::accumulate_cast(alpha,A,B);
    } else 
#endif
      B() += alpha*A();
  }
}

template<nda::MemoryArray A_t>
void scale(nda::get_value_t<A_t> alpha, A_t && A) {
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t> and nda::get_rank<A_t> < 5) {
      kernels::device::scale(alpha,A);
    } else
#endif
      nda::tensor::scale(alpha,A); 
}

template<nda::MemoryArray Arr>
void zero_imag(Arr && A)
{
  using value_type = nda::get_value_t<Arr>;
  if constexpr (nda::is_complex_v<value_type>) {
    if constexpr(nda::mem::on_host<Arr>) {
      for( auto& v : A ) v = value_type(v.real(),0.0);
    } else {
#if defined(ENABLE_DEVICE)
      kernels::device::zero_imag(A);
#else
      sfqmc::utils::check(false,"Error: Found device array without ENABLE_DEVICE.");
#endif
    }
  }
}

}
