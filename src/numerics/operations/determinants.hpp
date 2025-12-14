////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <complex>
#include "configuration.hpp"

#include "nda/nda.hpp"

namespace math
{

// Important to accumulate on res/ovlp!!!
namespace detail {
template<typename T, typename A, typename I, typename V>  
void log_determinant_from_getrf_impl(long n, long batchSize, A const& a, I const& pivot, V && res) { 
  static const auto pi = imag(std::log(T(-1)));
  T minus = T(-1.0);
  // this should be remove_complex<T>
  T small = std::numeric_limits<double>::min();
  for (int b = 0; b != batchSize; ++b ) { 
    for (int i = 0, ip = 1; i != n; i++, ip++) {
      if(pivot(b,i) == ip)
        res(b) += std::log(static_cast<T>(a(b,i,i))+small);
      else
        res(b) += std::log(minus*static_cast<T>(a(b,i,i))+small);
    }
    // bring imaginaty part to [-pi,pi]
    if(imag(res(b)) > pi) {
      while(imag(res(b)) > pi) {
        auto v = imag(res(b));
        res(b).imag(v-2*pi);
      }
    } else if (imag(res(b)) < -pi) { 
      while(imag(res(b)) < -pi) {
        auto v = imag(res(b));
        res(b).imag(v+2*pi);
      }
    } 
  }
}

template<typename T, typename A, typename I, typename V>
void log_determinant_from_geqrf_impl(long n, long batchSize, A const& a, I && scl, V && res) {
  T small = std::numeric_limits<double>::min();
  for (int b = 0; b != batchSize; ++b ) {
    for (int i = 0; i != n; i++) {
      if(std::real(a(b,i,i)) < 0)
        scl(b,i) = T(-1.);
      else
        scl(b,i) = T(1.);;
      res(b) += std::log(static_cast<T>(scl(b,i) * a(b,i,i))+small);
    }
  }
}

}

template<nda::MemoryArrayOfRank<3> A, nda::MemoryMatrix IPIV, nda::MemoryVector V>
requires(std::decay_t<A>::is_stride_order_C() and std::decay_t<IPIV>::is_stride_order_C() and
         nda::mem::have_compatible_addr_space<A,IPIV,V>)
void log_determinant_from_getrf(A const& a, IPIV const& ipiv, V && log_det) {
  using T = nda::get_value_t<V>;
  static_assert(nda::is_complex_v<T>, "log_determinant_from_getrf expects complex numbers.");
  sfqmc::utils::check(a.extent(0) == ipiv.extent(0), "Size mismatch");
  sfqmc::utils::check(a.extent(0) == log_det.extent(0), "Size mismatch");
  sfqmc::utils::check(a.extent(1) == ipiv.extent(1), "Size mismatch");
  sfqmc::utils::check(a.extent(1) == a.extent(2), "Size mismatch");
  if constexpr (nda::mem::have_device_compatible_addr_space<A,IPIV,V>) {
//    device::detail::log_determinant_from_getrf_impl(a,ipiv,log_det);
  } else {
    detail::log_determinant_from_getrf_impl<T>(a.extent(1),a.extent(0),a,ipiv,log_det);
  }
} 

template<nda::MemoryArrayOfRank<2> A, nda::MemoryVector IPIV>
requires(std::decay_t<A>::is_stride_order_C() and std::decay_t<IPIV>::is_stride_order_C() and
         nda::mem::have_compatible_addr_space<A,IPIV>)
void log_determinant_from_getrf(A const& aM, IPIV const& ipiv_v, nda::get_value_t<A>& val) {
  using T = nda::get_value_t<A>;
  static_assert(nda::is_complex_v<T>, "log_determinant_from_getrf expects complex numbers.");
  sfqmc::utils::check(aM.extent(0) == ipiv_v.extent(0), "Size mismatch");
  sfqmc::utils::check(aM.extent(0) == aM.extent(1), "Size mismatch");
  sfqmc::utils::check(aM.is_contiguous(), "Expects contiguos array. Fix if needed!");
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A>();
  auto a = nda::reshape(aM,std::array<long,3>{1,aM.extent(0),aM.extent(1)});
  memory::array_view<MEM,const nda::get_value_t<IPIV>,2> ipiv(std::array<long,2>{1,ipiv_v.size()},ipiv_v.data());
  memory::array_view<MEM,T,1> log_det(std::array<long,1>{1},&val);
  if constexpr (nda::mem::have_device_compatible_addr_space<A,IPIV>) {
//    device::detail::log_determinant_from_getrf_impl(a,ipiv,log_det);
  } else {
    detail::log_determinant_from_getrf_impl<T>(a.extent(1),a.extent(0),a,ipiv,log_det);
  }
}

template<nda::MemoryArrayOfRank<3> A, nda::MemoryMatrix S, nda::MemoryVector V>
requires(std::decay_t<A>::is_stride_order_C() and std::decay_t<S>::is_stride_order_C() and
         nda::mem::have_compatible_addr_space<A,S,V>)
void log_determinant_from_geqrf(A const& a, S && scl, V && log_det) {
  using T = nda::get_value_t<V>;
  static_assert(nda::is_complex_v<T>, "log_determinant_from_geqrf expects complex numbers.");
  sfqmc::utils::check(a.extent(0) == log_det.extent(0), "Size mismatch");
  sfqmc::utils::check(a.extent(0) == scl.extent(0), "Size mismatch");
  sfqmc::utils::check(scl.extent(1) >= std::min(a.extent(1),a.extent(2)), "Size mismatch");
  if constexpr (nda::mem::have_device_compatible_addr_space<A,S,V>) {
//    device::detail::log_determinant_from_geqrf_impl(a,scl,log_det);
  } else {
    detail::log_determinant_from_geqrf_impl<T>(std::min(a.extent(1),a.extent(2)),a.extent(0),a,scl,log_det);
  }
}


}
