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

#include <cmath>
#include <complex>
#include <algorithm>
#include "configuration.hpp"

#include "nda/nda.hpp"
#include "numerics/device_kernels/kernels.h"

namespace math
{

/*
 * Calculates s(i) = log( sum_p f(p,i) ), where a(p,i) = log( f(p,i) ) 
 * Uses the formula, which avoids computing f(p,i) directly:
 * s(i) = log( O(i) ) + log( sum_p exp(log(f(p,i)) - log(O(i))) ), where O(i) = max_p(a(p,i))
 *      = log( O(i) ) + log( sum_p exp(a(p,i) - log(O(i))) ) 
 * If n==0, we sum over the first axis, otherwise we sum over the second.
 */
template <nda::MemoryMatrix A, nda::MemoryVector S>
void log_of_sum(int n, A && a, S && s) 
requires(nda::have_same_value_type_v<A,S> and nda::is_blas_lapack_v<nda::get_value_t<A>> and
         nda::mem::have_compatible_addr_space<A,S> )
{
  using value_t = nda::get_value_t<A>;
  if constexpr(nda::mem::have_device_compatible_addr_space<A,S>) {
    // call kernel
    sfqmc::utils::check(false,"finish");
  } else {
    if(n==0) {
      //s(i) = log( O(i) ) + log( sum_p exp(a(p,i) - log(O(i))) ) 
      sfqmc::utils::check(s.size() >= a.extent(1), "Size mismatch.");
      for(int i=0; i<a.extent(1); ++i) {
        auto ai = a(nda::range::all,i);
        value_t log_m = *std::max_element(ai.begin(),ai.end()); 
        value_t sum = 0.0;
        for(int p=0; p<a.extent(0); ++p) 
          sum += std::exp( ai(p) - log_m ); 
        s(i) = log_m + std::log(sum);
      } 
    } else {
      //s(i) = log( O(i) ) + log( sum_p exp(a(i,p) - log(O(i))) ) 
      sfqmc::utils::check(s.size() >= a.extent(0), "Size mismatch.");
      for(int i=0; i<a.extent(0); ++i) {
        auto ai = a(i,nda::range::all);
        value_t log_m = *std::max_element(ai.begin(),ai.end());
        value_t sum = 0.0;
        for(int p=0; p<a.extent(1); ++p) 
          sum += std::exp( ai(p) - log_m );
        s(i) = log_m + std::log(sum);
      }
    } 
  }
}

}
