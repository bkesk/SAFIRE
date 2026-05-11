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


#include <numbers>
#if defined(__CUDACC__)
#include <cuda/std/limits>
#include <cuda/std/complex>
#else
#include <complex>
#include <limits>
#endif
#include "arch/arch.h"

namespace math
{

namespace detail {

template<typename V1, typename V2>
struct add_diagonal_impl
{
  //using T = typename std::decay_t<V1>::value_type; 
  V1 alpha;
  V2 A;

  __device__
  void operator()(long b)
  { 
#if defined(__CUDACC__)
    using ::cuda::std::complex;
#else
    using std::complex; 
    using std::imag;
    using std::log;
#endif
    for (long i = 0; i < A.extent(1); i++) {
        A(b,i,i) += alpha;
    }
  };
};

}

}
