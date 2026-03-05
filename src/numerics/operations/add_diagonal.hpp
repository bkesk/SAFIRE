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

#include "algorithm"
#include <complex>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "add_diagonal_impl.hpp"

#if defined(ENABLE_DEVICE)
#include "numerics/device_kernels/kernels.h"
#endif

namespace math
{

template<typename T, nda::MemoryArrayOfRank<3> A_t>
requires(std::decay_t<A_t>::is_stride_order_C())
void add_diagonal(T alpha, A_t && A)
{
  sfqmc::utils::check(A.extent(1) == A.extent(2), "Size mismatch");
#if defined(ENABLE_DEVICE)
    if constexpr (nda::mem::have_device_compatible_addr_space<A_t>){
      kernels::device::add_diagonal(alpha,A);
    } 
    else
#endif
    {
      auto F = detail::add_diagonal_impl<T,A_t>{alpha,A};
      std::ranges::for_each(nda::range(A.extent(0)),F);
    }
}

}
