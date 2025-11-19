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

#include "configuration.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"

namespace memory
{

#if defined(ENABLE_DEVICE)

  template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
  using shared_array = std::conditional_t<MEM==HOST_MEMORY, 
                                          math::shm::shared_array<nda::array_view<T,N,Layout>>,
                                          memory::array<MEM,T,N,Layout>>;  

#else

  template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
  using shared_array = math::shm::shared_array<nda::array_view<T,N,Layout>>; 

#endif

template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
auto make_shared_array(std::shared_ptr<sfqmc::utils::mpi_context_t<mpi3::communicator>> mpi,
                       std::array<long, N> shape) {
#if defined(ENABLE_DEVICE)
  if constexpr (MEM==HOST_MEMORY) {
    return memory::shared_array<MEM,T,N,Layout>(mpi,shape);
  } else {
    return memory::array<MEM,T,N,Layout>(shape);
  }
#else
  return memory::shared_array<MEM,T,N,Layout>(mpi,shape);
#endif  
}

} // memory
