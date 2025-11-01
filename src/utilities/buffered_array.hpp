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

#include "configuration.hpp"
#include "utilities/nda_addons/fallback.hpp"
#include "utilities/nda_addons/global_bucket.h"

namespace memory {

namespace detail
{

  template<MEMORY_SPACE MEM>
  using buffered_handle_t = nda::heap_basic<nda::mem::global_fallback<to_nda_address_space(MEM)>>;

}

  template<typename T, int N, typename Layout = nda::C_layout>
  using host_buffered_array = nda::array<T,N,Layout,detail::buffered_handle_t<HOST_MEMORY>>;

#if defined(ENABLE_DEVICE)

  template<typename T, int N, typename Layout = nda::C_layout>
  using device_buffered_array = nda::array<T,N,Layout,detail::buffered_handle_t<DEVICE_MEMORY>>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using unified_buffered_array = nda::array<T,N,Layout,detail::buffered_handle_t<UNIFIED_MEMORY>>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using default_buffered_array = device_buffered_array<T,N,Layout>; 

#else

  template<typename T, int N, typename Layout = nda::C_layout>
  using device_buffered_array = host_buffered_array<T,N,Layout>; 

  template<typename T, int N, typename Layout = nda::C_layout>
  using unified_buffered_array = host_buffered_array<T,N,Layout>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using default_buffered_array = host_buffered_array<T,N,Layout>;

#endif

  template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
  using buffered_array = std::conditional_t<MEM==HOST_MEMORY,    host_buffered_array<T,N,Layout>,
                         std::conditional_t<MEM==DEVICE_MEMORY,  device_buffered_array<T,N,Layout>,
                         std::conditional_t<MEM==UNIFIED_MEMORY, unified_buffered_array<T,N,Layout>,
                                                                 default_buffered_array<T,N,Layout>>>>;

} // sfqmc::afqmc
