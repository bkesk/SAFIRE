/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef MEMORY_CONFIG_H
#define MEMORY_CONFIG_H

#include "config.0.h"
#include "Utilities/check.hpp"
#include "Memory/custom_pointers.hpp"
#include "Memory/SharedMemory/shm_ptr_with_raw_ptr_dispatch.hpp"

#include "multi/memory/fallback.hpp"

#include "Memory/raw_pointers.hpp"
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
#include "Memory/device_pointers.hpp"
#endif

namespace sfqmc
{

// replace with memory namespace
namespace afqmc
{

// allocators
template<class T>
using shared_allocator = shm::allocator_shm_ptr_with_raw_ptr_dispatch<T>;
template<class T>
using shm_pointer = typename shared_allocator<T>::pointer;

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)

template<class T>
using device_allocator = device::device_allocator<T>;
template<class T>
using device_ptr = device::device_pointer<T>;
template<class T>
using localTG_allocator = device_allocator<T>;
template<class T>
using node_allocator = device_allocator<T>;
template<class T, class TG>
localTG_allocator<T> make_localTG_allocator(TG&)
{
  return localTG_allocator<T>{};
}
template<class T, class TG>
node_allocator<T> make_node_allocator(TG&)
{
  return node_allocator<T>{};
}
/*   Temporary fix for the conflict problem between cpu and gpu pointers. Find proper fix */
template<class T>
device_ptr<T> make_device_ptr(device_ptr<T> p)
{
  return p;
}
template<class T>
device_ptr<T> make_device_ptr(T* p)
{
  print_stacktrace;
  throw std::runtime_error(" Invalid pointer conversion: device_pointer<T> to T*.");
}
template<class T>
device_ptr<T> make_device_ptr(shm::shm_ptr_with_raw_ptr_dispatch<T> p)
{
  print_stacktrace;
  throw std::runtime_error(" Invalid pointer conversion: device_pointer<T> to T*.");
}

using device_memory_resource = device::memory_resource;
using shm_memory_resource    = device::memory_resource;
template<class T>
using device_constructor = device::constructor<T>;
template<class T>
using shm_constructor = device::constructor<T>;

#else

template<class T>
using device_allocator = std::allocator<T>;
template<class T>
using device_ptr = T*;
template<class T>
using localTG_allocator = shared_allocator<T>;
template<class T>
using node_allocator = shared_allocator<T>;
template<class T, class TG>
localTG_allocator<T> make_localTG_allocator(TG& t_)
{
  return localTG_allocator<T>{t_.TG_local()};
}
template<class T, class TG>
node_allocator<T> make_node_allocator(TG& t_)
{
  return node_allocator<T>{t_.Node()};
}
/*   Temporary fix for the conflict problem between cpu and gpu pointers. Find proper fix */
template<class T>
device_ptr<T> make_device_ptr(T* p)
{
  return p;
}
template<class T>
device_ptr<T> make_device_ptr(shm::shm_ptr_with_raw_ptr_dispatch<T> p)
{
  return device_ptr<T>{raw_pointer_cast(p)};
}

using device_memory_resource = boost::multi::memory::resource<>;
using shm_memory_resource    = shm::memory_resource_shm_ptr_with_raw_ptr_dispatch;
template<class T>
using device_constructor = device_allocator<T>;
template<class T>
using shm_constructor = shm::constructor_shm_ptr_with_raw_ptr_dispatch<T>; 

#endif

// useful functions
template<class V>
struct is_host_array
{
  static constexpr bool value = std::is_same<typename V::element_ptr,typename V::element*>::value;
};
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
template<class V>
struct is_device_array
{
  static constexpr bool value = std::is_same<typename V::element_ptr,device::device_pointer<typename V::element>>::value;
};
#else
template<class V>
struct is_device_array
{
  static constexpr bool value = false;
};
#endif
template<class V>
struct is_shm_array
{
  static constexpr bool value = std::is_same<typename V::element_ptr,shm::shm_ptr_with_raw_ptr_dispatch<typename V::element>>::value;
};
template<class V>
struct is_host_or_shm_array
{
  static constexpr bool value = (is_host_array<V>::value or is_shm_array<V>::value);
};


template<class T>
using host_constructor     = std::allocator<T>;
using host_memory_resource = boost::multi::memory::resource<>;

} // namespace afqmc

} // namespace qmcpluslus

#endif
