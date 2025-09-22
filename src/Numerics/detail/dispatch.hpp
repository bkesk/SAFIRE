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
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////


#ifndef MA_DETAIL_DISPATCH_HPP
#define MA_DETAIL_DISPATCH_HPP

#include "Memory/custom_pointers.hpp"

namespace ma
{

struct bad_backend {};
struct cpu_backend {}; 
struct openmp_backend {}; // intended for openmp routines available regardless of device

template<typename T>
struct ma_dispatch 
{
  using backend = bad_backend; 
};

template<typename T>
struct ma_dispatch<T*>
{
  using backend = cpu_backend;
};
template<typename T>
struct ma_dispatch<T const*>
{
  using backend = cpu_backend;
};
template<typename T>
struct ma_dispatch<T**>
{
  using backend = cpu_backend;
};
template<typename T>
struct ma_dispatch<T* const*>
{
  using backend = cpu_backend;
};

// shm_ptr_with_raw_ptr_dispatch dispatches to cpu as a raw pointer
template<typename T>
struct ma_dispatch<shm::shm_ptr_with_raw_ptr_dispatch<T>>
{
  using backend = cpu_backend;
};
template<typename T>
struct ma_dispatch<shm::shm_ptr_with_raw_ptr_dispatch<T> const>
{
  using backend = cpu_backend;
};
template<typename T>
struct ma_dispatch<shm::shm_ptr_with_raw_ptr_dispatch<T>*>
{
  using backend = cpu_backend;
};
template<typename T>
struct ma_dispatch<shm::shm_ptr_with_raw_ptr_dispatch<T> const*>
{
  using backend = cpu_backend;
};

#if defined(ENABLE_CUDA)

struct device_cuda_backend {}; 

template<typename T>
struct ma_dispatch<device::device_pointer<T>>
{
  using backend = device_cuda_backend;
};
template<typename T>
struct ma_dispatch<device::device_pointer<T> const>
{
  using backend = device_cuda_backend;
};
template<typename T>
struct ma_dispatch<device::device_pointer<T>*>
{
  using backend = device_cuda_backend;
};
template<typename T>
struct ma_dispatch<device::device_pointer<T> const*>
{
  using backend = device_cuda_backend;
};

#elif defined(ENABLE_HIP)

struct device_hip_backend {}; 

#if defined(ENABLE_ROCM)
struct device_rocm_backend {}; 
#endif

#elif defined(ENABLE_OPENMP)

struct device_openmp_backend {}; 

#endif

template<class Array>
auto select_backend() 
-> decltype(typename ma_dispatch<typename std::decay_t<Array>::element_ptr>::backend{})
{
  return typename ma_dispatch<typename std::decay_t<Array>::element_ptr>::backend{};
}

template<class Array1, class Array2>
auto select_backend()
{
#if defined(ENABLE_CUDA)
  if constexpr (std::is_same<typename ma_dispatch<typename std::decay_t<Array1>::element_ptr>::backend,
			     device_cuda_backend>::value or 
   		std::is_same<typename ma_dispatch<typename std::decay_t<Array2>::element_ptr>::backend,
			     device_cuda_backend>::value) 
  {
    return device_cuda_backend{}; 
  } 
  else
#endif
  if constexpr (std::is_same<typename ma_dispatch<typename std::decay_t<Array1>::element_ptr>::backend,
			     cpu_backend>::value and
   		std::is_same<typename ma_dispatch<typename std::decay_t<Array2>::element_ptr>::backend,
			     cpu_backend>::value) 
  {
    return cpu_backend{}; 
  } 
  else 
  {
    return bad_backend{}; 
  } 
}

} // namespace ma

#endif
