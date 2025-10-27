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

#ifndef DEVICE_POINTERS_HPP
#define DEVICE_POINTERS_HPP

#include "Numerics/device_kernels.hpp"

#if defined(ENABLE_CUDA)
//#include "multi/memory/adaptors/cuda/ptr.hpp"

//#include <thrust/device_allocator.h>
//#include <thrust/mr/memory_resource.h>
//#include "multi/memory/adaptors/cuda/algorithm.hpp"  

#include "multi/memory/adaptors/cuda/ptr.hpp"
#include "multi/memory/adaptors/cuda/allocator.hpp"
#include "multi/memory/adaptors/cuda/algorithm.hpp"
#include "Memory/CUDA/multi_memory_resource.hpp"
#include "Memory/CUDA/multi_cuda_ref_addons.hpp"

#elif defined(ENABLE_HIP)

#error

#endif

// write specific choices in different files, e.g. multi vs thrust vs custom, etc
// and include appropriate one with compiler flags

namespace device
{

#if defined(ENABLE_CUDA)

//  template<class T> using device_allocator = thrust::device_allocator<T>;
  template<class T> using device_allocator = boost::multi::memory::cuda::allocator<T>;

  template<class T> using device_reference = boost::multi::memory::cuda::ref<T>; 
  template<class T> using device_pointer = boost::multi::memory::cuda::ptr<T>; 
  using memory_resource = qmc_cuda::resource; 
  //using memory_resource = thrust::mr::memory_resource<thrust::device_ptr<void>>; 
  template<class T> using constructor = device_allocator<T>; 

#elif  defined(ENABLE_HIP)

  template<class T> using device_allocator = legacy::device::device_allocator<T>;
  template<class T> using device_reference = typename device::device_allocator<T>::reference; 
  template<class T> using device_pointer = typename device::device_allocator<T>::pointer; 
  template<class T> using memory_resource = legacy::device::memory_resource<T>;
  template<class T> using constructor = legacy::device::constructor<T>;

#endif

}

#if defined(ENABLE_CUDA)
namespace sfqmc
{
namespace afqmc
{

/************* copy_n_cast ****************/
template<typename T, typename Q, typename Size>
device::device_pointer<Q> copy_n_cast(device::device_pointer<T> const A, Size n, device::device_pointer<Q> B)
{
  if constexpr (std::is_same<std::decay_t<T>, Q>::value)
    return copy_n(A,n,B);
  else {
    kernels::copy_n_cast(raw_pointer_cast(A), n, raw_pointer_cast(B));
    return B + n;
  }
}

template<typename T, typename Size>
device::device_pointer<T> copy_n_cast(device::device_pointer<T> const A, Size n, device::device_pointer<T> B)
{
  return copy_n(A, n, B);
}

template<typename T, typename Q, typename Size>
device::device_pointer<Q> copy_n_cast(T* const A, Size n, device::device_pointer<Q> B)
{
  using decay_T = std::decay_t<T>;
  if constexpr (std::is_same<decay_T, Q>::value) {
    return copy_n(A,n,B);
  } else {
    //device::device_allocator<decay_T> alloc{};
    //device::device_pointer<decay_T> p(alloc.allocate(n));
    std::vector<Q> p(n);
    std::copy_n(A, n, p.data());
    return copy_n(p.data(), n, B);
    //alloc.deallocate(p, n);
  }
}

template<typename T, typename Q, typename Size>
Q* copy_n_cast(device::device_pointer<T> const A, Size n, Q* B)
{
  using decay_T = std::decay_t<T>;
  if constexpr (std::is_same<decay_T, Q>::value) {
    return copy_n(A,n,B);
  } else {
    std::vector<decay_T> p(n);
    copy_n(A, n, p.data());
    std::copy_n(p.data(), n, B);
    return B + n;
  }
}

/************* inplace_cast ****************/
template<typename T, typename Q, typename Size>
void inplace_cast(device::device_pointer<T> A, Size n)
{
  T* A_(raw_pointer_cast(A));
  Q* B_(reinterpret_cast<Q*>(A_));
  kernels::inplace_cast(n, A_, B_);
}

/************* fill2D ****************/
template<typename T, typename Q>
void fill2D(int n, int m, device::device_pointer<T> first, int lda, Q val)
{
  RUNTIME_CHECK(lda >= m, "");
  kernels::fill2D_n(n, m, raw_pointer_cast(first), lda, T(val));
}

/************* print ****************/
template<typename T>
void print(std::string str, device::device_pointer<T> p, int n)
{
  kernels::print(str, raw_pointer_cast(p), n);
}

/************* <</>> ****************/

template<typename T>
std::ostream& operator<<(std::ostream& os, device::device_reference<T> const& obj)
{
  os << T(obj);
  return os;
}

template<typename T>
std::istream& operator<<(std::istream& is, device::device_reference<T>& obj)
{
  T val;
  is >> val;
  obj = val;
  return is;
}

} // namespace sfqmc
} // namespace afqmc

/*
#include "Memory/SharedMemory/shm_ptr_with_raw_ptr_dispatch.hpp"
namespace shm
{

template<typename T, typename Size>
device::device_pointer<T> copy_n(shm_ptr_with_raw_ptr_dispatch<T> first, Size n, device::device_pointer<T> d_first)
{
  std::cout<<" copy_n shm_ptr -> cuda_ptr " <<std::endl;
  copy_n(raw_pointer_cast(first), n, d_first);
}

template<typename T, typename Size>
device::device_pointer<T> uninitialized_copy_n(shm_ptr_with_raw_ptr_dispatch<T> first, Size n, device::device_pointer<T> d_first)
{ 
  std::cout<<" uninitialized_copy_n shm_ptr -> cuda_ptr " <<std::endl;
  uninitialized_copy_n(raw_pointer_cast(first), n, d_first);
}
}
*/

#endif

#endif
