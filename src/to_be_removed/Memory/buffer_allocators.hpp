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

#ifndef BUFFER_HANDLER_HPP
#define BUFFER_HANDLER_HPP

#include <cstddef>

#include "Utilities/app_loggers.h"
#include "mpi3/shared_communicator.hpp"

// new allocators
#include "multi/memory/fallback.hpp"
#include "multi/memory/allocator.hpp"
#include "multi/memory/stack.hpp"

namespace sfqmc
{
namespace afqmc
{
template<class MemoryResource, class BaseConstructor = std::allocator<std::byte>, std::size_t Align = alignof(std::max_align_t)>
class BufferAllocatorGenerator
{
private:
  using memory_type = MemoryResource;
  using base_element = std::byte; 
  using raw_pointer  = decltype(std::declval<memory_type&>().allocate(0));
  using pointer      = typename std::pointer_traits<decltype(
      std::declval<memory_type&>().allocate(0))>::template rebind<base_element>;
  using stack_mr     = boost::multi::memory::stack<pointer, Align>;
  using fallback     = boost::multi::memory::fallback<stack_mr, memory_type>;
//  using Constructor = typename std::allocator_traits<BaseConstructor>::template rebind_alloc<base_element>; 

  memory_type base_mr;
  long _size     = 0;
  pointer _start = nullptr;
  fallback mr_;
//  Constructor constr_;

public:
  template<class T>
  using allocator = boost::multi::memory::allocator<T, fallback, typename std::allocator_traits<BaseConstructor>::template rebind_alloc<T>>;

  BufferAllocatorGenerator(MemoryResource const& a, long initial_size = 0) //, BaseConstructor const& c = {})
      : base_mr(a),
        _size(initial_size),
        _start(static_cast<pointer>(base_mr.allocate(_size, Align))),
        mr_({_start, _size}, std::addressof(base_mr))//,
//        constr_(c)
  {
    using std::fill_n;
    fill_n(_start, _size, base_element(0));
  }
   
  ~BufferAllocatorGenerator()
  {
    if (_start != nullptr)
      base_mr.deallocate(static_cast<raw_pointer>(_start), _size);
  }

  // sets _size to mr_.max_needed() and creates a new mr_ with capacity _size.
  // should check that stack allocator is empty
  void update() { resize(mr_.max_needed()); }

  // sets size of buffer to "at least" new_size elements
  void resize(long new_size)
  {
    if (new_size > _size)
    {
      app_log(2, "********************************************************* ");
      app_log(2, "      Resizing memory buffer to: {}  MBs. ", double(new_size) / 1024.0 / 1024.0);
      app_log(2, "********************************************************* \n");
      //          mr_.reset();
      if (_size > 0)
        base_mr.deallocate(static_cast<raw_pointer>(_start), _size);
      _size  = new_size + 1024;
      //_start = static_cast<pointer>(base_mr.allocate(_size, Align));
      _start = static_cast<pointer>(base_mr.allocate(_size));
      // useful to set to zero in GPUs
      //using thrust::fill_n;
      using std::fill_n;
      fill_n(_start, _size, base_element(0));
      mr_ = fallback{{_start, _size}, std::addressof(base_mr)};
    }
  }

  template<class T>
  allocator<T> get_allocator()
  {
    using Constructor_t = typename std::allocator_traits<BaseConstructor>::template rebind_alloc<T>;
    return allocator<T>{std::addressof(mr_), Constructor_t{}}; 
  }
};

} // namespace afqmc
} // namespace sfqmc

#endif
