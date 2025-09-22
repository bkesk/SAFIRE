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

#ifndef HOST_BUFFER_MANAGER_HPP
#define HOST_BUFFER_MANAGER_HPP

#include <memory>
#include <cstddef>

#include "Memory/config.h"
#include "Memory/buffer_allocators.hpp"

namespace sfqmc
{
namespace afqmc
{
// Class that manages the memory resource that generates allocators for host memory.
// Follows a monostate-type pattern. All variables are static and refer to a global instance
// of the resource.
class HostBufferManager
{
public:
  using generator_t = BufferAllocatorGenerator<host_memory_resource, host_constructor<std::byte>>;

  template<class T>
  using allocator_t = typename generator_t::template allocator<T>;

  HostBufferManager(size_t size)
  {
    if (not generator)
      generator = std::make_unique<generator_t>(host_memory_resource{}, size);//, host_constructor<std::byte>{});
  }

  HostBufferManager() { require(true); }

  ~HostBufferManager() {}

  void release()
  {
    if (generator)
    {
      generator.reset(nullptr);
      generator                    = nullptr;
      initialized_by_derived_class = false;
    }
  }

  generator_t& get_generator()
  {
    require(true);
    return *generator;
  }

protected:
  // protected constructor for use by derived class
  // this is done to allow for double initialization
  HostBufferManager([[maybe_unused]] size_t size, bool)
  {
    require(true);
    if (initialized_by_derived_class)
      throw std::runtime_error("Error: Incorrect global state in protected constructor.");
    initialized_by_derived_class = true;
  }

  // static pointers to global objects
  //static generator_t* generator;
  static std::unique_ptr<generator_t> generator;

  static bool initialized_by_derived_class;

  void require(bool ini)
  {
    if (ini && not generator)
    {
      throw std::runtime_error("Error: Incorrect global state in require (found uninitialized).");
    }
    else if (not ini && generator)
    {
      throw std::runtime_error("Error: Incorrect global state in require (found initialized).");
    }
  }
};

} // namespace afqmc
} // namespace sfqmc
#endif
