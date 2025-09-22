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

#ifndef DEVICE_BUFFER_MANAGER_HPP
#define DEVICE_BUFFER_MANAGER_HPP

#include <memory>
#include <cstddef>

#include "Memory/buffer_allocators.hpp"
#include "Memory/host_buffer_manager.hpp"

namespace sfqmc
{
namespace afqmc
{
// Class that manages the memory resource that generates allcators for device memory.
// Follows a monostate-type pattern. All variables are static and refer to a global instance
// of the resource.
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
class DeviceBufferManager
{
public:
  using generator_t = BufferAllocatorGenerator<device_memory_resource, device_constructor<std::byte>>;

  template<class T>
  using allocator_t = typename generator_t::template allocator<T>;

  DeviceBufferManager(size_t size)
  {
    require(false);
    generator = std::make_unique<generator_t>(device::memory_resource{}, size);//, device::constructor<std::byte>{});
  }

  DeviceBufferManager() { require(true); }

  ~DeviceBufferManager() {}

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
  DeviceBufferManager(size_t size, bool)
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
#else
//  using DeviceBufferManager = HostBufferManager;
class DeviceBufferManager : public HostBufferManager
{
public:
  using HostBufferManager::allocator_t;
  using HostBufferManager::generator_t;

  DeviceBufferManager(size_t size) : HostBufferManager(size, true) {}
  DeviceBufferManager() : HostBufferManager() {}
  ~DeviceBufferManager(){};

  void release() { HostBufferManager::release(); }

  generator_t& get_generator()
  {
    if (not initialized_by_derived_class)
      throw std::runtime_error("Error: Incorrect derived global state in get_generator. ");
    return HostBufferManager::get_generator();
  }

private:
  using HostBufferManager::initialized_by_derived_class;
};
#endif

} // namespace afqmc
} // namespace sfqmc
#endif
