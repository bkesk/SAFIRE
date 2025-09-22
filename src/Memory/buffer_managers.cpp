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

#include <memory>
#include "Memory/buffer_managers.h"

namespace sfqmc
{
namespace afqmc
{
std::unique_ptr<HostBufferManager::generator_t> HostBufferManager::generator(nullptr);
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
std::unique_ptr<DeviceBufferManager::generator_t> DeviceBufferManager::generator(nullptr);
bool DeviceBufferManager::initialized_by_derived_class(false);
#else
std::unique_ptr<LocalTGBufferManager::generator_t> LocalTGBufferManager::generator(nullptr);
#endif

bool HostBufferManager::initialized_by_derived_class(false);


void setup_memory_managers(boost::mpi3::shared_communicator& local, size_t size)
{
  // setup global memory resource mono state
  HostBufferManager host_buffer(size);
  DeviceBufferManager dev_buffer(size);
  LocalTGBufferManager local_buffer(local, size);
}

void setup_memory_managers(boost::mpi3::shared_communicator& node, size_t size, int nc)
{
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  setup_memory_managers(node, size);
#else
  boost::mpi3::shared_communicator local(
      node.split(node.rank() / ((nc < 1) ? (1) : (std::min(nc, node.size()))), node.rank()));
  setup_memory_managers(local, size);
#endif
}

void update_memory_managers()
{
  HostBufferManager host_buffer;
  host_buffer.get_generator().update();
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  DeviceBufferManager dev_buffer;
  dev_buffer.get_generator().update();
#else
  LocalTGBufferManager local_buffer;
  local_buffer.get_generator().update();
#endif
}

void release_memory_managers()
{
  HostBufferManager host_buffer;
  DeviceBufferManager dev_buffer;
  LocalTGBufferManager local_buffer;
  host_buffer.release();
  dev_buffer.release();
  local_buffer.release();
}

} // namespace afqmc
} // namespace sfqmc
