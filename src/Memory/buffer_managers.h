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

#ifndef BUFFER_MANAGERS_H
#define BUFFER_MANAGERS_H

#include "mpi3/shared_communicator.hpp"

#include "Memory/host_buffer_manager.hpp"
#include "Memory/device_buffer_manager.hpp"
#include "Memory/localTG_buffer_manager.hpp"

namespace sfqmc
{
namespace afqmc
{
void setup_memory_managers(boost::mpi3::shared_communicator& local, size_t size);
void setup_memory_managers(boost::mpi3::shared_communicator& node, size_t size, int nc);
void update_memory_managers();
void release_memory_managers();

} // namespace afqmc
} // namespace sfqmc

#endif
