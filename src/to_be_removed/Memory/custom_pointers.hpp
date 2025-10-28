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

#ifndef CUSTOM_POINTERS_HPP
#define CUSTOM_POINTERS_HPP

#include "Memory/raw_pointers.hpp"
#include "Memory/SharedMemory/shm_ptr_with_raw_ptr_dispatch.hpp"
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
#include "Memory/device_pointers.hpp"
#endif

#endif
