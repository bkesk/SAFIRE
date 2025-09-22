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

#ifndef ARCH_HPP
#define ARCH_HPP

#if defined(ENABLE_CUDA)
#include "Memory/CUDA/cuda_arch.h"
#elif defined(ENABLE_HIP)
#include "Memory/HIP/hip_arch.h"
#endif

#endif
