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

#ifndef MEMORY_UTILITIES_HPP
#define MEMORY_UTILITIES_HPP

#if defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#include "Memory/CUDA/cuda_utilities.h"
#elif defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#include "Memory/HIP/hip_utilities.h"
#endif
#include "Memory/OPENMP/OpenMP.hpp"

#if defined(ENABLE_CUDA)
namespace qmc_cuda
{
extern bool global_cuda_handles_init;
}
#elif defined(ENABLE_HIP)
namespace qmc_hip
{
extern bool global_hip_handles_init;
}
#endif

inline int number_of_devices()
{
  int num_devices = 0;
#if defined(ENABLE_CUDA)
  if (not qmc_cuda::global_cuda_handles_init)
    throw std::runtime_error(" Error: Uninitialized CUDA environment.");
  cudaGetDeviceCount(&num_devices);
#elif defined(ENABLE_HIP)
  if (not qmc_hip::global_hip_handles_init)
    throw std::runtime_error(" Error: Uninitialized HIP environment.");
  hipGetDeviceCount(&num_devices);
#endif
  return num_devices;
}

#endif
