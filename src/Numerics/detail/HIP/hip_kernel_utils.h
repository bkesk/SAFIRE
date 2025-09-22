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

#ifndef HIP_KERNEL_UTILITIES_HPP
#define HIP_KERNEL_UTILITIES_HPP

#include <cassert>
#include <hip/hip_runtime.h>

#include "hip_kernel_utils.h"
#include "rocrand/rocrand.h"

namespace qmc_hip
{
void hip_kernel_check(hipError_t sucess, std::string message = "");
void rocrand_check(rocrand_status sucess, std::string message = "");
} // namespace qmc_hip

#endif
