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

#ifndef BLAS_OPTIONS_HPP
#define BLAS_OPTIONS_HPP

#include <cassert>
#include "Numerics/detail/utilities.hpp"
#include "Numerics/detail/dispatch.hpp"
#include "Numerics/detail/CPU/blas.hpp"
#include "Numerics/detail/CPU/blas_extensions_cpu.hpp"
#if defined(ENABLE_CUDA)
#include "Numerics/detail/CUDA/blas_cuda_gpu_ptr.hpp"
#elif defined(ENABLE_HIP)
#include "Numerics/detail/HIP/blas_hip_gpu_ptr.hpp"
#endif

#endif
