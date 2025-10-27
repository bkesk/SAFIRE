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

#ifndef HIP_UTILITIES_HPP
#define HIP_UTILITIES_HPP

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <hip/hip_runtime.h>
#include "hipblas.h"
#include "hipsparse.h"
#include "rocsolver.h"
#include "rocrand/rocrand.h"

namespace qmc_hip
{
//  extern hiprandGenerator_t afqmc_curand_generator;
extern hipsparseMatDescr_t afqmc_hipsparse_matrix_descr;

extern std::vector<hipStream_t> afqmc_hip_streams;

// FDM: Temprorary hack to allow for easier grepping.
typedef rocsolver_status rocsolverStatus_t;
typedef rocsolver_status hipsolverStatus_t;
typedef rocsolver_handle hipsolverHandle_t;
typedef rocsolver_handle rocsolverHandle_t;
typedef rocblas_operation_ rocblasOperation_t;
typedef rocrand_status hiprandStatus_t;
typedef rocrand_generator hiprandGenerator_t;

void hip_check_error();
void hip_check(hipError_t sucess, std::string message = "");
void hipblas_check(hipblasStatus_t sucess, std::string message = "");
void hipsparse_check(hipsparseStatus_t sucess, std::string message = "");
void hiprand_check(hiprandStatus_t sucess, std::string message = "");
void hipsolver_check(hipsolverStatus_t sucess, std::string message = "");
hipblasOperation_t hipblasOperation(char A);
rocblasOperation_t rocblasOperation(char A);
hipsparseOperation_t hipsparseOperation(char A);

} // namespace qmc_hip

#endif
