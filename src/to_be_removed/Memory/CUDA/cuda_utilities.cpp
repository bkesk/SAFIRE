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

//#ifndef ENABLE_CUDA
//#error
//#endif

#include <cassert>
#include <complex>
#include <cstdlib>
#include <stdexcept>
#include "cuda_utilities.h"
#include "Memory/device_pointers.hpp"
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include "cublasXt.h"
#include "cusparse.h"
#include "cusolverDn.h"
#include "curand.h"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"


namespace qmc_cuda
{
bool global_cuda_handles_init = false;
cusparseMatDescr_t global_cusparse_matrix_descr;

std::vector<cudaStream_t> global_cuda_streams;

void cuda_check(cudaError_t sucess, std::string message)
{
  if (cudaSuccess != sucess)
  {
    std::cerr << message << std::endl;
    std::cerr << " cudaGetErrorName: " << cudaGetErrorName(sucess) << std::endl;
    std::cerr << " cudaGetErrorString: " << cudaGetErrorString(sucess) << std::endl;
    std::cerr.flush();
    throw std::runtime_error(" Error code returned by cuda. ");
  }
}

// cuBLAS API errors
static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

void cublas_check(cublasStatus_t sucess, std::string message)
{
  if (CUBLAS_STATUS_SUCCESS != sucess)
  {
    std::cerr << message << std::endl;
    std::cerr << " cublasGetErrorName: " <<_cublasGetErrorEnum(sucess) << std::endl;
    std::cerr.flush();
    throw std::runtime_error(" Error code returned by cublas. ");
  }
}

void cusparse_check(cusparseStatus_t sucess, std::string message)
{
  if (CUSPARSE_STATUS_SUCCESS != sucess)
  {
    std::cerr << message << std::endl;
    std::cerr.flush();
    throw std::runtime_error(" Error code returned by cusparse. ");
  }
}

void curand_check(curandStatus_t sucess, std::string message)
{
  if (CURAND_STATUS_SUCCESS != sucess)
  {
    std::cerr << message << std::endl;
    std::cerr.flush();
    throw std::runtime_error(" Error code returned by curand. ");
  }
}

void cusolver_check(cusolverStatus_t sucess, std::string message)
{
  if (CUSOLVER_STATUS_SUCCESS != sucess)
  {
    std::cerr << message << std::endl;
    std::cerr.flush();
    throw std::runtime_error(" Error code returned by cusolver. ");
  }
}

cublasOperation_t cublasOperation(char A)
{
  if (A == 'N' or A == 'n')
    return CUBLAS_OP_N;
  else if (A == 'T' or A == 't')
    return CUBLAS_OP_T;
  else if (A == 'C' or A == 'c' or A == 'H' or A == 'h')
    return CUBLAS_OP_C;
  else
  {
    throw std::runtime_error("unknown cublasOperation option");
  }
}

cusparseOperation_t cusparseOperation(char A)
{
  if (A == 'N' or A == 'n')
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (A == 'T' or A == 't')
    return CUSPARSE_OPERATION_TRANSPOSE;
  else if (A == 'C' or A == 'c' or A == 'H' or A == 'h')
    return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else
  {
    throw std::runtime_error("unknown cusparseOperation option");
  }
}

} // namespace qmc_cuda
