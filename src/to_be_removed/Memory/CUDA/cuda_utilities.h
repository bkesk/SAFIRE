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

#ifndef CUDA_UTILITIES_HPP
#define CUDA_UTILITIES_HPP

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include<complex>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include "cublasXt.h"
#include "cusparse.h"
#include "cusolverDn.h"
#include "curand.h"

namespace qmc_cuda
{
//  extern curandGenerator_t global_curand_generator;
extern cusparseMatDescr_t global_cusparse_matrix_descr;
extern std::vector<cudaStream_t> global_cuda_streams;

void cuda_check_error();
void cuda_check(cudaError_t sucess, std::string message = "");
void cublas_check(cublasStatus_t sucess, std::string message = "");
void cusparse_check(cusparseStatus_t sucess, std::string message = "");
void curand_check(curandStatus_t sucess, std::string message = "");
void cusolver_check(cusolverStatus_t sucess, std::string message = "");
cublasOperation_t cublasOperation(char A);
cusparseOperation_t cusparseOperation(char A);

// since when do these exist? Not sure, so keep guard for now.
#if CUSPARSE_VER_MAJOR > 10
template<typename T>
cusparseIndexType_t cusparse_index_type()
{
  return cusparseIndexType_t{};
}
template<>
inline cusparseIndexType_t cusparse_index_type<int>()
{
  return CUSPARSE_INDEX_32I;
}
template<>
inline cusparseIndexType_t cusparse_index_type<long>()
{
  return CUSPARSE_INDEX_64I;
}

template<typename T>
cudaDataType_t cusparse_data_type()
{
  return cudaDataType_t{};
}
template<>
inline cudaDataType_t cusparse_data_type<float>()
{
  return CUDA_R_32F;
}
template<>
inline cudaDataType_t cusparse_data_type<double>()
{
  return CUDA_R_64F;
}
template<>
inline cudaDataType_t cusparse_data_type<std::complex<float>>()
{
  return CUDA_C_32F;
}
template<>
inline cudaDataType_t cusparse_data_type<std::complex<double>>()
{
  return CUDA_C_64F;
}
#endif


} // namespace qmc_cuda

#endif
