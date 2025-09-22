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

#ifndef SPARSE_HIP_GPU_PTR_HPP
#define SPARSE_HIP_GPU_PTR_HPP

#include <type_traits>
#include <cassert>
#include <vector>
#include "Memory/custom_pointers.hpp"
#include "Numerics/detail/HIP/hipsparse_wrapper.hpp"
#include "Numerics/detail/HIP/hipblas_wrapper.hpp"
#include <cassert>
#include <complex>

#include "multi/array.hpp"

namespace ma
{
extern boost::multi::array<std::complex<double>, 1, device::device_allocator<std::complex<double>>>* hipsparse_buffer;


template<typename T, typename Q>
void csrmv(const char transa,
           const int M,
           const int K,
           const T alpha,
           const char* matdescra,
           device::device_pointer<T> const A,
           device::device_pointer<int> const indx,
           device::device_pointer<int> const pntrb,
           device::device_pointer<int> const pntre,
           device::device_pointer<Q> const x,
           const T beta,
           device::device_pointer<T> y)
{
  static_assert(std::is_same<typename std::decay<Q>::type, T>::value, "Wrong dispatch.\n");
  // somehow need to check if the matrix is compact!
  int pb, pe;
  arch::memcopy(std::addressof(pb), raw_pointer_cast(pntrb), sizeof(int), arch::memcopyD2H, "sparse_hip_gpu_ptr::csrmv");
  arch::memcopy(std::addressof(pe), raw_pointer_cast(pntre + (M - 1)), sizeof(int), arch::memcopyD2H,
                "sparse_hip_gpu_ptr::csrmv");
  int nnz = pe - pb;
  if (HIPSPARSE_STATUS_SUCCESS !=
      hipsparse::hipsparse_csrmv(*A.handles.hipsparse_handle, transa, M, K, nnz, alpha,
                                 qmc_hip::afqmc_hipsparse_matrix_descr, raw_pointer_cast(A), raw_pointer_cast(pntrb),
                                 raw_pointer_cast(indx), raw_pointer_cast(x), beta, raw_pointer_cast(y)))
    throw std::runtime_error("Error: hipsparse_csrmv returned error code.");
}

template<typename T, typename Q>
void csrmm(const char transa,
           const int M,
           const int N,
           const int K,
           const T alpha,
           const char* matdescra,
           device::device_pointer<T> A,
           device::device_pointer<int> indx,
           device::device_pointer<int> pntrb,
           device::device_pointer<int> pntre,
           device::device_pointer<Q> B,
           const int ldb,
           const T beta,
           device::device_pointer<T> C,
           const int ldc)
{
  static_assert(std::is_same<typename std::decay<Q>::type, T>::value, "Wrong dispatch.\n");
  // somehow need to check if the matrix is compact!
  int pb, pe;
  arch::memcopy(std::addressof(pb), raw_pointer_cast(pntrb), sizeof(int), arch::memcopyD2H, "sparse_hip_gpu_ptr::csrmm");
  arch::memcopy(std::addressof(pe), raw_pointer_cast(pntre + (M - 1)), sizeof(int), arch::memcopyD2H,
                "sparse_hip_gpu_ptr::csrmm");
  int nnz = pe - pb;
  if (transa == 'N')
  {
    /*
      // CSR_A * B = C  -->  (Fortran)  B^T * CSC_(CSR_A) = C^T
      if(HIPSPARSE_STATUS_SUCCESS != hipsparse::hipsparse_gemmi(*A.handles.hipsparse_handle,
            N,M,K,nnz,alpha,raw_pointer_cast(B),ldb,
            raw_pointer_cast(A),raw_pointer_cast(pntrb),raw_pointer_cast(indx),
            beta,raw_pointer_cast(C),ldc))
        throw std::runtime_error("Error: hipsparse_csrmm(gemmi) returned error code.");
*/
    char transb('T');
    // setup work space for column major matrix C
    if (hipsparse_buffer == nullptr)
    {
      hipsparse_buffer = new boost::multi::array<
          std::complex<double>, 1,
          device::device_allocator<std::complex<double>>>(typename boost::multi::layout_t<1u>::extensions_type{M * M},
                                                          device::device_allocator<std::complex<double>>{});
    }
    else if (hipsparse_buffer->num_elements() < M * N)
      hipsparse_buffer->reextent(typename boost::multi::layout_t<1u>::extensions_type{M * N});
    device::device_pointer<T> C_(hipsparse_buffer->origin().pointer_cast<T>());

    // if beta != 0, transpose C into C_
    if (std::abs(beta) > 1e-12)
    {
      auto status = hipblas::hipblas_geam(*A.handles.hipblas_handle, transb, transa, M, N, T(1), raw_pointer_cast(C), ldc,
                                          T(0), raw_pointer_cast(C_), M, raw_pointer_cast(C_), M);
      if (HIPBLAS_STATUS_SUCCESS != status)
      {
        std::cerr << "ERROR STATUS : " << status << std::endl;
        throw std::runtime_error("Error: hipblas_geam returned error code.");
      }
    }

    // call csrmm2 on C_
    auto status = hipsparse::hipsparse_csrmm2(*A.handles.hipsparse_handle, transa, transb, M, N, K, nnz, alpha,
                                              qmc_hip::afqmc_hipsparse_matrix_descr, raw_pointer_cast(A), raw_pointer_cast(pntrb),
                                              raw_pointer_cast(indx), raw_pointer_cast(B), ldb, beta, raw_pointer_cast(C_), M);
    if (HIPSPARSE_STATUS_SUCCESS != status)
    {
      std::cout << "THIS" << std::endl;
      std::cerr << "ERROR STATUS : " << status << std::endl;
      throw std::runtime_error("Error: hipsparse_csrmm returned error code.");
    }

    // transpose work matrix to row major on result C
    if (HIPBLAS_STATUS_SUCCESS !=
        hipblas::hipblas_geam(*A.handles.hipblas_handle, transb, transa, N, M, T(1), raw_pointer_cast(C_), M, T(0),
                              raw_pointer_cast(C), ldc, raw_pointer_cast(C), ldc))
      throw std::runtime_error("Error: hipblas_geam returned error code.");
    // */
  }
  else
  {
    char transT('T');
    char transN('N');
    // setup work space for column major matrix B,C
    if (hipsparse_buffer == nullptr)
    {
      hipsparse_buffer = new boost::multi::array<
          std::complex<double>, 1,
          device::device_allocator<std::complex<double>>>(typename boost::multi::layout_t<1u>::extensions_type{(M + K) *
                                                                                                               N},
                                                          device::device_allocator<std::complex<double>>{});
    }
    else if (hipsparse_buffer->num_elements() < (M + K) * N)
      hipsparse_buffer->reextent(typename boost::multi::layout_t<1u>::extensions_type{(M + K) * N});
    // A is MxK
    // B should be MxN
    device::device_pointer<T> B_(hipsparse_buffer->origin().pointer_cast<T>());
    // C should be KxN
    device::device_pointer<T> C_(B_ + M * N);

    // if beta != 0, transpose C into C_
    if (std::abs(beta) > 1e-12)
      if (HIPBLAS_STATUS_SUCCESS !=
          hipblas::hipblas_geam(*A.handles.hipblas_handle, transT, transN, K, N, T(1), raw_pointer_cast(C), ldc, T(0),
                                raw_pointer_cast(C_), K, raw_pointer_cast(C_), K))
        throw std::runtime_error("Error: hipblas_geam returned error code. C");

    auto st = hipblas::hipblas_geam(*A.handles.hipblas_handle, transT, transN, M, N, T(1), raw_pointer_cast(B), ldb, T(0),
                                    raw_pointer_cast(B_), M, raw_pointer_cast(B_), M);
    if (st != HIPBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Error: hipblas_geam returned error code. B");

    // call csrmm2 on C_
    auto status = hipsparse::hipsparse_csrmm(*A.handles.hipsparse_handle, transa, M, N, K, nnz, alpha,
                                             qmc_hip::afqmc_hipsparse_matrix_descr, raw_pointer_cast(A), raw_pointer_cast(pntrb),
                                             raw_pointer_cast(indx), raw_pointer_cast(B_), M, beta, raw_pointer_cast(C_), K);
    if (status != HIPSPARSE_STATUS_SUCCESS)
    {
      throw std::runtime_error("Error: hipsparse_csrmm returned error code.");
    }

    // transpose work matrix to row major on result C
    if (HIPBLAS_STATUS_SUCCESS !=
        hipblas::hipblas_geam(*A.handles.hipblas_handle, transT, transN, N, K, T(1), raw_pointer_cast(C_), K, T(0),
                              raw_pointer_cast(C), ldc, raw_pointer_cast(C), ldc))
      throw std::runtime_error("Error: hipblas_geam returned error code. C_");
  }
}

}; // namespace ma


#endif
