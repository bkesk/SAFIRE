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

#ifndef NUMERICS_BATCHED_OPERATIONS_CUDA_BACKEND_HPP
#define NUMERICS_BATCHED_OPERATIONS_CUDA_BACKEND_HPP

#include <cassert>

//#include "Memory/buffer_managers.h"
//#include "Numerics/ma_operations.hpp"
//#include "multi/array.hpp"
//#include "multi/array_ref.hpp"

#include "Numerics/detail/dispatch.hpp"
#include "Numerics/detail/CUDA/blas_cuda_gpu_ptr.hpp"
#include "Numerics/detail/CUDA/Kernels/vbias_from_v1.cuh"
#include "Numerics/detail/CUDA/Kernels/batched_dot_wabn_wban.cuh"
#include "Numerics/detail/CUDA/Kernels/batched_dot.cuh"
#include "Numerics/detail/CUDA/Kernels/batched_Tab_to_Klr.cuh"
#include "Numerics/detail/CUDA/Kernels/dot_wabn.cuh"
#include "Numerics/detail/CUDA/Kernels/Tab_to_Kl.cuh"
#include "Numerics/detail/CUDA/Kernels/Auwn_Bun_Cuw.cuh"
#include "Numerics/detail/CUDA/Kernels/spVi_Bij_yj.cuh"

namespace ma
{
template<typename T, typename Q>
void batched_Tab_to_Klr(int nterms,
                        int nwalk,
                        int nocc,
                        int nchol_max,
                        int ncholQ,
                        device::device_pointer<int> kdiag,
                        device::device_pointer<Q> Tab,
                        device::device_pointer<T> Kl,
			int ldkl,
                        device::device_pointer<T> Kr,
		 	int ldkr,
			device_cuda_backend)
{
  kernels::batched_Tab_to_Klr(nterms, nwalk, nocc, nchol_max, ncholQ, raw_pointer_cast(kdiag),
                              raw_pointer_cast(Tab), raw_pointer_cast(Kl), ldkl, raw_pointer_cast(Kr), ldkr);
}

// Not used.
template<typename T, typename Q>
void batched_Tanb_to_Klr(int nterms,
                         int nwalk,
                         int nocc,
                         int nchol_max,
                         int ncholQ,
                         device::device_pointer<int> kdiag,
                         device::device_pointer<Q> Tab,
                         device::device_pointer<T> Kl,
			 int ldkl,
                         device::device_pointer<T> Kr,
			 int ldkr,
			 device_cuda_backend)
{
  kernels::batched_Tanb_to_Klr(nterms, nwalk, nocc, nchol_max, ncholQ, raw_pointer_cast(kdiag),
                               raw_pointer_cast(Tab), raw_pointer_cast(Kl), ldkl, raw_pointer_cast(Kr), ldkr);
}

template<typename T, typename Q>
void Tab_to_Kl(int nwalk, int nocc, int nchol, device::device_pointer<Q> Tab, device::device_pointer<T> Kl, device_cuda_backend)
{
  kernels::Tab_to_Kl(nwalk, nocc, nchol, raw_pointer_cast(Tab), raw_pointer_cast(Kl));
}

template<typename T, typename Q>
void Tanb_to_Kl(int nwalk, int nocc, int nchol, int nact, device::device_pointer<Q> Tab, device::device_pointer<T> Kl, int ldkl, device_cuda_backend)
{
  kernels::Tanb_to_Kl(nwalk, nocc, nchol, nact, raw_pointer_cast(Tab), raw_pointer_cast(Kl), ldkl);
}

template<typename T, typename Q, typename R>
void batched_dot_wabn_wban(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           device::device_pointer<R> alpha,
                           device::device_pointer<Q> Tab,
                           device::device_pointer<T> y,
                           int incy,
			   device_cuda_backend)
{
  kernels::batched_dot_wabn_wban(nbatch, nwalk, nocc, nchol, raw_pointer_cast(alpha), raw_pointer_cast(Tab), 
				 raw_pointer_cast(y), incy);
}

template<typename T, typename Q, typename R>
void batched_dot_wanb_wbna(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           device::device_pointer<R> alpha,
                           device::device_pointer<Q> Tab,
                           device::device_pointer<T> y,
                           int incy,
			   device_cuda_backend)
{
  kernels::batched_dot_wanb_wbna(nbatch, nwalk, nocc, nchol, raw_pointer_cast(alpha), raw_pointer_cast(Tab), 
				 raw_pointer_cast(y), incy);
}

template<typename T, typename Q, typename R>
void dot_wabn(int nwalk, int nocc, int nchol, R alpha, device::device_pointer<Q> Tab, device::device_pointer<T> y, int incy, device_cuda_backend)
{
  kernels::dot_wabn(nwalk, nocc, nchol, alpha, raw_pointer_cast(Tab), raw_pointer_cast(y), incy);
}

// Not used.
template<typename T, typename Q, typename R>
void dot_wpan_waqn_Fwpq(int nwalk, int nocc, int nchol, R alpha, device::device_pointer<Q> Tab, device::device_pointer<T> F)
{
  kernels::dot_wpan_waqn_Fwpq(nwalk, nocc, nchol, alpha, raw_pointer_cast(Tab), raw_pointer_cast(F));
}

template<typename T, typename Q, typename R>
void dot_wanb(int nwalk, int nocc, int nact, int nchol, R alpha, device::device_pointer<Q> Tab, device::device_pointer<T> y, int incy, device_cuda_backend)
{
  kernels::dot_wanb(nwalk, nocc, nact, nchol, alpha, raw_pointer_cast(Tab), raw_pointer_cast(y), incy);
}

template<typename T, typename Q, typename R>
void vbias_from_v1(int nwalk,
                   int nkpts,
                   int nchol_max,
                   device::device_pointer<int> Qsym,
                   device::device_pointer<int> kminus,
                   device::device_pointer<int> ncholpQ,
                   device::device_pointer<int> ncholpQ0,
                   R alpha,
                   device::device_pointer<Q> v1,
                   device::device_pointer<T> vb,
		   device_cuda_backend)
{
  kernels::vbias_from_v1(nwalk, nkpts, nchol_max, raw_pointer_cast(Qsym), raw_pointer_cast(kminus), raw_pointer_cast(ncholpQ),
                         raw_pointer_cast(ncholpQ0), alpha, raw_pointer_cast(v1), raw_pointer_cast(vb));
}

template<typename T1, typename T2, typename T3, typename T4>
void Auwn_Bun_Cuw(int nu, int nw, int na, T1 alpha, device::device_pointer<T2> A, device::device_pointer<T3> B, device::device_pointer<T4> C, device_cuda_backend)
{
  kernels::Auwn_Bun_Cuw(nu, nw, na, alpha, raw_pointer_cast(A), raw_pointer_cast(B), raw_pointer_cast(C));
}

template<typename T1, typename T2, typename T3, typename T4>
void Awiu_Biu_Cuw(int nu,
                  int nw,
                  int ni,
                  T1 alpha,
                  device::device_pointer<T2> A,
                  device::device_pointer<T3> B,
                  int ldb,
                  device::device_pointer<T4> C,
                  int ldc,
		  device_cuda_backend)
{
  kernels::Awiu_Biu_Cuw(nu, nw, ni, alpha, raw_pointer_cast(A), raw_pointer_cast(B), ldb, raw_pointer_cast(C), ldc);
}

template<typename T1, typename T2, typename T3>
void Aijk_Bkj_Cik(int ni,
                  int nj,
                  int nk,
                  device::device_pointer<T1> A,
                  int lda,
                  int stride,
                  device::device_pointer<T2> B,
                  int ldb,
                  device::device_pointer<T3> C,
                  int ldc,
		  device_cuda_backend)
{
  kernels::Aijk_Bkj_Cik(ni, nj, nk, raw_pointer_cast(A), lda, stride, raw_pointer_cast(B), ldb, raw_pointer_cast(C), ldc);
}

// A[w][i][j] = B[i][w][j]
template<typename T, typename T1>
void viwj_vwij(int nw, int ni, int i0, int iN, device::device_pointer<T> B, device::device_pointer<T1> A, device_cuda_backend)
{
  kernels::viwj_vwij(nw, ni, i0, iN, raw_pointer_cast(B), raw_pointer_cast(A));
}

template<typename T1, typename T2, typename T3>
void element_wise_Aij_Bjk_Ckij(char transA,
                               int ni,
                               int nj,
                               int nk,
                               device::device_pointer<T1> A,
                               int lda,
                               device::device_pointer<T2> B,
                               int ldb,
                               device::device_pointer<T3> C,
                               int ldc,
                               int stride,
			       device_cuda_backend)
{
  kernels::element_wise_Aij_Bjk_Ckij(transA, ni, nj, nk, raw_pointer_cast(A), lda, raw_pointer_cast(B), ldb, raw_pointer_cast(C), ldc,
                                     stride);
}

template<typename T1, typename T2, typename T3>
void element_wise_Aij_Bjk_Ckji(int ni,
                               int nj,
                               int nk,
                               device::device_pointer<T1> A,
                               int lda,
                               device::device_pointer<T2> B,
                               int ldb,
                               device::device_pointer<T3> C,
                               int ldc,
                               int stride,
			       device_cuda_backend)
{
  kernels::element_wise_Aij_Bjk_Ckji(ni, nj, nk, raw_pointer_cast(A), lda, raw_pointer_cast(B), ldb, raw_pointer_cast(C), ldc, stride);
}

// for n in [0,N), y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} opA(A)[n,m] * opB(B)[n,m]
template<typename T, typename Q1, typename Q2>
void strided_batched_dot(char TA,
                 char TB,
                 int N,
                 int M,
                 T alpha,
                 device::device_pointer<Q1> A,
                 int lda,
                 device::device_pointer<Q2> B,
                 int ldb,
                 T beta,
                 device::device_pointer<T> y,
                 int incy,
		 device_cuda_backend)
{
  if( std::abs(beta - T(1.0)) > 1e-6  )
    ma::scal(N, beta, y, incy, device_cuda_backend{});
  kernels::strided_batched_dot(TA,TB,N,M,alpha,raw_pointer_cast(A),lda,raw_pointer_cast(B),ldb,raw_pointer_cast(y),incy);
}

// for i in [0,nbatch) 
//   for n in [0,N), 
//     C[i,n] = beta * C[i,n] + alpha sum_m^{0,M} opA(A[i])[n,m] * opB(B[i])[n,m]
template<typename T, typename Q1, typename Q2>
void strided_batched_dot(char TA,
                 char TB,
                 int nbatch,
                 int N,
                 int M,
                 T alpha,
                 device::device_pointer<Q1> A,
                 int lda,
                 long Astride,
                 device::device_pointer<Q2> B,
                 int ldb,
                 long Bstride,
                 T beta,
                 device::device_pointer<T> C,
                 int ldc,
                 long Cstride,
                 device_cuda_backend)
{
  if( std::abs(beta - T(1.0)) > 1e-6  )
    for(int b=0; b<nbatch; ++b)
      ma::scal(N, beta, C+b*Cstride, ldc, device_cuda_backend{});
  kernels::strided_batched_dot(TA,TB,nbatch,N,M,alpha,raw_pointer_cast(A),lda,Astride,raw_pointer_cast(B),ldb,Bstride,raw_pointer_cast(C),ldc,Cstride);
}

template<typename I, typename T, typename Q, typename T1>
void batched_ab_ba(device::device_pointer<I> n,
                   device::device_pointer<Q>* A,
                   int lda,
                   device::device_pointer<Q>* B,
                   int ldb,
                   T1 alpha,
                   device::device_pointer<T>* y,
                   int batchSize)
{
  APP_ABORT(" Error: batched_ab_ba not yet available in gpu.\n");
}

template<typename I, typename T, typename Q>
void batched_diagonal_sum(device::device_pointer<I> n,
                          device::device_pointer<Q>* A,
                          int lda,
                          T alpha,
                          device::device_pointer<T>* y,
                          int batchSize,
			  device_cuda_backend)
{
  APP_ABORT(" Error: batched_diagonal_sum not yet available in gpu.\n");
}

template<typename I1, typename T1, typename T2, typename T3>
void spVi_Bij_yj(int nj,
                 int nnz,
                 device::device_pointer<I1> index,
                 device::device_pointer<T1> values,
                 device::device_pointer<T2> B,
                 int ldb,
                 device::device_pointer<T3> y,
                 int incy,
		 device_cuda_backend)
{
  kernels::spVi_Bij_yj(nj,nnz,raw_pointer_cast(index),raw_pointer_cast(values),
                       raw_pointer_cast(B),ldb,raw_pointer_cast(y),incy);
}

} // namespace ma

#endif
