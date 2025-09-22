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

#ifndef NUMERICS_DETAIL_CPU_LAPACK_HPP
#define NUMERICS_DETAIL_CPU_LAPACK_HPP

#include "config.0.h"
#include "Numerics/detail/dispatch.hpp"
#include "Numerics/detail/CPU/lapack_cpu.hpp"
#include "Memory/buffer_managers.h"

namespace ma
{

template<typename T1, typename T2, typename T3, typename T4, typename T5>
inline void gesvd(char jobu, char jobvt, int m, int n, T1 A, int lda, T2 S, T3 U,
 		  int ldu, T4 Vt, int ldvt, T5 W, int lwork, int& info, cpu_backend)
{
  // check_dispatch<T2,T3>(cpu_backend);
  ma::cpu::gesvd(jobu, jobvt, m, n, raw_pointer_cast(A), lda, raw_pointer_cast(S), 
		 raw_pointer_cast(U), ldu, raw_pointer_cast(Vt), ldvt, raw_pointer_cast(W), lwork, info);
}

template<typename T>
inline void gesvd_bufferSize(const int m, const int n, T A, int& lwork, cpu_backend)
{
  ma::cpu::gesvd_bufferSize(m, n, raw_pointer_cast(A), lwork);
}

template<typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9>
inline void hevr(char JOBZ, char RANGE, char UPLO_param, int N, T1 A, int LDA, T2 VL, T3 VU, int IL, int IU,
                 T4 ABSTOL, int& M, T5 W, T6 Z, int LDZ, int* ISUPPZ, T7 WORK, int& LWORK, T8 RWORK, int& LRWORK,
                 T9 IWORK, int& LIWORK, int& INFO, cpu_backend)
{
  ma::cpu::hevr(JOBZ, RANGE, UPLO_param, N, raw_pointer_cast(A), LDA, VL, VU, IL, IU, ABSTOL, M,
                 raw_pointer_cast(W), raw_pointer_cast(Z), LDZ, raw_pointer_cast(ISUPPZ), raw_pointer_cast(WORK), 
                 LWORK, raw_pointer_cast(RWORK), LRWORK, raw_pointer_cast(IWORK), LIWORK, INFO);
}

template<typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9, typename T10>
inline void gvx(int ITYPE, char JOBZ, char RANGE, char UPLO_param, int N, T1 A, int LDA, T2 B, int LDB,  
		T3 VL, T4 VU, int IL, int IU, T5 ABSTOL, int& M, T6 W, T7 Z, int LDZ, 
		T8 WORK, int& LWORK, T9 RWORK, T10 IWORK, int* IFAIL, int& INFO, cpu_backend)
{
  ma::cpu::gvx(ITYPE, JOBZ, RANGE, UPLO_param, N, raw_pointer_cast(A), LDA, raw_pointer_cast(B), LDB, VL, VU, IL, IU, 
		 ABSTOL, M, raw_pointer_cast(W), raw_pointer_cast(Z), LDZ, raw_pointer_cast(WORK),
                 LWORK, raw_pointer_cast(RWORK), raw_pointer_cast(IWORK), raw_pointer_cast(IFAIL), INFO);
}

// getrf
template<typename T>
inline void getrf_bufferSize(int n, int m, T A, int lda, int& lwork, cpu_backend) 
{
  ma::cpu::getrf_bufferSize(n, m, raw_pointer_cast(A), lda, lwork);
}

template<typename T1, typename T2, typename T3>
inline void getrf(int n, int m, T1 A, int n0, T2 PIV, int& st, T3 WORK, cpu_backend) 
{
  ma::cpu::getrf(n, m, raw_pointer_cast(A), n0, raw_pointer_cast(PIV), st, raw_pointer_cast(WORK));
}

template<typename T1, typename T2, typename T3>
inline void getrfBatched(int n, T1* A, int lda, T2 PIV, T3 INFO, int batchSize, cpu_backend)
{ 
  ma::cpu::getrfBatched(n, A, lda, raw_pointer_cast(PIV), raw_pointer_cast(INFO), batchSize);
}


// getri
template<typename T>
inline void getri_bufferSize(int n, T A, int lda, int& lwork, cpu_backend)
{
  ma::cpu::getri_bufferSize(n, raw_pointer_cast(A), lda, lwork);
}

template<typename T1, typename T2, typename T3>
inline void getri(int n, T1 A, int n0, T2 PIV, T3 WORK, int lwork, int& st, cpu_backend)
{ 
  ma::cpu::getri(n, raw_pointer_cast(A), n0, raw_pointer_cast(PIV), raw_pointer_cast(WORK), lwork, st);
}

template<typename T1, typename T2, typename T3, typename T4>
inline void getriBatched(int n, T1* A, int lda, T2 PIV, T3* Ainv, int ldc, T4 INFO, int batchSize, cpu_backend)
{
  ma::cpu::getriBatched(n, A, lda, raw_pointer_cast(PIV), Ainv, ldc,
			raw_pointer_cast(INFO), batchSize);
}

// geqrf
template<typename T>
inline void geqrf_bufferSize(int m, int n, T A, int lda, int& lwork, cpu_backend)
{
  ma::cpu::geqrf_bufferSize(m, n, raw_pointer_cast(A), lda, lwork);
}

template<typename T1, typename T2, typename T3>
inline void geqrf(int m, int n, T1 A, int lda, T2 TAU, T3 WORK, int lwork, int& info, cpu_backend)
{
  ma::cpu::geqrf(m, n, raw_pointer_cast(A), lda, raw_pointer_cast(TAU), raw_pointer_cast(WORK), lwork, info);
}


template<typename T1, typename T2, typename T3>
inline void geqrfStrided(int M, int N, T1 A, int lda, int Astride, T2 TAU, const int Tstride,
			 T3 info, int batchSize, cpu_backend) 
{
  int lwork=-1;
  ma::cpu::geqrf_bufferSize(M,N,raw_pointer_cast(A),lda,lwork);
  using element = typename std::pointer_traits<T1>::element_type;
  using sfqmc::afqmc::HostBufferManager;
  HostBufferManager hbuffer_manager;
  auto halloc(hbuffer_manager.get_generator().template get_allocator<element>());
  auto WORK = halloc.allocate(lwork);
  for(int b=0; b<batchSize; b++)
  {
    ma::cpu::geqrf(M, N, raw_pointer_cast(A+b*Astride), lda, raw_pointer_cast(TAU+b*Tstride), 
		   WORK, lwork, raw_pointer_cast(info[b]));
  }   
  halloc.deallocate(WORK, lwork);
}

template<typename T>
inline void gqr_bufferSize(int m, int n, int k, T A, int lda, int& lwork, cpu_backend)
{
  ma::cpu::gqr_bufferSize(m, n, k, raw_pointer_cast(A), lda, lwork);
}

template<typename T1, typename T2, typename T3>
inline void gqr(int m, int n, int k, T1 A, int lda, T2 TAU, T3 WORK, int lwork, int& info, cpu_backend)
{
  ma::cpu::gqr(m, n, k, raw_pointer_cast(A), lda, raw_pointer_cast(TAU), raw_pointer_cast(WORK), lwork, info);
}

template<typename T1, typename T2, typename T3, typename T4>
inline void gqrStrided(int M, int N, int K, T1 A, int lda, int Astride, T2 TAU, const int Tstride,
                         T3 WORK, int lwork, T4 info, int batchSize, cpu_backend)
{
  for(int b=0; b<batchSize; b++)
  {
    ma::cpu::gqr(M, N, K, raw_pointer_cast(A+b*Astride), lda, raw_pointer_cast(TAU+b*Tstride),
                 raw_pointer_cast(WORK+b*lwork), lwork, raw_pointer_cast(info[b]));
  }
}

// gelqf
template<typename T>
inline void gelqf_bufferSize(int m, int n, T A, int lda, int& lwork, cpu_backend)
{
  ma::cpu::gelqf_bufferSize(m, n, raw_pointer_cast(A), lda, lwork);
}

template<typename T1, typename T2, typename T3>
inline void gelqf(int m, int n, T1 A, int lda, T2 TAU, T3 WORK, int lwork, int& info, cpu_backend)
{
  ma::cpu::gelqf(m, n, raw_pointer_cast(A), lda, raw_pointer_cast(TAU), raw_pointer_cast(WORK), lwork, info);
}

template<typename T1, typename T2, typename T3>
inline void glq(int m, int n, int k, T1 A, int lda, T2 TAU, T3 WORK, int lwork, int& info, cpu_backend)
{
  ma::cpu::glq(m, n, k, raw_pointer_cast(A), lda, raw_pointer_cast(TAU), raw_pointer_cast(WORK), lwork, info);
}

template<typename T>
inline void glq_bufferSize(int m, int n, int k, T A, int lda, int& lwork, cpu_backend)
{
  ma::cpu::glq_bufferSize(m, n, k, raw_pointer_cast(A), lda, lwork);
}

// matinv
template<typename T1, typename T2, typename T3>
inline void matinvBatched(int n, T1* A, int lda, T2* Ainv, int ldainv, T3 info, int batchSize, cpu_backend)
{
  ma::cpu::matinvBatched(n, A, lda, Ainv, ldainv, raw_pointer_cast(info), batchSize);
}

} // namespace ma

#endif
