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

#ifndef NUMERICS_BATCHED_OPERATIONS_HPP
#define NUMERICS_BATCHED_OPERATIONS_HPP

#include "Numerics/detail/batched_operations.hpp"
#include "Utilities/check.hpp"


namespace ma
{

template<class MA1,
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 3>
        >
MA3&& element_wise_Aij_Bjk_Ckij(char transA, MA1&& A, MA2&& B, MA3&& C)
{
  RUNTIME_CHECK(A.size(0) == C.size(1), "");
  RUNTIME_CHECK(A.size(1) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == C.size(2), "");
  RUNTIME_CHECK(B.size(1) == C.size(0), "");
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(B.stride(1) == 1, "");
  RUNTIME_CHECK(C.stride(2) == 1, "");
  // notice the uncommon convention in passing C strides  
  ma::element_wise_Aij_Bjk_Ckij(transA, A.size(0), A.size(1), B.size(1), pointer_dispatch(A.origin()), A.stride(0),
                pointer_dispatch(B.origin()), B.stride(0), 
		pointer_dispatch(C.origin()), C.stride(1), C.stride(0), 
		select_backend<MA1>());
  return std::forward<MA3>(C);
}

template<class MA1,
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 3>
	>
MA3&& element_wise_Aij_Bjk_Ckji(MA1&& A, MA2&& B, MA3&& C)
{
  RUNTIME_CHECK(A.size(0) == C.size(2), "");
  RUNTIME_CHECK(A.size(1) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == C.size(1), "");
  RUNTIME_CHECK(B.size(1) == C.size(0), "");
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(B.stride(1) == 1, "");
  RUNTIME_CHECK(C.stride(2) == 1, "");
  ma::element_wise_Aij_Bjk_Ckji(A.size(0), A.size(1), B.size(1), pointer_dispatch(A.origin()), A.stride(0),  
		pointer_dispatch(B.origin()), B.stride(0), pointer_dispatch(C.origin()), C.stride(1),
		C.stride(0), select_backend<MA1>());
  return std::forward<MA3>(C); 
}

template<typename T, 
         class MA1,
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 2>
        >
MA3&& Auwn_Bun_Cuw(T alpha, MA1&& A, MA2&& B, MA3&& C)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(0) == C.size(0), "");
  RUNTIME_CHECK(A.size(1) == C.size(1), "");
  RUNTIME_CHECK(A.size(2) == B.size(1), "");
  // A contiguous
  RUNTIME_CHECK(A.stride(0) == A.size(1)*A.size(2), "");
  RUNTIME_CHECK(A.stride(1) == A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == 1, "");
  // B contiguous
  RUNTIME_CHECK(B.stride(0) == B.size(1), "");
  RUNTIME_CHECK(B.stride(1) == 1, "");
  // C contiguous
  RUNTIME_CHECK(C.stride(0) == C.size(1), "");
  RUNTIME_CHECK(C.stride(1) == 1, "");
  ma::Auwn_Bun_Cuw(A.size(0), A.size(1), A.size(2), alpha, pointer_dispatch(A.origin()),
                pointer_dispatch(B.origin()), pointer_dispatch(C.origin()), 
                select_backend<MA1>());
  return std::forward<MA3>(C);
}

template<typename T, 
	 class MA1,
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 2>
        >
MA3&& Awiu_Biu_Cuw(T alpha, MA1&& A, MA2&& B, MA3&& C)
{
  RUNTIME_CHECK(A.size(0) == C.size(1), "");
  RUNTIME_CHECK(A.size(1) == B.size(0), "");
  RUNTIME_CHECK(A.size(2) == B.size(1), "");
  RUNTIME_CHECK(A.size(2) == C.size(0), "");
  // implementation assumes A contiguous
  RUNTIME_CHECK(A.stride(0) == A.size(1)*A.size(2), "");
  RUNTIME_CHECK(A.stride(1) == A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(B.stride(1) == 1, "");
  RUNTIME_CHECK(C.stride(1) == 1, "");
  ma::Awiu_Biu_Cuw(C.size(0), C.size(1), B.size(0), alpha, pointer_dispatch(A.origin()), 
                pointer_dispatch(B.origin()), B.stride(0), pointer_dispatch(C.origin()), C.stride(0),
                select_backend<MA1>());
  return std::forward<MA3>(C);
}

template<class MA1,
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 2>
        >
MA3&& Aijk_Bkj_Cik(MA1&& A, MA2&& B, MA3&& C)
{
  RUNTIME_CHECK(A.size(0) == C.size(0), "");
  RUNTIME_CHECK(A.size(1) == B.size(1), "");
  RUNTIME_CHECK(A.size(2) == B.size(0), "");
  RUNTIME_CHECK(A.size(2) == C.size(1), "");
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(B.stride(1) == 1, "");
  RUNTIME_CHECK(C.stride(1) == 1, "");
  ma::Aijk_Bkj_Cik(A.size(0), A.size(1), A.size(2), pointer_dispatch(A.origin()), A.stride(1), A.stride(0),
                pointer_dispatch(B.origin()), B.stride(0), pointer_dispatch(C.origin()), C.stride(0),
                select_backend<MA1>());
  return std::forward<MA3>(C);
}

template<class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 3>
        >
MA2&& viwj_vwij(int i0, int iN, MA1&& A, MA2&& B)
{
  RUNTIME_CHECK(A.size(0) == A.size(2), "");
  RUNTIME_CHECK(A.size(0) == B.size(1), "");
  RUNTIME_CHECK(A.size(1) == B.size(0), "");
  RUNTIME_CHECK(A.size(2) == B.size(2), "");
  RUNTIME_CHECK(i0 <= A.size(0), "");
  RUNTIME_CHECK(iN <= A.size(0), "");
  RUNTIME_CHECK(i0 <= iN, "");
  RUNTIME_CHECK(A.stride(0) == A.size(1)*A.size(2), "");
  RUNTIME_CHECK(A.stride(1) == A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(B.stride(0) == B.size(1)*A.size(2), "");
  RUNTIME_CHECK(B.stride(1) == B.size(2), "");
  RUNTIME_CHECK(B.stride(2) == 1, "");
  ma::viwj_vwij(A.size(1), A.size(0), i0, iN, pointer_dispatch(A.origin()), 
                pointer_dispatch(B.origin()), select_backend<MA1>());
  return std::forward<MA2>(B);
}


// B[w] = sum_{labn} alpha[l] * A[2*l,w,a,b,n] A[2*l+1,w,b,a,n]
template<class MA1, 
	 class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 1>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 5>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 1>
        >
void Apwabn_Apwban_Bw(MA1&& alpha, MA2&& A, MA3&& B)  
{
  RUNTIME_CHECK(A.size(1) == B.size(0), "");
  RUNTIME_CHECK(2*alpha.size(0) == A.size(0), "");
  RUNTIME_CHECK(A.stride(0) == A.size(4)*A.size(3)*A.size(2)*A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == A.size(4)*A.size(3)*A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == A.size(4)*A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == A.size(4), "");
  RUNTIME_CHECK(A.stride(4) == 1, "");
  ma::batched_dot_wabn_wban(alpha.size(0), A.size(1), A.size(2), A.size(4), pointer_dispatch(alpha.origin()), 
		pointer_dispatch(A.origin()), pointer_dispatch(B.origin()), B.stride(0), select_backend<MA3>());
}

/* not used
// B[w] = sum_{labn} alpha[l] * A[2*l,w,a,n,b] A[2*l+1,w,b,n,a]
template<class MA1,  
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 1>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 5>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 1>
        >
void Apwanb_Apwbna_Bw(MA1&& alpha, MA2&& A, MA3&& B)
{
  RUNTIME_CHECK(A.size(1) == B.size(0), "");
  RUNTIME_CHECK(2*alpha.size(0) == A.size(0), "");
  RUNTIME_CHECK(A.stride(0) == A.size(4)*A.size(3)*A.size(2)*A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == A.size(4)*A.size(3)*A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == A.size(4)*A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == A.size(4), "");
  RUNTIME_CHECK(A.stride(4) == 1, "");
  ma::batched_dot_wanb_wbna(alpha.size(0), A.size(1), A.size(2), A.size(3), pointer_dispatch(alpha.origin()), 
                pointer_dispatch(A.origin()), pointer_dispatch(B.origin()), B.stride(0), select_backend<MA3>());
}
*/

// B[w] = sum_{abn} alpha * A[w,a,b,n] A[w,b,a,n]
template<typename T,
         class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 4>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 1>
        >
void Awabn_Awban_Bw(T alpha, MA1&& A, MA2&& B)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == A.size(2), ""); 
  RUNTIME_CHECK(A.stride(0) == A.size(3)*A.size(2)*A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == A.size(3)*A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == 1, "");
  ma::dot_wabn(A.size(0), A.size(1), A.size(3), alpha, pointer_dispatch(A.origin()), 
	       pointer_dispatch(B.origin()), B.stride(0), select_backend<MA2>());
}

// B[w] = sum_{abn} alpha * A[w,a,n,b] A[w,b,n,a]
template<typename T,
         class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 4>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 1>
        >
void Awanb_Awbna_Bw(T alpha, MA1&& A, MA2&& B)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.stride(0) == A.size(3)*A.size(2)*A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == A.size(3)*A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == 1, "");
  ma::dot_wanb(A.size(0), A.size(1), A.size(3), A.size(2), alpha, pointer_dispatch(A.origin()), 
               pointer_dispatch(B.origin()), B.stride(0), select_backend<MA2>());
}

/*
void dot_wpan_waqn_Fwpq(int nwalk,
                        int nmo,
                        int nchol,
                        std::complex<Q> alpha,
                        std::complex<Q> const* Tab,
                        std::complex<T>* F,
*/

// B1wn += Apwaan  
// B2wn += Apwaan  
template<class MA1,
         class MA2,
         class MA3,
         class MA4,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 1>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 5>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA4>::dimensionality == 2>
        >
void Apwaan_Bwn(MA1&& index, MA2&& A, MA3&& B1, MA4&& B2)
{ 
  RUNTIME_CHECK(B1.size(1) == B2.size(1), "");
  RUNTIME_CHECK(A.size(1) == B1.size(0), "");
  RUNTIME_CHECK(A.size(1) == B2.size(0), "");
  RUNTIME_CHECK(A.size(2) == A.size(3), "");
  RUNTIME_CHECK(A.stride(0) == A.size(4)*A.size(3)*A.size(2)*A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == A.size(4)*A.size(3)*A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == A.size(4)*A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == A.size(4), "");
  RUNTIME_CHECK(A.stride(4) == 1, "");

  ma::batched_Tab_to_Klr(index.size(0), A.size(1), A.size(2), A.size(4), B1.size(1), 
		pointer_dispatch(index.origin()), pointer_dispatch(A.origin()),
		pointer_dispatch(B1.origin()), B1.stride(0), 
		pointer_dispatch(B2.origin()), B2.stride(0), select_backend<MA3>());
}


// B1wn += Apwaan  
// B2wn += Apwaan  
template<class MA1,
         class MA2,
         class MA3,
         class MA4,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 1>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 5>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA4>::dimensionality == 2>
        >
void Apwana_Bwn(MA1&& index, MA2&& A, MA3&& B1, MA4&& B2)
{
  RUNTIME_CHECK(B1.size(1) == B2.size(1), "");
  RUNTIME_CHECK(A.size(1) == B1.size(0), "");
  RUNTIME_CHECK(A.size(1) == B2.size(0), "");
  RUNTIME_CHECK(A.size(2) == A.size(3), "");
  RUNTIME_CHECK(A.stride(0) == A.size(4)*A.size(3)*A.size(2)*A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == A.size(4)*A.size(3)*A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == A.size(4)*A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == A.size(4), "");
  RUNTIME_CHECK(A.stride(4) == 1, "");

  ma::batched_Tanb_to_Klr(index.size(0), A.size(1), A.size(2), A.size(3), B1.size(1),
                pointer_dispatch(index.origin()), pointer_dispatch(A.origin()),
                pointer_dispatch(B1.origin()), B1.stride(0),
                pointer_dispatch(B2.origin()), B2.stride(0), select_backend<MA3>());
}

// Bwn += Awaan  
template<class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 4>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>
        >
void Awaan_Bwn(MA1&& A, MA2&& B)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == A.size(2), "");
  RUNTIME_CHECK(A.size(3) == B.size(1), "");
  RUNTIME_CHECK(A.stride(0) == A.size(3)*A.size(2)*A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == A.size(3)*A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == 1, "");

  ma::Tab_to_Kl(A.size(0), A.size(1), A.size(3), pointer_dispatch(A.origin()),
        	pointer_dispatch(B.origin()), select_backend<MA2>());
}


// Bwn += Awana  
template<class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 4>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>
        >
void Awana_Bwn(MA1&& A, MA2&& B)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(2) == B.size(1), "");
  RUNTIME_CHECK(A.stride(0) == A.size(3)*A.size(2)*A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == A.size(3)*A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == 1, "");

  ma::Tanb_to_Kl(A.size(0), A.size(1), A.size(2), A.size(3), pointer_dispatch(A.origin()),
                pointer_dispatch(B.origin()), B.stride(0), select_backend<MA2>());
}

// for n in [0,N), y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} opA(A)[n,m] * opB(B)[n,m]
template<typename T1, 
	 typename T2,
	 class MA1,
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 1>
        >
MA3&& dot(char TA, char TB, T1 alpha, MA1&& A, MA2&& B, T2 beta, MA3&& y)
{
  bool tA_(TA == 'H' || TA == 'T');
  int N = (tA_ ? A.size(1) : A.size(0));
  int M = (tA_ ? A.size(0) : A.size(1));
  if( TB == 'T' or TB == 'H' ) {
    RUNTIME_CHECK(B.size(0)  == M, "");	
    RUNTIME_CHECK(B.size(1)  == N, "");	
  } else {
    RUNTIME_CHECK(B.size(0)  == N, "");	
    RUNTIME_CHECK(B.size(1)  == M, "");	
  }
  RUNTIME_CHECK(y.size() == N, "");

  ma::strided_batched_dot(TA, TB, N, M, alpha, pointer_dispatch(A.origin()), A.stride(0),
	pointer_dispatch(B.origin()), B.stride(0), beta, pointer_dispatch(y.origin()), y.stride(0), 
	select_backend<MA3>());
  return std::forward<MA3>(y);
}

// for b in [0, nbatch)
//  for n in [0,N), 
//   C[b,n] = beta * C[b,n] + alpha sum_m^{0,M} opA(A[b])[n,m] * opB(B[b])[n,m]
template<typename T1,
         typename T2,
         class MA1,
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 2>,
         typename = void
        >
MA3&& dot(char TA, char TB, char TC, T1 alpha, MA1&& A, MA2&& B, T2 beta, MA3&& C)
{
  bool tA_(TA == 'H' || TA == 'T');
  int N = (tA_ ? A.size(2) : A.size(1));
  int M = (tA_ ? A.size(1) : A.size(2));
  int nbatch = A.size(0);
  utils::check( TC == 'N' or TC == 'T', "dot: Invalid TC");
  utils::check( B.size(0)  == nbatch, "dot: Shape mismatch."  );
  if( TB == 'T' or TB == 'H' ) {
    utils::check( B.size(1)  == M, "dot: Shape mismatch."  );
    utils::check( B.size(2)  == N, "dot: Shape mismatch." );
  } else {
    utils::check( B.size(1)  == N, "dot: Shape mismatch."  );
    utils::check( B.size(2)  == M, "dot: Shape mismatch."  );
  }
  if( TC == 'N' ) {
    utils::check( C.size(0) == nbatch, "dot: Shape mismatch."  );
    utils::check( C.size(1) == N, "dot: Shape mismatch."  );
    ma::strided_batched_dot(TA, TB, nbatch, N, M, alpha, 
        pointer_dispatch(A.origin()), A.stride(1), A.stride(0),
        pointer_dispatch(B.origin()), B.stride(1), B.stride(0), 
        beta, pointer_dispatch(C.origin()), C.stride(1), C.stride(0),
        select_backend<MA3>());
  } else {
    utils::check( C.size(0) == nbatch, "dot: Shape mismatch."  );
    utils::check( C.size(1) == N, "dot: Shape mismatch."  );
    ma::strided_batched_dot(TA, TB, nbatch, N, M, alpha,
        pointer_dispatch(A.origin()), A.stride(1), A.stride(0),
        pointer_dispatch(B.origin()), B.stride(1), B.stride(0),
        beta, pointer_dispatch(C.origin()), C.stride(0), C.stride(1),
        select_backend<MA3>());
  }
  return std::forward<MA3>(C);
}

// for b in [0, nbatch)
//  for n in [0,N), 
//   C[b,n] = beta * C[b,n] + alpha sum_m^{0,M} opA(A[b])[n,m] * opB(B)[n,m]
template<typename T1,
         typename T2,
         class MA1,
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 2>,
         typename = void,
         typename = void
        >
MA3&& dot(char TA, char TB, char TC, T1 alpha, MA1&& A, MA2&& B, T2 beta, MA3&& C)
{
  bool tA_(TA == 'H' || TA == 'T');
  int N = (tA_ ? A.size(2) : A.size(1));
  int M = (tA_ ? A.size(1) : A.size(2));
  int nbatch = A.size(0);
  utils::check( TC == 'N' or TC == 'T', "dot: Invalid TC");
  if( TB == 'T' or TB == 'H' ) {
    utils::check( B.size(0)  == M, "dot: Shape mismatch."  );
    utils::check( B.size(1)  == N, "dot: Shape mismatch." );
  } else {
    utils::check( B.size(0)  == N, "dot: Shape mismatch."  );
    utils::check( B.size(1)  == M, "dot: Shape mismatch."  );
  }
  if( TC == 'N' ) {
    utils::check( C.size(0) == nbatch, "dot: Shape mismatch."  );
    utils::check( C.size(1) == N, "dot: Shape mismatch."  );
    ma::strided_batched_dot(TA, TB, nbatch, N, M, alpha,
        pointer_dispatch(A.origin()), A.stride(1), A.stride(0),
        pointer_dispatch(B.origin()), B.stride(0), 0,
        beta, pointer_dispatch(C.origin()), C.stride(1), C.stride(0),
        select_backend<MA3>());
  } else {
    utils::check( C.size(1) == nbatch, "dot: Shape mismatch."  );
    utils::check( C.size(0) == N, "dot: Shape mismatch."  );
    ma::strided_batched_dot(TA, TB, nbatch, N, M, alpha,
        pointer_dispatch(A.origin()), A.stride(1), A.stride(0),
        pointer_dispatch(B.origin()), B.stride(0), 0,
        beta, pointer_dispatch(C.origin()), C.stride(0), C.stride(1),
        select_backend<MA3>());
  }

  return std::forward<MA3>(C);
}


template<class MA1,
         class MA2,
         class MA3,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA3>::dimensionality == 1>
        >
MA3&& dot(char TA, char TB, MA1&& A, MA2&& B, MA3&& y)
{
  using valy = typename std::decay<MA3>::type::element; 
  return dot(TA,TB,valy(1.0),std::forward<MA1>(A),std::forward<MA2>(B),
		   valy(0.0),std::forward<MA3>(y));
}

template<class CSR, class MA1, class MA2,
         typename = std::enable_if_t<std::decay_t<CSR>::dimensionality == -1>,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 1>
        >
MA2&& spAi_Bij_yj(CSR&& spA, MA1&& B, MA2&& y)
{
  RUNTIME_CHECK(B.stride(1) == 1, "");
  RUNTIME_CHECK(spA.size() == B.size(0), "");
  RUNTIME_CHECK(y.size() == B.size(1), "");
  ma::spVi_Bij_yj(y.size(0), pointer_dispatch(spA.num_non_zero_elements()),
                             pointer_dispatch(spA.non_zero_indices2_data()),
                             pointer_dispatch(spA.non_zero_values_data()),
                             pointer_dispatch(B.origin()), B.stride(0),
                             pointer_dispatch(y.origin()), y.stride(0),
                             select_backend<MA1>());
  return std::forward<MA2>(y);
}

// these don't have MA interface, just hide dispatch

template<typename T1, typename T2, typename T3, typename T4, typename T5>
void batched_ab_ba(T1 n, T2* A, int lda, T3* B, int ldb, T4 alpha, T5* y, int batchSize)
{
  ma::batched_ab_ba(pointer_dispatch(n), A, lda, B, ldb, 
		    alpha, y, batchSize, typename ma_dispatch<T1>::backend{});
}

template<typename T1, typename T2, typename T3, typename T4>
void batched_diagonal_sum(T1 n, T2* A, int lda, T3 alpha, T4* y, int batchSize)
{
  ma::batched_diagonal_sum(pointer_dispatch(n), A, lda, 
                    alpha, y, batchSize, typename ma_dispatch<T1>::backend{});
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
void vbias_from_v1(int nwalk, int nkpts, int nchol_max, T1 Qsym, T2 kminus, T3 ncholpQ, 
		   T4 ncholpQ0, T5 alpha, T6 v1, T7 vb)
{
  ma::vbias_from_v1(nwalk, nkpts, nchol_max, pointer_dispatch(Qsym), pointer_dispatch(kminus), 
		    pointer_dispatch(ncholpQ), pointer_dispatch(ncholpQ0), 
                    static_cast<typename std::pointer_traits<T7>::element_type>(alpha), 
	   	    pointer_dispatch(v1), pointer_dispatch(vb), typename ma_dispatch<T1>::backend{});
}

} // namespace ma

#endif
