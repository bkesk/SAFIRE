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

#pragma once

#include "AFQMC/config.h"
#include "utilities/check_strides.hpp"
#include "numerics/operations/determinants.hpp"
#include "numerics/nda_functions.hpp"
#include "nda/tensor.hpp"

namespace sfqmc
{
namespace afqmc
{
namespace det_ops 
{

  using math::sparse::CSRMatrix;

namespace detail
{

template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<1> O_t,
         nda::MemoryArrayOfRank<3> T_t>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, B_t, O_t, T_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<T_t>::is_stride_order_C()
        )
void log_overlap_impl(A_t const& A, B_t const& B, O_t && ovlp, T_t && TNN, bool herm = true, bool invert = false)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  if(herm)
    utils::check(A.shape() == std::array<long,2>{NEL,NMO}, "Size mismatch");
  else
    utils::check(A.shape() == std::array<long,2>{NMO,NEL}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  utils::check(TNN.shape() == std::array<long,3>{nbatch,NEL,NEL}, "Size mismatch"); 

  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  if constexpr (CSRMatrix<A_t>) {
    if(herm)
      math::sparse::csrmm<'N'>(A,B,TNN);
    else
      math::sparse::csrmm<'H'>(A,B,TNN);
  } else {
    if(herm)
      nda::tensor::contract(A,"ij",B,"njk",TNN,"nik");
    else
      nda::tensor::contract(nda::conj(A),"ji",B,"njk",TNN,"nik");
  }

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert
  if(invert)
    nda::lapack::getri(TNN,ipiv,work);
}

template<nda::MemoryArrayOfRank<3> A_t, nda::MemoryArrayOfRank<3> B_t, 
         nda::MemoryArrayOfRank<1> O_t, nda::MemoryArrayOfRank<3> T_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, B_t, O_t, T_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C() and
          std::decay_t<T_t>::is_stride_order_C() 
        )
void log_overlap_impl(A_t const& A, B_t const& B, O_t && ovlp, T_t && TNN, bool invert = false)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  utils::check(A.shape() == B.shape(), "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  utils::check(TNN.shape() == std::array<long,3>{nbatch,NEL,NEL}, "Size mismatch");

  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  nda::tensor::contract(nda::conj(A),"nji",B,"njk",TNN,"nik");

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert
  if(invert)
    nda::lapack::getri(TNN,ipiv,work);
}

//inverse and log(det)
template<typename A_t, typename O_t, typename T_t>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, O_t, T_t> and
          std::decay_t<T_t>::is_stride_order_C()
        )
void inverse_logdet(A_t const& A, O_t && ovlp, T_t && TNN)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<A_t>;

  auto NMO = A.shape()[0];

  utils::check(A.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  utils::check(TNN.shape() == std::array<long,3>{1,NMO,NMO}, "Size mismatch"); 

  memory::buffered_array<MEM,int,2> ipiv(1,NMO);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  TNN(0,nda::range(NMO),nda::range(NMO)) = A;

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert
  nda::lapack::getri(TNN,ipiv,work);
}

template<nda::MemoryArrayOfRank<3> A_t, typename O_t, nda::MemoryArrayOfRank<3> T_t>
requires( nda::mem::have_compatible_addr_space<A_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, O_t, T_t> and
          std::decay_t<A_t>::is_stride_order_C() and
          std::decay_t<T_t>::is_stride_order_C()
        )
void inverse_logdet(A_t const& A, O_t && ovlp, T_t && TNN)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<A_t>;

  auto [nbatch, NMO, NMO2] = A.shape();

  //utils::check(A.shape() == std::array<long,3>{NMO,NMO}, "Size mismatch");
  utils::check(TNN.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch"); 

  memory::buffered_array<MEM,int,2> ipiv(nbatch,NMO);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  TNN = A;

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert
  nda::lapack::getri(TNN,ipiv,work);
}

template<nda::MemoryArrayOfRank<2> A_t, nda::MemoryArrayOfRank<2> B_t,
                     nda::MemoryArrayOfRank<2> C_t, nda::MemoryArrayOfRank<1> O_t,
                     nda::MemoryArrayOfRank<1> T_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,C_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, O_t, T_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C() and
          std::decay_t<C_t>::is_stride_order_C() 
        )
void splitDmatrix(A_t const& A, B_t&& B, C_t&& C, O_t&& logdet, T_t const& scl0)
{

  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO] = B.shape();
  utils::check(A.shape() == B.shape(), "Size mismatch");
  utils::check(B.shape() == C.shape(), "Size mismatch");
  utils::check(logdet.size() >= nbatch, "");
  utils::check(scl0.size() == nbatch, "");

  logdet() = 0.0; //

  for (int nb = 0; nb < nbatch; nb++){
    for (int i = 0; i < NMO; i++)
    {
      double ksi = log(A(nb,i).real()) + scl0(nb).real();
          
      if(ksi > 0.0)
      {
        B(nb,i) = 1.0; // Dmin
        if(ksi >= 32*log(10.0)){
          C(nb,i) = 0.0; // Dmax^-1
        }
        else{
          C(nb,i) = exp(-1.0*ksi);
        }
        logdet(nb) += ksi; // store log(det(Dmax))
      }
      else
      {
        if(ksi <= -32*log(10.0)){
          B(nb,i) = 0.0;
        }
        else{
          B(nb,i) = exp(ksi);
        }
        C(nb,i) = 1.0;
      }
    }
  }

}

template<nda::MemoryArrayOfRank<1> A_t, nda::MemoryArrayOfRank<1> B_t,
                     nda::MemoryArrayOfRank<1> C_t, typename T_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,C_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, T_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C() and
          std::decay_t<C_t>::is_stride_order_C() 
        )
auto splitDmatrix(A_t const& A, B_t&& B, C_t&& C, T_t const& scl0)
{

  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto NMO = B.size();
  utils::check(A.shape() == B.shape(), "Size mismatch");
  utils::check(B.shape() == C.shape(), "Size mismatch");

  auto logdet = 0.0;

  for (int i = 0; i < NMO; i++)
  {
    double ksi = log(A(i).real()) + scl0.real();
        
    if(ksi > 0.0)
    {
      B(i) = 1.0; // Dmin
      if(ksi >= 32*log(10.0)){
        C(i) = 0.0; // Dmax^-1
      }
      else{
        C(i) = exp(-1.0*ksi);
      }
      logdet += ksi; // store log(det(Dmax))
    }
    else
    {
      if(ksi <= -32*log(10.0)){
        B(i) = 0.0;
      }
      else{
        B(i) = exp(ksi);
      }
      C(i) = 1.0;
    }
  }

  return logdet;

}


template<nda::MemoryArrayOfRank<3> A_t, nda::MemoryArrayOfRank<3> B_t, typename O_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,O_t> and
          nda::have_same_value_type_v<A_t, B_t, O_t> and
          std::decay_t<A_t>::is_stride_order_C() and
          std::decay_t<B_t>::is_stride_order_C()
        )
void LUsolve(A_t && A, B_t && B, O_t && ovlp)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<A_t>;

  auto [nbatch, NMO, NMO2] = A.shape();

  utils::check(B.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch"); 

  memory::buffered_array<MEM,Type,3,nda::F_layout> AT(NMO,NMO,nbatch);
  memory::buffered_array<MEM,Type,3,nda::F_layout> T0(NMO,NMO,nbatch);
  memory::buffered_array<MEM,int,2,nda::F_layout> ipiv(NMO,nbatch);
  
  memory::buffered_array<MEM,Type,1> work;
 
  ipiv() = 0;

  //nda::tensor::add(A,"nij",AT,"ijn");
  //permute indices
  AT = nda::permuted_indices_view<nda::encode(nda::permutations::cycle<3>(1))>(std::forward<A_t>(A));

  //nda::tensor::add(B,"nij",T0,"ijn");
  //permute indices
  T0 = nda::permuted_indices_view<nda::encode(nda::permutations::cycle<3>(1))>(std::forward<B_t>(B));

  // LU 
  nda::lapack::getrf(AT,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(AT,ipiv,ovlp);

  // solve Ax = b
  nda::lapack::getrs(AT,T0,ipiv);

  //nda::tensor::add(T0,"ijn",B,"nij");
  //permute indices
  B = nda::permuted_indices_view<nda::encode(nda::permutations::cycle<3>(2))>(T0);

}

}


template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<1> O_t>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t> and 
          nda::have_same_value_type_v<A_t, B_t, O_t> and
          std::decay_t<B_t>::is_stride_order_C() 
        ) 
void Log_Overlap(A_t const& A, B_t const& B, O_t && ovlp, bool herm = true)
{
  utils::check_strides(B,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>(); 
  using Type = nda::get_value_t<B_t>;
  auto [nbatch, NMO, NEL] = B.shape();
  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL); 

  detail::log_overlap_impl(A,B,ovlp,TNN,herm);
}

template<nda::MemoryArrayOfRank<3> A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<1> O_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,O_t> and
          nda::have_same_value_type_v<A_t, B_t, O_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()
        )
void Log_Overlap(A_t const& A, B_t const& B, O_t && ovlp)
{
  utils::check_strides(A,B,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;
  auto [nbatch, NMO, NEL] = A.shape();
  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);

  detail::log_overlap_impl(A,B,ovlp,TNN);
}

template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<1> O_t,
         nda::MemoryArrayOfRank<3> QQ0_t, nda::MemoryVector IVec>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t,QQ0_t,IVec> and
          nda::have_same_value_type_v<A_t, B_t, O_t, QQ0_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<QQ0_t>::is_stride_order_C()
        )
void Log_OverlapForWoodbury(A_t const& A, B_t const& B, O_t && ovlp, QQ0_t && QQ0, IVec && ref)
{
  utils::check_strides(B,ovlp,QQ0,ref);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  auto NACT = A.extent(0);
  utils::check(A.shape() == std::array<long,2>{NACT,NMO}, "Size mismatch");
  utils::check(QQ0.shape() == std::array<long,3>{nbatch,NACT,NEL}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "Size mismatch");
  utils::check(ref.size() == NEL, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TMN(nbatch,NACT,NEL);
  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);
  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  if constexpr (CSRMatrix<A_t>) {
    math::sparse::csrmm<'N'>(A,B,TMN);
  } else {
    utils::check_strides(A);
    nda::tensor::contract(A,"ij",B,"njk",TMN,"nik");
  }

  // TNN(i,:) = TMN(ref(i),:)
  nda::copy_select(false, 0, ref, Type(1.0), TMN, Type(0.0), TNN);

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert 
  nda::lapack::getri(TNN,ipiv,work);

  // fill_if_zero()

  // QQ0 = TMN * inv(TNN)
  nda::tensor::contract(TMN,"nij",TNN,"njk",QQ0,"nik");
}

// Density Matrices
template<typename A_t, 
         nda::MemoryArrayOfRank<3> B_t, 
         nda::MemoryArrayOfRank<3> C_t, 
         nda::MemoryArrayOfRank<1> O_t>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,C_t,O_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, O_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<C_t>::is_stride_order_C()
        )
void MixedDensityMatrix(A_t const& A, B_t const& B, C_t && C, O_t && ovlp, bool compact = true, bool herm = true)
{
  utils::check_strides(B,C,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  if(herm)
    utils::check(A.shape() == std::array<long,2>{NEL,NMO}, "Size mismatch");
  else
    utils::check(A.shape() == std::array<long,2>{NMO,NEL}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  if(compact)
    utils::check(C.shape() == std::array<long,3>{nbatch, NEL,NMO}, "Size mismatch");
  else
    utils::check(C.shape() == std::array<long,3>{nbatch, NMO,NMO}, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);

  // A*B and overlap
  detail::log_overlap_impl(A,B,ovlp,TNN,herm,true);

  // zero out TNN if determinant is zero
//  math::fill_if_zero(TNN, ovlp, Type(0.0));

  if(compact) {

    nda::tensor::contract(TNN,"nji",B,"nkj",C,"nik");

  } else {
    memory::buffered_array<MEM,Type,3> TNM(nbatch,NEL,NMO);

    if constexpr (CSRMatrix<A_t>) {
      if (herm)
      {
        // T2 = T(T1) * T(B)
        nda::tensor::contract(TNN,"nji",B,"nkj",TNM,"nik");

        // C = conj(A) * T2
        math::sparse::csrmm<'T'>(A,TNM,C);
      }
      else
      {
        // T2 = T1 * H(A)
        // can't do TNN*H(A), what to do???
        //ma::productStridedBatched(TNN3D, ma::H(hermA), TNM3D);  
        sfqmc::utils::check(false, "finish implementation");

        // T2 = T(T1) * T(B)
        // C = T( B * T2) = T(T2) * T(B)
        nda::tensor::contract(TNM,"nji",B,"nkj",C,"nik");
      }
    } else {
      if (herm)
      { 
        // T2 = T(T1) * T(B)
        nda::tensor::contract(TNN,"nji",B,"nkj",TNM,"nik");
        
        // C = conj(A) * T2
        nda::tensor::contract(A,"ji",TNM,"njk",C,"nik");
      }
      else
      { 
        // T2 = T1 * H(A)
        nda::tensor::contract(TNN,"nij",nda::conj(A),"kj",TNM,"nik");
        
        // T2 = T(T1) * T(B)
        // C = T( B * T2) = T(T2) * T(B)
        nda::tensor::contract(TNM,"nji",B,"nkj",C,"nik");
      }
    }
  }
}

template<nda::MemoryArrayOfRank<3> A_t, 
         nda::MemoryArrayOfRank<3> B_t, 
         nda::MemoryArrayOfRank<3> C_t, 
         nda::MemoryArrayOfRank<1> O_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,C_t,O_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, O_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C() and 
          std::decay_t<C_t>::is_stride_order_C()
        )
void MixedDensityMatrix(A_t const& A, B_t const& B, C_t && C, O_t && ovlp, bool compact = true)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = A.shape();
  utils::check(A.shape() == B.shape(), "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  if(compact)
    utils::check(C.shape() == std::array<long,3>{nbatch, NEL,NMO}, "Size mismatch");
  else
    utils::check(C.shape() == std::array<long,3>{nbatch, NMO,NMO}, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);

  // A*B and overlap
  detail::log_overlap_impl(A,B,ovlp,TNN,true);

  // zero out TNN if determinant is zero
//  math::fill_if_zero(TNN, ovlp, Type(0.0));

  if(compact) {

    nda::tensor::contract(TNN,"nji",B,"nkj",C,"nik");

  } else {
    memory::buffered_array<MEM,Type,3> TNM(nbatch,NEL,NMO);

    // T2 = T1 * H(A)
    nda::tensor::contract(TNN,"nij",nda::conj(A),"kj",TNM,"nik");
        
    // T2 = T(T1) * T(B)
    // C = T( B * T2) = T(T2) * T(B)
    nda::tensor::contract(TNM,"nji",B,"nkj",C,"nik");
  }
}

template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<3> C_t, 
         nda::MemoryArrayOfRank<1> O_t, nda::MemoryArrayOfRank<3> QQ0_t, nda::MemoryVector IVec>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t,C_t,QQ0_t,IVec> and
          nda::have_same_value_type_v<A_t, B_t, O_t, C_t,QQ0_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<QQ0_t>::is_stride_order_C()
          and std::decay_t<C_t>::is_stride_order_C()
        )
void MixedDensityMatrixForWoodbury(A_t const& A, B_t const& B, C_t &&C, O_t && ovlp, QQ0_t && QQ0, IVec && ref, bool compact)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  auto NACT = A.extent(0);
  utils::check(A.shape() == std::array<long,2>{NACT,NMO}, "Size mismatch");
  if(compact)
    utils::check(C.shape() == std::array<long,3>{nbatch, NEL,NMO}, "Size mismatch");
  else
    utils::check(C.shape() == std::array<long,3>{nbatch, NMO,NMO}, "Size mismatch");
  utils::check(QQ0.shape() == std::array<long,3>{nbatch,NACT,NEL}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "Size mismatch");
  utils::check(ref.size() == NEL, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);
  memory::buffered_array<MEM,Type,3> TAB(nbatch,NACT,NEL);
  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  if constexpr (CSRMatrix<A_t>) {
    math::sparse::csrmm<'N'>(A,B,TAB);
  } else {
    nda::tensor::contract(A,"ij",B,"njk",TAB,"nik");
  }

  // TNN(i,:) = TAB(ref(i),:)
  nda::copy_select(false, 0, ref, Type(1.0), TAB, Type(0.0), TNN);

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert 
  nda::lapack::getri(TNN,ipiv,work);

  // fill_if_zero()

  // QQ0 = TAB * inv(TNN)
  nda::tensor::contract(TAB,"nij",TNN,"njk",QQ0,"nik");

  if(compact) {

    // C = T(TNN) * T(B)
    nda::tensor::contract(TNN,"nji",B,"nkj",C,"nik");

  } else {

    memory::buffered_array<MEM,Type,3> TNM(nbatch,NEL,NMO);

    // TNM = T(TNN) * T(B)
    nda::tensor::contract(TNN,"nji",B,"nkj",TNM,"nik");

    // C = conj(A) * TNM
    if constexpr (CSRMatrix<A_t>) {
      math::sparse::csrmm<'T'>(A,TNM,C);
    } else {
      nda::tensor::contract(A,"nji",TNM,"njk",C,"nik");
    }

  } 
}

template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<3> C_t,
         nda::MemoryArrayOfRank<1> O_t, nda::MemoryVector IVec>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t,C_t,IVec> and
          nda::have_same_value_type_v<A_t, B_t, O_t, C_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<C_t>::is_stride_order_C()
        )
void MixedDensityMatrixFromConfiguration(A_t const& A, B_t const& B, C_t &&C, O_t && ovlp, IVec && ref, bool compact)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  auto NACT = A.extent(0);
  utils::check(A.shape() == std::array<long,2>{NACT,NMO}, "Size mismatch");
  if(compact)
    utils::check(C.shape() == std::array<long,3>{nbatch, NEL,NMO}, "Size mismatch");
  else
    utils::check(C.shape() == std::array<long,3>{nbatch, NMO,NMO}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "Size mismatch");
  utils::check(ref.size() == NEL, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);
  memory::buffered_array<MEM,Type,3> TAB(nbatch,NACT,NEL);
  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  if constexpr (CSRMatrix<A_t>) {
    math::sparse::csrmm<'N'>(A,B,TAB);
  } else {
    nda::tensor::contract(A,"ij",B,"njk",TAB,"nik");
  }

  // TNN(i,:) = TAB(ref(i),:)
  nda::copy_select(false, 0, ref, Type(1.0), TAB, Type(0.0), TNN);

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert 
  nda::lapack::getri(TNN,ipiv,work);

  // fill_if_zero()

  if(compact) {

    // C = T(TNN) * T(B)
    nda::tensor::contract(TNN,"nji",B,"nkj",C,"nik");

  } else {

    memory::buffered_array<MEM,Type,3> TNM(nbatch,NEL,NMO);

    // TNM = T(TNN) * T(B)
    nda::tensor::contract(TNN,"nji",B,"nkj",TNM,"nik");

    // C = conj(A) * TNM
    if constexpr (CSRMatrix<A_t>) {
      math::sparse::csrmm<'T'>(A,TNM,C);
    } else {
      nda::tensor::contract(A,"nji",TNM,"njk",C,"nik");
    }

  } 
}

// Finite temperature Density Matrices
template<typename A_t, typename B_t, typename C_t,
         nda::MemoryArrayOfRank<3> D_t,
         nda::MemoryArrayOfRank<2> E_t,
         nda::MemoryArrayOfRank<3> F_t,
         nda::MemoryArrayOfRank<3> G_t, 
         nda::MemoryArrayOfRank<1> O_t,
         typename SL_t,
         nda::MemoryArrayOfRank<1> SR_t>
requires( ((CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
           nda::MemoryVector<B_t> and
           (CSRMatrix<C_t> or nda::MemoryMatrix<C_t>)) and
          nda::mem::have_compatible_addr_space<A_t,B_t,C_t,D_t,E_t,F_t,G_t,O_t,SR_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, D_t, E_t, F_t, G_t, O_t, SR_t> and
          std::decay_t<D_t>::is_stride_order_C() and std::decay_t<E_t>::is_stride_order_C() and
          std::decay_t<F_t>::is_stride_order_C() and std::decay_t<G_t>::is_stride_order_C()
        )
void MixedDensityMatrix(A_t const& UL, B_t const& DL, C_t const& VL, 
                        D_t const& UR, E_t const& DR, F_t const& VR,
                        G_t && G, O_t && ovlp, SL_t const& sclL, SR_t const& sclR, 
                        bool unitaryL = false, bool unitaryR = false)
{
  utils::check_strides(UR,DR,VR,G,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = UR.shape();
  
  utils::check(UL.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  utils::check(DL.shape() == std::array<long,1>{NMO}, "Size mismatch");
  utils::check(VL.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  utils::check(DR.shape() == std::array<long,2>{nbatch,NMO}, "Size mismatch");
  utils::check(VR.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  utils::check(G.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch");

  memory::buffered_array<MEM,Type,3> UL_inv_3D(1,NMO,NMO); // 3D to be compatible with log_determinant_from_getrf
  memory::buffered_array<MEM,Type,2> UL_inv(NMO,NMO);
  memory::buffered_array<MEM,Type,2> DL_UL(NMO,NMO);
  memory::buffered_array<MEM,Type,3> UR_inv(nbatch,NMO,NMO);
  memory::buffered_array<MEM,Type,3> M1(nbatch,NMO,NMO);
  memory::buffered_array<MEM,Type,3> M2(nbatch,NMO,NMO);
  memory::buffered_array<MEM,Type,3> M3(nbatch,NMO,NMO);

  Type logdetDL;
  // matrices to store terms to compute log(P_T)
  memory::buffered_array<MEM,Type,1> logdetDL_vec(nbatch,Type(0.0));
  memory::buffered_array<MEM,Type,1> logdetUL(1,Type(0.0));
  memory::buffered_array<MEM,Type,1> logdetUL_vec(nbatch,Type(0.0));
  memory::buffered_array<MEM,Type,1> logdetDR(nbatch,Type(0.0));
  memory::buffered_array<MEM,Type,1> logdetUR(nbatch,Type(0.0));
  memory::buffered_array<MEM,Type,1> logdetM(nbatch,Type(0.0));

  // matrices to store terms to Dmax^-1, Dmin
  memory::buffered_array<MEM,Type,2> DRmin(nbatch,NMO);
  memory::buffered_array<MEM,Type,2> DRmax_inv(nbatch,NMO);
  memory::buffered_array<MEM,Type,1> DLmin(NMO);
  memory::buffered_array<MEM,Type,1> DLmax_inv(NMO);

  logdetDL = detail::splitDmatrix(DL, DLmin, DLmax_inv, sclL);
  logdetDL_vec() = logdetDL;
  detail::splitDmatrix(DR, DRmin, DRmax_inv, logdetDR, sclR);

  // UL^-1
  detail::inverse_logdet(UL,logdetUL,UL_inv_3D);
  UL_inv() = UL_inv_3D(0,nda::ellipsis{});
  logdetUL_vec() = logdetUL(0);
  // UR^-1
  detail::inverse_logdet(UR,logdetUR,UR_inv);

  // UR^-1*UL^-1
  nda::tensor::contract(UR_inv,"nik",UL_inv,"kj",M1,"nij");

  // DRmax^-1*UR^-1*UL^-1
  nda::tensor::contract(M1,"nij",DRmax_inv,"ni",M2,"nij");

  // DRmax^-1*UR^-1*UL^-1*DLmax^-1
  nda::tensor::contract(M2,"nij",DLmax_inv,"j",M1,"nij");

  // VR * VL
  nda::tensor::contract(VR,"nij",VL,"jk",M2,"nik");

  // DRmin*VR*VL
  nda::tensor::contract(M2,"nij",DRmin,"ni",M3,"nij");
  
  // DRmin*VR*VL*DLmin
  nda::tensor::contract(M3,"nij",DLmin,"j",M2,"nij");

  //M2 <-- M1 + M2;
  nda::tensor::add(ComplexType(1.0),M1,"nij",ComplexType(1.0),M2,"nij");

  // DRmax^-1*UR^-1
  nda::tensor::contract(UR_inv,"nij",DRmax_inv,"ni",M1,"nij");

  // LU solve for [DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
  detail::LUsolve(M2,M1,logdetM);

  // UL^-1*DLmax^-1
  nda::tensor::contract(UL_inv,"ij",DLmax_inv,"j",DL_UL,"ij");

  // Gp^T = <c_i c_j^+>^T
  //    = UL^-1*DLmax^-1*[DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
  nda::tensor::contract(DL_UL,"ij",M1,"njk",G,"nki");

  // FIX: is there a better way to construct a tensor of
  // identity matrices? Or can we do the addition without
  // needing a tensor of identity matrices?
  memory::buffered_array<MEM,Type,3> It(nbatch,NMO,NMO);
  for(int i = 0; i < nbatch; ++i)
    It(i,nda::range(NMO),nda::range(NMO)) = nda::eye<Type>(NMO);

  // G = <c_i^+ c_j> --> I - Gp^T
  nda::tensor::add(ComplexType(1.0),It,"nij",ComplexType(-1.0),G,"nij");

  // log(PT) = log(detUR) + log(detDRmax) + log(detM3) + log(detDLmax) + log(detUL)
  ovlp(nda::range(nbatch)) = logdetUR + logdetDR + logdetM + logdetDL_vec + logdetUL_vec; 

}

} // namespace det_ops 

} // namespace afqmc

} // namespace sfqmc

