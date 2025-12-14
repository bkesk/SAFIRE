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
  auto all = nda::range::all;
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
  for(int n=0; n<nbatch; ++n)
    nda::copy_select(false, 0, ref, Type(1.0), TMN(n,all,all), Type(0.0), TNN(n,all,all));

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
  auto all = nda::range::all;
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
  for(int n=0; n<nbatch; ++n)
    nda::copy_select(false, 0, ref, Type(1.0), TAB(n,all,all), Type(0.0), TNN(n,all,all));

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
  auto all = nda::range::all;
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
  for(int n=0; n<nbatch; ++n)
    nda::copy_select(false, 0, ref, Type(1.0), TAB(n,all,all), Type(0.0), TNN(n,all,all));

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

} // namespace det_ops 

} // namespace afqmc

} // namespace sfqmc

