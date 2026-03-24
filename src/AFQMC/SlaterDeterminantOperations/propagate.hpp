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
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/sparse/sparse.hpp"
#include "numerics/operations/product.hpp"

namespace sfqmc
{
namespace afqmc
{
namespace det_ops 
{

namespace detail
{

/*
 * Calculate S = exp(im*V)*S using a Taylor expansion of exp(V)
 * Can be used for fully polarized, closed shell, collinear (call each spin separately)
 * or noncollinear with full spin-orbit potential.
 */
template<char TA, nda::MemoryArrayOfRank<3> V_t, nda::MemoryArrayOfRank<3> S_t>
requires( nda::mem::have_compatible_addr_space<V_t,S_t> and math::is_valid_op(TA) and
          std::decay_t<V_t>::is_stride_order_C() and std::decay_t<S_t>::is_stride_order_C()
        )
void apply_expM(V_t const& V, S_t && S, int order = 6)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<S_t>();
  using Type = nda::get_value_t<S_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, Nel] = S.shape();
  utils::check( V.shape() == std::array<long,3>{Nw,M,M}, "Shape mismatch");
  memory::buffered_array<MEM,Type,3> T1(Nw,M,Nel); 
  memory::buffered_array<MEM,Type,3> T2(Nw,M,Nel); 

  Type zero(0.), one(1.);
  Type im(0.0, 1.0);
  if constexpr (TA == 'H' || TA == 'h')
    im = ComplexType(0.0, -1.0);
  auto pT1=std::addressof(T1);
  auto pT2=std::addressof(T2);

  T1() = S();
  for (int n = 1; n <= order; n++)
  {
    Type fact = im * static_cast<Type>(1.0 / static_cast<double>(n));
    math::product<TA>(fact,V,*pT1,zero,*pT2); 
    nda::tensor::add(one,*pT2,one,S);
    std::swap(pT1, pT2);
  }
}

/*
 * Calculate S = exp(im*V)*S using a Taylor expansion of exp(V)
 * In this case, V is a csr_matrix with dimensions [Nw*M, Nw*M].
 * Can be used for fully polarized, closed shell, collinear (call each spin separately)
 * or noncollinear with full spin-orbit potential.
 */
template<char TA, math::sparse::CSRMatrix V_t, nda::MemoryArrayOfRank<3> S_t>
requires(nda::mem::have_compatible_addr_space<V_t,S_t> and std::decay_t<S_t>::is_stride_order_C() and math::is_valid_op(TA))
void apply_expM(V_t const& V, S_t && S, int order = 6)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<S_t>();
  using Type = nda::get_value_t<S_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, Nel] = S.shape();
  utils::check( V.shape() == std::array<long,2>{Nw*M,Nw*M}, "Shape mismatch");
  memory::buffered_array<MEM,Type,2> T1(Nw*M,Nel);
  memory::buffered_array<MEM,Type,2> T2(Nw*M,Nel);

  Type zero(0.), one(1.);
  Type im(0.0, 1.0);
  if constexpr (TA == 'H' || TA == 'h')
    im = ComplexType(0.0, -1.0);

  auto str = S.strides();
  utils::check(str[0] == M*str[1], "Stride mismatch");
  std::array<long,2> shape = {Nw*M,Nel};
  std::array<long,2> strides = {str[1],str[2]};
  nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm(shape,strides);
  memory::array_view<MEM,Type,2> S_(idxm,S.data());

  auto pT1=std::addressof(T1);
  auto pT2=std::addressof(T2);

  T1() = S_();
  for (int n = 1; n <= order; n++)
  {
    Type fact = im * static_cast<Type>(1.0 / static_cast<double>(n));
    math::sparse::csrmm<TA>(fact,V,*pT1,zero,*pT2);
    nda::tensor::add(one,*pT2,"ab",one,S_,"ab");
    std::swap(pT1, pT2);
  }
}

/*
 * Calculate S = exp(im*V)*S using a Taylor expansion of exp(V)
 * Version for non_collinear calculations with a diagonal potential in the spin sector. 
 */
template<char TA, nda::MemoryArrayOfRank<4> V_t, nda::MemoryArrayOfRank<4> S_t>
requires( nda::mem::have_compatible_addr_space<V_t,S_t> and math::is_valid_op(TA) and
          std::decay_t<V_t>::is_stride_order_C() and std::decay_t<S_t>::is_stride_order_C()
        )
void apply_expM(V_t const& V, S_t && S, int order = 6)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<S_t>();
  using Type = nda::get_value_t<S_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, npol, M, Nel] = S.shape();
  utils::check( V.shape() == std::array<long,4>{Nw,npol,M,M}, "Shape mismatch");
  memory::buffered_array<MEM,Type,4> T1(Nw,npol,M,Nel);
  memory::buffered_array<MEM,Type,4> T2(Nw,npol,M,Nel);

  Type zero(0.), one(1.);
  Type im(0.0, 1.0);
  if constexpr (TA == 'H' || TA == 'h')
    im = ComplexType(0.0, -1.0);
  auto pT1=std::addressof(T1);
  auto pT2=std::addressof(T2);

  T1() = S();
  for (int n = 1; n <= order; n++)
  {
    Type fact = im * static_cast<Type>(1.0 / static_cast<double>(n));
    if constexpr (MEM==HOST_MEMORY) {
      if (TA == 'H' || TA == 'h')
        for(int i=0; i<V.extent(0); ++i)
          for(int p=0; p<V.extent(1); ++p)
            nda::blas::gemm(fact,nda::dagger(V(i,p,nda::ellipsis{})),(*pT1)(i,p,nda::ellipsis{}),zero,(*pT2)(i,p,nda::ellipsis{}));
      else if (TA == 'T' || TA == 't')
        for(int i=0; i<V.extent(0); ++i)
          for(int p=0; p<V.extent(1); ++p)
            nda::blas::gemm(fact,nda::transpose(V(i,p,nda::ellipsis{})),(*pT1)(i,p,nda::ellipsis{}),zero,(*pT2)(i,p,nda::ellipsis{}));
      else
        for(int i=0; i<V.extent(0); ++i)
          for(int p=0; p<V.extent(1); ++p)
            nda::blas::gemm(fact,V(i,p,nda::ellipsis{}),(*pT1)(i,p,nda::ellipsis{}),zero,(*pT2)(i,p,nda::ellipsis{}));
    } else {
      if (TA == 'H' || TA == 'h')
        nda::tensor::contract(fact,nda::conj(V),"npji",*pT1,"npjk",zero,*pT2,"npik");
      else if (TA == 'T' || TA == 't')
        nda::tensor::contract(fact,V,"npji",*pT1,"npjk",zero,*pT2,"npik");
      else
        nda::tensor::contract(fact,V,"npij",*pT1,"npjk",zero,*pT2,"npik");
    }
    nda::tensor::add(one,*pT2,one,S);
    std::swap(pT1, pT2);
  }
}

}  // namespace detail

// SM[nbatch][M][NEL]
// P1[M][M]
// V[nbatch][M][M]
template<char TA, nda::MemoryArrayOfRank<3> S_t, typename P_t, typename V_t>
requires( (nda::MemoryArrayOfRank<V_t,3> or math::sparse::CSRMatrix<V_t>) and
          (nda::MemoryArrayOfRank<P_t,2> or math::sparse::CSRMatrix<P_t>) and
          nda::mem::have_compatible_addr_space<V_t,S_t,P_t> and
          std::decay_t<S_t>::is_stride_order_C() and math::is_valid_op(TA) 
        )
void Propagate(S_t && SM, P_t const& P1, V_t const& V, int order = 6)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<S_t>();
  using Type = nda::get_value_t<S_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, Nel] = SM.shape();
  utils::check( P1.shape() == std::array<long,2>{M,M}, "Shape mismatch");

  memory::buffered_array<MEM,Type,3> TMN(Nw,M,Nel);
  // Apply P1     
  math::product<TA>(P1,SM,TMN);
  // Apply exp(i*V)  
  detail::apply_expM<TA>(V, TMN, order);
  // Apply P1
  math::product<TA>(P1,TMN,SM);
}

// Special case for non-collinear calculations with dense, spin-diagonal, potentials.
// SM[nbatch][npol*M][NEL]
// P1[npol*M][npol*M]
// V[nbatch][npol*M][M]
template<char TA, nda::MemoryArrayOfRank<3> S_t, typename P_t, nda::MemoryArrayOfRank<3> V_t>
requires( (nda::MemoryArrayOfRank<P_t,2> or math::sparse::CSRMatrix<P_t>) and
          nda::mem::have_compatible_addr_space<V_t,S_t,P_t> and math::is_valid_op(TA) and
          std::decay_t<S_t>::is_stride_order_C() and std::decay_t<V_t>::is_stride_order_C()
        )
void Propagate_pol(long npol, S_t && SM, P_t const& P1, V_t const& V, int order = 6)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<S_t>();
  using Type = nda::get_value_t<S_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, Mtot, Nel] = SM.shape();
  utils::check(Mtot%npol==0, "npol:{} incompatible with M:{}",npol,Mtot);
  long M = Mtot/npol;
  utils::check( V.shape() == std::array<long,3>{Nw,Mtot,M}, "Shape mismatch");
  utils::check( P1.shape() == std::array<long,2>{Mtot,Mtot}, "Shape mismatch");

  auto str = V.strides();
  std::array<long,4> shape = {Nw,npol,M,M};
  std::array<long,4> strides = {str[0],M*str[1],str[1],str[2]};
  nda::idx_map<4, 0, nda::C_stride_order<4>, nda::layout_prop_e::none> idxm(shape,strides);
  memory::array_view<MEM,const Type,4> V4d(idxm,V.data());

  memory::buffered_array<MEM,Type,3> TMN(Nw,npol*M,Nel);
  auto TMN_4d = nda::reshape(TMN,std::array<long,4>{Nw,npol,M,Nel});
  // Apply P1
  math::product<TA>(P1,SM,TMN);
  // Apply exp(i*V)  
  detail::apply_expM<TA>(V4d, TMN_4d, order);
  // Apply P1
  math::product<TA>(P1,SM,TMN);
}

// Propagate a WalkerSet
// P1(nspin)(npol*NMO,npol*NMO): The matrix can be csr_matrix or nda::MemoryMatrix
// V: vHS[nspin][nw][npol*NMO][npol*NMO]
template<MEMORY_SPACE MEM, char TA, typename WlkSet, typename P_t, typename V_t>
requires( std::decay_t<V_t>::is_stride_order_C() and math::is_valid_op(TA) and 
          (nda::MemoryArrayOfRank<P_t,3> or nda::MemoryArrayOfRank<P_t,1>) and
          (nda::MemoryArrayOfRank<V_t,4> or nda::MemoryArrayOfRank<V_t,1>) ) 
void PropagateWlkSet(WlkSet& wset, P_t const& P1, V_t const& V, int order = 6)
{
  auto all = nda::range::all;
  int nwalk        = wset.size();
  auto walker_type = wset.getWalkerType();
  utils::check(V.extent(1) == nwalk, "Size mismatch");
  long nspin_P1 = P1.extent(0);
  
// MAM: wrong is npol_in_file == 1 in NONCOLLINEAR, fix fix fix!!!
  if constexpr ( nda::MemoryArrayOfRank<V_t,4> ) {
    long nspin_V = V.extent(0);
    if constexpr( nda::MemoryArrayOfRank<P_t,3> ) {
      Propagate<TA>(wset.SlaterMatrices(Alpha),P1(0,nda::ellipsis{}),V(0,all,all,all),order);
      if(walker_type==COLLINEAR)
        Propagate<TA>(wset.SlaterMatrices(Beta),P1(1%nspin_P1,nda::ellipsis{}),V(1%nspin_V,all,all,all),order);
    } else {
      Propagate<TA>(wset.SlaterMatrices(Alpha),P1(0),V(0,all,all,all),order);
      if(walker_type==COLLINEAR)
        Propagate<TA>(wset.SlaterMatrices(Beta),P1(1%nspin_P1),V(1%nspin_V,all,all,all),order);
    }
  } else {
    long nspin_V = V.extent(0);
    if constexpr( nda::MemoryArrayOfRank<P_t,3> ) {
      Propagate<TA>(wset.SlaterMatrices(Alpha),P1(0,nda::ellipsis{}),V(0),order);
      if(walker_type==COLLINEAR)
        Propagate<TA>(wset.SlaterMatrices(Beta),P1(1%nspin_P1,nda::ellipsis{}),V(1%nspin_V),order);
    } else {
      Propagate<TA>(wset.SlaterMatrices(Alpha),P1(0),V(0),order);
      if(walker_type==COLLINEAR)
        Propagate<TA>(wset.SlaterMatrices(Beta),P1(1%nspin_P1),V(1%nspin_V),order);
    }
  }
}

} // namespace det_ops 

} // namespace afqmc

} // namespace sfqmc

