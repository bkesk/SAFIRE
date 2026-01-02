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
  
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "AFQMC/config.h"
#include "nda/nda.hpp"
#include "nda/tensor.hpp" 
#include "numerics/sparse/sparse.hpp"
#include "numerics/operations/determinants.hpp"
  
namespace sfqmc
{ 
namespace afqmc
{ 
namespace det_ops 
{ 

template<nda::MemoryArrayOfRank<3> A_t, nda::MemoryVector B_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()
        )
void orthogonalize(A_t && A, B_t && log_detR) 
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<A_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, Nel] = A.shape();
  if(A.size()==0) return;
  utils::check( log_detR.extent(0) >= Nw, "Size mismatch");

  // transposing for now, can call geqlf if available in principle
  memory::buffered_array<MEM,Type,3> Q(Nw,Nel,M);
  memory::buffered_array<MEM,Type,2> tau(Nw,Nel);
  memory::buffered_array<MEM,Type,2> scl(Nw,Nel);
  memory::buffered_array<MEM,Type,1> work;

  nda::tensor::add(A,"nab",Q,"nba");  
  nda::lapack::geqrf(nda::transpose(Q),tau,work);

  // log(Det)
  math::log_determinant_from_geqrf(Q,scl,log_detR(nda::range(Nw)));
  
  // Q
  nda::lapack::gqr(nda::transpose(Q),tau,work);

  // copy back
  nda::tensor::add(Q,"nab",A,"nba");  

  // scale A by scl, to make sign of determinant consistent
  if constexpr (nda::mem::have_device_compatible_addr_space<A_t>) {
    nda::tensor::elementwise(ComplexType(1.0), scl, "wn", ComplexType(1.0), A, "win", nda::tensor::op::MUL);
  } else {
    for (int i = 0; i < M; ++i)
      A(nda::range::all,i,nda::range::all) *= scl();
  }
}

template<typename WlkSet, nda::MemoryVector Vec>
requires( not nda::Array<WlkSet> )
void orthogonalize(WlkSet &wset, Vec && ldet, bool importance_sampling = true) 
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<Vec>();
  utils::check(MEM == wset.get_memory_space(), "Memory space mismatch");
  memory::check_memory_space<MEM>(ldet);
  auto walker_type = wset.getWalkerType();
  const int nspin = ( (walker_type == COLLINEAR) ? 2 : 1 );
  const int nwalk = wset.size();
  utils::check(ldet.size() >= nwalk, "Size mismatch");
  ldet() = ComplexType(0.0);
  if(importance_sampling) {
    orthogonalize( wset.template SlaterMatrices<MEM>(Alpha), ldet);
    if(walker_type == COLLINEAR)
      orthogonalize( wset.template SlaterMatrices<MEM>(Beta), ldet);
  } else {
    double scl = ( walker_type == CLOSED ? 2.0 : 1.0 );
    orthogonalize( wset.template SlaterMatrices<MEM>(Alpha), ldet);
    if(walker_type == COLLINEAR)
      orthogonalize( wset.template SlaterMatrices<MEM>(Beta), ldet);
    memory::buffered_array<MEM,ComplexType,1> wgt(nwalk);
    wset.getProperty(WEIGHT, wgt);
    auto wgt_h = nda::to_host(wgt);
    auto ldet_h = nda::to_host(ldet);
    wgt_h() *= nda::exp(scl*ldet_h());
    wgt() = wgt_h();
    wset.setProperty(WEIGHT, wgt);
  }
}

} // namespace det_ops 

} // namespace afqmc

} // namespace sfqmc
