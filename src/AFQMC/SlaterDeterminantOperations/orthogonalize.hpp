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
  log_detR() = Type(0.); 
  math::log_determinant_from_geqrf(Q,scl,log_detR);
  
  // Q
  nda::lapack::gqr(nda::transpose(Q),tau,work);

  // copy back
  nda::tensor::add(Q,"nab",A,"nba");  

  // scale A by scl, to make sign of determinant consistent
  if constexpr (nda::mem::have_device_compatible_addr_space<A_t>) {
    nda::tensor::elementwise(scl, "wn", A, "win", nda::tensor::op::MUL);
  } else {
    for (int i = 0; i < M; ++i)
      A(nda::range::all,i,nda::range::all) *= scl();
  }
}

template<nda::MemoryArrayOfRank<3> A_t>
requires( nda::mem::have_compatible_addr_space<A_t> and std::decay_t<A_t>::is_stride_order_C() )
void orthogonalize(A_t && A)
{ 
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<A_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, Nel] = A.shape();
  if(A.size()==0) return;
  memory::buffered_array<MEM,Type,1> ldet(Nw,Type(0.));
  orthogonalize(std::forward<A_t>(A),ldet);
}

} // namespace det_ops 

} // namespace afqmc

} // namespace sfqmc
