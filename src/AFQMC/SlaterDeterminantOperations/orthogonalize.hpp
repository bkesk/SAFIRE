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
#include "numerics/operations/tensor.hpp"
#include "numerics/nda_linalg_lapack_extensions.hpp"
  
namespace sfqmc
{ 
namespace afqmc
{ 
namespace det_ops 
{ 

  namespace detail
  {

    template<nda::MemoryVector A_t, nda::MemoryVector B_t>
      requires(nda::mem::have_compatible_addr_space<A_t,B_t> and
              std::decay_t<A_t>::is_stride_order_Fortran() and
              std::decay_t<B_t>::is_stride_order_Fortran())
    int partition(A_t && A, B_t && B, int low, int high) {

      // Selecting last element as the pivot
      auto pivot = A(high);

      // Index of elemment just before the last element
      // It is used for swapping
      int i = (low - 1);

      for (int j = low; j <= high - 1; j++) {
        // If current element is great than pivot
        if (A(j).real() >= pivot.real()) {
            i++;
            std::swap(A(i), A(j));
            std::swap(B(i), B(j));
        }
      }

      // Put pivot to its position
      std::swap(A(i + 1), A(high));
      std::swap(B(i + 1), B(high));
    
      // Return the point of partition
      return (i + 1);
    }

    /*
    Routine to sort D vector and return sorted indices
    */
    template<nda::MemoryVector A_t, nda::MemoryVector B_t>
      requires(nda::mem::have_compatible_addr_space<A_t,B_t> and
              std::decay_t<A_t>::is_stride_order_Fortran() and
              std::decay_t<B_t>::is_stride_order_Fortran())
    void quick_sort(A_t && A, B_t && B, int low, int high)
    {
      // Base case: This part will be executed until the starting
      // index low is lesser than the ending index high
      if (low < high) {

        // pi is Partitioning Index, arr[p] is now at
        // right place
        int pi = partition(A, B, low, high);

        // Separately sort elements before and after the
        // Partition Index pi
        quick_sort(A, B, low, pi - 1);
        quick_sort(A, B, pi + 1, high);
      }
    }

    template<nda::MemoryArrayOfRank<2> A_t, nda::MemoryArrayOfRank<2> B_t>
      requires(nda::mem::have_compatible_addr_space<A_t,B_t> and
              std::decay_t<A_t>::is_stride_order_Fortran() and
              std::decay_t<B_t>::is_stride_order_Fortran())
    void quick_sort(A_t && A, B_t && B)
    {
      auto const [n, nbatch] = A.shape();

      for(int b = 0; b < nbatch; ++b){
        quick_sort(A(nda::range::all,b),B(nda::range::all,b),0,n-1);
      }
    }

    /*
      Routine to get Q, R matrices from QR decomposition
    */
   /*
    template <nda::MemoryArrayOfRank<3> A, nda::MemoryMatrix TAU>
      requires(nda::mem::have_host_compatible_addr_space<A, TAU> and nda::have_same_value_type_v<A, TAU> 
              and nda::is_blas_lapack_v<nda::get_value_t<A>>)
    auto get_qr_matrices(A const &a, TAU const &tau, bool complete = false) {

      constexpr MEMORY_SPACE MEM = memory::get_memory_space<A>();

      auto all = nda::range::all;

      auto const [m, n, nbatch] = a.shape();
      auto const min_mn = std::min(m, n);
      auto const k      = (complete ? m : min_mn);
      auto Q            = memory::buffered_array<MEM, nda::get_value_t<A>, 3, nda::F_layout>::zeros(m, k, nbatch);
      auto R            = memory::buffered_array<MEM, nda::get_value_t<A>, 3, nda::F_layout>::zeros(k, n, nbatch);

      // compute Q matrix
      for(int b = 0; b < nbatch; ++b){
        Q(all, nda::range(min_mn),b) = a(all, nda::range(min_mn), b);
        int info{};
        if constexpr (nda::is_complex_v<nda::get_value_t<A>>) {
          info = nda::lapack::ungqr(Q(all,all,b), tau(all,b));
        } else {
          info = nda::lapack::orgqr(Q(all,all,b), tau(all,b));
        }
        if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::qr_in_place: orgqr/ungqr returned a non-zero value: info = " << info;

        // extract R matrix
        for (int i = 0; i < min_mn; ++i) R(nda::range(i + 1), i, b) = a(nda::range(i + 1), i, b);
        for (int i = min_mn; i < n; ++i) R(all, i, b) = a(all, i, b);
      }
      return std::make_tuple(Q, R);

    }
    */

  }

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


// Finite temperature w/QR
template<nda::MemoryArrayOfRank<3> U_t, nda::MemoryArrayOfRank<2> D_t,
         nda::MemoryArrayOfRank<3> V_t, nda::MemoryVector B_t>
requires( nda::mem::have_compatible_addr_space<U_t,D_t,V_t,B_t> and
          std::decay_t<U_t>::is_stride_order_C() and std::decay_t<D_t>::is_stride_order_C() and
          std::decay_t<V_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()
        )
void orthogonalize_wQR(U_t && U, D_t && D, V_t && V, B_t && scl) 
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<U_t>();
  using Type = nda::get_value_t<U_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, M2] = U.shape();
  if(U.size()==0) return;
  utils::check( scl.extent(0) >= Nw, "Size mismatch");

  memory::buffered_array<MEM,Type,3,nda::F_layout> UT(M,M,Nw);
  memory::buffered_array<MEM,Type,3,nda::F_layout> VT(M,M,Nw);
  memory::buffered_array<MEM,Type,3,nda::F_layout> P(M,M,Nw);
  memory::buffered_array<MEM,Type,2,nda::F_layout> DT(M,Nw);
  memory::buffered_array<MEM,Type,1,nda::F_layout> scl_new(Nw);
  memory::buffered_array<MEM,Type,2,nda::F_layout> Dinv(M,Nw);

  memory::buffered_array<MEM,int,2,nda::F_layout> jpvt(M,Nw);
  memory::buffered_array<MEM,Type,2,nda::F_layout> tau(M,Nw);

  memory::buffered_array<MEM,Type,1,nda::F_layout> work;

  memory::buffered_array<MEM,int,2,nda::F_layout> Psort_vec(M,Nw);
  memory::buffered_array<MEM,Type,3,nda::F_layout> Psort(M,M,Nw);

  //U*D --> to fortran order
  nda::tensor::contract(D, "nj", U, "nij", UT, "ijn");

  // pivoted QR : U*D = Q*R*P^T
  nda::lapack::geqp3(UT,jpvt,tau,work);
  // get Q, R
  //std::tie(UT,VT) = detail::get_qr_matrices(UT, tau, true);
  std::tie(UT,VT) = nda::linalg::get_qr_matrices(UT, tau, true);

  // get permutation matrices as tensor
  P = nda::linalg::get_permutation_array<Type>(jpvt);

  // FIX : any way to get rid of these loops?
  // check performance later
  for(int b = 0; b < Nw; ++b)
  {
    //utility array for sort
    Psort_vec(nda::range::all,b) = nda::arange(M); 
    for(int i = 0; i < M; ++i){ 
      // D(i) = norm(R(i,:))
      DT(i,b) = nda::norm(VT(i,nda::range(i,M),b));
    }
  }
  
  nda::tensor::contract(VT,"ijn",P,"jkn",VT,"ikn");

  Dinv() = DT;

  nda::tensor::scale(Type(1.0),Dinv,nda::tensor::op::RCP);

  nda::tensor::contract(Dinv,"in",VT,"ijn",VT,"ijn");

  detail::quick_sort(DT,Psort_vec);

  // make compatible with get_permutation_array that
  // assumes Fortran indexing
  Psort_vec += 1;

  Psort = nda::linalg::get_permutation_array<Type>(Psort_vec);

  nda::tensor::contract(Psort,"ijn",VT,"jkn",VT,"ikn");

  nda::tensor::contract(UT,"ijn",Psort,"jkn",UT,"ikn");

  // FIX : any way to get rid of this loop?
  for(int b = 0; b < Nw; ++b){
   scl_new(b) = 0.5 * ( std::log(DT(0,b).real()) + std::log(DT(M-1,b).real()) );
   // FIX : replace with elementwise?
   DT(nda::range::all,b) *= std::exp(-scl_new(b)); 
  }

  scl += scl_new;

  nda::tensor::contract(VT,"jin",V,"nik",V,"njk");
  nda::tensor::add(UT,"ijn",U,"nij");
  nda::tensor::add(DT,"in",D,"ni");

}

// Finite temperature w/SVD
template<nda::MemoryArrayOfRank<3> U_t, nda::MemoryArrayOfRank<2> D_t,
         nda::MemoryArrayOfRank<3> V_t, nda::MemoryVector B_t>
requires( nda::mem::have_compatible_addr_space<U_t,D_t,V_t,B_t> and
          std::decay_t<U_t>::is_stride_order_C() and std::decay_t<D_t>::is_stride_order_C() and
          std::decay_t<V_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()
        )
void orthogonalize_wSVD(U_t && U, D_t && D, V_t && V, B_t && scl) 
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<U_t>();
  using Type = nda::get_value_t<U_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, Q] = U.shape();
  if(U.size()==0) return;
  utils::check( scl.extent(0) >= Nw, "Size mismatch");

  memory::buffered_array<MEM,Type,3,nda::F_layout> UT(M,M,Nw);
  memory::buffered_array<MEM,Type,3,nda::F_layout> VT(M,M,Nw);
  memory::buffered_array<MEM,Type,3,nda::F_layout> M2(M,M,Nw); 
  memory::buffered_array<MEM,Type,3,nda::C_layout> M1(Nw,M,M);
  memory::buffered_array<MEM,nda::remove_complex_t<Type>,2,nda::F_layout> S(M,Nw);
  memory::buffered_array<MEM,Type,2,nda::F_layout> DT(M,Nw);
  memory::buffered_array<MEM,nda::remove_complex_t<Type>,1,nda::F_layout> scl_new(Nw);
  
  //U*D --> to fortran order
  nda::tensor::contract(D, "nj", U, "nij", VT, "ijn");
  //nda::tensor::contract(D, "nj", U, "nij", M2, "ijn");

  nda::lapack::gesvd(VT,S,UT,VT);  

  // FIX : any way to get rid of this loop?
  for(int b = 0; b < Nw; ++b){
   scl_new(b) = 0.5 * ( std::log(S(0,b)) + std::log(S(M-1,b)) );
   // FIX : replace with elementwise?
   S(nda::range::all,b) *= std::exp(-scl_new(b)); 
  }

  scl += scl_new;

  //nda::tensor::contract(UT,"kin",V,"nkj",VT,"ijn");

  //nda::tensor::add(V,"nij",U,"nji");

  nda::tensor::add(UT,"ijn",U,"nij");

  // FIX : need V^+ (not just V^T)
  nda::tensor::add(VT,"ijn",M1,"nij");

  // V <-- V'*V
  nda::tensor::contract(V,"nij",nda::conj(M1),"nki",V,"nkj");

  nda::copy_cast(S,DT);

  nda::tensor::add(DT,"in",D,"ni");

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
    orthogonalize( wset.SlaterMatrices(Alpha), ldet);
    if(walker_type == COLLINEAR)
      orthogonalize( wset.SlaterMatrices(Beta), ldet);
  } else {
    double scl = ( walker_type == CLOSED ? 2.0 : 1.0 );
    orthogonalize( wset.SlaterMatrices(Alpha), ldet);
    if(walker_type == COLLINEAR)
      orthogonalize( wset.SlaterMatrices(Beta), ldet);
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
