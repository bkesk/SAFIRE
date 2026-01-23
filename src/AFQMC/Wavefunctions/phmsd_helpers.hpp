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
#include "AFQMC/Wavefunctions/detail/phmsd_impl.hpp"
#include "AFQMC/Wavefunctions/Excitations.hpp"
#include "numerics/operations/small_mat_ops.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

namespace sfqmc::afqmc
{

template<nda::MemoryArrayOfRank<3> T_t, nda::MemoryMatrix Mat>
void calculate_overlaps(int spin, ph_excitations<int, ComplexType, memory::get_memory_space<Mat>()>& abij,  T_t const& T, Mat&& ov)
{
  using nda::range;
  auto all = range::all;
  static_assert(memory::get_memory_space<T_t>() == memory::get_memory_space<Mat>(), "Memory space mismatch");
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<Mat>();
  ov(0,all)=ComplexType(1.0); 
  int nw = ov.extent(1);
  utils::check(T.extent(0) == nw, "Size mismatch"); 
  if constexpr (MEM==HOST_MEMORY) {
    for (int nex = 1, nd = 1; nex < abij.maximum_excitation_number()[spin]; nex++) {
      // expanding some of them by hand for efficiency
      if (nex == 1) {
        for (auto it = abij.unique_begin(1)[spin]; it < abij.unique_end(1)[spin]; ++it, ++nd) {
          auto e = *it;
          ov(nd,all) = T(all,e[1],e[0]);
        }
      } else if (nex == 2) {
        for (auto it = abij.unique_begin(2)[spin]; it < abij.unique_end(2)[spin]; ++it, ++nd) {
          auto e = *it;
          for(int iw=0; iw<nw; ++iw) 
            ov(nd,iw) = math::D2x2(T(iw,e[2],e[0]), T(iw,e[2],e[1]), T(iw,e[3],e[0]), T(iw,e[3],e[1]));
        }
      } else if (nex == 3) {
        for (auto it = abij.unique_begin(3)[spin]; it < abij.unique_end(3)[spin]; ++it, ++nd)
        {
          auto e = *it;
          for(int iw=0; iw<nw; ++iw)
            ov(nd,iw) = math::D3x3(T(iw,e[3],e[0]), T(iw,e[3],e[1]), T(iw,e[3],e[2]), T(iw,e[4],e[0]), T(iw,e[4],e[1]), T(iw,e[4],e[2]),
                            T(iw,e[5],e[0]), T(iw,e[5],e[1]), T(iw,e[5],e[2]));
        }
      } else if (nex == 4) {
        for (auto it = abij.unique_begin(4)[spin]; it < abij.unique_end(4)[spin]; ++it, ++nd)
        {
          auto e = *it;
          for(int iw=0; iw<nw; ++iw)
            ov(nd,iw) = math::D4x4(T(iw,e[4],e[0]), T(iw,e[4],e[1]), T(iw,e[4],e[2]), T(iw,e[4],e[3]), T(iw,e[5],e[0]), T(iw,e[5],e[1]),
                                   T(iw,e[5],e[2]), T(iw,e[5],e[3]), T(iw,e[6],e[0]), T(iw,e[6],e[1]), T(iw,e[6],e[2]), T(iw,e[6],e[3]),
                                   T(iw,e[7],e[0]), T(iw,e[7],e[1]), T(iw,e[7],e[2]), T(iw,e[7],e[3]));
        }
      } else if (nex == 5) {
        for (auto it = abij.unique_begin(5)[spin]; it < abij.unique_end(5)[spin]; ++it, ++nd)
        {
          auto e = *it;
          for(int iw=0; iw<nw; ++iw)
            ov(nd,iw) = math::D5x5(T(iw,e[5],e[0]), T(iw,e[5],e[1]), T(iw,e[5],e[2]), T(iw,e[5],e[3]), T(iw,e[5],e[4]), T(iw,e[6],e[0]),
                                   T(iw,e[6],e[1]), T(iw,e[6],e[2]), T(iw,e[6],e[3]), T(iw,e[6],e[4]), T(iw,e[7],e[0]), T(iw,e[7],e[1]),
                                   T(iw,e[7],e[2]), T(iw,e[7],e[3]), T(iw,e[7],e[4]), T(iw,e[8],e[0]), T(iw,e[8],e[1]), T(iw,e[8],e[2]),
                                   T(iw,e[8],e[3]), T(iw,e[8],e[4]), T(iw,e[9],e[0]), T(iw,e[9],e[1]), T(iw,e[9],e[2]), T(iw,e[9],e[3]),
                                   T(iw,e[9],e[4]));
        }
      } else {
        memory::buffered_array<MEM,ComplexType,2> Qwork(nex,nex); 
        memory::buffered_array<MEM,ComplexType,1> Qwork2(nex*nex); 
        memory::buffered_array<MEM,int,1> ipiv(nex); 
        for (auto it = abij.unique_begin(nex)[spin]; it < abij.unique_end(nex)[spin]; ++it, ++nd)
        {
          auto exct = *it;
          for(int iw=0; iw<nw; ++iw) {
            for (int p = 0; p < nex; p++)
              for (int q = 0; q < nex; q++) 
                Qwork(p,q) = T(iw,exct[p + nex],exct[q]);
            nda::lapack::getrf(Qwork,ipiv,Qwork2);
            math::log_determinant_from_getrf(Qwork,ipiv,ov(nd,iw)); 
            ov(nd,iw) = std::exp(ov(nd,iw));
          }
        }
      }
    } // nex
  } else { // MEM
    // custom kernel dispatch
    utils::check(false,"finish");
  }
}

// T(nw,nact,nel)
// weights(nd,nw)
// R(nw,nel,nact)
template<nda::MemoryArrayOfRank<3> T_t, nda::MemoryMatrix Mat, nda::MemoryArrayOfRank<3> R_t>
void calculate_R(int spin, ph_excitations<int, ComplexType, memory::get_memory_space<Mat>()>& abij,  T_t const& T, Mat&& weights, R_t && R)
{
  using nda::range;
  auto all = range::all;
  static_assert(::nda::mem::have_compatible_addr_space<T_t,Mat,R_t>, "Memory space mismatch");
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<R_t>(); 
  int NEL = T.extent(2);
  int nw = T.extent(0);
  nda::array<int,1> orbs(NEL);
  ComplexType ov_a;
  R() = ComplexType(0);
  if constexpr (MEM==HOST_MEMORY) {
    {
      auto refc   = abij.reference_configuration(spin);
      // Wrong if ndown < nup!!! FIX FIX FIX
      for (int i = 0; i < NEL; ++i)
        R(all,i,refc[i]) += weights(0,all);
    }
    for (int nex = 1, nd = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
    {
      memory::buffered_array<MEM,ComplexType,2> Q(nex,nex); 
      memory::buffered_array<MEM,int,1> ipiv(nex); 
      memory::buffered_array<MEM,ComplexType,1> work(nex*nex); 
      for (auto it = abij.unique_begin(nex)[spin]; it < abij.unique_end(nex)[spin]; ++it, ++nd)
      {
        auto e = *it;
        abij.get_configuration(spin, nd, orbs);
        for(int iw=0; iw<nw; ++iw) {
          if (nex == 1)
          {
            ov_a    = T(iw,e[1],e[0]);
            Q(0,0) = 1.0 / ov_a;
          }
          else if (nex == 2)
          {
            ov_a = math::I2x2(T(iw,e[2],e[0]), T(iw,e[2],e[1]), T(iw,e[3],e[0]), T(iw,e[3],e[1]), Q);
          }
          else if (nex == 3)
          {
            ov_a = math::I3x3(T(iw,e[3],e[0]), T(iw,e[3],e[1]), T(iw,e[3],e[2]), T(iw,e[4],e[0]), T(iw,e[4],e[1]), T(iw,e[4],e[2]),
                          T(iw,e[5],e[0]), T(iw,e[5],e[1]), T(iw,e[5],e[2]), Q);
          }
          else
          {
            for (int p = 0; p < nex; p++)
              for (int q = 0; q < nex; q++)
                Q(p,q) = T(iw,e[p + nex],e[q]);
            nda::lapack::getrf(Q,ipiv,work);
            math::log_determinant_from_getrf(Q,ipiv,ov_a);
            // how to handle cases where this is basically zero?
            ov_a = std::exp(ov_a);
            if(std::abs(ov_a) != 0.0) 
              nda::lapack::getri(Q,ipiv,work);
          }
          ComplexType w(weights(nd,iw));
          if (std::abs(ov_a) != 0.0)
          {
            // add term coming from identity
            for (int i = 0; i < NEL; ++i)
              R(iw,i,orbs(i)) += w;
            for (int p = 0; p < nex; ++p)
            {
              auto Rp = R(iw,e[p],all);
              auto Ip = Q(p,all);
              for (int q = 0; q < nex; ++q)
              {
                auto Ipq = Ip(q);
                auto Tq  = T(iw,e[q + nex],all);
                for (int i = 0; i < NEL; ++i)
                  Rp(orbs(i)) -= w * Ipq * Tq(i);
                Rp(orbs(e[q])) += w * Ipq;
              }
            }
          }
        } // iw
      } // it
    } // nex
  } else {
    // custom kernel dispatch
    utils::check(false,"finish");
  }
}

// R[nwalk,ndet,nex,nact]
// T(nwalk,nact,nelec]
template<MEMORY_SPACE MEM>
void get_compact_ph_R_matrices(nda::MemoryVector auto const& iexcit,
     nda::MemoryVector auto const& refc, nda::MemoryArrayOfRank<3> auto&& Tw, 
     nda::MemoryArrayOfRank<4> auto&& Rw) 
{
  using nda::range;
  auto all = range::all;
  memory::check_memory_space<MEM>(Tw,Rw);
  auto [nwalk,ndet,nex,nact] = Rw.shape();
  int nelec = Tw.extent(2);
  utils::check(Tw.extent(0) == nwalk and Tw.extent(1)==nact, "Size mismatch");
  memory::buffered_array<MEM,ComplexType,2> Q(nex,nex); 
  memory::buffered_array<MEM,ComplexType,1> work(nex*nex); 
  memory::buffered_array<MEM,int,1> ipiv(nex); 
  memory::buffered_array<MEM,int,1> orbs(nelec); 
  ComplexType ov_a;
  if constexpr (MEM==HOST_MEMORY) {
    for(int iw=0; iw<nwalk; iw++) 
    {  
      auto T = Tw(iw,all,all);
      for(int nd=0; nd<ndet; nd++) 
      {
        auto R = Rw(iw,nd,all,all); 
        auto e = iexcit.data()+2*nex*nd;
        for(int i=0; i<nelec; i++) {
          orbs(i) = refc(i);
          for(int p=0; p<nex; p++) {
             if(e[p]==i) {
              orbs(i) = e[nex+p];
              break;
            }  
          }  
        }
        if (nex == 1)
        {
          ov_a    = T(e[1],e[0]);
          Q(0,0) = 1.0 / ov_a;
        }
        else if (nex == 2)
        {
          ov_a = math::I2x2(T(e[2],e[0]), T(e[2],e[1]), T(e[3],e[0]), T(e[3],e[1]), Q);
        }
        else if (nex == 3)
        {
          ov_a = math::I3x3(T(e[3],e[0]), T(e[3],e[1]), T(e[3],e[2]), 
                          T(e[4],e[0]), T(e[4],e[1]), T(e[4],e[2]),
                          T(e[5],e[0]), T(e[5],e[1]), T(e[5],e[2]), Q);
        }
        else
        {
          for (int p = 0; p < nex; p++)
            for (int q = 0; q < nex; q++)
              Q(p,q) = T(e[p + nex],e[q]);
          nda::lapack::getrf(Q,ipiv,work);
          math::log_determinant_from_getrf(Q,ipiv,ov_a);
          // how to handle cases where this is basically zero?
          ov_a = std::exp(ov_a);
          if(std::abs(ov_a) != 0.0) nda::lapack::getri(Q,ipiv,work);
        }
        if( std::abs(ov_a) != 0.0) {
          // compact notation:
          // R[p,nact] for p in [0, nex)
          // R[nex,nact] for diagonal term    
          for (int p = 0; p < nex; ++p)
          {
            auto Ip = Q(p,all);
            for (int q = 0; q < nex; ++q)
            {
              auto Ipq = Q(p,q);
              auto Tq  = T(e[q + nex],all);
              for (int i = 0; i < nelec; ++i)
                R(p,orbs(i)) -= Ipq * Tq(i);
              R(p,orbs(e[q])) += Ipq;
            }
          }
        }
      } // nd
    } // iw
  } else {
    // custom kernel dispatch
    utils::check(false,"finish");
  }
}

// Calculates the Alpha/Alpha (XXX_first_step) contribution to E1, EJ, EX and KEright.
// Loops over excitation shells and calls Op.ph_excited_energy 
// wgt[ndet,nwalk]
// T(nwalk,nact,nelec]
// E[nwalk,3]
// KE[ndet,nwalk,nke]
template<class Op, nda::MemoryMatrix Mat>
void ph_excited_energies_first_step(ph_excitations<int, ComplexType, memory::get_memory_space<Mat>()>& abij,
         Mat&& wgt, nda::MemoryArrayOfRank<3> auto T,
         nda::MemoryMatrix auto&& E, nda::MemoryArrayOfRank<3> auto&& KE, Op& HamOps)
{
  using nda::range;
  auto all = range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<decltype(T)>(); 
  memory::check_memory_space<MEM>(wgt,T,E,KE);
  auto [nwalk,nact,nelec] = T.shape();
  int nconfg = KE.extent(0);
  int nke = KE.extent(2);
  utils::check(E.extent(0) == nwalk and E.extent(1)==3, "Size mismatch");
  utils::check(wgt.extent(0) == nconfg and wgt.extent(1)==nwalk, "Size mismatch");
  utils::check(KE.extent(1) == nwalk, "Size mismatch");
  E() = ComplexType(0.0);
  
  int spin(0);
  auto refc=abij.get_reference_configuration_device(spin);
  // no use of blocks yet! 
  long max2=0, max3=0;
  for (int nex = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if( ndet > max2 ) max2 = ndet;
    if( ndet * nex * nact > max3 ) max3 = ndet * nex * nact;
  }

  for (int nex = 1, idet=1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex);
      memory::buffered_array<MEM,ComplexType,4> R(nwalk, ndet, nex, nact);
      R() = ComplexType(0.0);

      // generate R matrices for current excitation shell
      get_compact_ph_R_matrices<MEM>(iexcit, refc, T, R); 

      // assumes no TG_local parallelization for now, needs TG for this
      // calculate E and KE for current shell of excitations
      auto KEr=KE(range(idet,idet+ndet),all,all);
      HamOps.ph_excited_energy(Alpha, nelec, iexcit, refc, E, 
                wgt(range(idet, idet+ndet),all), R, KEr, true); 

      idet += ndet;
    }
  }
}

// Calculates the Alpha/Alpha (XXX_second_step) contribution to E1, EJ, EX and KEright.
// Loops over excitation shells and calls Op.ph_excited_energy 
// wgt[ndet,nwalk]
// T(nwalk,nact,nelec]
// E[nwalk,3]
// KE[ndet,nwalk,nke]
template<class Op, nda::MemoryMatrix Mat>
void ph_excited_energies_second_step(ph_excitations<int, ComplexType, 
         memory::get_memory_space<Mat>()>& abij, Mat&& wgt, nda::MemoryArrayOfRank<3> auto T,
         nda::MemoryMatrix auto&& E, nda::MemoryArrayOfRank<3> auto&& KE, Op& HamOps)
{
  using nda::range;
  auto all = range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<decltype(T)>(); 
  memory::check_memory_space<MEM>(wgt,T,E,KE);
  auto [nwalk,nact,nelec] = T.shape();
  int nconfg = KE.extent(0);
  int nke = KE.extent(2);
  utils::check(E.extent(0) == nwalk and E.extent(1)==3, "Size mismatch");
  utils::check(wgt.extent(0) == nconfg and wgt.extent(1)==nwalk, "Size mismatch");
  utils::check(KE.extent(1) == nwalk, "Size mismatch");
  E() = ComplexType(0.0);

  int spin(1);
  auto refc=abij.get_reference_configuration_device(spin);
  // no use of blocks yet!
  long max2=0, max3=0;
  for (int nex = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if( ndet > max2 ) max2 = ndet;
    if( ndet * nex * nact > max3 ) max3 = ndet * nex * nact;
  }

  for (int nex = 1, idet=1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex);
      memory::buffered_array<MEM,ComplexType,4> R(nwalk, ndet, nex, nact);
      memory::buffered_array<MEM,ComplexType,3> KEl(ndet, nwalk, nke);
      R() = ComplexType(0.0);
      KEl() = ComplexType(0.0);

      // generate R matrices for current excitation shell
      get_compact_ph_R_matrices<MEM>(iexcit, refc, T, R);

      // calculate E and KE for current shell of excitations
      HamOps.ph_excited_energy(Beta, nelec, iexcit, refc, E, 
                wgt(range(idet, idet+ndet),all), R, KEl, true); 

      // eloc[iw] = sum_d_ke KE[d,iw,ke] KEl[d,iw,ke]
      nda::tensor::contract(ComplexType(1.0),KE(range(idet,idet+ndet),all,all),"dwn",KEl,"dwn",
                            ComplexType(1.0),E(all,2),"w");
 
      idet += ndet;
    }
  }
}
/*
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
template<class MArray, class MatA, class PH_EXCT,
         typename = std::enable_if_t< is_device_array<MatA>::value >,
         typename = std::enable_if_t< is_device_array<std::decay_t<MArray>>::value >,
         typename = std::enable_if_t< MatA::dimensionality == 3 >,
         typename = std::enable_if_t< std::decay_t<MArray>::dimensionality == 2 >
        >
inline void calculate_overlaps(int rank, int ngrp, int spin, PH_EXCT& abij, MatA const& T, 
                               MArray&& Ovlps, int buffer_size_in_MB = 2048)
{
#ifndef NDEBUG
  // set to a low value in debug runs, nvcc takes for ever to compile large values
  const int max_nex_phmsd_det = 3;
#else
  const int max_nex_phmsd_det = 7;
#endif
  RUNTIME_CHECK(T.size(0) == Ovlps.size(1), "");
  using kernels::extract_overlap_matrix;
  using kernels::phmsd_det;
  using ma::getrfBatched;
  using ma::strided_determinant_from_getrf;
  using Cptr = device_ptr<ComplexType>;  
  using Ibuffer_alloc_type       = DeviceBufferManager::template allocator_t<int>;
  using buffer_alloc_type       = DeviceBufferManager::template allocator_t<ComplexType>;
  
  bool OvContiguous =  (Ovlps.stride(0) == Ovlps.size(1));
  int sx = (OvContiguous ? 0 : 1);
  int nwalk = T.size(0);  
  DeviceBufferManager buffer_manager;
  // no use of blocks yet!
  long max1=0, max2=0, max3=0;
  for (int nex = max_nex_phmsd_det+1 / * note start * /; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if( ndet * nex > max1 ) max1 = ndet * nex; 
    if( ndet * nex * nex > max2 ) max2 = ndet * nex * nex; 
    if( ndet > max3 ) max3 = ndet;
  }
  StaticVector<int, Ibuffer_alloc_type> IWORK(iextensions<1u>{nwalk*max1},
               buffer_manager.get_generator().template get_allocator<int>());
  StaticVector<ComplexType, buffer_alloc_type> Qwork(iextensions<1u>{nwalk*max2},
                buffer_manager.get_generator().template get_allocator<ComplexType>());
  StaticMatrix<ComplexType, buffer_alloc_type> OvWork({sx*max3, sx*nwalk},
                buffer_manager.get_generator().template get_allocator<ComplexType>());

  using Tptr = ComplexType*;
  using Tptrbuffer_alloc_type       = DeviceBufferManager::template allocator_t<Tptr>;
  StaticVector<Tptr, Tptrbuffer_alloc_type> Odev(iextensions<1u>{nwalk*max3},
               buffer_manager.get_generator().template get_allocator<Tptr>());
  std::vector<Tptr> Ptrs;
  Ptrs.resize(nwalk*max3);
    
  // det_cnt starts 
  // set reference Ov to 1.0, since this calculates overlap ratios wrt to the reference
  ma::fill(Ovlps[0], ComplexType(1.0));
  for (int nex = 1, idet=1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin]; 
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex);  
      if(nex <= max_nex_phmsd_det ) {
        phmsd_det(nwalk, ndet, nex, raw_pointer_cast(iexcit), raw_pointer_cast(T.origin()),
                  T.stride(1), T.stride(0), raw_pointer_cast(Ovlps[idet].origin()), Ovlps.stride(0));      
      } else {  
        extract_overlap_matrix(nwalk, ndet, nex, raw_pointer_cast(iexcit), 
                 raw_pointer_cast(T.origin()), T.stride(1), T.stride(0), raw_pointer_cast(Qwork.origin()), true);
        Ptrs[0] = raw_pointer_cast(Qwork.origin());  
        for (int i = 1; i < ndet*nwalk; i++)
          Ptrs[i] = Ptrs[0] + i*nex*nex;
        arch::memcopy(raw_pointer_cast(Odev.origin()), Ptrs.data(), nwalk*ndet*sizeof(Tptr), arch::memcopyH2D);  
        cublas::cublas_getrfBatched(arch::global_cublas_handle, nex, 
                raw_pointer_cast(Odev.origin()), nex, raw_pointer_cast(IWORK.origin()), 
                raw_pointer_cast(IWORK.origin())+nwalk*ndet*nex, nwalk*ndet);
        if(OvContiguous) {
          strided_determinant_from_getrf(nex, Qwork.origin(), nex, nex*nex, IWORK.origin(), nex, 
               ComplexType(0.0), raw_pointer_cast(Ovlps[idet].origin()), 1, nwalk*ndet);  
        } else {
          strided_determinant_from_getrf(nex, Qwork.origin(), nex, nex*nex, IWORK.origin(), nex, 
               ComplexType(0.0), raw_pointer_cast(OvWork.origin()), 1, nwalk*ndet);  
          ma::add(ComplexType(1.0),OvWork.sliced(0,ndet),ComplexType(0.0),OvWork.sliced(0,ndet),
              Ovlps.sliced(idet,idet+ndet));
        }
      }  
      idet += ndet;
    }
  }
}

template<class MArray, class MatA, class MatC, class PH_EXCT,
         typename = std::enable_if_t< is_device_array<std::decay_t<MatA>>::value >,
         typename = std::enable_if_t< is_device_array<std::decay_t<MatC>>::value >,
         typename = std::enable_if_t< is_device_array<std::decay_t<MArray>>::value >,
         typename = std::enable_if_t< std::decay_t<MatA>::dimensionality == 3 >,
         typename = std::enable_if_t< std::decay_t<MatC>::dimensionality == 3 >,
         typename = std::enable_if_t< std::decay_t<MArray>::dimensionality == 2 >
        >
inline void calculate_R(int rank, int ngrp, int spin, PH_EXCT& abij, 
                 MatA&& T, MArray&& weights, MatC&& R, int buffer_size_in_MB = 2048)
{
#ifndef NDEBUG
  const int max_nex_phmsd_inv = 3;
#else
  const int max_nex_phmsd_inv = 5;
#endif
  RUNTIME_CHECK(T.size(0) == R.size(0), "");
  RUNTIME_CHECK(T.size(0) == weights.size(1), "");
  using kernels::extract_overlap_matrix;
  using kernels::construct_phmsd_R;
  using kernels::reduce_phmsd_R;
  using kernels::phmsd_inv;
  using ma::matinvBatched;
  using Cptr = device_ptr<ComplexType>;  
  using Ibuffer_alloc_type       = DeviceBufferManager::template allocator_t<int>;
  using buffer_alloc_type       = DeviceBufferManager::template allocator_t<ComplexType>;

  int nwalk = R.size(0);
  int nelec = R.size(1);
  int nact = R.size(2);
  DeviceBufferManager buffer_manager;
  // no use of blocks yet!
  long max1=0, max2=0, max3=0;
  for (int nex = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if( ndet * nex * nex > max1 ) max1 = ndet * nex * nex; 
    if( ndet > max2 ) max2 = ndet;
    //if( ndet > max2 and nex > max_nex_phmsd_inv ) max2 = ndet;
    if( ndet * nex * nact > max3 ) max3 = ndet * nex * nact;
  }
  StaticVector<ComplexType, buffer_alloc_type> Minv(iextensions<1u>{nwalk*max1},
               buffer_manager.get_generator().template get_allocator<ComplexType>());
  StaticVector<ComplexType, buffer_alloc_type> Buff(iextensions<1u>{nwalk*std::max(max1,max3)},
               buffer_manager.get_generator().template get_allocator<ComplexType>());
  StaticVector<int, Ibuffer_alloc_type> IWORK(iextensions<1u>{nwalk*max2},
               buffer_manager.get_generator().template get_allocator<int>());
  using Tptr = ComplexType*;
  using Tptrbuffer_alloc_type       = DeviceBufferManager::template allocator_t<Tptr>;
  StaticVector<Tptr, Tptrbuffer_alloc_type> Odev(iextensions<1u>{2*nwalk*max2},
               buffer_manager.get_generator().template get_allocator<Tptr>());
  std::vector<Tptr> Ptrs;
  Ptrs.resize(2*nwalk*max2);

  auto refc=abij.get_reference_configuration_device(spin);
    
  {
    // add ontribution from reference determinant
    // R[w,i, refc[i] ] += weights[w]
    kernels::add_diagonal(R.size(1), raw_pointer_cast(refc), 
                          raw_pointer_cast(weights.origin()), 1,  
                          raw_pointer_cast(R.origin()), R.stride(1), R.stride(0),R.size(0));
  }
  for (int nex = 1, idet=1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin]; 
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex);  
      if(nex <= max_nex_phmsd_inv) {
        phmsd_inv(nwalk, ndet, nex, raw_pointer_cast(iexcit), raw_pointer_cast(T.origin()),
                  T.stride(1), T.stride(0), //raw_pointer_cast(Ov[idet].origin()), Ov.stride(0), 
                  raw_pointer_cast(Minv.origin()));
      } else {
        // loop over blocks of excitations if memory becomes an issue
        Ptrs[0] = raw_pointer_cast(Buff.origin());
        Ptrs[nwalk*ndet] = raw_pointer_cast(Minv.origin());
        for (int i = 1; i < nwalk*ndet; i++) { 
          Ptrs[i] = Ptrs[0] + i*nex*nex;
          Ptrs[nwalk*ndet+i] = Ptrs[nwalk*ndet] + i*nex*nex;
        }  
        arch::memcopy(raw_pointer_cast(Odev.origin()), Ptrs.data(), 2*nwalk*ndet*sizeof(Tptr), arch::memcopyH2D);  
        extract_overlap_matrix(nwalk, ndet, nex, raw_pointer_cast(iexcit), 
                    raw_pointer_cast(T.origin()), T.stride(1), T.stride(0), Ptrs[0]);
        qmc_cuda::cublas_check( cublas::cublas_matinvBatched(arch::global_cublas_handle,
                nex, raw_pointer_cast(Odev.origin()), nex, raw_pointer_cast(Odev.origin())+nwalk*ndet, nex,
                raw_pointer_cast(IWORK.origin()), nwalk*ndet),
                "phmsd_helpers::calculate_R" );
        Array_ref<ComplexType,3,Cptr> Minv_( Minv.origin(), {nwalk*ndet, nex, nex});
        ma::fill_if_non_zero(Minv_,IWORK.sliced(0,nwalk*ndet),ComplexType(0.0));
      }
      fill_n(Buff.origin(), nwalk*ndet*nex*nact, ComplexType(0.0));  
      construct_phmsd_R(nwalk, ndet, nex, nact, nelec, raw_pointer_cast(iexcit), raw_pointer_cast(refc),
                        raw_pointer_cast(T.origin()), T.stride(1), T.stride(0),
                        raw_pointer_cast(Minv.origin()), raw_pointer_cast(Buff.origin())); 
      reduce_phmsd_R(nwalk, ndet, nex, nact, nelec, raw_pointer_cast(iexcit), raw_pointer_cast(refc),
                        raw_pointer_cast(weights[idet].origin()), weights.stride(0),
                        raw_pointer_cast(Buff.origin()), raw_pointer_cast(R.origin()));
      idet += ndet;
    }
  }
}

// R[nwalk,ndet,nex,nact]
// T(nwalk,nact,nelec]
template<class MatA, class MatC, class PH_EXCT, class Iptr,
         typename = std::enable_if_t< is_device_array<std::decay_t<MatA>>::value >,
         typename = std::enable_if_t< is_device_array<std::decay_t<MatC>>::value >,
         typename = void
        >
inline void get_compact_ph_R_matrices(int rank, int ngrp, int spin, int nwalk, int ndet, int nex,
               int nelec, int nact, Iptr const iexcit, Iptr const refc,
               PH_EXCT& abij, MatA&& Tw, MatC&& Rw, int buffer_size_in_MB=0)
{
#ifndef NDEBUG
  const int max_nex_phmsd_inv = 3;
#else
  const int max_nex_phmsd_inv = 5;
#endif
  RUNTIME_CHECK(Tw.size(0) == Rw.size(0), "");
  using kernels::extract_overlap_matrix;
  using kernels::construct_phmsd_R;
  using kernels::phmsd_inv;
  using ma::matinvBatched;
  using Cptr = device_ptr<ComplexType>;
  using Ibuffer_alloc_type       = DeviceBufferManager::template allocator_t<int>;
  using buffer_alloc_type       = DeviceBufferManager::template allocator_t<ComplexType>;

  DeviceBufferManager buffer_manager;
  StaticVector<ComplexType, buffer_alloc_type> Minv(iextensions<1u>{nwalk * ndet * nex * nex},
               buffer_manager.get_generator().template get_allocator<ComplexType>());

  if(nex <= max_nex_phmsd_inv) {
    phmsd_inv(nwalk, ndet, nex, raw_pointer_cast(iexcit), raw_pointer_cast(Tw.origin()),
              Tw.stride(1), Tw.stride(0), raw_pointer_cast(Minv.origin()));
  } else {
    StaticVector<int, Ibuffer_alloc_type> IWORK(iextensions<1u>{nwalk*ndet},
               buffer_manager.get_generator().template get_allocator<int>());
    using Tptr = ComplexType*;
    using Tptrbuffer_alloc_type       = DeviceBufferManager::template allocator_t<Tptr>;
    StaticVector<Tptr, Tptrbuffer_alloc_type> Odev(iextensions<1u>{2*nwalk*ndet},
               buffer_manager.get_generator().template get_allocator<Tptr>());
    std::vector<Tptr> Ptrs;
    Ptrs.resize(2*nwalk*ndet);
    StaticVector<ComplexType, buffer_alloc_type> Buff(iextensions<1u>{nwalk*ndet*nex*nex},
               buffer_manager.get_generator().template get_allocator<ComplexType>());
    Ptrs[0] = raw_pointer_cast(Buff.origin());
    Ptrs[nwalk*ndet] = raw_pointer_cast(Minv.origin());
    for (int i = 1; i < nwalk*ndet; i++) {
      Ptrs[i] = Ptrs[0] + i*nex*nex;
      Ptrs[nwalk*ndet+i] = Ptrs[nwalk*ndet] + i*nex*nex;
    }
    arch::memcopy(raw_pointer_cast(Odev.origin()), Ptrs.data(), 2*nwalk*ndet*sizeof(Tptr), arch::memcopyH2D);
    extract_overlap_matrix(nwalk, ndet, nex, raw_pointer_cast(iexcit),
                raw_pointer_cast(Tw.origin()), Tw.stride(1), Tw.stride(0), Ptrs[0]);
    qmc_cuda::cublas_check( cublas::cublas_matinvBatched(arch::global_cublas_handle,
            nex, raw_pointer_cast(Odev.origin()), nex, raw_pointer_cast(Odev.origin())+nwalk*ndet, nex,
            raw_pointer_cast(IWORK.origin()), nwalk*ndet),
            "phmsd_helpers::get_compact_ph_R_matrices" );
    Array_ref<ComplexType,3,Cptr> Minv_( Minv.origin(), {nwalk*ndet, nex, nex});
    ma::fill_if_non_zero(Minv_,IWORK.sliced(0,nwalk*ndet),ComplexType(0.0));
  }
  fill_n(Rw.origin(), Rw.num_elements(), typename std::decay_t<MatC>::element_type(0.0));
  construct_phmsd_R(nwalk, ndet, nex, nact, nelec, raw_pointer_cast(iexcit), raw_pointer_cast(refc),
                    raw_pointer_cast(Tw.origin()), Tw.stride(1), Tw.stride(0),
                    raw_pointer_cast(Minv.origin()), raw_pointer_cast(Rw.origin()));
}
#endif
*/

} // namespace sfqmc::afqmc


