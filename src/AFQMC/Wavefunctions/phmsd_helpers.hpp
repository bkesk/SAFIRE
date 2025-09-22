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

#ifndef SFQMC_AFQMC_PHMSD_HELPERS_HPP
#define SFQMC_AFQMC_PHMSD_HELPERS_HPP

#include "AFQMC/config.h"
#include "Memory/arch.hpp"
#include "Memory/buffer_allocators.hpp"
#include "Numerics/ma_operations.hpp"
#include "Numerics/ma_small_mat_ops.hpp"
//#include "Numerics/device_kernels.hpp"
#include "Numerics/batched_operations.hpp"
#include "AFQMC/Wavefunctions/detail/phmsd_impl.hpp"

namespace sfqmc
{
namespace afqmc
{

// CPU version (GPU version below)
// using simple round-robin scheme for parallelization within TG_local
// assumes that reference determinant is already on [0]
template<class Array1D, class MatA, class PH_EXCT
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
// temporary hack! implement is_shm_array and add it here
         ,typename = std::enable_if_t< is_host_array<std::decay_t<MatA>>::value >,
         typename = std::enable_if_t< is_host_array<std::decay_t<Array1D>>::value >
#endif
>
inline void calculate_overlaps(int rank, 
                               int ngrp, 
                               int spin, 
                               PH_EXCT& abij, 
                               MatA const& T, 
                               Array1D&& ov, 
                               [[maybe_unused]] int buffer_size_in_MB = 0)
{
  using buffer_alloc_type       = HostBufferManager::template allocator_t<ComplexType>;
  HostBufferManager buffer_manager;
  int max_exct_n(std::max(abij.maximum_excitation_number()[0], abij.maximum_excitation_number()[1]));
  StaticVector<ComplexType, buffer_alloc_type> Qwork(iextensions<1u>{2 * max_exct_n * max_exct_n},
                   buffer_manager.get_generator().template get_allocator<ComplexType>());  
  std::vector<int> IWORK(abij.maximum_excitation_number()[spin]);
  if(rank==0) ov[0]=1.0; 
  for (int nex = 1, nd = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    // expanding some of them by hand for efficiency
    if (nex == 1)
    {
      for (auto it = abij.unique_begin(1)[spin]; it < abij.unique_end(1)[spin]; ++it, ++nd)
        if (nd % ngrp == rank)
        {
          ov[nd] = T[*((*it) + 1)][*(*it)];
        }
    }
    else if (nex == 2)
    {
      for (auto it = abij.unique_begin(2)[spin]; it < abij.unique_end(2)[spin]; ++it, ++nd)
        if (nd % ngrp == rank)
        {
          auto e = *it;
          ov[nd] = ma::D2x2(T[e[2]][e[0]], T[e[2]][e[1]], T[e[3]][e[0]], T[e[3]][e[1]]);
        }
    }
    else if (nex == 3)
    {
      for (auto it = abij.unique_begin(3)[spin]; it < abij.unique_end(3)[spin]; ++it, ++nd)
        if (nd % ngrp == rank)
        {
          auto e = *it;
          ov[nd] = ma::D3x3(T[e[3]][e[0]], T[e[3]][e[1]], T[e[3]][e[2]], T[e[4]][e[0]], T[e[4]][e[1]], T[e[4]][e[2]],
                            T[e[5]][e[0]], T[e[5]][e[1]], T[e[5]][e[2]]);
        }
    }
    else if (nex == 4)
    {
      for (auto it = abij.unique_begin(4)[spin]; it < abij.unique_end(4)[spin]; ++it, ++nd)
        if (nd % ngrp == rank)
        {
          auto e = *it;
          ov[nd] = ma::D4x4(T[e[4]][e[0]], T[e[4]][e[1]], T[e[4]][e[2]], T[e[4]][e[3]], T[e[5]][e[0]], T[e[5]][e[1]],
                            T[e[5]][e[2]], T[e[5]][e[3]], T[e[6]][e[0]], T[e[6]][e[1]], T[e[6]][e[2]], T[e[6]][e[3]],
                            T[e[7]][e[0]], T[e[7]][e[1]], T[e[7]][e[2]], T[e[7]][e[3]]);
        }
    }
    else if (nex == 5)
    {
      for (auto it = abij.unique_begin(5)[spin]; it < abij.unique_end(5)[spin]; ++it, ++nd)
        if (nd % ngrp == rank)
        {
          auto e = *it;
          ov[nd] = ma::D5x5(T[e[5]][e[0]], T[e[5]][e[1]], T[e[5]][e[2]], T[e[5]][e[3]], T[e[5]][e[4]], T[e[6]][e[0]],
                            T[e[6]][e[1]], T[e[6]][e[2]], T[e[6]][e[3]], T[e[6]][e[4]], T[e[7]][e[0]], T[e[7]][e[1]],
                            T[e[7]][e[2]], T[e[7]][e[3]], T[e[7]][e[4]], T[e[8]][e[0]], T[e[8]][e[1]], T[e[8]][e[2]],
                            T[e[8]][e[3]], T[e[8]][e[4]], T[e[9]][e[0]], T[e[9]][e[1]], T[e[9]][e[2]], T[e[9]][e[3]],
                            T[e[9]][e[4]]);
        }
    }
    else
    {
      boost::multi::array_ref<ComplexType, 2> Qwork_(raw_pointer_cast(Qwork.origin()), {nex, nex});
      boost::multi::array_ref<ComplexType, 1> Qwork2_(Qwork_.origin() + Qwork_.num_elements(),
                                                      iextensions<1u>{nex * nex});
      for (auto it = abij.unique_begin(nex)[spin]; it < abij.unique_end(nex)[spin]; ++it, ++nd)
        if (nd % ngrp == rank)
        {
          auto exct = *it;
          for (int p = 0; p < nex; p++)
            for (int q = 0; q < nex; q++)
              Qwork_[p][q] = T[exct[p + nex]][exct[q]];
          ov[nd] = ma::determinant<ComplexType>(Qwork_, IWORK, Qwork2_, 0.0);
        }
    }
  }
}

// using simple round-robin scheme for parallelization within TG_local
// assumes that reference determinant is already on [0]
template<class Array1D, class MatA, class MatC, class PH_EXCT 
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
         ,typename = std::enable_if_t< is_host_array<std::decay_t<MatA>>::value >,
         typename = std::enable_if_t< is_host_array<std::decay_t<MatC>>::value >,
         typename = std::enable_if_t< is_host_array<std::decay_t<Array1D>>::value >
#endif
        >
inline void calculate_R(int rank,
                        int ngrp,
                        int spin,
                        PH_EXCT& abij,
                        MatA&& T,
                        Array1D&& weights,
                        MatC&& R,
                        [[maybe_unused]] int buffer_size_in_MB = 0)
{
  using ma::conj;
  using std::get;
  using buffer_alloc_type       = HostBufferManager::template allocator_t<ComplexType>;
  HostBufferManager buffer_manager;
  int max_exct_n(std::max(abij.maximum_excitation_number()[0], abij.maximum_excitation_number()[1]));
  StaticVector<ComplexType, buffer_alloc_type> Qwork(iextensions<1u>{2 * max_exct_n * max_exct_n},
                   buffer_manager.get_generator().template get_allocator<ComplexType>());
  std::vector<int> IWORK(abij.maximum_excitation_number()[spin]);
  std::vector<ComplexType> WORK(abij.maximum_excitation_number()[spin] * abij.maximum_excitation_number()[spin]);
  for (int i = 0; i < R.size(0); i++)
    std::fill_n(R[i].origin(), R.size(1), ComplexType(0));
  int NEL = T.size(1);
  std::vector<int> orbs(NEL);
  ComplexType ov_a;
  // add reference contribution!!!
  if (rank == 0)
  {
    auto refc   = abij.reference_configuration(spin);
    ComplexType w(weights[0]);
    // Wrong if NAEB < NAEA!!! FIX FIX FIX
    for (int i = 0; i < NEL; ++i)
      R[i][refc[i]] += w;
  }
  for (int nex = 1, nd = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    boost::multi::array_ref<ComplexType, 2> Q(raw_pointer_cast(Qwork.origin()), {nex, nex});
    for (auto it = abij.unique_begin(nex)[spin]; it < abij.unique_end(nex)[spin]; ++it, ++nd)
    {
      if (nd % ngrp == rank)
      {
        auto e = *it;
        abij.get_configuration(spin, nd, orbs);
        if (nex == 1)
        {
          ov_a    = T[*((*it) + 1)][*(*it)];
          Q[0][0] = 1.0 / ov_a;
        }
        else if (nex == 2)
        {
          ov_a = ma::I2x2(T[e[2]][e[0]], T[e[2]][e[1]], T[e[3]][e[0]], T[e[3]][e[1]], Q);
        }
        else if (nex == 3)
        {
          ov_a = ma::I3x3(T[e[3]][e[0]], T[e[3]][e[1]], T[e[3]][e[2]], T[e[4]][e[0]], T[e[4]][e[1]], T[e[4]][e[2]],
                          T[e[5]][e[0]], T[e[5]][e[1]], T[e[5]][e[2]], Q);
        }
        else
        {
          for (int p = 0; p < nex; p++)
            for (int q = 0; q < nex; q++)
              Q[p][q] = T[e[p + nex]][e[q]];
          ov_a = ma::invert<ComplexType>(Q, IWORK, WORK, 0.0);
        }
        ComplexType w(weights[nd]);
        if (std::abs(ov_a) != 0.0)
        {
          // add term coming from identity
          for (int i = 0; i < NEL; ++i)
            R[i][orbs[i]] += w;
          for (int p = 0; p < nex; ++p)
          {
            auto Rp = R[e[p]];
            auto Ip = Q[p];
            for (int q = 0; q < nex; ++q)
            {
              auto Ipq = Ip[q];
              auto Tq  = T[e[q + nex]];
              for (int i = 0; i < NEL; ++i)
                Rp[orbs[i]] -= w * Ipq * Tq[i];
              Rp[orbs[e[q]]] += w * Ipq;
            }
          }
        }
      }
    }
  }
}

// R[nwalk][ndet][nex][nact]
// T[nwalk][nact][nelec]
template<class MatA, class MatC, class PH_EXCT, class Iptr
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
         ,typename = std::enable_if_t< is_host_array<std::decay_t<MatA>>::value >,
         typename = std::enable_if_t< is_host_array<std::decay_t<MatC>>::value >
#endif
        >
inline void get_compact_ph_R_matrices(int rank, int ngrp, [[maybe_unused]] int spin, int nwalk, 
               int ndet, int nex, int nelec, int nact, Iptr const iexcit, Iptr const refc, 
               [[maybe_unused]] PH_EXCT& abij, MatA&& Tw, MatC&& Rw, [[maybe_unused]] int buffer_size_in_MB=0)
{
  using ma::conj;
  using std::get;
  using buffer_alloc_type       = HostBufferManager::template allocator_t<ComplexType>;
  HostBufferManager buffer_manager;
  StaticMatrix<ComplexType, buffer_alloc_type> Q({nex, nex},
                   buffer_manager.get_generator().template get_allocator<ComplexType>());
  std::vector<int> IWORK(nex+1);
  std::vector<ComplexType> WORK(nex*nex);
  RUNTIME_CHECK(Rw.size(0) == nwalk, "");
  RUNTIME_CHECK(Rw.size(1) == ndet, "");
  RUNTIME_CHECK(Rw.size(2) == nex, "");
  RUNTIME_CHECK(Rw.size(3) == nact, "");
  RUNTIME_CHECK(Tw.size(0) == nwalk, "");
  RUNTIME_CHECK(Tw.size(1) == nact, "");
  RUNTIME_CHECK(Tw.size(2) == nelec, "");
  std::vector<int> orbs(nelec);
  ComplexType ov_a;
  for(int iw=0; iw<nwalk; iw++) 
  {  
    auto T = Tw[iw]; 
    for(int nd=0; nd<ndet; nd++) 
    {
      if (nd % ngrp == rank)
      {
        auto R = Rw[iw][nd]; 
        auto e = iexcit+2*nex*nd;
        for(int i=0; i<nelec; i++) {
          orbs[i] = refc[i];
          for(int p=0; p<nex; p++) {
            if(e[p]==i) {
              orbs[i] = e[nex+p];
              break;
            }  
          }  
        }
        if (nex == 1)
        {
          ov_a    = T[e[1]][e[0]];
          Q[0][0] = 1.0 / ov_a;
        }
        else if (nex == 2)
        {
          ov_a = ma::I2x2(T[e[2]][e[0]], T[e[2]][e[1]], T[e[3]][e[0]], T[e[3]][e[1]], Q);
        }
        else if (nex == 3)
        {
          ov_a = ma::I3x3(T[e[3]][e[0]], T[e[3]][e[1]], T[e[3]][e[2]], 
                          T[e[4]][e[0]], T[e[4]][e[1]], T[e[4]][e[2]],
                          T[e[5]][e[0]], T[e[5]][e[1]], T[e[5]][e[2]], Q);
        }
        else
        {
          for (int p = 0; p < nex; p++)
            for (int q = 0; q < nex; q++)
              Q[p][q] = T[e[p + nex]][e[q]];
          ov_a = ma::invert<ComplexType>(Q, IWORK, WORK, 0.0);
        }
        if( std::abs(ov_a) != 0.0) {
          // compact notation:
          // R[p][nact] for p in [0, nex)
          // R[nex][nact] for diagonal term    
          for (int p = 0; p < nex; ++p)
          {
            auto Rp = R[p];
            auto Ip = Q[p];
            for (int q = 0; q < nex; ++q)
            {
              auto Ipq = Ip[q];
              auto Tq  = T[e[q + nex]];
              for (int i = 0; i < nelec; ++i)
                Rp[orbs[i]] -= Ipq * Tq[i];
              Rp[orbs[e[q]]] += Ipq;
            }
          }
        }
      }
    }
  }
}

// Calculates the Alpha/Alpha (XXX_first_step) contribution to E1, EJ, EX and KEright.
// Loops over excitation shells and calls Op.ph_excited_energy 
// wgt[ndet][nwalk]
// T[nwalk][nact][nelec]
// E[nwalk][3]
// KE[ndet][nwalk][nke]
template<class PH_EXCT, class MatW, class MatT, class MatE, class MatK, class Op>
inline void ph_excited_energies_first_step(int nelec, int nact, PH_EXCT& abij, MatW&& wgt,
                    MatT&& T, MatE&& E, MatK&& KE, Op& HamOps, [[maybe_unused]] int buffer_size_in_MB = 2048)
{
  using VType = typename std::decay_t<MatK>::element_type;
  using ptr = device_ptr<VType>;  
  using buffer_alloc_type       = DeviceBufferManager::template allocator_t<VType>;

  DeviceBufferManager buffer_manager;
  int nwalk = E.size(0);
  int spin(0);
  RUNTIME_CHECK(T.size(0) == nwalk, "");
  RUNTIME_CHECK(T.size(1) == nact, "");
  RUNTIME_CHECK(T.size(2) == nelec, "");
  RUNTIME_CHECK(wgt.size(1) == nwalk, "");
  RUNTIME_CHECK(E.size(1) == 3, "");
  RUNTIME_CHECK(KE.size(0) == wgt.size(0), "");
  RUNTIME_CHECK(KE.size(1) == nwalk, "");
  // no use of blocks yet! 
  long max2=0, max3=0;
  for (int nex = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if( ndet > max2 ) max2 = ndet;
    if( ndet * nex * nact > max3 ) max3 = ndet * nex * nact;
  }
  StaticVector<VType, buffer_alloc_type> RBuff(iextensions<1u>{nwalk*max3},
                buffer_manager.get_generator().template get_allocator<VType>());
  auto refc=abij.get_reference_configuration_device(spin);

  ma::fill(E, ComplexType(0.0));  
    
  for (int nex = 1, idet=1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex);
      Array_ref<VType, 4, ptr> R(RBuff.origin(), {nwalk, ndet, nex, nact});

      // generate R matrices for current excitation shell
      fill_n(R.origin(), R.num_elements(), VType(0.0));  
      get_compact_ph_R_matrices(0, 1, spin, nwalk, ndet, nex, nelec, nact, iexcit, refc, abij, T, R); 

      // assumes no TG_local parallelization for now, needs TG for this
      // calculate E and KE for current shell of excitations
      auto KEr=KE.sliced(idet,idet+ndet);
      HamOps.ph_excited_energy(Alpha, ndet, nex, nelec, nact, iexcit, refc, E, 
                wgt.sliced(idet, idet+ndet), R, KEr, true); 

      idet += ndet;
    }
  }
}

// Calculates the Alpha/Alpha (XXX_second_step) contribution to E1, EJ, EX and KEright.
// Loops over excitation shells and calls Op.ph_excited_energy 
// wgt[ndet][nwalk]
// T[nwalk][nact][nelec]
// E[nwalk][3]
// KE[ndet][nwalk][nke]
template<class PH_EXCT, class MatW, class MatT, class MatE, class MatKr, class Op>
inline void ph_excited_energies_second_step(int nelec, int nact, PH_EXCT& abij, MatW&& wgt,
       MatT&& T, MatE&& E, MatKr&& KE, Op& HamOps, [[maybe_unused]] int buffer_size_in_MB = 2048)
{
  using VType = typename std::decay_t<MatKr>::element_type;
  using ptr = device_ptr<VType>;  
  using buffer_alloc_type       = DeviceBufferManager::template allocator_t<VType>;
  using Cbuffer_alloc_type       = DeviceBufferManager::template allocator_t<ComplexType>;

  DeviceBufferManager buffer_manager;
  int nwalk = E.size(0);
  int spin(1);
  int nke = KE.size(2);
  RUNTIME_CHECK(T.size(0) == nwalk, "");
  RUNTIME_CHECK(T.size(1) == nact, "");
  RUNTIME_CHECK(T.size(2) == nelec, "");
  RUNTIME_CHECK(wgt.size(1) == nwalk, "");
  RUNTIME_CHECK(E.size(1) == 3, "");
  RUNTIME_CHECK(KE.size(0) == wgt.size(0), "");
  RUNTIME_CHECK(KE.size(1) == nwalk, "");
  // no use of blocks yet!
  long max2=0, max3=0;
  for (int nex = 1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if( ndet > max2 ) max2 = ndet;
    if( ndet * nex * nact > max3 ) max3 = ndet * nex * nact;
  }
  StaticVector<VType, buffer_alloc_type> RBuff(iextensions<1u>{nwalk*max3},
                buffer_manager.get_generator().template get_allocator<VType>());
  StaticVector<VType, buffer_alloc_type> KEBuff(iextensions<1u>{nwalk*max2*nke},
                buffer_manager.get_generator().template get_allocator<VType>());
  StaticVector<ComplexType, Cbuffer_alloc_type> eloc(iextensions<1u>{nwalk},
                buffer_manager.get_generator().template get_allocator<ComplexType>());
  auto refc=abij.get_reference_configuration_device(spin);

  ma::fill(E, ComplexType(0.0));  
    
  for (int nex = 1, idet=1; nex < abij.maximum_excitation_number()[spin]; nex++)
  {
    int ndet = abij.number_of_unique_excitations(nex)[spin];
    if(ndet > 0) {
      auto iexcit=abij.get_excitation_list_device(spin, nex);
      Array_ref<VType, 4, ptr> R(RBuff.origin(), {nwalk, ndet, nex, nact});
      Array_ref<VType, 3, ptr> KEl(KEBuff.origin(), {ndet, nwalk, nke});

      // generate R matrices for current excitation shell
      fill_n(R.origin(), R.num_elements(), VType(0.0));  
      get_compact_ph_R_matrices(0, 1, spin, nwalk, ndet, nex, nelec, nact, iexcit, refc, abij, T, R);

      // assumes no TG_local parallelization for now, needs TG for this
      // calculate E and KE for current shell of excitations
      HamOps.ph_excited_energy(Beta, ndet, nex, nelec, nact, iexcit, refc, E, 
                wgt.sliced(idet, idet+ndet), R, KEl, true); 

      ma::fill(eloc, ComplexType(0.0));
      // MAM: FIX why loop over idet? Run all simultaneously!	
      // eloc[iw] = sum_d_ke KE[d][iw][ke] KEl[d][iw][ke]
      for(int d=0; d<ndet; d++)  
	ma::dot('N','N', ComplexType(1.0), KE[idet+d], KEl[d], ComplexType(1.0), eloc);
      ma::axpy(ComplexType(1.0), eloc, E({0,nwalk},2));
 
      idet += ndet;
    }
  }
}

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
  for (int nex = max_nex_phmsd_det+1 /* note start */; nex < abij.maximum_excitation_number()[spin]; nex++)
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
    // R[w][i][ refc[i] ] += weights[w]
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

// R[nwalk][ndet][nex][nact]
// T[nwalk][nact][nelec]
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

} // namespace afqmc
} // namespace sfqmc


#endif
