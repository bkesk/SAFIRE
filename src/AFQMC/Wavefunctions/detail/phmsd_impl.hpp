/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

#include <cassert>
#include "nda/nda.hpp"
#include "nda/tensor.hpp"

/*
#if defined(ENABLE_CUDA)
#include "Numerics/detail/CUDA/Kernels/phmsd_energy.cuh"
#include "Numerics/detail/CUDA/Kernels/phmsd_determinants.cuh"
#include "Numerics/detail/CUDA/Kernels/phmsd_inverse.cuh"
#include "Numerics/detail/CUDA/Kernels/extract_overlap_matrix.cuh"
#include "Numerics/detail/CUDA/Kernels/construct_phmsd_R.cuh"
#endif
*/

namespace sfqmc::afqmc 
{

// E[w] = sum_abpqdn w[d][w] T[w][i[p]][n][a] * T[w][i[q]][n][b] * Rwdpa * Rwdqb 
// KE[d][w][n] = sum_pa T[w][i[p]][a][n] * Rwdpa
// R is the compact version of the ph R matrix, where for each determinant there
// are nex rows
void ph_excited_2body_energy_dense_cholesky(nda::MemoryVector auto const& iexcit, nda::MemoryVector auto const& refc,
              nda::MemoryArrayOfRank<4> auto const& Twina, nda::MemoryArrayOfRank<4> auto const& R,
              nda::MemoryMatrix auto const& wgt, nda::MemoryVector auto&& EX,
              nda::MemoryVector auto&& EJ, nda::MemoryArrayOfRank<3> auto && KE)
{
  using nda::range;
  auto all = range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<decltype(Twina)>();
  memory::check_memory_space<MEM>(iexcit,refc,Twina,R,wgt,EX,EJ,KE);
  auto [nwalk, ndet, nex, nact] = R.shape();
  long nelec = Twina.extent(1);
  long nchol = Twina.extent(2);

  utils::check(wgt.shape() == std::array<long,2>{ndet,nwalk}, "Size mismatch");
  utils::check(EX.shape() == std::array<long,1>{nwalk}, "Size mismatch");
  utils::check(EJ.shape() == std::array<long,1>{nwalk}, "Size mismatch");
  utils::check(KE.shape() == std::array<long,3>{ndet,nwalk,nchol}, "Size mismatch");
  utils::check(Twina.extent(0)==nwalk and Twina.extent(3)==nact, "Size mismatch");

  nda::vector<int> occps(nelec);
  ComplexType zero(0.0), one(1.0), two(2.0), half(0.5);  
  memory::buffered_array<MEM,ComplexType,2> Fbuff(2,std::max(nact, nchol));
  nda::array_view<nda::get_value_t<decltype(iexcit)> const,3> iex(std::array<long,3>{ndet,2,nex}, iexcit.data());

  for(int iw=0; iw<nwalk; iw++) {
    for(int idet=0; idet<ndet; idet++) {

      auto Fa = Fbuff(0,range(nact));
      auto Fn = Fbuff(0,range(nchol));
      auto Fn1 = Fbuff(1,range(nchol));
      auto Kn = KE(idet,iw,all);
      auto Rxa = R(iw,idet,all,all); 
      Kn() = zero; 

      ComplexType eX(0.0);
      for(int i=0; i<nelec; i++) {
        occps(i) = refc(i);
        for(int ie1=0; ie1<nex; ie1++) {
          int ip = iex(idet,0,ie1);
          if(ip == i) {
            occps(i) = iex(idet,1,ie1); 
            break;
          }
        }
      }
      // spin-diagonal part of the kinetic energy of the reference configuration,
      // since the routine produces EJ-EJref (including only the spin-diagonal part) 
      // Fn = sum_i Twina(iw,i,n,refc(i))
      Fn = zero;
      for(int i=0; i<nelec; i++) 
        Fn += Twina(iw,i,range(nchol),refc(i));

      ComplexType eJ0 = nda::sum( Fn*Fn );

      for(int ie1=0; ie1<nex; ie1++) {
        int ip = iex(idet,0,ie1);

        // ie1==ie2 term
        nda::tensor::contract(one,Twina(iw,ip,all,all),"na",Rxa(ie1,all),"a",zero,Fn,"n");
        eX += nda::sum( Fn*Fn );
        Kn() += Fn;

        // R[p]*R[q] terms
        for(int ie2=ie1+1; ie2<nex; ie2++) {
          int iq = iex(idet,0,ie2);
          // Twq[n][a] * Rwp[a] = Fn
          nda::tensor::contract(one,Twina(iw,iq,all,all),"na",Rxa(ie1,all),"a",zero,Fn,"n");
          // Twp[n][b] * Rwq[b] = Fn
          nda::tensor::contract(one,Twina(iw,ip,all,all),"na",Rxa(ie2,all),"a",zero,Fn1,"n");
          eX += two * nda::sum( Fn*Fn1 ); 
        }

        // R[p]*R[diagonal] term
        for(int j=0; j<nelec; j++) {
          int Oj = occps(j); 
          nda::tensor::contract(one,Twina(iw,j,all,all),"na",Twina(iw,ip,all,Oj),"n",zero,Fa,"a");
          eX += two * nda::sum( Fa*Rxa(ie1,all) ); 
        }

      }  

      // Rdiag-Rdiag terms
      for(int i=0; i<nelec; i++) {
        int Oi = occps(i);
        int ri = refc(i);
        if( Oi!=ri )
          eX += nda::sum( Twina(iw,i,all,Oi)*Twina(iw,i,all,Oi) ) - nda::sum( Twina(iw,i,all,ri)*Twina(iw,i,all,ri) ); 

        for(int j=i+1; j<nelec; j++) {
          int Oj = occps(j);
          int rj = refc(j);
          // Rdiag-Rdiag terms
          if( Oi!=ri or Oj!=rj )
            eX += two * ( nda::sum( Twina(iw,i,all,Oj)*Twina(iw,j,all,Oi) ) - nda::sum( Twina(iw,i,all,rj)*Twina(iw,j,all,ri) )); 
        }
      }
      // R[diagonal]*R[diagonal] J-term
      for(int i=0; i<nelec; i++) 
        Kn() += Twina(iw,i,all,occps(i));

      EX(iw) -= half * wgt(idet,iw) * eX;
      EJ(iw) += half * wgt(idet,iw) * ( nda::sum( Kn*Kn ) - eJ0 );
    }
  }
}

void ph_excited_1body_energy(nda::MemoryVector auto const& iexcit, nda::MemoryVector auto const& refc, 
                  nda::MemoryArrayOfRank<3> auto const& S, nda::MemoryArrayOfRank<4> auto const& R, 
                  nda::MemoryMatrix auto const& wgt, nda::MemoryVector auto&& E)
{
  using nda::range;
  auto all = range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<decltype(S)>(); 
  memory::check_memory_space<MEM>(iexcit,refc,S,R,wgt,R);
  auto [nwalk, ndet, nex, nact] = R.shape();
  int nelec = S.extent(1);

  utils::check(S.shape() == std::array<long,3>{nwalk,nelec,nact}, "Size mismatch");
  utils::check(wgt.shape() == std::array<long,2>{ndet,nwalk}, "Size mismatch");
  utils::check(E.shape() == std::array<long,1>{nwalk}, "Size mismatch");

  if constexpr (MEM==HOST_MEMORY) {
    nda::array_view<nda::get_value_t<decltype(iexcit)> const,3> iex(std::array<long,3>{ndet,2,nex}, iexcit.data());
    for (int iw = 0; iw < nwalk; iw++) {
      for (int d = 0; d < ndet; d++) {
        ComplexType e_(0.0);
        for (int p = 0; p < nex; p++) {
          int ip = iex(d,0,p); 
          e_ += nda::sum(S(iw,ip,all)*R(iw,d,p,all)) 
               + S(iw,ip,iex(d,1,p)) - S(iw,ip,refc(ip));
        }
        E(iw) += wgt(d,iw) * e_;
      }
    }
  } else {
    // custom kernel???
    utils::check(false,"finish");
//  kernels::ph_excited_1body_energy(iexcit,refc,S,R,wgt,E);
  }
}
/*

#if defined(ENABLE_DEVICE)
namespace ma 
{

template<typename I1, class MatT, class MatR, class WVec, class EVec, class MatK,
          typename = typename std::enable_if_t<(std::decay<MatT>::type::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay<MatR>::type::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay<WVec>::type::dimensionality == 2)>,
          typename = typename std::enable_if_t<(std::decay<EVec>::type::dimensionality == 1)>,
          typename = typename std::enable_if_t<(std::decay<MatK>::type::dimensionality == 3)>,
          typename = std::enable_if_t< is_device_array<std::decay_t<MatT>>::value >,
          typename = std::enable_if_t< is_device_array<std::decay_t<MatR>>::value >,
          typename = std::enable_if_t< is_device_array<std::decay_t<WVec>>::value >,
          typename = std::enable_if_t< is_device_array<std::decay_t<EVec>>::value >,
          typename = std::enable_if_t< is_device_array<std::decay_t<MatK>>::value >
         >
void ph_excited_2body_energy_dense_cholesky_Tpan(device::device_pointer<I1> iexcit, device::device_pointer<I1> refc, 
	MatT&& T, MatR&& R, WVec&& wgt, EVec&& EX, EVec&& EJ, MatK&& KE)
{
  APP_ABORT(" Error: ph_excited_2body_energy_dense_cholesky_Tpan should not be called in GPU. "); 
}

template<typename I1, class MatT, class MatR, class WVec, class EVec, class MatK,
          typename = typename std::enable_if_t<(std::decay_t<MatT>::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay_t<MatR>::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay_t<WVec>::dimensionality == 2)>,
          typename = typename std::enable_if_t<(std::decay_t<EVec>::dimensionality == 1)>,
          typename = typename std::enable_if_t<(std::decay_t<MatK>::dimensionality == 3)>,
          typename = std::enable_if_t< is_device_array<std::decay_t<MatT>>::value >,
          typename = std::enable_if_t< is_device_array<std::decay_t<MatR>>::value >,
          typename = std::enable_if_t< is_device_array<std::decay_t<WVec>>::value >,
          typename = std::enable_if_t< is_device_array<std::decay_t<EVec>>::value >,
          typename = std::enable_if_t< is_device_array<std::decay_t<MatK>>::value >
         >
void ph_excited_2body_energy_dense_cholesky_Tpna(device::device_pointer<I1> iexcit, device::device_pointer<I1> refc, 
	MatT&& Twina, MatR&& R, WVec&& wgt, EVec&& EX, EVec&& EJ, MatK&& KE)
{
  int nwalk = R.size(0);
  int ndet  = R.size(1);
  int nex   = R.size(2);
  int nact  = R.size(3);
  int nelec = Twina.size(1);
  int nchol = Twina.size(2);

  RUNTIME_CHECK(Twina.size(0) == nwalk, "");
  RUNTIME_CHECK(Twina.size(3) == nact, "");
  RUNTIME_CHECK(wgt.size(0) == ndet, "");
  RUNTIME_CHECK(wgt.size(1) == nwalk, "");
  RUNTIME_CHECK(EJ.size(0) == nwalk, "");
  RUNTIME_CHECK(EX.size(0) == nwalk, "");
  RUNTIME_CHECK(KE.size(0) == ndet, "");
  RUNTIME_CHECK(KE.size(1) == nwalk, "");
  RUNTIME_CHECK(KE.size(2) == nchol, "");

  // Twina continuous 
  RUNTIME_CHECK(Twina.stride(0) == Twina.size(3) * Twina.size(2) * Twina.size(1), ""); 
  RUNTIME_CHECK(Twina.stride(1) == Twina.size(3) * Twina.size(2), ""); 
  RUNTIME_CHECK(Twina.stride(2) == Twina.size(3), ""); 
  RUNTIME_CHECK(Twina.stride(3) == 1, ""); 
  // R continuous 
  RUNTIME_CHECK(R.stride(0) == R.size(3) * R.size(2) * R.size(1), ""); 
  RUNTIME_CHECK(R.stride(1) == R.size(3) * R.size(2), ""); 
  RUNTIME_CHECK(R.stride(2) == R.size(3), ""); 
  RUNTIME_CHECK(R.stride(3) == 1, ""); 

  RUNTIME_CHECK(wgt.stride(1) == 1, "");

  kernels::ph_excited_2body_energy_dense_cholesky_Tpna(nwalk, ndet, nex, nact, nelec, nchol, 
        raw_pointer_cast(iexcit), raw_pointer_cast(refc), raw_pointer_cast(Twina.origin()), 
	raw_pointer_cast(R.origin()), raw_pointer_cast(wgt.origin()), wgt.stride(0),
        raw_pointer_cast(EX.origin()), EX.stride(0), raw_pointer_cast(EJ.origin()), EJ.stride(0), 
	raw_pointer_cast(KE.origin()), KE.stride(1), KE.stride(0));
}

template<typename I1, class MatS, class MatR, class MatW, class EVec,
          typename = typename std::enable_if_t<(std::decay_t<MatS>::dimensionality == 3)>,
          typename = typename std::enable_if_t<(std::decay_t<MatR>::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay_t<MatW>::dimensionality == 2)>,
          typename = typename std::enable_if_t<(std::decay_t<EVec>::dimensionality == 1)>
          ,typename = std::enable_if_t< is_device_array<std::decay_t<MatS>>::value >
          ,typename = std::enable_if_t< is_device_array<std::decay_t<MatR>>::value >
          ,typename = std::enable_if_t< is_device_array<std::decay_t<MatW>>::value >
          ,typename = std::enable_if_t< is_device_array<std::decay_t<EVec>>::value >
         >
void ph_excited_1body_energy(device::device_pointer<I1> iexcit, device::device_pointer<I1> refc,
	MatS&& S, MatR&& R, MatW&& wgt, EVec&& E)
{
  int nwalk = R.size(0);
  int ndet  = R.size(1);
  int nex   = R.size(2);
  int nact  = R.size(3);
  int nelec = S.size(1);
  
  RUNTIME_CHECK(S.size(0) == nwalk, "");
  RUNTIME_CHECK(S.size(2) == nact, "");
  RUNTIME_CHECK(wgt.size(0) == ndet, "");
  RUNTIME_CHECK(wgt.size(1) == nwalk, "");
  RUNTIME_CHECK(E.size(0) == nwalk, "");

  kernels::ph_excited_1body_energy(nwalk, ndet, nex, nact, nelec, 
        raw_pointer_cast(iexcit), raw_pointer_cast(refc), raw_pointer_cast(S.origin()), 
	raw_pointer_cast(R.origin()), raw_pointer_cast(wgt.origin()), wgt.stride(0),
        raw_pointer_cast(E.origin()), E.stride(0));
}


} // namespace ma
#endif
*/

} // sfqmc::afqmc

