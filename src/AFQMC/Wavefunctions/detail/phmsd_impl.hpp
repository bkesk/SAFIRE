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

#ifndef PHMSD_DETAIL_IMPLEMENTATION_HPP 
#define PHMSD_DETAIL_IMPLEMENTATION_HPP 

#include <cassert>

#if defined(ENABLE_CUDA)
#include "Numerics/detail/CUDA/Kernels/phmsd_energy.cuh"
#include "Numerics/detail/CUDA/Kernels/phmsd_determinants.cuh"
#include "Numerics/detail/CUDA/Kernels/phmsd_inverse.cuh"
#include "Numerics/detail/CUDA/Kernels/extract_overlap_matrix.cuh"
#include "Numerics/detail/CUDA/Kernels/construct_phmsd_R.cuh"
#elif defined(ENABLE_HIP)
// Need to finish kernels!!!
#error
#endif

#include "Memory/buffer_managers.h"
#include "Numerics/ma_operations.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"

namespace ma
{
#if defined (ENABLE_DEVICE)
using sfqmc::afqmc::is_host_array;
using sfqmc::afqmc::is_device_array;
#endif

// E[w] = sum_abpqdn w[d][w] T[w][i[p]][n][a] * T[w][i[q]][n][b] * Rwdpa * Rwdqb 
// KE[d][w][n] = sum_pa T[w][i[p]][a][n] * Rwdpa
// R is the compact version of the ph R matrix, where for each determinant there
// are nex rows
template<class MatT, class MatR, class MatW, class EVec, class MatK,
          typename = typename std::enable_if_t<(std::decay_t<MatT>::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay_t<MatR>::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay_t<MatW>::dimensionality == 2)>,
          typename = typename std::enable_if_t<(std::decay_t<EVec>::dimensionality == 1)>,
          typename = typename std::enable_if_t<(std::decay_t<MatK>::dimensionality == 3)>
#if defined (ENABLE_DEVICE)
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatT>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatR>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatW>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<EVec>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatK>>::value >
#endif
         >
void ph_excited_2body_energy_dense_cholesky_Tpna(int const* iexcit, int const* refc, MatT&& Twina, MatR&& R, MatW&& wgt,
        EVec&& EX, EVec&& EJ, MatK&& KE)
{
  using EType = typename std::decay_t<EVec>::element_type;
  using KType = typename std::decay_t<MatK>::element_type;
  static_assert( std::is_same<KType, std::decay_t<typename std::decay_t<MatT>::element_type>>::value, "Wrong type." );
  static_assert( std::is_same<KType, std::decay_t<typename std::decay_t<MatR>::element_type>>::value, "Wrong type." );

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

  std::vector<int> occps(nelec);
  using sfqmc::afqmc::HostBufferManager;
  using boost::multi::array;  
  EType TWO(2.0);  
  EType HALF(0.5);  
  HostBufferManager buffer_manager;
  array<KType, 2, HostBufferManager::template allocator_t<KType>> Fa({2, std::max(nact, nchol)},
            buffer_manager.get_generator().template get_allocator<KType>());
  using std::fill_n;
  for(int iw=0; iw<nwalk; iw++) {
    for(int idet=0; idet<ndet; idet++) {

      auto&& K1D(KE[idet][iw]);
      ma::fill(K1D,KType(0.0));

      EType eX(0.0);
      for(int i=0; i<nelec; i++) {
        occps[i] = refc[i];
        for(int ie1=0; ie1<nex; ie1++) {
          int ip = iexcit[ idet*2*nex + ie1 ];
          if(ip == i) {
            occps[i] = iexcit[ idet*2*nex + ie1 + nex ];
            break;
          }
        }
      }
      // spin-diagonal part of the kinetic energy of the reference configuration,
      // since the routine produces EJ-EJref (including only the spin-diagonal part) 
      fill_n(Fa[0].origin(),nchol,KType(0.0));  
      for(int i=0; i<nelec; i++) 
        ma::axpy(KType(1.0), Twina[iw][i]({0,nchol}, refc[i]), Fa[0].sliced(0,nchol));
      EType eJ0 = static_cast<EType>(ma::dot(Fa[0].sliced(0,nchol), Fa[0].sliced(0,nchol)));

      for(int ie1=0; ie1<nex; ie1++) {
        int ip = iexcit[ idet*2*nex + ie1 ]; 
        // ie1==ie2 term

        ma::product(Twina[iw][ip],R[iw][idet][ie1],Fa[0].sliced(0,nchol));
        eX += static_cast<EType>( ma::dot(Fa[0].sliced(0,nchol),Fa[0].sliced(0,nchol)) );  
        ma::axpy(KType(1.0), Fa[0].sliced(0, nchol), K1D);

        // R[p]*R[q] terms
        for(int ie2=ie1+1; ie2<nex; ie2++) {
          int iq = iexcit[ idet*2*nex + ie2 ];
          // Twq[n][a] * Rwp[a] = Fn
          ma::product(Twina[iw][iq],R[iw][idet][ie1],Fa[0].sliced(0,nchol)); 
          // Twp[n][b] * Rwq[b] = Fn
          ma::product(Twina[iw][ip],R[iw][idet][ie2],Fa[1].sliced(0,nchol)); 
          eX += TWO * static_cast<EType>( ma::dot(Fa[0].sliced(0,nchol),Fa[1].sliced(0,nchol)) );  
        }

        // R[p]*R[diagonal] term
        for(int j=0; j<nelec; j++) {
          int Oj = occps[j]; 
          ma::product(ma::T(Twina[iw][j]),Twina[iw][ip]({0, nchol}, Oj), Fa[0].sliced(0, nact));  
          //eX += static_cast<EType>(ma::dot(Fa[0].sliced(0, nact),R[iw][idet][ie1]));
          eX += TWO * static_cast<EType>(ma::dot(Fa[0].sliced(0, nact),R[iw][idet][ie1]));
        }

      }  

      // Rdiag-Rdiag terms
      for(int i=0; i<nelec; i++) {
        int Oi = occps[i];
        int ri = refc[i];
        if( not(Oi==ri) )
            eX += static_cast<EType>( ma::dot(Twina[iw][i]({0, nchol},Oi), Twina[iw][i]({0, nchol}, Oi)) ) -
                  static_cast<EType>( ma::dot(Twina[iw][i]({0, nchol},ri), Twina[iw][i]({0, nchol},ri)) ); 
        for(int j=i+1; j<nelec; j++) {
          int Oj = occps[j];
          int rj = refc[j];
          // Rdiag-Rdiag terms
          if( Oi!=ri or Oj!=rj )
            eX += TWO * (static_cast<EType>( ma::dot(Twina[iw][i]({0, nchol},Oj), Twina[iw][j]({0, nchol},Oi)) ) -
                 static_cast<EType>( ma::dot(Twina[iw][i]({0, nchol},rj), Twina[iw][j]({0, nchol},ri)) )); 
        }
      }

      // R[diagonal]*R[diagonal] J-term
      for(int i=0; i<nelec; i++) {
        int Oi = occps[i];
        ma::axpy(KType(1.0), Twina[iw][i]({0, nchol}, Oi), K1D);
      }

      EX[iw] -= HALF * static_cast<EType>(wgt[idet][iw]) * eX;
      EType eJ = static_cast<EType>(ma::dot(K1D,K1D)) - eJ0;
      EJ[iw] += HALF * static_cast<EType>(wgt[idet][iw]) * eJ;
    }
  }
}
 
// E[w] = sum_abpqdn w[d][w] T[w][i[p]][a][n] * T[w][i[q]][b][n] * Rwdpa * Rwdqb 
// KE[d][w][n] = sum_pa T[w][i[p]][a][n] * Rwdpa
// R is the compact version of the ph R matrix, where for each determinant there
// are nex rows
template<class MatT, class MatR, class MatW, class EVec, class MatK, 
          typename = typename std::enable_if_t<(std::decay_t<MatT>::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay_t<MatR>::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay_t<MatW>::dimensionality == 2)>,
          typename = typename std::enable_if_t<(std::decay_t<EVec>::dimensionality == 1)>,
          typename = typename std::enable_if_t<(std::decay_t<MatK>::dimensionality == 3)>
#if defined (ENABLE_DEVICE)
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatT>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatR>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatW>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<EVec>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatK>>::value >
#endif
         >
void ph_excited_2body_energy_dense_cholesky_Tpan(int const* iexcit, int const* refc, MatT&& Twian, MatR&& R, MatW&& wgt, 
        EVec&& EX, EVec&& EJ, MatK&& KE)
{
  using EType = typename std::decay_t<EVec>::element_type;
  using KType = typename std::decay_t<MatK>::element_type;
  static_assert( std::is_same<KType, std::decay_t<typename std::decay_t<MatT>::element_type>>::value, "Wrong type." );
  static_assert( std::is_same<KType, std::decay_t<typename std::decay_t<MatR>::element_type>>::value, "Wrong type." );

  int nwalk = R.size(0);
  int ndet  = R.size(1);
  int nex   = R.size(2);
  int nact  = R.size(3);
  int nelec = Twian.size(1); 
  int nchol = Twian.size(3);

  RUNTIME_CHECK(Twian.size(0) == nwalk, "");
  RUNTIME_CHECK(Twian.size(2) == nact, "");
  RUNTIME_CHECK(wgt.size(0) == ndet, "");
  RUNTIME_CHECK(wgt.size(1) == nwalk, "");
  RUNTIME_CHECK(EJ.size(0) == nwalk, "");
  RUNTIME_CHECK(EX.size(0) == nwalk, "");
  RUNTIME_CHECK(KE.size(0) == ndet, "");
  RUNTIME_CHECK(KE.size(1) == nwalk, "");
  RUNTIME_CHECK(KE.size(2) == nchol, "");

  std::vector<int> occps(nelec);
  using sfqmc::afqmc::HostBufferManager;
  using boost::multi::array;  
  EType TWO(2.0);  
  EType HALF(0.5);  
  HostBufferManager buffer_manager;
  array<KType, 2, HostBufferManager::template allocator_t<KType>> Fa({2, std::max(nact, nchol)},
            buffer_manager.get_generator().template get_allocator<KType>());
  using std::fill_n;
  for(int iw=0; iw<nwalk; iw++) {
    for(int idet=0; idet<ndet; idet++) {

      auto&& K1D(KE[idet][iw]);		
      ma::fill(K1D,KType(0.0));

      EType eX(0.0);
      for(int i=0; i<nelec; i++) {
        occps[i] = refc[i];
        for(int ie1=0; ie1<nex; ie1++) {
          int ip = iexcit[ idet*2*nex + ie1 ];
          if(ip == i) {
            occps[i] = iexcit[ idet*2*nex + ie1 + nex ];
            break;
          }
        }
      }
      // spin-diagonal part of the kinetic energy of the reference configuration,
      // since the routine produces EJ-EJref (including only the spin-diagonal part) 
      fill_n(Fa[0].origin(),nchol,KType(0.0));  
      for(int i=0; i<nelec; i++) 
        ma::axpy(KType(1.0), Twian[iw][i][refc[i]], Fa[0].sliced(0,nchol));
      EType eJ0 = static_cast<EType>(ma::dot(Fa[0].sliced(0,nchol), Fa[0].sliced(0,nchol)));

      for(int ie1=0; ie1<nex; ie1++) {
        int ip = iexcit[ idet*2*nex + ie1 ]; 
        // ie1==ie2 term

        ma::product(ma::T(Twian[iw][ip]),R[iw][idet][ie1],Fa[0].sliced(0,nchol));
        eX += static_cast<EType>( ma::dot(Fa[0].sliced(0,nchol),Fa[0].sliced(0,nchol)) );  
        ma::axpy(KType(1.0), Fa[0].sliced(0, nchol), K1D);

        // R[p]*R[q] terms
        for(int ie2=ie1+1; ie2<nex; ie2++) {
          int iq = iexcit[ idet*2*nex + ie2 ];
          // Twq[a][n] * Rwp[a] = Fn
          ma::product(ma::T(Twian[iw][iq]),R[iw][idet][ie1],Fa[0].sliced(0,nchol)); 
          // Twp[b][n] * Rwq[b] = Fn
          ma::product(ma::T(Twian[iw][ip]),R[iw][idet][ie2],Fa[1].sliced(0,nchol)); 
          eX += TWO * static_cast<EType>( ma::dot(Fa[0].sliced(0,nchol),Fa[1].sliced(0,nchol)) );  
        }

        // R[p]*R[diagonal] term
        for(int j=0; j<nelec; j++) {
          int Oj = occps[j]; 
          ma::product(Twian[iw][j],Twian[iw][ip][Oj], Fa[0].sliced(0, nact));  
          //eX += static_cast<EType>(ma::dot(Fa[0].sliced(0, nact),R[iw][idet][ie1]));
          eX += TWO * static_cast<EType>(ma::dot(Fa[0].sliced(0, nact),R[iw][idet][ie1]));
        }

      }  

      // Rdiag-Rdiag terms
      for(int i=0; i<nelec; i++) {
        int Oi = occps[i];
        int ri = refc[i];
        if( not(Oi==ri) )
            eX += static_cast<EType>( ma::dot(Twian[iw][i][Oi], Twian[iw][i][Oi]) ) -
                  static_cast<EType>( ma::dot(Twian[iw][i][ri], Twian[iw][i][ri]) ); 
        for(int j=i+1; j<nelec; j++) {
          int Oj = occps[j];
          int rj = refc[j];
          // Rdiag-Rdiag terms
          if( Oi!=ri or Oj!=rj )
            eX += TWO * (static_cast<EType>( ma::dot(Twian[iw][i][Oj], Twian[iw][j][Oi]) ) -
                 static_cast<EType>( ma::dot(Twian[iw][i][rj], Twian[iw][j][ri]) )); 
        }
      }

      // R[diagonal]*R[diagonal] J-term
      for(int i=0; i<nelec; i++) {
        int Oi = occps[i];
        ma::axpy(KType(1.0), Twian[iw][i][Oi], K1D);
      }

      EX[iw] -= HALF * static_cast<EType>(wgt[idet][iw]) * eX;
      EType eJ = static_cast<EType>(ma::dot(K1D,K1D)) - eJ0;
      EJ[iw] += HALF * static_cast<EType>(wgt[idet][iw]) * eJ;
    }
  }
}

template<class MatS, class MatR, class MatW, class EVec,
          typename = typename std::enable_if_t<(std::decay_t<MatS>::dimensionality == 3)>,
          typename = typename std::enable_if_t<(std::decay_t<MatR>::dimensionality == 4)>,
          typename = typename std::enable_if_t<(std::decay_t<MatW>::dimensionality == 2)>,
          typename = typename std::enable_if_t<(std::decay_t<EVec>::dimensionality == 1)>
#if defined (ENABLE_DEVICE)
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatS>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatR>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<MatW>>::value >
          ,typename = std::enable_if_t< is_host_array<std::decay_t<EVec>>::value >
#endif
         >
void ph_excited_1body_energy(int const* iexcit, int const* refc, MatS&& S, MatR&& R, 
        MatW&& wgt, EVec&& E)
{
  using EType = typename std::decay_t<EVec>::element_type;
  int nwalk = R.size(0);
  int ndet  = R.size(1);
  int nex   = R.size(2);
  int nact  = R.size(3);

  RUNTIME_CHECK(S.size(0) == nwalk, "");
  RUNTIME_CHECK(S.size(2) == nact, "");
  RUNTIME_CHECK(wgt.size(0) == ndet, "");
  RUNTIME_CHECK(wgt.size(1) == nwalk, "");
  RUNTIME_CHECK(E.size(0) == nwalk, "");

  boost::multi::array_ref<int const, 3> iex(iexcit, {ndet, 2, nex});
  for (int iw = 0; iw < nwalk; iw++) {
    for (int d = 0; d < ndet; d++) {
      EType e_(0.0);
      for (int p = 0; p < nex; p++) {
        int ip = iex[d][0][p];
        auto S_(S[iw][ip]);
        auto R_(R[iw][d][p]);
        for(int a=0; a<nact; a++)
          e_ += static_cast<EType>(S_[a])*static_cast<EType>(R_[a]);
        e_ += static_cast<EType>(S_[iex[d][1][p]]) - 
              static_cast<EType>(S_[refc[ip]]);
      }
      E[iw] += static_cast<EType>(wgt[d][iw]) * e_;
    }
  }
}

} // namespace ma

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


#endif
