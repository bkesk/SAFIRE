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

#ifndef NUMERICS_TENSOR_OPERATIONS_CPU_BACKEND_HPP
#define NUMERICS_TENSOR_OPERATIONS_CPU_BACKEND_HPP

#include "AFQMC/config.h"
#include "Numerics/detail/dispatch.hpp"

namespace ma
{
using sfqmc::afqmc::is_host_or_shm_array;

template<typename Q, typename T>
void KaKjw_to_KKwaj(int nwalk,
                    int nkpts,
                    int npol,
                    int nmo_max,
                    int nmo_tot,
                    int nocc_max,
                    int* nopk,
                    [[maybe_unused]] int* nopk0,
                    int* nelpk,
                    [[maybe_unused]] int* nelpk0,
                    Q const* A,
                    T* B,
		    cpu_backend)
{ 
  // OpenMP: Combine Ka,Kj loops into single loop and call parallel for
  int napj = nocc_max * npol * nmo_max;
  int na0  = 0; 
  for (int Ka = 0; Ka < nkpts; Ka++)
  {
    int na  = nelpk[Ka];
    int nj0 = 0;
    for (int Kj = 0; Kj < nkpts; Kj++)
    {
      int nj = nopk[Kj];
      auto G_{B + (Ka * nkpts + Kj) * nwalk * nocc_max * npol * nmo_max};
      for (int a = 0; a < na; a++)
      {
        int apj = a * npol * nmo_max;
        for (int p = 0; p < npol; p++)
        {
          auto Gc_{A + ((na0 + a) * npol + p) * nmo_tot * nwalk + nj0 * nwalk};
          for (int j = 0; j < nj; j++, apj++)
          {
            for (int w = 0, wapj = 0; w < nwalk; w++, ++Gc_, wapj += napj)
              G_[wapj + apj] = static_cast<T>(*Gc_);
          }
        }
      }
      nj0 += nj;
    }
    na0 += na;
  }
}

template<typename T, typename T1>
void KaKjw_to_QKajw(int nwalk,
                    int nkpts,
                    int npol,
                    int nmo_max,
                    int nmo_tot,
                    int nocc_max,
                    int* nmo,
                    int* nmo0,
                    int* nocc,
                    int* nocc0,
                    int* QKtok2,
                    T1 const* A,
                    T* B,
		    cpu_backend)
{
  // OpenMP: Combine Q,K loops into single loop and call parallel for
  for (int Q = 0; Q < nkpts; Q++)
  {
    for (int K = 0; K < nkpts; K++)
    {
      int Ka  = K;
      int Kj  = QKtok2[Q * nkpts + Ka];
      int na  = nocc[Ka];
      int nj  = nmo[Kj];
      int na0 = nocc0[Ka];
      int nj0 = nmo0[Kj];
      auto G_{B + (Q * nkpts + K) * nwalk * nocc_max * npol * nmo_max};
      for (int a = 0, a0 = 0; a < na; a++)
      {
        for (int p = 0; p < npol; p++, a0 += nmo_max * nwalk)
        {
          auto Gc_{A + ((na0 + a) * npol + p) * nmo_tot * nwalk + nj0 * nwalk};
          for (int j = 0, apj = a0; j < nj; j++, apj += nwalk)
          {
            for (int w = 0; w < nwalk; w++, ++Gc_)
            {
              G_[apj + w] = static_cast<T>(*Gc_);
            }
          }
        }
      }
    }
  }
}

template<typename T, typename Q>
void vKKwij_to_vwKiKj(int nwalk, int nkpts, int nmo_max, int nmo_tot, int* kk, int* nopk, int* nopk0, Q const* A, T* B, cpu_backend)
{
  for (int w = 0; w < nwalk; w++)
  {
    for (int Ki = 0; Ki < nkpts; Ki++)
    {
      for (int Kj = 0; Kj < nkpts; Kj++)
      {
        int ni  = nopk[Ki];
        int nj  = nopk[Kj];
        int ni0 = nopk0[Ki];
        int nj0 = nopk0[Kj];
        // setup copy/transpose tags
        // 1: copy from [Ki][Kj] without rho^+ term
        // 2: transpose from [Ki][Kj] without rho^+ term
        // -P: copy from [Ki][Kj] and transpose from [nkpts+P-1][Kj]
        int key = kk[Ki * nkpts + Kj];
        if (key == 3)
          continue;
        if (key == 2)
        { // transpose
          auto vb_{B + w * nmo_tot * nmo_tot + ni0 * nmo_tot + nj0};
          auto v_{A + ((Ki * nkpts + Kj) * nwalk + w) * nmo_max * nmo_max};
          for (int i = 0; i < ni; i++)
            for (int j = 0; j < nj; j++)
              vb_[i * nmo_tot + j] += static_cast<T>(v_[j * nmo_max + i]);
        }
        else if ((key == 1) || (key < 0))
        { // copy
          for (int i = 0; i < ni; i++)
          {
            auto vb_{B + w * nmo_tot * nmo_tot + (ni0 + i) * nmo_tot + nj0};
            auto v_{A + (((Ki * nkpts + Kj) * nwalk + w) * nmo_max + i) * nmo_max};
            for (int j = 0; j < nj; j++)
              vb_[j] += static_cast<T>(v_[j]);
          }
        }
        else
        {
          APP_ABORT(" Error: Programming error. \n");
        }
        if (key < 0)
        { // transpose
          key = (-key) - 1;
          auto vb_{B + w * nmo_tot * nmo_tot + nj0 * nmo_tot + ni0};
          auto v_{A + (((nkpts + key) * nkpts + Kj) * nwalk + w) * nmo_max * nmo_max};
          for (int i = 0; i < ni; i++)
            for (int j = 0; j < nj; j++)
              vb_[j * nmo_tot + i] += static_cast<T>(v_[i * nmo_max + j]);
        }
      }
    }
  }
}

template<typename T, typename Q>
void transpose_wabn_to_wban(int nwalk, int na, int nb, int nchol, T const* Tab, Q* Tba, cpu_backend)
{
  for (int w = 0; w < nwalk; w++)
  {
    for (int b = 0; b < nb; b++)
    {
      for (int a = 0; a < na; a++)
      {
        T const* Tn(Tab + ((w * na + a) * nb + b) * nchol);
        for (int n = 0; n < nchol; n++, ++Tba, ++Tn)
          *Tba = static_cast<Q>(*Tn);
      }
    }
  }
}

// C[n][w] = sum_a Aup[ I[n] ][a] B[w][a][ J[n] ]
// where I[n] = n2IJ[n]/M
//       J[n] = n2IJ[n]%M 
template<typename T, typename Q1, typename Q2>
void getGIJ_impl(int nw, int nIJ, int nspin, int M, int nel_a, int nel_b,
        Q1 const* Aup, int ldau, Q1 const* Adn, int ldad, Q2 const* B, int ldb, long strideB, 
        T* C, int ldc, size_t const* n2IJ, cpu_backend)
{
  size_t M_ = size_t(M);
  Q1 const* A_;
  Q2 const* B_;
  int nt;
  for(int iw=0; iw<nw; ++iw) {
    for(int n=0; n<nIJ; ++n) {
      int I = int(n2IJ[n]/M_);
      int J = int(n2IJ[n]%M_);
      if( nspin == 2 ) {
        if( I < M ) {
          nt = nel_a;
          A_ = (Aup + I*ldau);
          B_ = (B + iw*strideB + J);
        } else {
          nt = nel_b;
          A_ = (Adn + (I-M)*ldad);
          B_ = (B + iw*strideB + nel_a*ldb + J);
        }
      } else {
        nt = nel_a;
        A_ = (Aup + I*ldau);
        B_ = (B + iw*strideB + J);
      }
      T res(0.0);
      for(int a=0; a<nt; ++a, ++A_, B_+=ldb)
        res += static_cast<T>(*A_) * static_cast<T>(*B_);
      C[ n*ldc + iw ] = res;
    }
  }
}

// C[g] = sum_n,w,i,j Xw[w] * A[n][g][i][w][j] * B[n][g][j][w][i]
template<class MatX, class MatA, class MatB, class MatC,
    typename = std::enable_if_t< is_host_or_shm_array<std::decay_t<MatA>>::value >,
    typename = std::enable_if_t< is_host_or_shm_array<std::decay_t<MatB>>::value >,
    typename = std::enable_if_t< is_host_or_shm_array<std::decay_t<MatC>>::value >,
    typename = std::enable_if_t< is_host_or_shm_array<std::decay_t<MatX>>::value >
>
inline static void AGiwj_BGjwi_CG(MatX const& Xw, MatA const& AGiwj, MatB const& BGjwi, MatC&& CG, cpu_backend)
{
  static_assert(std::decay_t<MatA>::dimensionality == 5, "Wrong dimensionality.");
  static_assert(std::decay_t<MatB>::dimensionality == 5, "Wrong dimensionality.");
  static_assert(std::decay_t<MatX>::dimensionality == 1, "Wrong dimensionality.");
  static_assert(std::decay_t<MatC>::dimensionality == 1, "Wrong dimensionality.");

  RUNTIME_CHECK(AGiwj.size(0) == BGjwi.size(0), "");
  RUNTIME_CHECK(AGiwj.size(1) == BGjwi.size(1), "");
  RUNTIME_CHECK(AGiwj.size(2) == BGjwi.size(4), "");
  RUNTIME_CHECK(AGiwj.size(3) == BGjwi.size(3), "");
  RUNTIME_CHECK(AGiwj.size(4) == BGjwi.size(2), "");
  RUNTIME_CHECK(Xw.size(0) == AGiwj.size(3), "");
  RUNTIME_CHECK(CG.size(0) == AGiwj.size(1), "");

  using Type = typename std::decay_t<MatC>::element_type;

  int ni = AGiwj.size(2);
  int nj = AGiwj.size(4);   
  for(int ib=0; ib<AGiwj.size(0); ib++) {
    for(int ig=0; ig<AGiwj.size(1); ig++) {
      auto A{AGiwj[ib][ig]};
      auto B{BGjwi[ib][ig]};    
      for(int iw=0; iw<AGiwj.size(3); iw++) {
        Type v(0.0); 
// use blocking!!!!! poor performance as is!!!
        for(int i=0; i<ni; i++) {
          for(int j=0; j<nj; j++) {
            v += static_cast<Type>(A[i][iw][j]) * static_cast<Type>(B[j][iw][i]);
          }
        }
        CG[ig] += static_cast<Type>(Xw[iw]) * v;
      }
    }
  }
}

} // namespace ma

#endif
