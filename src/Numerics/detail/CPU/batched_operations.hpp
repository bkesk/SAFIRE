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

#ifndef NUMERICS_BATCHED_OPERATIONS_CPU_BACKEND_HPP
#define NUMERICS_BATCHED_OPERATIONS_CPU_BACKEND_HPP

#include <cassert>
#include "Numerics/detail/dispatch.hpp"
#include "Numerics/detail/CPU/blas_cpu.hpp"
//#include "Memory/buffer_managers.h"
//#include "Numerics/ma_operations.hpp"
//#include "multi/array.hpp"
//#include "multi/array_ref.hpp"

namespace ma
{
/** Contract 3D tensor over for use in exchange calculation.
 * E_x[w] ~ sum_{labn} T1[l,w,a,b,n] T2[l,w,b,a,n]
 * l is the batching parameter (like Q vectors in k-point code)
 *
 * \param[in]     nbatch Number of tensors to batch over.
 * \param[in]     nwalk  Number of walkers.
 * \param[in]     nocc   Number of electrons.
 * \param[in]     nchol  Number of Cholesky vectors.
 * \param[in]     alpha  Pointer to array of nbatch scaling factors.
 * \param[in]     Tab    Pointer to packed tensors {T1[l0,w0,a,b,n],T2[[l0,w0,a,b,n]...}.
 *                       Should be allocated on device.
 * \param[in,out] y      Pointer to accumulator array. Typically E[w].
 * \param[in]     incy   Stride for y.
*/
template<typename T, typename Q>
void batched_dot_wabn_wban(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<Q> const* alpha,
                           std::complex<Q> const* Tab,
                           std::complex<T>* y,
                           int incy,
			   cpu_backend)
{
  int nocc2nc = nocc * nocc * nchol;
  for (int batch = 0; batch < nbatch; ++batch)
  {
    for (int w = 0; w < nwalk; ++w)
    {
      std::complex<Q> E_(0.0);
      auto A_ = Tab + (2 * batch * nwalk + w) * nocc2nc;
      auto B_ = Tab + ((2 * batch + 1) * nwalk + w) * nocc2nc;
      for (int a = 0; a < nocc; ++a)
        for (int b = 0; b < nocc; ++b)
          E_ += ma::cpu::dot(nchol, A_ + (a * nocc + b) * nchol, 1, B_ + (b * nocc + a) * nchol, 1);
      y[w * incy] += static_cast<std::complex<T>>(alpha[batch] * E_);
    }
  }
}

template<typename T, typename Q>
void batched_dot_wanb_wbna(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<Q> const* alpha,
                           std::complex<Q> const* Tab,
                           std::complex<T>* y,
                           int incy,
			   cpu_backend)
{
  int nocc2nc = nocc * nocc * nchol;
  for (int batch = 0; batch < nbatch; ++batch)
  {
    for (int w = 0; w < nwalk; ++w)
    {
      std::complex<Q> E_(0.0);
      auto A_ = Tab + (2 * batch * nwalk + w) * nocc2nc;
      auto B_ = Tab + ((2 * batch + 1) * nwalk + w) * nocc2nc;
      for (int a = 0; a < nocc; ++a)
        for (int b = 0; b < nocc; ++b)
          E_ += ma::cpu::dot(nchol, A_ + a * nocc * nchol + b, nocc, B_ + b * nocc * nchol + a, nocc);
      y[w * incy] += static_cast<std::complex<T>>(alpha[batch] * E_);
    }
  }
}

template<typename T, typename Q>
void dot_wabn(int nwalk,
              int nocc,
              int nchol,
              std::complex<Q> alpha,
              std::complex<Q> const* Tab,
              std::complex<T>* y,
              int incy,
	      cpu_backend)
{
  int nocc2nc = nocc * nocc * nchol;
  for (int w = 0; w < nwalk; ++w)
  {
    std::complex<Q> E_(0.0);
    auto A_ = Tab + w * nocc2nc;
    for (int a = 0; a < nocc; ++a)
      for (int b = 0; b < nocc; ++b)
        E_ += ma::cpu::dot(nchol, A_ + (a * nocc + b) * nchol, 1, A_ + (b * nocc + a) * nchol, 1);
    y[w * incy] += static_cast<std::complex<T>>(alpha * E_);
  }
}

/** Construct generalized Fock matrix.
 *
 * \param[in]     nwalk  Number of walkers.
 * \param[in]     nmo    Number of basis functions.
 * \param[in]     nchol  Number of Cholesky vectors.
 * \param[in]     alpha  Scale factor.
 * \param[in]     Tab    Pointer to packed tensors {T1[w0,a,b,n],T2[[w0,a,b,n]...}.
 *                       Should be allocated on device.
 * \param[out]    F      Pointer to buffer for generalised Fock matrix.
*/
template<typename T, typename Q>
void dot_wpan_waqn_Fwpq(int nwalk,
                        int nmo,
                        int nchol,
                        std::complex<Q> alpha,
                        std::complex<Q> const* Tab,
                        std::complex<T>* F,
		        cpu_backend)
{
  std::complex<T> alp(static_cast<std::complex<T>>(alpha));
  for (int w = 0; w < nwalk; ++w)
  {
    auto A_ = Tab + w * nmo * nmo * nchol;
    for (int p = 0; p < nmo; ++p)
      for (int q = 0; q < nmo; ++q, ++F)
        for (int a = 0; a < nmo; ++a)
          *F += alp *
              static_cast<std::complex<T>>(
                    ma::cpu::dot(nchol, A_ + (p * nmo + a) * nchol, 1, A_ + (a * nmo + q) * nchol, 1));
  }
}

// T: {nwalk, nocc, nchol, nact}
template<typename T, typename Q>
void dot_wanb(int nwalk,
              int nocc,
              int nact,
              int nchol,
              std::complex<Q> alpha,
              std::complex<Q> const* Tab,
              std::complex<T>* y,
              int incy,
	      cpu_backend)
{
  int nocc2nc = nocc * nchol * nact;
  for (int w = 0; w < nwalk; ++w)
  {
    std::complex<Q> E_(0.0);
    auto A_ = Tab + w * nocc2nc;
    for (int a = 0; a < nocc; ++a)
      for (int b = 0; b < nocc; ++b)
        E_ += ma::cpu::dot(nchol, A_ + a * nact * nchol + b, nact, A_ + b * nact * nchol + a, nact);
    y[w * incy] += static_cast<std::complex<T>>(alpha * E_);
  }
}


/** .
 *
 * \param[in]     nters      Number terms batching over.
 * \param[in]     nwalk      Number of walkers.
 * \param[in]     nocc       Number of electrons.
 * \param[in]     nchol_max  Max number of Cholesky vectors per kpoint.
 * \param[in]     ncholQ     Number of Cholesky for Q vector.
 * \param[in]     kdiag      Pointer to array of number of k point pairs for each batch.
 * \param[in]     Tab        Pointer to buffer containing Tab, packed.
 * \param[out]    Kl         Pointer to buffer containing Kl.
 * \param[in]     ldKl       leading dimension of Kl 
 * \param[out]    Kr         Pointer to buffer containing Kr.
 * \param[in]     ldKr       leading dimension of Kr 
*/
template<typename T, typename Q>
void batched_Tab_to_Klr(int nterms,
                        int nwalk,
                        int nocc,
                        int nchol_max,
                        int ncholQ,
                        int* kdiag,
                        Q const* Tab,
                        T* Kl,
                        int ldkl, 
                        T* Kr,
                        int ldkr, 
			cpu_backend)
{
  for (int w = 0; w < nwalk; ++w)
  {
    for (int k = 0; k < nterms; k++)
    {
      int batch = kdiag[k];
      for (int a = 0; a < nocc; a++)
      {
        auto Tba_ = Tab + batch * nwalk * nocc * nocc * nchol_max + ((w * nocc + a) * nocc + a) * nchol_max;
        auto Kr_ = Kr + w * ldkr;
        for (int c = 0; c < ncholQ; ++c)
          Kr_[c] += static_cast<T>(Tba_[c]);
      }
    }
    for (int k = 0; k < nterms; k++)
    {
      int batch = kdiag[k];
      for (int a = 0; a < nocc; a++)
      {
        auto Tab_=Tab + (batch + 1) * nwalk * nocc * nocc * nchol_max + ((w * nocc + a) * nocc + a) * nchol_max;
        auto Kl_=Kl + w * ldkl; 
        for (int c = 0; c < ncholQ; ++c)
          Kl_[c] += static_cast<T>(Tab_[c]);
      }
    }
  }
}

// Not used.
template<typename T, typename Q>
void batched_Tanb_to_Klr(int nterms,
                         int nwalk,
                         int nocc,
                         int nchol_max,
                         int ncholQ,
                         int* kdiag,
                         Q const* Tab,
                         T* Kl,
                         int ldkl, 
                         T* Kr,
                         int ldkr, 
			 cpu_backend)
{
  for (int w = 0; w < nwalk; ++w)
  {
    for (int k = 0; k < nterms; k++)
    {
      int batch = kdiag[k];
      for (int a = 0; a < nocc; a++)
      {
        auto Tba_=Tab + batch * nwalk * nocc * nocc * nchol_max + ((w * nocc + a) * nocc) * nchol_max + a;
        auto Kr_=Kr + w * ldkr; 
        for (int c = 0; c < ncholQ; ++c)
          Kr_[c] += static_cast<T>(Tba_[c * nocc]);
      }
    }
    for (int k = 0; k < nterms; k++)
    {
      int batch = kdiag[k];
      for (int a = 0; a < nocc; a++)
      {
        auto Tab_=Tab + (batch + 1) * nwalk * nocc * nocc * nchol_max + ((w * nocc + a) * nocc) * nchol_max + a;
        auto Kl_=Kl + w * ldkl; 
        for (int c = 0; c < ncholQ; ++c)
          Kl_[c] += static_cast<T>(Tab_[c * nocc]);
      }
    }
  }
}

template<typename T, typename Q>
void Tab_to_Kl(int nwalk, int nocc, int nchol, Q const* Tab, T* Kl, cpu_backend)
{
  for (int w = 0; w < nwalk; ++w)
  {
    for (int a = 0; a < nocc; a++)
    {
      auto Tab_=Tab + ((w * nocc + a) * nocc + a) * nchol;
      auto Kl_=Kl + w * nchol;
      for (int c = 0; c < nchol; ++c)
        Kl_[c] += static_cast<T>(Tab_[c]);
    }
  }
}

template<typename T, typename Q>
void Tanb_to_Kl(int nwalk, int nocc, int nchol, int nact, Q const* Tab, T* Kl, int ldkl, cpu_backend)
{
  for (int w = 0; w < nwalk; ++w)
  {
    for (int a = 0; a < nocc; a++)
    {
      auto Tab_=Tab + ((w * nocc + a) * nchol) * nact + a;
      auto Kl_=Kl + w * ldkl;
      for (int c = 0; c < nchol; ++c)
        Kl_[c] += static_cast<T>(Tab_[c * nact]);
    }
  }
}

template<typename T, typename T1>
void vbias_from_v1(int nwalk,
                   int nkpts,
                   int nchol_max,
                   int* Qsym,
                   int* kminus,
                   int* ncholpQ,
                   int* ncholpQ0,
                   std::complex<T> const alpha,
                   std::complex<T1> const* v1,
                   std::complex<T>* vb,
		   cpu_backend)
{
  for (int Q = 0; Q < nkpts; Q++)
  {
    if (Qsym[Q] < 0)
      return;
    int Qm   = kminus[Q];
    int nc0  = ncholpQ0[Q];
    int nc   = ncholpQ[Q];
    int Qm_  = Qm;
    int ntot = nc * nwalk;
    if (Qsym[Q] > 0)
      Qm_ = nkpts + Qsym[Q] - 1;

    // v+
    auto vb_=vb + nc0 * nwalk;
    auto v1_=v1 + Q * nchol_max * nwalk;
    auto v2_=v1 + Qm_ * nchol_max * nwalk;
    // v+ = a*(v[Q]+v[-Q])
    for (int n = 0; n < ntot; ++n)
      vb_[n] += alpha * static_cast<std::complex<T>>(v1_[n]);
    for (int n = 0; n < ntot; ++n)
      vb_[n] += alpha * static_cast<std::complex<T>>(v2_[n]);
    // v-
    vb_ = (vb + (nc0 + nc) * nwalk);
    // v- = -a*i*(v[Q]-v[-Q])
    auto ialpha = alpha * std::complex<T>(0.0, 1.0);
    for (int n = 0; n < ntot; ++n)
      vb_[n] -= ialpha * static_cast<std::complex<T>>(v1_[n]);
    for (int n = 0; n < ntot; ++n)
      vb_[n] += ialpha * static_cast<std::complex<T>>(v2_[n]);
  }
}

// for n in [0,N), y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} opA(A)[n,m] * opB(B)[n,m]
template<typename T, typename Q>
void strided_batched_dot(char TA,
                 char TB,
                 int N,
                 int M,
                 std::complex<T> const alpha,
                 std::complex<Q> const* A,
                 int lda,
                 std::complex<Q> const* B,
                 int ldb,
                 std::complex<T> const beta,
                 std::complex<T>* y,
                 int incy,
		 cpu_backend)
{
  using Tc = typename std::complex<T>;  
  bool cA(TA == 'H' || TA == 'C');
  bool cB(TB == 'H' || TB == 'C');
  bool tA(TA == 'H' || TA == 'T');
  bool tB(TB == 'H' || TB == 'T');
  if (not tA && not tB)
  {
    for (int n = 0; n < N; n++)
    {
      std::complex<T> r(0.0, 0.0);
      auto an=A + n * lda;
      auto bn=B + n * ldb;
      if (cA && cB)
      {
        for (int m = 0; m < M; m++, an++, bn++)
          r += std::conj(static_cast<Tc>(*an)) * std::conj(static_cast<Tc>(*bn));
      }
      else if (cA && not cB)
      {
        for (int m = 0; m < M; m++, an++, bn++)
          r += std::conj(static_cast<Tc>(*an)) * static_cast<Tc>(*bn);
      }
      else if (not cA && cB)
      {
        for (int m = 0; m < M; m++, an++, bn++)
          r += static_cast<Tc>(*an) * std::conj(static_cast<Tc>(*bn));
      }
      else
      {
        for (int m = 0; m < M; m++, an++, bn++)
          r += static_cast<Tc>(*an) * static_cast<Tc>(*bn);
      }
      y[incy * n] = beta * y[incy * n] + alpha * r;
    }
  }
  else if (tA && not tB)
  {
    for (int n = 0; n < N; n++)
    {
      std::complex<T> r(0.0, 0.0);
      auto an=A + n;
      auto bn=B + n * ldb;
      if (cA && cB)
      {
        for (int m = 0; m < M; m++, an += lda, bn++)
          r += std::conj(static_cast<Tc>(*an)) * std::conj(static_cast<Tc>(*bn));
      }
      else if (cA && not cB)
      {
        for (int m = 0; m < M; m++, an += lda, bn++)
          r += std::conj(static_cast<Tc>(*an)) * static_cast<Tc>(*bn);
      }
      else if (not cA && cB)
      {
        for (int m = 0; m < M; m++, an += lda, bn++)
          r += static_cast<Tc>(*an) * std::conj(static_cast<Tc>(*bn));
      }
      else
      {
        for (int m = 0; m < M; m++, an += lda, bn++)
          r += static_cast<Tc>(*an) * static_cast<Tc>(*bn);
      }
      y[incy * n] = beta * y[incy * n] + alpha * r;
    }
  }
  else if (not tA && tB)
  {
    for (int n = 0; n < N; n++)
    {
      std::complex<T> r(0.0, 0.0);
      auto an = A + n * lda;
      auto bn = B + n;
      if (cA && cB)
      {
        for (int m = 0; m < M; m++, an++, bn += ldb)
          r += std::conj(static_cast<Tc>(*an)) * std::conj(static_cast<Tc>(*bn));
      }
      else if (cA && not cB)
      {
        for (int m = 0; m < M; m++, an++, bn += ldb)
          r += std::conj(static_cast<Tc>(*an)) * static_cast<Tc>(*bn);
      }
      else if (not cA && cB)
      {
        for (int m = 0; m < M; m++, an++, bn += ldb)
          r += static_cast<Tc>(*an) * std::conj(static_cast<Tc>(*bn));
      }
      else
      {
        for (int m = 0; m < M; m++, an++, bn += ldb)
          r += static_cast<Tc>(*an) * static_cast<Tc>(*bn);
      }
      y[incy * n] = beta * y[incy * n] + alpha * r;
    }
  }
  else
  { // special case, tA && tB
    for (int n = 0; n < N; n++)
      y[incy * n] *= beta;
    for (int m = 0; m < M; m++)
    {
      auto am = A + m * lda;
      auto bm = B + m * ldb;
      if (cA && cB)
      {
        for (int n = 0; n < N; n++, am++, bm++)
          y[incy * n] += alpha * std::conj(static_cast<Tc>(*am)) * std::conj(static_cast<Tc>(*bm));
      }
      else if (cA && not cB)
      {
        for (int n = 0; n < N; n++, am++, bm++)
          y[incy * n] += alpha * std::conj(static_cast<Tc>(*am)) * static_cast<Tc>(*bm);
      }
      else if (not cA && cB)
      {
        for (int n = 0; n < N; n++, am++, bm++)
          y[incy * n] += alpha * static_cast<Tc>(*am) * std::conj(static_cast<Tc>(*bm));
      }
      else
      {
        for (int n = 0; n < N; n++, am++, bm++)
          y[incy * n] += alpha * static_cast<Tc>(*am) * static_cast<Tc>(*bm);
      }
    }
  }
}

// for i in [0,nbatch) 
//   for n in [0,N), 
//     C[i,n] = beta * C[i,n] + alpha sum_m^{0,M} opA(A[i])[n,m] * opB(B[i])[n,m]
template<typename T, typename Q>
void strided_batched_dot(char TA,
                 char TB,
                 int nbatch,
                 int N,
                 int M,
                 std::complex<T> const alpha,
                 std::complex<Q> const* A,
                 int lda,
                 long Astride,
                 std::complex<Q> const* B,
                 int ldb,
                 long Bstride,
                 std::complex<T> const beta,
                 std::complex<T>* C,
                 int ldc,
                 long Cstride,
                 cpu_backend)
{
  for( long b=0; b<nbatch; b++) 
    strided_batched_dot(TA,TB,N,M,alpha,A+b*Astride,lda,B+b*Bstride,ldb,beta,C+b*Cstride,ldc,cpu_backend{});
}

// y[s] = y[s] + sum_ab A[s][a][b] * B[s][b][a]
// shapes of arrays are in packed form in n array
template<typename T, typename Q>
void batched_ab_ba(int* n,
                   std::complex<Q>* const* A,
                   int lda,
                   std::complex<Q>* const* B,
                   int ldb,
                   std::complex<T> alpha,
                   std::complex<T>** y,
                   int batchSize,
		   cpu_backend)
{
  // not optimal, blocked algorithm is faster
  for (int b = 0; b < batchSize; b++)
  {
    int n1(n[2 * b]);
    int n2(n[2 * b + 1]);
    std::complex<Q> const* A_(A[b]);
    std::complex<Q> const* B_(B[b]);
    std::complex<T> y_(0.0);
    for (int i = 0; i < n1; i++)
      for (int j = 0; j < n2; j++)
        y_ += static_cast<std::complex<T>>(A_[i * lda + j] * B_[j * ldb + i]);
    *(y[b]) += alpha * y_;
  }
}

template<typename T, typename Q>
void batched_diagonal_sum(int* n,
                          std::complex<Q>* const* A,
                          int lda,
                          std::complex<T> alpha,
                          std::complex<T>** y,
                          int batchSize,
			  cpu_backend)
{
  for (int b = 0; b < batchSize; b++)
  {
    std::complex<Q> const* A_(A[b]);
    std::complex<T> y_(0.0);
    int n_(n[b]);
    for (int i = 0; i < n_; i++)
      y_ += static_cast<std::complex<T>>(A_[i * lda + i]);
    *(y[b]) = alpha * y_;
  }
}

// C[u][w] = a * sum_n A[u][w][n] * B[u][n]
// should be called: Aijk_Bik_Cij
template<typename T>
void Auwn_Bun_Cuw(int nu, int nw, int na, T alpha, T const* A, T const* B, T* C, cpu_backend)
{
  for (int u = 0; u < nu; ++u, B += na)
    for (int iw = 0; iw < nw; ++iw, ++C, A += na)
      (*C) = alpha * ma::cpu::dot(na, A, 1, B, 1);
}

// C[u][w] = alpha * sum_i A[w][i][u] * B[i][u]
template<typename T, typename Q>
// Aijk_Bjk_Cki
void Awiu_Biu_Cuw(int nu, int nw, int ni, T alpha, T const* A, Q const* B, int ldb, T* C, int ldc, cpu_backend)
{
  for (int w = 0; w < nw; ++w)
  {
    for (int i = 0; i < ni; ++i)
    {
      auto Ci = C + w;
      auto Bi = B + i * ldb;
      for (int u = 0; u < nu; ++u, ++A, ++Bi, Ci += ldc)
        *Ci += alpha * (*A) * static_cast<T>(*Bi);
    }
  }
}

// C[i][k] = sum_i A[i][j][k] * B[k][j]
template<typename T, typename T1>
void Aijk_Bkj_Cik(int ni, int nj, int nk, T const* A, int lda, int stride, T1 const* B, int ldb, T* C, int ldc, cpu_backend)
{
  for (int i = 0; i < ni; ++i)
  {
    auto Ci = C + i * ldc;
    auto Ai_ = A + i * stride;
    for (int k = 0; k < nk; ++k, ++Ci)
    {
      auto Ak = Ai_ + k;
      auto Bk = B + k * ldb;
      for (int j = 0; j < nj; ++j, Ak += lda, ++Bk)
        *Ci += (*Ak) * static_cast<T>(*Bk);
    }
  }
}

// A[w][i][j] = B[i][w][j]
template<typename T, typename T1>
void viwj_vwij(int nw, int ni, int i0, int iN, T const* B, T1* A, cpu_backend)
{
  for (int w = 0; w < nw; ++w)
  {
    for (int i = i0; i < iN; ++i)
    {
      auto A_ = A + (w * ni + i) * ni;
      auto B_ = B + (i * nw + w) * ni;
      for (int j = 0; j < ni; ++j, ++A_, ++B_)
        *A_ = static_cast<T1>(*B_);
    }
  }
}

// Ckij = transA(Aij) * Bjk
//        conj(Aij)?
template<typename T>
void element_wise_Aij_Bjk_Ckij(char transA,
                               int ni,
                               int nj,
                               int nk,
                               T const* A,
                               int lda,
                               T const* B,
                               int ldb,
                               T* C,
                               int ldc,
                               int stride,
			       cpu_backend)
{
  if (transA == 'N')
  {
    for (int k = 0; k < nk; ++k)
    {
      for (int i = 0; i < ni; ++i)
      {
        auto A_=A + i * lda;
        auto B_=B + k;
        auto C_=C + k * stride + i * ldc;
        for (int j = 0; j < nj; ++j, ++C_, ++A_, B_ += ldb)
          (*C_) = (*A_) * (*B_);
      }
    }
  }
  else if (transA == 'C')
  {
    for (int k = 0; k < nk; ++k)
    {
      for (int i = 0; i < ni; ++i)
      {
        auto A_=A + i * lda;
        auto B_=B + k;
        auto C_=C + k * stride + i * ldc;
        for (int j = 0; j < nj; ++j, ++C_, ++A_, B_ += ldb)
          (*C_) = ma::conj(*A_) * (*B_);
      }
    }
  }
  else
    throw std::runtime_error(" Invalid parameter in element_wise_Aij_Bjk_Ckij. ");
}

template<typename T1, typename T2>
void element_wise_Aij_Bjk_Ckji(int ni,
                               int nj,
                               int nk,
                               T1 const* A,
                               int lda,
                               T2 const* B,
                               int ldb,
                               T2* C,
                               int ldc,
                               int stride,
			       cpu_backend)
{
  for (int k = 0; k < nk; k++)
  {
    for (int j = 0; j < nj; j++)
    {
      auto A_=A + j;
      auto B_=*(B + j * ldb + k);
      auto C_=C + k * stride + j * ldc;
      for (int i = 0; i < ni; i++, A_ += lda, ++C_)
        (*C_) = (*A_) * B_;
    }
  }
}

template<typename I1, typename T1, typename T2, typename T3>
void spVi_Bij_yj(int nj,
                 int nnz,
                 I1 const* index,
                 T1 const* values,
                 T2 const* B,
                 int ldb,
                 T3* y,
                 int incy,
		 cpu_backend)
{ 
  I1 ldb_ = static_cast<I1>(ldb);  
  for( size_t iz=0; iz<size_t(nnz); iz++) {
    T3 v_ = static_cast<T3>( values[iz] );
    auto B_ = B + index[iz] * ldb_ ;
    for(int j=0; j<nj; j++)
      y[ j * incy ] += static_cast<T3>(B_[j])* v_;
  }
}

} // namespace ma
#endif
