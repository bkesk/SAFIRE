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

#ifndef AFQMC_APPLY_EXPM_HPP
#define AFQMC_APPLY_EXPM_HPP

#include "Numerics/ma_operations.hpp"
#include "Utilities/FairDivide.hpp"

namespace sfqmc
{
namespace afqmc
{
namespace SlaterDeterminantOperations
{

/*
 * Calculate S = exp(im*V)*S using a Taylor expansion of exp(V)
 */
template<class MatA, class MatB, class MatC,
        typename = typename std::enable_if_t<std::decay_t<MatA>::dimensionality == 2>,
        typename = typename std::enable_if_t<std::decay_t<MatB>::dimensionality == 2>,
        typename = typename std::enable_if_t<std::decay_t<MatC>::dimensionality == 2>
        >
inline void apply_expM(const MatA& V, MatB&& S, MatC& T1, MatC& T2, int order = 6, char TA = 'N')
{
  RUNTIME_CHECK(V.size(0) == V.size(1), "");
  RUNTIME_CHECK(V.size(1) == S.size(0), "");
  RUNTIME_CHECK(S.size(0) == T1.size(0), "");
  RUNTIME_CHECK(S.size(1) == T1.size(1), "");
  RUNTIME_CHECK(S.size(0) == T2.size(0), "");
  RUNTIME_CHECK(S.size(1) == T2.size(1), "");

  using ma::H;
  using ma::T;
  using ComplexType = typename std::decay<MatB>::type::element;
  ComplexType zero(0.);
  auto pT1=std::addressof(T1);
  auto pT2=std::addressof(T2);

  ComplexType im(0.0, 1.0);
  if (TA == 'H' || TA == 'h')
    im = ComplexType(0.0, -1.0);

  // getting around issue in multi, fix later
  //T1 = S;
  T1.sliced(0, T1.size(0)) = S;
  for (int n = 1; n <= order; n++)
  {
    ComplexType fact = im * static_cast<ComplexType>(1.0 / static_cast<double>(n));
    if (TA == 'H' || TA == 'h')
      ma::product(fact, ma::H(V), *pT1, zero, *pT2);
    else if (TA == 'T' || TA == 't')
      ma::product(fact, ma::T(V), *pT1, zero, *pT2);
    else
      ma::product(fact, V, *pT1, zero, *pT2);
    ma::add(ComplexType(1.0), *pT2, ComplexType(1.0), S, S);
    std::swap(pT1, pT2);
  }
}

/*
 * Calculate S = exp(im*V)*S using a Taylor expansion of exp(V)
 * V, S, T1, T2 are expected to be in shared memory.  
 */
template<class MatA, class MatB, class MatC, class communicator>
inline void apply_expM(const MatA& V, MatB&& S, MatC& T1, MatC& T2, communicator& comm, int order = 6, char TA = 'N')
{
  RUNTIME_CHECK(V.size(0) == S.size(0), "");
  RUNTIME_CHECK(V.size(1) == S.size(0), "");
  RUNTIME_CHECK(S.size(0) == T1.size(0), "");
  RUNTIME_CHECK(S.size(1) == T1.size(1), "");
  RUNTIME_CHECK(S.size(0) == T2.size(0), "");
  RUNTIME_CHECK(S.size(1) == T2.size(1), "");

  using ComplexType = typename std::decay<MatB>::type::element;

  const ComplexType zero(0.);
  ComplexType im(0.0, 1.0);
  if (TA == 'H' || TA == 'h')
    im = ComplexType(0.0, -1.0);

  auto pT1=std::addressof(T1);
  auto pT2=std::addressof(T2);

  auto [M0, Mn] = FairDivideBoundary(comm.rank(), int(S.size(0)), comm.size());

  RUNTIME_CHECK(M0 <= Mn, "");
  RUNTIME_CHECK(M0 >= 0, "");

  T1.sliced(M0, Mn) = S.sliced(M0, Mn);
  comm.barrier();
  for (int n = 1; n <= order; n++)
  {
    const ComplexType fact = im * static_cast<ComplexType>(1.0 / static_cast<double>(n));
    if (TA == 'H' || TA == 'h')
      ma::product(fact, ma::H(V(V.extension(0), {M0, Mn})), *pT1, zero, (*pT2).sliced(M0, Mn));
    else if (TA == 'T' || TA == 't')
      ma::product(fact, ma::T(V(V.extension(0), {M0, Mn})), *pT1, zero, (*pT2).sliced(M0, Mn));
    else
      ma::product(fact, V.sliced(M0, Mn), *pT1, zero, (*pT2).sliced(M0, Mn));
    // overload += ???
/*
    for (int i = M0; i < Mn; i++)
      for (int j = 0, je = S.size(1); j < je; j++)
        S[i][j] += (*pT2)[i][j];
*/
    ma::axpy(ComplexType(1.0), pT2->flatted(), S.flatted());
    comm.barrier();
    std::swap(pT1, pT2);
  }
}

/*
 * Calculate S = exp(im*V)*S using a Taylor expansion of exp(V)
 */
template<class MatA, class MatB, class MatC,
        typename = typename std::enable_if_t<std::decay_t<MatA>::dimensionality == 3>,
        typename = typename std::enable_if_t<std::decay_t<MatB>::dimensionality == 3>,
        typename = typename std::enable_if_t<std::decay_t<MatC>::dimensionality == 3>,
        typename = void
        >
inline void apply_expM(const MatA& V, MatB&& S, MatC& T1, MatC& T2, int order = 6, char TA = 'N')
{
  RUNTIME_CHECK(V.size(0) == S.size(0), "");
  RUNTIME_CHECK(V.size(0) == T1.size(0), "");
  RUNTIME_CHECK(V.size(0) == T2.size(0), "");
  RUNTIME_CHECK(V.size(1) == V.size(2), "");
  RUNTIME_CHECK(V.size(2) == S.size(1), "");
  RUNTIME_CHECK(S.size(1) == T1.size(1), "");
  RUNTIME_CHECK(S.size(2) == T1.size(2), "");
  RUNTIME_CHECK(S.size(1) == T2.size(1), "");
  RUNTIME_CHECK(S.size(2) == T2.size(2), "");
  // for now limit to continuous
  RUNTIME_CHECK(S.stride(0) == S.size(1) * S.size(2), "");
  RUNTIME_CHECK(T1.stride(0) == T1.size(1) * T1.size(2), "");
  RUNTIME_CHECK(T2.stride(0) == T2.size(1) * T2.size(2), "");
  RUNTIME_CHECK(S.stride(1) == S.size(2), "");
  RUNTIME_CHECK(T1.stride(1) == T1.size(2), "");
  RUNTIME_CHECK(T2.stride(1) == T2.size(2), "");
  RUNTIME_CHECK(S.stride(2) == 1, "");
  RUNTIME_CHECK(T1.stride(2) == 1, "");
  RUNTIME_CHECK(T2.stride(2) == 1, "");

  using ComplexType = typename std::decay<MatB>::type::element;
  ComplexType zero(0.);
  ComplexType im(0.0, 1.0);
  if (TA == 'H' || TA == 'h')
    im = ComplexType(0.0, -1.0);
  auto pT1=std::addressof(T1);
  auto pT2=std::addressof(T2);

  // getting around issue in multi, fix later
  //T1 = S;
  using std::copy_n;
  copy_n(S.origin(), S.num_elements(), T1.origin());
  for (int n = 1; n <= order; n++)
  {
    ComplexType fact = im * static_cast<ComplexType>(1.0 / static_cast<double>(n));
    if (TA == 'H' || TA == 'h')
      ma::productStridedBatched(fact, ma::H(V), *pT1, zero, *pT2);
    else if (TA == 'T' || TA == 't')
      ma::productStridedBatched(fact, ma::T(V), *pT1, zero, *pT2);
    else
      ma::productStridedBatched(fact, V, *pT1, zero, *pT2);
    //ma::add(ComplexType(1.0),*pT2,ComplexType(1.0),S,S);
    ma::axpy(ComplexType(1.0), pT2->flatted().flatted(), S.flatted().flatted());
    std::swap(pT1, pT2);
  }
}

/*
 * Calculate S = exp(im*V)*S using a Taylor expansion of exp(V)
 * Version for non_collinear calculations, where there are 2 S matrices per V in the batch.
 */
template<class MatA, class MatB, class MatC,
        typename = typename std::enable_if_t<std::decay_t<MatA>::dimensionality == 3>,
        typename = typename std::enable_if_t<std::decay_t<MatB>::dimensionality == 3>,
        typename = typename std::enable_if_t<std::decay_t<MatC>::dimensionality == 3>
        >
inline void apply_expM_noncollinear(const MatA& Vup, const MatA& Vdn, MatB&& S, MatC& T1, MatC& T2, int order = 6, char TA = 'N')
{
  RUNTIME_CHECK(Vup.size(0) * 2 == S.size(0), "");
  RUNTIME_CHECK(Vup.size(0) * 2 == T1.size(0), "");
  RUNTIME_CHECK(Vup.size(0) * 2 == T2.size(0), "");
  RUNTIME_CHECK(Vup.size(1) == Vup.size(2), "");
  RUNTIME_CHECK(Vup.size(2) == S.size(1), "");
  RUNTIME_CHECK(Vdn.size(0) == Vup.size(0), "");
  RUNTIME_CHECK(Vdn.size(1) == Vup.size(1), "");
  RUNTIME_CHECK(Vdn.size(2) == Vup.size(2), "");
  RUNTIME_CHECK(Vdn.stride(1) == Vup.stride(1), "");
  RUNTIME_CHECK(S.size(1) == T1.size(1), "");
  RUNTIME_CHECK(S.size(2) == T1.size(2), "");
  RUNTIME_CHECK(S.size(1) == T2.size(1), "");
  RUNTIME_CHECK(S.size(2) == T2.size(2), "");
  // for now limit to continuous
  RUNTIME_CHECK(S.stride(0) == S.size(1) * S.size(2), "");
  RUNTIME_CHECK(T1.stride(0) == T1.size(1) * T1.size(2), "");
  RUNTIME_CHECK(T2.stride(0) == T2.size(1) * T2.size(2), "");
  RUNTIME_CHECK(S.stride(1) == S.size(2), "");
  RUNTIME_CHECK(T1.stride(1) == T1.size(2), "");
  RUNTIME_CHECK(T2.stride(1) == T2.size(2), "");
  RUNTIME_CHECK(S.stride(2) == 1, "");
  RUNTIME_CHECK(T1.stride(2) == 1, "");
  RUNTIME_CHECK(T2.stride(2) == 1, "");

  using ComplexType = typename std::decay<MatB>::type::element;
  ComplexType zero(0.);
  ComplexType im(0.0, 1.0);
  if (TA == 'H' || TA == 'h')
    im = ComplexType(0.0, -1.0);
  auto pT1=std::addressof(T1);
  auto pT2=std::addressof(T2);

  using pointerC = typename std::decay<MatC>::type::element_ptr;

  int nbatch = S.size(0);
  int ldv    = Vup.stride(1);
  int M      = T2.size(2);
  int N      = T2.size(1);
  int K      = T1.size(1);

  std::vector<decltype(ma::pointer_dispatch(Vup[0].origin()))> Vi;
  std::vector<pointerC> T1i;
  std::vector<pointerC> T2i;
  Vi.reserve(2 * Vup.size(0));
  T1i.reserve(T1.size(0));
  T2i.reserve(T2.size(0));
  for (int i = 0; i < Vup.size(0); i++)
  {
    Vi.emplace_back(ma::pointer_dispatch(Vup[i].origin()));
    Vi.emplace_back(ma::pointer_dispatch(Vdn[i].origin()));
  }
  for (int i = 0; i < T1.size(0); i++)
    T1i.emplace_back(ma::pointer_dispatch(T1[i].origin()));
  for (int i = 0; i < T2.size(0); i++)
    T2i.emplace_back(ma::pointer_dispatch(T2[i].origin()));

  auto pT1i=std::addressof(T1i);
  auto pT2i=std::addressof(T2i);

  // getting around issue in multi, fix later
  //T1 = S;
  using std::copy_n;
  copy_n(S.origin(), S.num_elements(), T1.origin());
  for (int n = 1; n <= order; n++)
  {
    ComplexType fact = im * static_cast<ComplexType>(1.0 / static_cast<double>(n));
    using ma::gemmBatched;
    // careful with fortran ordering
    gemmBatched('N', TA, M, N, K, fact, pT1i->data(), (*pT1).stride(1), Vi.data(), ldv, zero, pT2i->data(),
                (*pT2).stride(1), nbatch);
    ma::axpy(ComplexType(1.0), pT2->flatted().flatted(), S.flatted().flatted());
    std::swap(pT1, pT2);
    std::swap(pT1i, pT2i);
  }
}

/*
 * Calculate S = exp(im*V)*S using a Taylor expansion of exp(V)
 * V is given as a block-diagonal sparse matrix. Each block is the V of a term in the batch.
 */
template<class MatA, class MatB, class MatC,
        typename = typename std::enable_if_t<std::decay_t<MatA>::dimensionality == -2>,
        typename = typename std::enable_if_t<std::decay_t<MatB>::dimensionality == 3>,
        typename = typename std::enable_if_t<std::decay_t<MatC>::dimensionality == 3>,
        typename = void,
        typename = void
        >
inline void apply_expM(const MatA& V, MatB&& S, MatC& T1, MatC& T2, int order = 6, char TA = 'N')
{
  RUNTIME_CHECK(S.size(0) == T1.size(0), "");
  RUNTIME_CHECK(S.size(0) == T2.size(0), "");
  RUNTIME_CHECK(S.size(1) == T1.size(1), "");
  RUNTIME_CHECK(S.size(2) == T1.size(2), "");
  RUNTIME_CHECK(S.size(1) == T2.size(1), "");
  RUNTIME_CHECK(S.size(2) == T2.size(2), "");
  // check special constructino of V
  // V[ nbatch*M, nbatch*M ]
  RUNTIME_CHECK(V.size(0) == V.size(1), "");
  RUNTIME_CHECK(V.size(0) == S.size(0)*S.size(1), "");
  // for now limit to continuous
  RUNTIME_CHECK(S.stride(0) == S.size(1) * S.size(2), "");
  RUNTIME_CHECK(T1.stride(0) == T1.size(1) * T1.size(2), "");
  RUNTIME_CHECK(T2.stride(0) == T2.size(1) * T2.size(2), "");
  RUNTIME_CHECK(S.stride(1) == S.size(2), "");
  RUNTIME_CHECK(T1.stride(1) == T1.size(2), "");
  RUNTIME_CHECK(T2.stride(1) == T2.size(2), "");
  RUNTIME_CHECK(S.stride(2) == 1, "");
  RUNTIME_CHECK(T1.stride(2) == 1, "");
  RUNTIME_CHECK(T2.stride(2) == 1, "");

  using ComplexType = typename std::decay<MatB>::type::element;
  ComplexType zero(0.);
  ComplexType im(0.0, 1.0);
  if (TA == 'H' || TA == 'h')
    im = ComplexType(0.0, -1.0);

  // generate new 'view' of T1/T2, with dimensions [size(0)*size(1), size(2)]
  auto&& T1_= T1.flatted();
  auto&& T2_= T2.flatted();

  auto pT1=std::addressof(T1_);
  auto pT2=std::addressof(T2_);

  // getting around issue in multi, fix later
  //T1 = S;
  using std::copy_n;
  copy_n(S.origin(), S.num_elements(), T1.origin());
  for (int n = 1; n <= order; n++)
  {
    ComplexType fact = im * static_cast<ComplexType>(1.0 / static_cast<double>(n));
    if (TA == 'H' || TA == 'h')
      ma::product(fact, ma::H(V), *pT1, zero, *pT2);
    else if (TA == 'T' || TA == 't')
      ma::product(fact, ma::T(V), *pT1, zero, *pT2);
    else
      ma::product(fact, V, *pT1, zero, *pT2);
    //ma::add(ComplexType(1.0),*pT2,ComplexType(1.0),S,S);
    ma::axpy(ComplexType(1.0), pT2->flatted(), S.flatted().flatted());
    std::swap(pT1, pT2);
  }
}

} // namespace SlaterDeterminantOperations

} // namespace afqmc

} // namespace sfqmc

#endif
