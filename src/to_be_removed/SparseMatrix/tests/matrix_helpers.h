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

#ifndef SPARSE_MATRIX_HELPER_H
#define SPARSE_MATRIX_HELPER_H

#include <stdio.h>
#include <string>
#include <complex>
#include <type_traits>

using std::complex;
using std::string;

namespace sfqmc
{
template<typename T>
void myREQUIRE(T const& a, T const& b)
{
  REQUIRE(a == Approx(b));
}

template<typename T>
void myREQUIRE(std::complex<T> const& a, std::complex<T> const& b)
{
  REQUIRE(a.real() == Approx(b.real()));
  REQUIRE(a.imag() == Approx(b.imag()));
}

template<class M1,
         class M2,
         typename = typename std::enable_if<(M1::dimensionality == 1)>::type,
         typename = typename std::enable_if<(M2::dimensionality == 1)>::type>
void verify_approx(M1 const& A, M2 const& B)
{
  // casting in case operator[] returns a fancy reference
  using element1 = typename std::decay<M1>::type::element;
  using element2 = typename std::decay<M2>::type::element;
  REQUIRE(A.size(0) == B.size(0));
  for (int i = 0; i < A.size(0); i++)
    myREQUIRE(element1(A[i]), element2(B[i]));
}

template<class M1,
         class M2,
         typename = typename std::enable_if<(M1::dimensionality > 1)>::type,
         typename = typename std::enable_if<(M2::dimensionality > 1)>::type,
         typename = void>
void verify_approx(M1 const& A, M2 const& B)
{
  REQUIRE(A.size(0) == B.size(0));
  for (int i = 0; i < A.size(0); i++)
    verify_approx(A[i], B[i]);
}

} // namespace sfqmc

#endif
