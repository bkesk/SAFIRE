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

#ifndef CSR_OPERATIONS_HPP
#define CSR_OPERATIONS_HPP

#include <tuple>
#include <complex>
#include "SparseMatrix/csr_matrix.hpp"

#include <type_traits> // enable_if

using std::complex;
using std::tuple;

namespace csr
{
inline double const& conj(double const& d) { return d; }
inline float const& conj(float const& f) { return f; }

// Dot product between 2 sparse vectors
template<class T, class integer, class VPtr, class JPtr>
inline T csrvv(char TA, char TB, std::tuple<integer, VPtr, JPtr> const& V1, std::tuple<integer, VPtr, JPtr> const& V2)
{
  RUNTIME_CHECK((TA == 'N' or TA == 'C'), "");
  RUNTIME_CHECK((TB == 'N' or TB == 'C'), "");
  using std::get;
  T res     = T(0);
  integer i = 0, j = 0;
  const integer n1 = get<0>(V1);
  const VPtr A1    = get<1>(V1);
  const JPtr indx1 = get<2>(V1);
  const integer n2 = get<0>(V2);
  const VPtr A2    = get<1>(V2);
  const JPtr indx2 = get<2>(V2);

  if (TA == 'N' && TB == 'N')
  {
    while (i < n1 && j < n2)
    {
      if (*(indx1 + i) < *(indx2 + j))
        ++i;
      else if (*(indx2 + j) < *(indx1 + i))
        ++j;
      else
      {
        res += *(A1 + i) * (*(A2 + j));
        ++i;
        ++j;
      }
    }
  }
  else if (TA == 'N' && TB == 'C')
  {
    while (i < n1 && j < n2)
    {
      if (*(indx1 + i) < *(indx2 + j))
        ++i;
      else if (*(indx2 + j) < *(indx1 + i))
        ++j;
      else
      {
        res += *(A1 + i) * conj(*(A2 + j));
        ++i;
        ++j;
      }
    }
  }
  else if (TA == 'C' && TB == 'N')
  {
    while (i < n1 && j < n2)
    {
      if (*(indx1 + i) < *(indx2 + j))
        ++i;
      else if (*(indx2 + j) < *(indx1 + i))
        ++j;
      else
      {
        res += conj(*(A1 + i)) * (*(A2 + j));
        ++i;
        ++j;
      }
    }
  }
  else if (TA == 'C' && TB == 'C')
  {
    while (i < n1 && j < n2)
    {
      if (*(indx1 + i) < *(indx2 + j))
        ++i;
      else if (*(indx2 + j) < *(indx1 + i))
        ++j;
      else
      {
        res += conj(*(A1 + i)) * conj(*(A2 + j));
        ++i;
        ++j;
      }
    }
  }
  return res;
}

} // namespace csr

#endif
