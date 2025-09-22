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

#ifndef AFQMC_HSPOTENTIAL_HELPERS_H
#define AFQMC_HSPOTENTIAL_HELPERS_H

#include <cstdlib>
#include <algorithm>
#include <complex>
#include <iostream>
#include <vector>
#include <numeric>

#include "config.h"
#include "Utilities/AppAbort.hpp"

#include <boost/type_traits/is_complex.hpp>

#include "AFQMC/config.h"
#include "Utilities/FairDivide.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "SparseMatrix/csr_matrix.hpp"
#include "Numerics/ma_operations.hpp"
#include "Numerics/csr_blas.hpp"
#include "Numerics/detail/utilities.hpp"

/********************************************************************
*  You get 2 potentials per Cholesky std::vector
*
*    vn(+-)_{i,k} = 0.5*( L^n_{i,k} +- ma::conj(L^n_{k,i}) )
********************************************************************/

namespace sfqmc
{
namespace afqmc
{
namespace HamHelper
{

namespace
{

template<class csrVec>
void count_over_cholvec(double cut,
                        std::vector<std::size_t>& count,
                        int c0,
                        int c1,
                        csrVec const& Lik,
                        csrVec const& Lki)
{
  using VType = typename csrVec::element_type;
  RUNTIME_CHECK(c1 >= c0, "");
  if (c0 == c1)
    return;
  auto cik     = std::lower_bound(raw_pointer_cast(Lik.non_zero_indices2_data()),
                              raw_pointer_cast((Lik.non_zero_indices2_data() + Lik.num_non_zero_elements())), c0);
  auto cik_end = std::lower_bound(cik, raw_pointer_cast((Lik.non_zero_indices2_data() + Lik.num_non_zero_elements())), c1);
  auto vik     = raw_pointer_cast(Lik.non_zero_values_data()) + std::distance(raw_pointer_cast(Lik.non_zero_indices2_data()), cik);

  auto cki     = std::lower_bound(raw_pointer_cast(Lki.non_zero_indices2_data()),
                              raw_pointer_cast((Lki.non_zero_indices2_data() + Lki.num_non_zero_elements())), c0);
  auto cki_end = std::lower_bound(cki, raw_pointer_cast((Lki.non_zero_indices2_data() + Lki.num_non_zero_elements())), c1);
  auto vki     = raw_pointer_cast(Lki.non_zero_values_data()) + std::distance(raw_pointer_cast(Lki.non_zero_indices2_data()), cki);

  // ignoring factor of 0.5 to keep it consistent with the old code for now
  using ma::conj;
  using std::abs;
  using std::size_t;
  while (cik != cik_end && cki != cki_end)
  {
    if (*cik == *cki)
    { // both Lik and Lki have components on Chol Vec *cik==*cki
      if (abs(*vik + ma::conj(*vki)) > cut)
        count[2 * (*cik)] += size_t(2); // Lik + Lki* and Lki + Lik*
      if (abs(*vik - ma::conj(*vki)) > cut)
        count[2 * (*cik) + 1] += size_t(2); // Lik - Lki* and Lki - Lik*
      ++cik;
      ++vik;
      ++cki;
      ++vki;
    }
    else if (*cik < *cki)
    { // not on the same chol vector, only operate on the smallest
      if (abs(*vik) > cut)
      {
        count[2 * (*cik)] += size_t(2); // Lik + 0
	if constexpr (boost::is_complex<VType>::value)
            count[2 * (*cik) + 1] += size_t(2); // Lik - 0
      }
      ++cik;
      ++vik;
    }
    else
    {
      if (abs(*vki) > cut)
      {
        count[2 * (*cki)] += size_t(2); // Lki + 0
	if constexpr (boost::is_complex<VType>::value)
          count[2 * (*cki) + 1] += size_t(2); // Lki - 0
      }
      ++cki;
      ++vki;
    }
  }
  // either cik or cki are at end, check if any terms are missing
  while (cik != cik_end)
  {
    if (abs(*vik) > cut)
    {
      count[2 * (*cik)] += size_t(2); // Lik + 0
      if constexpr (boost::is_complex<VType>::value)
        count[2 * (*cik) + 1] += size_t(2); // Lik - 0
    }
    ++cik;
    ++vik;
  }
  while (cki != cki_end)
  {
    if (abs(*vki) > cut)
    {
      count[2 * (*cki)] += size_t(2); // Lki + 0
      if constexpr (boost::is_complex<VType>::value)
        count[2 * (*cki) + 1] += size_t(2); // Lki - 0
    }
    ++cki;
    ++vki;
  }
}

template<class csrVec>
void count_over_cholvec(double cut,
                        std::vector<std::size_t>& count,
                        int c0,
                        int c1,
                        csrVec const& Lii)
{
  using VType = typename csrVec::element_type;
  RUNTIME_CHECK(c1 >= c0, "");
  if (c0 == c1)
    return;
  auto ci     = std::lower_bound(raw_pointer_cast(Lii.non_zero_indices2_data()),
                             raw_pointer_cast((Lii.non_zero_indices2_data() + Lii.num_non_zero_elements())), c0);
  auto ci_end = std::lower_bound(ci, raw_pointer_cast((Lii.non_zero_indices2_data() + Lii.num_non_zero_elements())), c1);
  auto vi     = raw_pointer_cast(Lii.non_zero_values_data()) + std::distance(raw_pointer_cast(Lii.non_zero_indices2_data()), ci);

  // ignoring factor of 0.5 to keep it consistent with the old code for now
  using ma::conj;
  using std::abs;
  using std::size_t;
  while (ci != ci_end)
  {
    if (abs(*vi + ma::conj(*vi)) > cut)
      ++count[2 * (*ci)]; // Lii + Lii*
    if constexpr (boost::is_complex<VType>::value)
      if (abs(*vi - ma::conj(*vi)) > cut)
        ++count[2 * (*ci) + 1]; // Lii - Lii*
    ++ci;
    ++vi;
  }
}

// In this case, c0/c1 refer to the ranges in the expanded list of CVs, e.g. 2*n/2*n+1
template<class csrVec>
void count_nnz(double cut,
               std::size_t& nik,
               std::size_t& nki,
               int c0,
               int c1,
               csrVec const& Lik,
               csrVec const& Lki)
{
  using VType = typename csrVec::element_type;
  using ma::conj;
  using std::abs;
  using std::size_t;
  RUNTIME_CHECK(c1 >= c0, "");
  nik = nki = size_t(0);
  if (c0 == c1)
    return;
  auto cik = std::lower_bound(raw_pointer_cast(Lik.non_zero_indices2_data()),
                              raw_pointer_cast((Lik.non_zero_indices2_data() + Lik.num_non_zero_elements())), c0 / 2);
  auto cik_end =
      std::lower_bound(cik, raw_pointer_cast((Lik.non_zero_indices2_data() + Lik.num_non_zero_elements())), (c1 + 1) / 2);
  auto vik = raw_pointer_cast(Lik.non_zero_values_data()) + std::distance(raw_pointer_cast(Lik.non_zero_indices2_data()), cik);

  auto cki = std::lower_bound(raw_pointer_cast(Lki.non_zero_indices2_data()),
                              raw_pointer_cast((Lki.non_zero_indices2_data() + Lki.num_non_zero_elements())), c0 / 2);
  auto cki_end =
      std::lower_bound(cki, raw_pointer_cast((Lki.non_zero_indices2_data() + Lki.num_non_zero_elements())), (c1 + 1) / 2);
  auto vki = raw_pointer_cast(Lki.non_zero_values_data()) + std::distance(raw_pointer_cast(Lki.non_zero_indices2_data()), cki);

  // ignoring factor of 0.5 to keep it consistent with the old code for now
  while (cik != cik_end && cki != cki_end)
  {
    if (*cik == *cki)
    {                                         // both Lik and Lki have components on Chol Vec *cik==*cki
      if (abs(*vik + ma::conj(*vki)) > cut && // Lik + Lki* and Lki + Lik*
          2 * (*cik) >= c0 && 2 * (*cik) < c1)
      {
        nik++;
        nki++;
      }
      if constexpr (boost::is_complex<VType>::value)
        if (abs(*vik - ma::conj(*vki)) > cut && // Lik - Lki* and Lki - Lik*
            2 * (*cik) + 1 >= c0 && 2 * (*cik) + 1 < c1)
        {
          nik++;
          nki++;
        }
      ++cik;
      ++vik;
      ++cki;
      ++vki;
    }
    else if (*cik < *cki)
    { // not on the same chol vector, only operate on the smallest
      if constexpr (boost::is_complex<VType>::value) {
        if (abs(*vik) > cut)
        {
          if (2 * (*cik) >= c0 && 2 * (*cik) < c1)
          {
            ++nik;
            ++nki;
          }
          if (2 * (*cik) + 1 >= c0 && 2 * (*cik) + 1 < c1)
          {
            ++nik;
            ++nki;
          }
        } 
      } else {
        if (abs(*vik) > cut && 2 * (*cik) >= c0 && 2 * (*cik) < c1)
        {
          ++nik;
         ++nki;
        } 
      }
      ++cik;
      ++vik;
    }
    else
    {
      if constexpr (boost::is_complex<VType>::value) {
        if (abs(*vki) > cut)
        {
          if (2 * (*cki) >= c0 && 2 * (*cki) < c1)
          {
            ++nik;
            ++nki;
          }
          if (2 * (*cki) + 1 >= c0 && 2 * (*cki) + 1 < c1)
          {
            ++nik;
            ++nki;
          }
        }
      } else {
        if (abs(*vki) > cut && 2 * (*cki) >= c0 && 2 * (*cki) < c1)
        {
          ++nik;
          ++nki;
        }
      }
      ++cki;
      ++vki;
    }
  }
  while (cik != cik_end)
  {
    if constexpr (boost::is_complex<VType>::value) {
      if (abs(*vik) > cut)
      {
        if (2 * (*cik) >= c0 && 2 * (*cik) < c1)
        {
          ++nik;
          ++nki;
        }
        if (2 * (*cik) + 1 >= c0 && 2 * (*cik) + 1 < c1)
        {
          ++nik;
          ++nki;
        }
      }
    } else {
      if (abs(*vik) > cut && 2 * (*cik) >= c0 && 2 * (*cik) < c1)
      {
        ++nik;
        ++nki;
      }
    }
    ++cik;
    ++vik;
  }
  while (cki != cki_end)
  {
    if constexpr (boost::is_complex<VType>::value) {
      if (abs(*vki) > cut)
      {
        if (2 * (*cki) >= c0 && 2 * (*cki) < c1)
        {
          ++nik;
          ++nki;
        }
        if (2 * (*cki) + 1 >= c0 && 2 * (*cki) + 1 < c1)
        {
          ++nik;
          ++nki;
        }
      }
    } else {
      if (abs(*vki) > cut && 2 * (*cki) >= c0 && 2 * (*cki) < c1)
      {
        ++nik;
        ++nki;
      }
    }
    ++cki;
    ++vki;
  }
}

template<class csrVec>
void count_nnz(double cut, size_t& ni, int c0, int c1, csrVec const& Lii)
{
  using VType = typename csrVec::element_type;
  using ma::conj;
  using std::abs;
  using std::size_t;
  RUNTIME_CHECK(c1 >= c0, "");
  ni = size_t(0);
  if (c0 == c1)
    return;
  auto ci = std::lower_bound(raw_pointer_cast(Lii.non_zero_indices2_data()),
                             raw_pointer_cast((Lii.non_zero_indices2_data() + Lii.num_non_zero_elements())), c0 / 2);
  auto ci_end =
      std::lower_bound(ci, raw_pointer_cast((Lii.non_zero_indices2_data() + Lii.num_non_zero_elements())), (c1 + 1) / 2);
  auto vi = raw_pointer_cast(Lii.non_zero_values_data()) + std::distance(raw_pointer_cast(Lii.non_zero_indices2_data()), ci);

  // ignoring factor of 0.5 to keep it consistent with the old code for now
  while (ci != ci_end)
  {
    if (abs(*vi + ma::conj(*vi)) > cut && 2 * (*ci) >= c0 && 2 * (*ci) < c1)
      ++ni; // Lii + Lii*
    if constexpr (boost::is_complex<VType>::value) 
      if (abs(*vi - ma::conj(*vi)) > cut && 2 * (*ci) + 1 >= c0 && 2 * (*ci) + 1 < c1)
        ++ni; // Lii - Lii*
    ++ci;
    ++vi;
  }
}

template<class csrMat>
void add_to_vn(csrMat& vn,
               std::vector<int> const& map_,
               double cut,
               int ik,
               int c0,
               int c1,
               typename csrMat::reference const& Lii)
{
  using VType = typename csrMat::element_type;
  using ma::conj;
  using std::abs;
  using std::size_t;
  RUNTIME_CHECK(c1 >= c0, "");
  if (c0 == c1)
    return;
  auto ci = std::lower_bound(raw_pointer_cast(Lii.non_zero_indices2_data()),
                             raw_pointer_cast((Lii.non_zero_indices2_data() + Lii.num_non_zero_elements())), c0 / 2);
  auto ci_end =
      std::lower_bound(ci, raw_pointer_cast((Lii.non_zero_indices2_data() + Lii.num_non_zero_elements())), (c1 + 1) / 2);
  auto vi = raw_pointer_cast(Lii.non_zero_values_data()) + std::distance(raw_pointer_cast(Lii.non_zero_indices2_data()), ci);

  int c_origin = vn.global_origin()[1];
  VType half(0.5);
  typename to_complex<VType>::type im(0.0, 1.0);
  while (ci != ci_end)
  {
    if (abs(*vi + ma::conj(*vi)) > cut && 2 * (*ci) >= c0 && 2 * (*ci) < c1)
    {
      RUNTIME_CHECK(map_[2 * (*ci)] >= 0, "");
      RUNTIME_CHECK(map_[2 * (*ci)] - c_origin < vn.size(1), "");
      vn.emplace_back({ik, (map_[2 * (*ci)] - c_origin)},
                      static_cast<VType>(half * (*vi + ma::conj(*vi)))); // Lii + Lii*
    }
    if constexpr (boost::is_complex<VType>::value) { 
      if (abs(*vi - ma::conj(*vi)) > cut && 2 * (*ci) + 1 >= c0 && 2 * (*ci) + 1 < c1)
      {
        RUNTIME_CHECK(map_[2 * (*ci) + 1] >= 0, "");
        RUNTIME_CHECK(map_[2 * (*ci) + 1] - c_origin < vn.size(1), "");
        vn.emplace_back({ik, (map_[2 * (*ci) + 1] - c_origin)},
                      static_cast<VType>(half * im * (*vi - ma::conj(*vi)))); // Lii - Lii*
      }
    }
    ++ci;
    ++vi;
  }
}

template<class csrMat>
void add_to_vn(csrMat& vn,
               std::vector<int> const& map_,
               double cut,
               int ik,
               int ki,
               int c0,
               int c1,
               typename csrMat::reference const& Lik,
               typename csrMat::reference const& Lki)
{
  using VType = typename csrMat::element_type;
  using ma::conj;
  using std::abs;
  using std::size_t;
  RUNTIME_CHECK(c1 >= c0, "");
  if (c0 == c1)
    return;
  auto cik = std::lower_bound(raw_pointer_cast(Lik.non_zero_indices2_data()),
                              raw_pointer_cast((Lik.non_zero_indices2_data() + Lik.num_non_zero_elements())), c0 / 2);
  auto cik_end =
      std::lower_bound(cik, raw_pointer_cast((Lik.non_zero_indices2_data() + Lik.num_non_zero_elements())), (c1 + 1) / 2);
  auto vik = raw_pointer_cast(Lik.non_zero_values_data()) + std::distance(raw_pointer_cast(Lik.non_zero_indices2_data()), cik);

  auto cki = std::lower_bound(raw_pointer_cast(Lki.non_zero_indices2_data()),
                              raw_pointer_cast((Lki.non_zero_indices2_data() + Lki.num_non_zero_elements())), c0 / 2);
  auto cki_end =
      std::lower_bound(cki, raw_pointer_cast((Lki.non_zero_indices2_data() + Lki.num_non_zero_elements())), (c1 + 1) / 2);
  auto vki = raw_pointer_cast(Lki.non_zero_values_data()) + std::distance(raw_pointer_cast(Lki.non_zero_indices2_data()), cki);

  typename to_complex<VType>::type im(0.0, 1.0);
  VType half(0.5);
  int c_origin = vn.global_origin()[1];
  while (cik != cik_end && cki != cki_end)
  {
    if (*cik == *cki)
    { // both Lik and Lki have components on Chol Vec *cik==*cki
      if (abs(*vik + ma::conj(*vki)) > cut)
      { // Lik + Lki* and Lki + Lik*
        if (2 * (*cik) >= c0 && 2 * (*cik) < c1)
        {
          RUNTIME_CHECK(map_[2 * (*cik)] >= 0, "");
          RUNTIME_CHECK(map_[2 * (*cik)] - c_origin < vn.size(1), "");
          vn.emplace_back({ik, (map_[2 * (*cik)] - c_origin)},
                          static_cast<VType>(half * (*vik + ma::conj(*vki)))); // Lik + Lki*
          vn.emplace_back({ki, (map_[2 * (*cki)] - c_origin)},
                          static_cast<VType>(half * (*vki + ma::conj(*vik)))); // Lki + Lik*
        }
      }
      if constexpr (boost::is_complex<VType>::value) { 
        if (abs(*vik - ma::conj(*vki)) > cut)
        { // Lik - Lki* and Lki - Lik*
          if (2 * (*cik) + 1 >= c0 && 2 * (*cik) + 1 < c1)
          {
            RUNTIME_CHECK(map_[2 * (*cik) + 1] >= 0, "");
            RUNTIME_CHECK(map_[2 * (*cik) + 1] - c_origin < vn.size(1), "");
            vn.emplace_back({ik, (map_[2 * (*cik) + 1] - c_origin)},
                          static_cast<VType>(half * im * (*vik - ma::conj(*vki)))); // Lik - Lki*
            vn.emplace_back({ki, (map_[2 * (*cki) + 1] - c_origin)},
                          static_cast<VType>(half * im * (*vki - ma::conj(*vik)))); // Lki - Lik*
          }
        }
      }
      ++cik;
      ++vik;
      ++cki;
      ++vki;
    }
    else if (*cik < *cki)
    { // not on the same chol vector, only operate on the smallest
      if (abs(*vik) > cut)
      {
        if (2 * (*cik) >= c0 && 2 * (*cik) < c1)
        {
          RUNTIME_CHECK(map_[2 * (*cik)] >= 0, "");
          RUNTIME_CHECK(map_[2 * (*cik)] - c_origin < vn.size(1), "");
          vn.emplace_back({ik, (map_[2 * (*cik)] - c_origin)},
                          static_cast<VType>(half * (*vik))); // Lik + 0
          vn.emplace_back({ki, (map_[2 * (*cik)] - c_origin)},
                          static_cast<VType>(half * ma::conj(*vik))); // Lik + 0
        }
        if constexpr (boost::is_complex<VType>::value) { 
          if (2 * (*cik) + 1 >= c0 && 2 * (*cik) + 1 < c1)
          {
            RUNTIME_CHECK(map_[2 * (*cik) + 1] >= 0, "");
            RUNTIME_CHECK(map_[2 * (*cik) + 1] - c_origin < vn.size(1), "");
            vn.emplace_back({ik, (map_[2 * (*cik) + 1] - c_origin)},
                          static_cast<VType>(half * im * (*vik))); // Lik - 0
            vn.emplace_back({ki, (map_[2 * (*cik) + 1] - c_origin)},
                          static_cast<VType>(-half * im * ma::conj(*vik))); // Lik - 0
          }
        }
      }
      ++cik;
      ++vik;
    }
    else
    {
      if (abs(*vki) > cut)
      {
        if (2 * (*cki) >= c0 && 2 * (*cki) < c1)
        {
          RUNTIME_CHECK(map_[2 * (*cki)] >= 0, "");
          RUNTIME_CHECK(map_[2 * (*cki)] - c_origin < vn.size(1), "");
          vn.emplace_back({ik, (map_[2 * (*cki)] - c_origin)},
                          static_cast<VType>(half * ma::conj(*vki))); // Lki + 0
          vn.emplace_back({ki, (map_[2 * (*cki)] - c_origin)},
                          static_cast<VType>(half * (*vki))); // Lki + 0
        }
        if constexpr (boost::is_complex<VType>::value) { 
          if (2 * (*cki) + 1 >= c0 && 2 * (*cki) + 1 < c1)
          {
            RUNTIME_CHECK(map_[2 * (*cki) + 1] >= 0, "");
            RUNTIME_CHECK(map_[2 * (*cki) + 1] - c_origin < vn.size(1), "");
            vn.emplace_back({ik, (map_[2 * (*cki) + 1] - c_origin)},
                          static_cast<VType>(-half * im * ma::conj(*vki))); // Lki - 0
            vn.emplace_back({ki, (map_[2 * (*cki) + 1] - c_origin)},
                          static_cast<VType>(half * im * (*vki))); // Lki - 0
          }
        }
      }
      ++cki;
      ++vki;
    }
  }
  while (cik != cik_end)
  {
    if (abs(*vik) > cut)
    {
      if (2 * (*cik) >= c0 && 2 * (*cik) < c1)
      {
        RUNTIME_CHECK(map_[2 * (*cik)] >= 0, "");
        RUNTIME_CHECK(map_[2 * (*cik)] - c_origin < vn.size(1), "");
        vn.emplace_back({ik, (map_[2 * (*cik)] - c_origin)},
                        static_cast<VType>(half * (*vik))); // Lik + 0
        vn.emplace_back({ki, (map_[2 * (*cik)] - c_origin)},
                        static_cast<VType>(half * ma::conj(*vik))); // Lik + 0
      }
      if constexpr (boost::is_complex<VType>::value) { 
        if (2 * (*cik) + 1 >= c0 && 2 * (*cik) + 1 < c1)
        {
          RUNTIME_CHECK(map_[2 * (*cik) + 1] >= 0, "");
          RUNTIME_CHECK(map_[2 * (*cik) + 1] - c_origin < vn.size(1), "");
         vn.emplace_back({ik, (map_[2 * (*cik) + 1] - c_origin)},
                         static_cast<VType>(half * im * (*vik))); // Lik - 0
          vn.emplace_back({ki, (map_[2 * (*cik) + 1] - c_origin)},
                        static_cast<VType>(-half * im * ma::conj(*vik))); // Lik - 0
        }
      }
    }
    ++cik;
    ++vik;
  }
  while (cki != cki_end)
  {
    if (abs(*vki) > cut)
    {
      if (2 * (*cki) >= c0 && 2 * (*cki) < c1)
      {
        RUNTIME_CHECK(map_[2 * (*cki)] >= 0, "");
        RUNTIME_CHECK(map_[2 * (*cki)] - c_origin < vn.size(1), "");
        vn.emplace_back({ik, (map_[2 * (*cki)] - c_origin)},
                        static_cast<VType>(half * ma::conj(*vki))); // Lki + 0
        vn.emplace_back({ki, (map_[2 * (*cki)] - c_origin)},
                        static_cast<VType>(half * (*vki))); // Lki + 0
      }
      if constexpr (boost::is_complex<VType>::value) { 
        if (2 * (*cki) + 1 >= c0 && 2 * (*cki) + 1 < c1)
        {
          RUNTIME_CHECK(map_[2 * (*cki) + 1] >= 0, "");
          RUNTIME_CHECK(map_[2 * (*cki) + 1] - c_origin < vn.size(1), "");
          vn.emplace_back({ik, (map_[2 * (*cki) + 1] - c_origin)},
                        static_cast<VType>(-half * im * ma::conj(*vki))); // Lki - 0
          vn.emplace_back({ki, (map_[2 * (*cki) + 1] - c_origin)},
                        static_cast<VType>(half * im * (*vki))); // Lki - 0
        }
      }
    }
    ++cki;
    ++vki;
  }
}

} // namespace

template<typename csrMat>
std::vector<std::size_t> count_nnz_per_cholvec(double cut, TaskGroup_& TG, csrMat& V2, int NMO)
{
  if (TG.getNumberOfTGs() > 1)
    APP_ABORT("Error: count_nnz_per_cholvec is not designed for distributed CholMat. ");

  if (V2.size(0) != NMO * NMO)
    APP_ABORT(" Error in count_nnz_per_cholvec: V2.size(0) ! NMO*NMO ");
  int ik0, ikN;
  // only upper triangular part since both ik/ki are processed simultaneously
  std::tie(ik0, ikN) = FairDivideBoundary(TG.Global().rank(), int(NMO * (NMO + 1) / 2), TG.Global().size());

  if (cut < 1e-12)
    cut = 1e-12;
  std::vector<std::size_t> counts(2 * V2.size(1));
  int nvecs = V2.size(1);

  for (int i = 0, cnt = 0; i < NMO; i++)
    for (int k = i; k < NMO; k++, cnt++)
    {
      if (cnt < ik0)
        continue;
      if (cnt >= ikN)
        break;
      if (i == k)
        count_over_cholvec(cut, counts, 0, nvecs, V2[i * NMO + k]);
      else
        count_over_cholvec(cut, counts, 0, nvecs, V2[i * NMO + k], V2[k * NMO + i]);
    }
  TG.Global().all_reduce_in_place_n(counts.begin(), counts.size(), std::plus<>());

  return counts;
}

template<typename csrMat>
std::vector<std::size_t> count_nnz_per_ik(double cut,
                                          TaskGroup_& TG,
                                          csrMat& V2,
                                          int NMO,
                                          int cv0,
                                          int cvN)
{
  RUNTIME_CHECK(cv0 >= 0 && cvN <= 2 * V2.size(1), "");
  if (TG.getNumberOfTGs() > 1)
    APP_ABORT("Error: count_nnz_per_ik is not designed for distributed CholMat. ");

  int ncores = TG.getTotalCores(), coreid = TG.getCoreID();

  if (V2.size(0) != NMO * NMO)
    APP_ABORT(" Error in count_nnz_per_cholvec: V2.size(0) ! NMO*NMO ");
  int ik0, ikN;
  // only upper triangular part since both ik/ki are processed simultaneously
  // it is possible to ssubdivide this over equivalent nodes and then reduce over them
  std::tie(ik0, ikN) = FairDivideBoundary(coreid, int(NMO * (NMO + 1) / 2), ncores);

  if (cut < 1e-12)
    cut = 1e-12;
  std::vector<std::size_t> counts(V2.size(0));

  for (int i = 0, cnt = 0; i < NMO; i++)
    for (int k = i; k < NMO; k++, cnt++)
    {
      if (cnt < ik0)
        continue;
      if (cnt >= ikN)
        break;
      if (i == k)
        count_nnz(cut, counts[i * NMO + k], cv0, cvN, V2[i * NMO + k]);
      else
        count_nnz(cut, counts[i * NMO + k], counts[k * NMO + i], cv0, cvN, V2[i * NMO + k], V2[k * NMO + i]);
    }
  TG.Node().all_reduce_in_place_n(counts.begin(), counts.size(), std::plus<>());
  // if divided over equivalent nodes, reduce over the Core_Equivalent communicator
  //TG.EqvCore().all_reduce_in_place_n(counts.begin(),counts.size(),std::plus<>());

  return counts;
}

template<typename csrMat>
void generateHSPotential(csrMat& vn,
                         std::vector<int> const& map_,
                         double cut,
                         TaskGroup_& TG,
                         csrMat& V2,
                         int NMO,
                         int cv0,
                         int cvN)
{
  RUNTIME_CHECK(cv0 >= 0 && cvN <= 2 * V2.size(1), "");
  if (TG.getNumberOfTGs() > 1)
    APP_ABORT("Error: generateHSPotential is not designed for distributed CholMat. ");

  int ncores = TG.getTotalCores(), coreid = TG.getCoreID();

  if (V2.size(0) != NMO * NMO)
    APP_ABORT(" Error in generateHSPotential: V2.size(0) ! NMO*NMO ");
  int ik0, ikN;
  // only upper triangular part since both ik/ki are processed simultaneously
  // it is possible to ssubdivide this over equivalent nodes and then reduce over them
  std::tie(ik0, ikN) = FairDivideBoundary(coreid, int(NMO * (NMO + 1) / 2), ncores);

  if (cut < 1e-12)
    cut = 1e-12;

  for (int i = 0, cnt = 0; i < NMO; i++)
    for (int k = i; k < NMO; k++, cnt++)
    {
      if (cnt < ik0)
        continue;
      if (cnt >= ikN)
        break;
      if (i == k)
        add_to_vn(vn, map_, cut, i * NMO + k, cv0, cvN, V2[i * NMO + k]);
      else
        add_to_vn(vn, map_, cut, i * NMO + k, k * NMO + i, cv0, cvN, V2[i * NMO + k], V2[k * NMO + i]);
    }
  // if distributed over equivalent nodes, perform communication step here

  TG.Node().barrier();
}

} // namespace HamHelper

} // namespace afqmc

} // namespace sfqmc

#endif
