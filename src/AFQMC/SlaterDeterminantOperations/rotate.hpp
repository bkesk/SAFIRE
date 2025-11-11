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

#ifndef AFQMC_ROTATE_HPP
#define AFQMC_ROTATE_HPP

#include <numeric>
#include "AFQMC/config.h"
#include "Utilities/FairDivide.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "SparseMatrix/csr_matrix.hpp"
#include "Numerics/ma_operations.hpp"
#include "Numerics/csr_blas.hpp"
#include "SparseMatrix/matrix_emplace_wrapper.hpp"
#include "multi/array_ref.hpp"
#include "AFQMC/Utilities/afqmc_TTI.hpp"
#include "mpi.h"

namespace sfqmc
{
namespace afqmc
{
namespace det_ops 
{
/*
 * Constructs the following contraction of the Slater matrix of the trial wfn and the Cholesky Matrix (Likn).
 * Case:
 *   - Closed/Collinear:  L[a][n][k] = sum_i A[a][i] L[i][k][n]
 *       - In collinear case, two separate calls are made for each spin channel.
 *   - Non-collinear: L[a][n][sk] = sum_i A[a][si] L[i][k][n]   // [si] == [s][i] combined spinor index
 *       - In this case, to preserve matrix dimenions, [s][k] --> [sk] is kept as a single index.
 */
template<class MultiArray2DA, class MultiArray3DB, class MultiArray3DC, class MultiArray2D>
void getLank(MultiArray2DA&& Aai,
             MultiArray3DB&& Likn,
             MultiArray3DC&& Lank,
             MultiArray2D&& buff,
             bool noncollinear = false)
{
  int npol = noncollinear ? 2 : 1;
  int na   = Aai.size(0);
  if (na == 0)
    return;
  int ni    = Aai.size(1) / npol;
  int nk    = Likn.size(1);
  int nchol = Likn.size(2);
  RUNTIME_CHECK(Likn.size(0) == ni, "");
  RUNTIME_CHECK(Lank.size(0) == na, "");
  RUNTIME_CHECK(Lank.size(1) == nchol, "");
  RUNTIME_CHECK(Lank.size(2) == nk * npol, "");
  RUNTIME_CHECK(buff.size(0) >= npol * nk, "");
  RUNTIME_CHECK(buff.size(1) >= nchol, "");
  if (noncollinear)
    RUNTIME_CHECK(Aai.stride(0) == Aai.size(1), "Aai is not contiguous."); // make sure it is contiguous

  using elementA = typename std::decay<MultiArray2DA>::type::element;
  using element  = typename std::decay<MultiArray3DC>::type::element;
  boost::multi::array_ref<elementA, 2> Aas_i(raw_pointer_cast(Aai.origin()), {na * npol, ni});
  boost::multi::array_ref<element, 2> Li_kn(raw_pointer_cast(Likn.origin()), {ni, nk * nchol});
  boost::multi::array_ref<element, 2> Las_kn(raw_pointer_cast(Lank.origin()), {na * npol, nk * nchol});

  ma::product(Aas_i, Li_kn, Las_kn);
  for (int a = 0; a < na; a++)
  {
    boost::multi::array_ref<element, 2> Lskn(raw_pointer_cast(Lank[a].origin()), {npol * nk, nchol});
    boost::multi::array_ref<element, 2> Lnsk(raw_pointer_cast(Lank[a].origin()), {nchol, npol * nk});
    buff({0, npol * nk}, {0, nchol}) = Lskn;
    ma::transpose(buff({0, npol * nk}, {0, nchol}), Lnsk);
  }
}

/*
 * Constructs the following contraction of the Slater matrix of the trial wfn and the Cholesky Matrix (Likn).
 * Case:
 *   - Closed/Collinear:  L[a][n][k] = sum_i A[a][i] conj(L[k][i][n])
 *       - In collinear case, two separate calls are made for each spin channel.
 *   - Non-collinear: L[a][n][sk] = sum_i A[a][si] conj(L[k][i][n])   // [si] == [s][i] combined spinor index
 *       - In this case, to preserve matrix dimenions, [s][k] --> [sk] is kept as a single index.
 */
template<class MultiArray2DA, class MultiArray3DB, class MultiArray3DC, class MultiArray2D>
void getLank_from_Lkin(MultiArray2DA&& Aai,
                       MultiArray3DB&& Lkin,
                       MultiArray3DC&& Lank,
                       MultiArray2D&& buff,
                       bool noncollinear = false)
{
  int npol = noncollinear ? 2 : 1;
  int na   = Aai.size(0);
  if (na == 0)
    return;
  int ni    = Aai.size(1) / npol;
  int nk    = Lkin.size(0);
  int nchol = Lkin.size(2);
  RUNTIME_CHECK(Lkin.size(1) == ni, "");
  RUNTIME_CHECK(Lank.size(0) == na, "");
  RUNTIME_CHECK(Lank.size(1) == nchol, "");
  RUNTIME_CHECK(Lank.size(2) == nk * npol, "");
  RUNTIME_CHECK(buff.num_elements() >= na * npol * nchol, "");
  if (noncollinear)
    RUNTIME_CHECK(Aai.stride(0) == Aai.size(1), "Aai is not contiguous."); // make sure it is contiguous

  using Type     = typename std::decay<MultiArray3DC>::type::element;
  using elementA = typename std::decay<MultiArray2DA>::type::element;
  boost::multi::array_ref<elementA, 2> Aas_i(raw_pointer_cast(Aai.origin()), {na * npol, ni});
  boost::multi::array_ref<Type, 2> bnas(raw_pointer_cast(buff.origin()), {nchol, na * npol});
  // Lank[a][n][k] = sum_i Aai[a][i] conj(Lkin[k][i][n])
  // Lank[as][n][k] = sum_i Aai[as][i] conj(Lkin[k][i][n])
  for (int k = 0; k < nk; k++)
  {
    ma::product(ma::H(Lkin[k]), ma::T(Aas_i), bnas);
    for (int a = 0; a < na; a++)
      for (int n = 0; n < nchol; n++)
        for (int p = 0; p < npol; p++)
          Lank[a][n][p * nk + k] = bnas[n][a * npol + p];
  }
}

} // namespace ma_rotate

namespace ma_rotate_padded
{
// designed for padded arrays
template<class MultiArray2DA, class MultiArray3DB, class MultiArray3DC>
void getLakn_Lank(MultiArray2DA&& Aai,
                  MultiArray3DB&& Likn,
                  MultiArray3DC&& Lakn,
                  MultiArray3DC&& Lank,
                  bool noncollinear = false)
{
  int npol = noncollinear ? 2 : 1;
  int na   = Aai.size(0);
  if (na == 0)
    return;
  int ni = Aai.size(1) / npol;

  int nmo   = Likn.size(0);
  int nchol = Likn.size(2);
  RUNTIME_CHECK(Likn.size(1) == nmo, "");

  RUNTIME_CHECK(Lakn.size(1) == npol * nmo, "");
  RUNTIME_CHECK(Lakn.size(2) == nchol, "");

  RUNTIME_CHECK(Lakn.size(0) >= na, "");
  RUNTIME_CHECK(Lakn.size(0) == Lank.size(0), "");
  RUNTIME_CHECK(Lank.size(1) == nchol, "");
  RUNTIME_CHECK(Lank.size(2) == npol * nmo, "");

  if (noncollinear)
    RUNTIME_CHECK(Aai.stride(0) == Aai.size(1), "Aai is not contiguous."); // make sure it is contiguous

  using elmA = typename std::decay<MultiArray2DA>::type::element;
  using elmB = typename std::decay<MultiArray3DB>::type::element;
  using elmC = typename std::decay<MultiArray3DC>::type::element;

  boost::multi::array_ref<elmA, 2> Aas_i(raw_pointer_cast(Aai.origin()), {na * npol, ni});
  boost::multi::array_ref<elmB, 2, decltype(Likn.origin())> Li_kn(Likn.origin(), {ni, nmo * nchol});
  boost::multi::array_ref<elmC, 2, decltype(Lakn.origin())> Las_kn(Lakn.origin(), {na * npol, nmo * nchol});

  ma::product(Aas_i, Li_kn, Las_kn);
  for (int a = 0; a < na; a++)
    ma::transpose(Lakn[a], Lank[a]);
}

template<class MultiArray2DA, class MultiArray3DB, class MultiArray3DC, class MultiArray2D>
void getLakn_Lank_from_Lkin(MultiArray2DA&& Aai,
                            MultiArray3DB&& Lkin,
                            MultiArray3DC&& Lakn,
                            MultiArray3DC&& Lank,
                            MultiArray2D&& buff,
                            bool noncollinear = false)
{
  int npol = noncollinear ? 2 : 1;
  int na   = Aai.size(0);
  if (na == 0)
    return;
  int ni = Aai.size(1) / npol;

  int nmo   = Lkin.size(0);
  int nchol = Lkin.size(2);
  RUNTIME_CHECK(Lkin.size(1) == nmo, "");

  RUNTIME_CHECK(Lakn.size(1) == npol * nmo, "");
  RUNTIME_CHECK(Lakn.size(2) == nchol, "");

  RUNTIME_CHECK(Lakn.size(0) >= na, "");
  RUNTIME_CHECK(Lakn.size(0) == Lank.size(0), "");
  RUNTIME_CHECK(Lank.size(1) == nchol, "");
  RUNTIME_CHECK(Lank.size(2) == npol * nmo, "");

  if (noncollinear)
    RUNTIME_CHECK(Aai.stride(0) == Aai.size(1), "Aai is not contiguous."); // make sure it is contiguous

  RUNTIME_CHECK(buff.num_elements() >= na * npol * nchol, "");

  using ptr2 = typename std::decay<MultiArray2D>::type::element_ptr;
  using elm2 = typename std::decay<MultiArray2D>::type::element;
  using elmA = typename std::decay<MultiArray2DA>::type::element;

  boost::multi::array_ref<elmA, 2> Aas_i(raw_pointer_cast(Aai.origin()), {na * npol, ni});
  boost::multi::array_ref<elm2, 2, ptr2> bnas(buff.origin(), {nchol, na * npol});
  // Lakn[a][sk][n] = sum_i Aai[as][i] conj(Lkin[k][i][n])
  for (int k = 0; k < nmo; k++)
  {
    ma::product(ma::H(Lkin[k].sliced(0, ni)), ma::T(Aas_i), bnas);
    for (int a = 0; a < na; a++)
      for (int p = 0; p < npol; p++)
        Lakn[a][p * nmo + k] = bnas({0, nchol}, a * npol + p);
  }
  for (int a = 0; a < na; a++)
    ma::transpose(Lakn[a], Lank[a]);
}

} // namespace

} // namespace afqmc

} // namespace sfqmc

#endif
