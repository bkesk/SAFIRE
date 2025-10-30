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

#pragma once

#include <vector>
#include <tuple>
#include <mpi.h>
#include <algorithm>
#include <numeric>

#include "AFQMC/config.h"
#include "AFQMC/Utilities/afqmc_TTI.hpp"

namespace sfqmc
{
namespace afqmc
{
// due to generalized Slater matrices, the conditions change
template<class PsiT_Type>
inline void check_wavefunction_consistency(WALKER_TYPES type, PsiT_Type const& A, PsiT_Type const& B, int NMO, int NAEA, int NAEB)
{
  std::string err("Error: Incorrect Slater Matrix dimensions in check_wavefunction_consistency():\n");
  if (type == CLOSED)
  {
    utils::check(A.size(1) == NMO and A.size(0) >= NAEA, 
                    err + " wfn_type=CLOSED, NMO={}, NAEA={}, PsiT.shape: ({}, {}) ",
                    NMO, NAEA, A.size(0), A.size(1));
  }
  else if (type == COLLINEAR)
  {
    utils::check(A.size(1) == NMO and A.size(0) >= NAEA and 
                 B.size(1) == NMO and B.size(0) >= NAEB, 
                    err + " wfn_type=COLLINEAR, NMO={}, NAEA={}, NAEB={}, PsiT_up.shape: ({}, {}), PsiT_dn.shape: ({}, {}) ", 
                    NMO, NAEA, NAEB, A.size(0), A.size(1), B.size(0), B.size(1));
  }
  else if (type == NONCOLLINEAR)
  {
    utils::check(A.size(1) == 2 * NMO and A.size(0) >= (NAEB + NAEA),
                    err + " wfn_type=NONCOLLINEAR, NMO={}, NAEA={}, PsiT.shape: ({}, {}) ",
                    NMO, NAEA, A.size(0), A.size(1));
  }
  else
  {
    utils::check(false," Error: Unacceptable walker_type in check_wavefunction_consistency(): {}", int(type));
  }
}

template<MEMORY_SPACE MEM>
inline auto rotateHij(WALKER_TYPES walker_type, 
                      nda::MemoryArrayOfRank<2> auto&& H1,
                      PsiT_Matrix<MEM> const& Alpha,
                      PsiT_Matrix<MEM> const* Beta = nullptr) 
{
  int NAEA = Alpha.size(0);
  int NMO  = Alpha.size(1);

  memory::array<MEM, ComplexType, 2> N;
/*

  // 1-body part
  if (walker_type == CLOSED || walker_type == NONCOLLINEAR || walker_type == FULLYPOLARIZED)
  {
    N.reextent(iextensions<1u>{NAEA * NMO});
    boost::multi::array_ref<ComplexType, 2> N_(N.origin(), {NAEA, NMO});

    ma::product(*Alpha, H1, N_);
    if (walker_type == CLOSED)
      ma::scal(ComplexType(2.0), N);
  }
  else if (walker_type == COLLINEAR)
  {
    RUNTIME_CHECK(Beta != nullptr, "");
    if( (H1.size(0)!=NMO and H1.size(0)!=2*NMO) or (H1.size(1) != NMO) )
      APP_ABORT(" rotateHij: Smape mismatch."); 
    int NAEB = Beta->size(0);

    N.reextent(iextensions<1u>{(NAEA + NAEB) * NMO});
    boost::multi::array_ref<ComplexType, 2> NA_(N.origin(), {NAEA, NMO});
    boost::multi::array_ref<ComplexType, 2> NB_(N.origin() + NAEA * NMO, {NAEB, NMO});

    if(H1.size(0) == NMO) {
      ma::product(*Alpha, H1, NA_);
      ma::product(*Beta, H1, NB_);
    } else {
      ma::product(*Alpha, H1.sliced(0,NMO), NA_);
      ma::product(*Beta, H1.sliced(NMO,2*NMO), NB_);
    }
  }
*/
  return N;
}

} // namespace afqmc

} // namespace sfqmc

