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

#ifndef SFQMC_AFQMC_WALKERCONFIG_HPP
#define SFQMC_AFQMC_WALKERCONFIG_HPP

// basic definitions used by walker set classes
namespace sfqmc
{
namespace afqmc
{
// wlk_descriptor: [ nmo, naea, naeb, nback_prop, nCV, nRefs, nHist]
using wlk_descriptor = std::array<int, 8>;
using wlk_indices    = std::array<int, 17>;
enum walker_data
{
  SM,
  WEIGHT,
  PHASE,
  PHASE1,
  PHASE2,
  PHASE3,
  PSEUDO_ELOC_,
  E1_,
  EXX_,
  EJ_,
  OVLP,
  SMN,
  SM_AUX,
  FIELDS,
  WEIGHT_FAC,
  WEIGHT_HISTORY,
  THETA,
};

} // namespace afqmc
} // namespace sfqmc

enum LOAD_BALANCE_ALGORITHM
{
  UNDEFINED_LOAD_BALANCE,
  SIMPLE,
  ASYNC
};
enum BRANCHING_ALGORITHM
{
  UNDEFINED_BRANCHING,
  PAIR,
  COMB,
  MIN_BRANCH,
  SERIAL_COMB
};

#endif
