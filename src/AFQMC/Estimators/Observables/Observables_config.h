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

#ifndef SFQMC_AFQMC_OBSERVABLES_CONFIG_HPP
#define SFQMC_AFQMC_OBSERVABLES_CONFIG_HPP

namespace sfqmc
{
namespace afqmc
{
enum observables
{
  GFockOpa, // numerator of Extended Koopman's Theorem approach
  GFockOpb, // numerator of Extended Koopman's Theorem approach
  OneRDMFull,
  TwoRDMFull,
  OneRDMc,
  TwoRDMc,
  OneRSDMFull,
  TwoRSDMFull,
  OneRSDMc,
  TwoRSDMc,
  energy,
  force,
  PairCorr,
  SpinSpin,
}; // 12 observables
std::array<std::string, 12> hdf_ids = {"GFockOpa_",
                                       "GFockOpb_",
                                       "one_rdm_",
                                       "two_rdm_",
                                       "contracted_one_rdm_",
                                       "contracted_two_rdm_",
                                       "one_rsdm_",
                                       "two_rsdm_",
                                       "contracted_one_rsdm_",
                                       "contracted_two_rsdm_",
                                       "bp_energy_",
                                       "bp_force_",
                                       "spinspin_"};


} // namespace afqmc
} // namespace sfqmc


#endif
