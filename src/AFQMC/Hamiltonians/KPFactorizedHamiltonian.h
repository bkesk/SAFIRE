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


#ifndef SFQMC_AFQMC_KPFACTORIZEDHAMILTONIAN_H
#define SFQMC_AFQMC_KPFACTORIZEDHAMILTONIAN_H

#include <iostream>
#include <vector>
#include <map>
#include <fstream>

#include "hdf/hdf_archive.h"
#include "io/ptree/ptree_utilities.hpp"
#include "Utilities/app_loggers.h"

#include "AFQMC/config.h"
#include "Memory/utilities.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "Numerics/ma_operations.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

namespace sfqmc
{
namespace afqmc
{
class KPFactorizedHamiltonian : public AFQMCInfo 
{
public:

  KPFactorizedHamiltonian(AFQMCInfo const& info,
                          ptree pt_in,
                          TaskGroup_& tg_,
                          ComplexType nucE = 0,
                          ComplexType fzcE = 0)
      : AFQMCInfo(info),
        TG(tg_),
	NuclearCoulombEnergy(nucE),
	FrozenCoreEnergy(fzcE),	
        fileName(""),
        batched(false),
        out_of_core(false),
        memory(4096)
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nKPFactorizedHamiltonian input:");
    app_log(2, "{}", io::to_string(pt));
    // initialize using verbose input
    fileName  = pt.get<std::string>("filename");
    name      = pt.get<std::string>("name");
    batched   = pt.get<bool>("batched");
    out_of_core  = pt.get<bool>("out_of_core");
    memory   = pt.get<int>("memory");
    nsampleQ = pt.get<int>("nsampleQ");
    cutoff_cholesky = pt.get<double>("cutoff_cholesky");
  }

  ~KPFactorizedHamiltonian() {}

  KPFactorizedHamiltonian(KPFactorizedHamiltonian const& other) = delete;
  KPFactorizedHamiltonian(KPFactorizedHamiltonian&& other)      = default;
  KPFactorizedHamiltonian& operator=(KPFactorizedHamiltonian const& other) = delete;
  KPFactorizedHamiltonian& operator=(KPFactorizedHamiltonian&& other) = delete;

  ComplexType getNuclearCoulombEnergy() const { return NuclearCoulombEnergy; }

  HamiltonianTypes getHamType() const { return KPFactorized; }

  template<bool MP>
  HamiltonianOperations<MP> getHamiltonianOperations(WALKER_TYPES type,
                                                     std::vector<PsiT_Matrix>& PsiT,
                                                     TaskGroup_& TGprop,
                                                     TaskGroup_& TGwfn,
                                                     hdf_archive& hdf_restart);

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string name, filename;
    bool batched, out_of_core;
    double cutoff_cholesky;
    int memory, nsampleQ;
    bool batched_default = false;
    if (number_of_devices() > 0) batched_default = true;
    filename  = pt0.get<std::string>("filename");
    name      = pt0.get<std::string>("name", "ham0");
    batched   = pt0.get<bool>("batched", batched_default);
    out_of_core  = pt0.get<bool>("out_of_core", false);
    memory   = pt0.get<int>("memory", 4096);
    nsampleQ = pt0.get<int>("nsampleQ", -1);
    cutoff_cholesky = pt0.get<double>("cutoff_cholesky", 1e-6);
    // validate inputs
    if ((omp_get_num_threads() > 1) && (not batched))
    {
      app_warning(" Found OMP_NUM_THREADS > 1 with batched=false.");
      app_warning(" This will lead to low performance. Set batched=true. ");
    }
    if ((number_of_devices() < 1) && out_of_core)
    {
      app_warning(" out-of-core Cholesky energy matrix movement without device.");
      app_warning(" This options does nothing. ");
    }
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("filename", filename);
    pt1.put("batched", batched);
    pt1.put("out_of_core", out_of_core);
    pt1.put("cutoff_cholesky", cutoff_cholesky);
    pt1.put("memory", memory);
    pt1.put("nsampleQ", nsampleQ);
    std::unordered_set<std::string> pass_through_keys = {
      "system"
    };
    io::compare_known_keys("K-point Factorized Cholesky Hamiltonian",pt1, pt0,pass_through_keys);
    return pt1;
  }

protected:
  // for hamiltonian distribution
  TaskGroup_& TG;

  // nuclear coulomb term
  ComplexType NuclearCoulombEnergy;
  ComplexType FrozenCoreEnergy;

  std::string fileName;

  bool batched;

  bool out_of_core;

  int memory;

  double cutoff_cholesky;

  int nsampleQ = -1;

#if !defined(ENABLE_DEVICE)
  template<bool MP>
  HamiltonianOperations<MP> getHamiltonianOperations_shared(WALKER_TYPES type,
                                                           std::vector<PsiT_Matrix>& PsiT,
                                                           TaskGroup_& TGprop,
                                                           TaskGroup_& TGwfn,
                                                           hdf_archive& dump);
#endif

  template<bool MP>
  HamiltonianOperations<MP> getHamiltonianOperations_batched(WALKER_TYPES type,
                                                             std::vector<PsiT_Matrix>& PsiT,
                                                             TaskGroup_& TGprop,
                                                             TaskGroup_& TGwfn,
                                                             hdf_archive& dump);
};

} // namespace afqmc
} // namespace sfqmc

#endif
