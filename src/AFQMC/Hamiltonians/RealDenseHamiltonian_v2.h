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


#ifndef SFQMC_AFQMC_REALDENSEHAMILTONIAN_V2_H
#define SFQMC_AFQMC_REALDENSEHAMILTONIAN_V2_H

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
class RealDenseHamiltonian_v2 : public AFQMCInfo 
{
public:
  RealDenseHamiltonian_v2(AFQMCInfo const& info,
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
        max_memory_MB(2000)
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nRealDense GPU input:");
    app_log(2, "{}", io::to_string(pt));
    // initialize using verbose input
    fileName  = pt.get<std::string>("filename");
    name      = pt.get<std::string>("name");
    batched   = pt.get<bool>("batched");
    max_memory_MB = pt.get<int>("max_memory");
  }

  ~RealDenseHamiltonian_v2() {}

  RealDenseHamiltonian_v2(RealDenseHamiltonian_v2 const& other) = delete;
  RealDenseHamiltonian_v2(RealDenseHamiltonian_v2&& other)      = default;
  RealDenseHamiltonian_v2& operator=(RealDenseHamiltonian_v2 const& other) = delete;
  RealDenseHamiltonian_v2& operator=(RealDenseHamiltonian_v2&& other) = delete;

  ComplexType getNuclearCoulombEnergy() const { return NuclearCoulombEnergy; }

  template<bool MP>
  HamiltonianOperations<MP> getHamiltonianOperations(WALKER_TYPES type,
                                                     std::vector<PsiT_Matrix>& PsiT,
                                                     TaskGroup_& TGprop,
                                                     TaskGroup_& TGwfn,
                                                     hdf_archive& hdf_restart);

  HamiltonianTypes getHamType()
  {
    return RealDenseFactorized;
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string name, filename;
    bool batched;
    bool batched_default = false;
    int mmem_mb;
    if (number_of_devices() > 0) batched_default = true;
    name      = pt0.get<std::string>("name", "ham0");
    filename  = pt0.get<std::string>("filename");
    batched   = pt0.get<bool>("batched", batched_default);
    mmem_mb = pt0.get<int>("max_memory", 2000);
    // validate inputs
    if ((omp_get_num_threads() > 1) && (not batched))
    {
      app_warning(" Found OMP_NUM_THREADS > 1 with batched=false.");
      app_warning(" This will lead to low performance. Set batched=true. ");
    }
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("filename", filename);
    pt1.put("batched", batched);
    pt1.put("max_memory", mmem_mb);
    std::unordered_set<std::string> pass_through_keys = {
      "system"
    };
    io::compare_known_keys("Dense Generic Factorized (Choesky) Hamiltonian",pt1, pt0,pass_through_keys);
    return pt1;
  }

protected:
  // for hamiltonian distribution
  TaskGroup_& TG;

  ComplexType NuclearCoulombEnergy;
  ComplexType FrozenCoreEnergy;

  std::string fileName;

  bool batched;

  int max_memory_MB;

  template<bool MP, bool REAL_ONEBODY>
  HamiltonianOperations<MP> getHamiltonianOperations_impl(WALKER_TYPES type,
                                                         std::vector<PsiT_Matrix>& PsiT,
                                                         TaskGroup_& TGprop,   
                                                         TaskGroup_& TGwfn,    
                                                         hdf_archive& hdf_restart);

};

} // namespace afqmc
} // namespace sfqmc

#endif
