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


#ifndef SFQMC_AFQMC_KPTHCHAMILTONIAN_H
#define SFQMC_AFQMC_KPTHCHAMILTONIAN_H

#include <iostream>
#include <vector>
#include <map>
#include <fstream>

#include "hdf/hdf_archive.h"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "Numerics/ma_operations.hpp"

#include "AFQMC/Hamiltonians/OneBodyHamiltonian.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.hpp"

namespace sfqmc
{
namespace afqmc
{
class KPTHCHamiltonian : public OneBodyHamiltonian
{
public:
  using shmSpMatrix = boost::multi::array<SPComplexType, 2, shared_allocator<SPComplexType>>;
  using CMatrix     = boost::multi::array<ComplexType, 2>;

  KPTHCHamiltonian(AFQMCInfo const& info,
                   ptree pt_in,
                   boost::multi::array<ValueType, 2>&& h,
                   TaskGroup_& tg_,
                   ValueType nucE = 0,
                   ValueType fzcE = 0)
      : OneBodyHamiltonian(info, std::move(h), nucE, fzcE), TG(tg_), fileName("")
  {
    if (TG.getNumberOfTGs() > 1)
      APP_ABORT(" Error: Distributed KPTHCHamiltonian not yet implemented.");
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(1,"KPTHCHamiltonian input:");
    app_log() << io::to_string(pt) << std::endl;
    // initialize using verbose input
    fileName  = pt.get<std::string>("filename");
    name      = pt.get<std::string>("name");
    cutoff_cholesky = pt.get<double>("cutoff_cholesky");
  }

  ~KPTHCHamiltonian() {}

  KPTHCHamiltonian(KPTHCHamiltonian const& other) = delete;
  KPTHCHamiltonian(KPTHCHamiltonian&& other)      = default;
  KPTHCHamiltonian& operator=(KPTHCHamiltonian const& other) = delete;
  KPTHCHamiltonian& operator=(KPTHCHamiltonian&& other) = delete;

  ValueType getNuclearCoulombEnergy() const { return OneBodyHamiltonian::NuclearCoulombEnergy; }

  boost::multi::array<ValueType, 2> getH1() const { return OneBodyHamiltonian::getH1(); }

  HamiltonianOperations getHamiltonianOperations(bool pureSD,
                                                 bool addCoulomb,
                                                 WALKER_TYPES type,
                                                 std::vector<PsiT_Matrix>& PsiT,
                                                 double cutvn,
                                                 double cutv2,
                                                 TaskGroup_& TGprop,
                                                 TaskGroup_& TGwfn,
                                                 hdf_archive& dump);

  ValueType H(IndexType I, IndexType J) const { return OneBodyHamiltonian::H(I, J); }

  // this should never be used outside initialization routines.
  ValueType H(IndexType I, IndexType J, IndexType K, IndexType L) const
  {
    APP_ABORT("Error: Calling H(I,J,K,L) in KPTHCHamiltonian. ");
    return ValueType(0.0);
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string name, filename;
    double cutoff_cholesky;
    filename  = pt0.get<std::string>("filename");
    name      = pt0.get<std::string>("name", "ham0");
    cutoff_cholesky = pt0.get<double>("cutoff_cholesky", 1e-6);
    // validate inputs
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("filename", filename);
    pt1.put("cutoff_cholesky", cutoff_cholesky);
    std::unordered_set<std::string> pass_through_keys = {
      "system"
    };
    io::compare_known_keys("K-point factorized THC Hamiltonian",pt1, pt0,pass_through_keys);
    return pt1;
  }

protected:
  // for hamiltonian distribution
  TaskGroup_& TG;

  std::string fileName;

  double cutoff_cholesky;
};

} // namespace afqmc
} // namespace sfqmc

#endif
