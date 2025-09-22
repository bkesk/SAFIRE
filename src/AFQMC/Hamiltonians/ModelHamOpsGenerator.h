/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef SFQMC_AFQMC_MODELHAMOPSGENERATOR_H
#define SFQMC_AFQMC_MODELHAMOPSGENERATOR_H

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

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.hpp"

namespace sfqmc
{
namespace afqmc
{
class ModelHamOpsGenerator : public AFQMCInfo 
{
public:
  ModelHamOpsGenerator(AFQMCInfo const& info,
                       ptree pt_in,
                       TaskGroup_& tg_,
                       ComplexType nucE = 0,
                       ComplexType fzcE = 0)
      : AFQMCInfo(info), 
        TG(tg_), 
        NuclearCoulombEnergy(nucE), 
        FrozenCoreEnergy(fzcE)
  {
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nModelHamiltonian input:");
    app_log(2, "{}", io::to_string(pt));
    // initialize using verbose input
    fileName  = pt.get<std::string>("filename");
    name      = pt.get<std::string>("name");
    sparse_g_eval = pt.get<bool>("sparse_gf_eval");
    sparse_1body = pt.get<bool>("sparse_1body");
    shift_1body = pt.get<bool>("shift_1body");
  }

  ~ModelHamOpsGenerator() {}

  ModelHamOpsGenerator(ModelHamOpsGenerator const& other) = delete;
  ModelHamOpsGenerator(ModelHamOpsGenerator&& other)      = default;
  ModelHamOpsGenerator& operator=(ModelHamOpsGenerator const& other) = delete;
  ModelHamOpsGenerator& operator=(ModelHamOpsGenerator&& other) = delete;

  ComplexType getNuclearCoulombEnergy() const { return NuclearCoulombEnergy; }

  template<bool MP>
  HamiltonianOperations<MP> getHamiltonianOperations(WALKER_TYPES type,
                                                     std::vector<PsiT_Matrix>& PsiT,
                                                     TaskGroup_& TGprop,
                                                     TaskGroup_& TGwfn,
                                                     hdf_archive& hdf_restart);

  HamiltonianTypes getHamType()
  {
    return ModelHamiltonian;
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string name, filename;
    name      = pt0.get<std::string>("name", "ham0");
    filename  = pt0.get<std::string>("filename");
    bool sparse_gf = pt0.get<bool>("sparse_gf_eval", true);
    bool sparse_1b = pt0.get<bool>("sparse_1body", true);
    bool shift_1b = pt0.get<bool>("shift_1body", false);
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("filename", filename);
    pt1.put("sparse_gf_eval", sparse_gf);
    pt1.put("sparse_1body", sparse_1b);
    pt1.put("shift_1body", shift_1b);
    std::unordered_set<std::string> pass_through_keys = {
      "system"
    };
    io::compare_known_keys("Lattice Model",pt1, pt0,pass_through_keys);
    return pt1;
  }

protected:

  // for hamiltonian distribution
  TaskGroup_& TG;

  // nuclear coulomb term
  ComplexType NuclearCoulombEnergy = 0.0;
  ComplexType FrozenCoreEnergy = 0.0;

  std::string fileName = "";
  bool sparse_g_eval = true;
  bool sparse_1body = true;
  bool shift_1body = false;

  template<class csrM>
  csrM spin_to_walker_type(WALKER_TYPES type, std::string stype, csrM& hij); 

  template<bool MP, bool REAL, class csrM>
  SparseEnergy<MP,REAL> make_SparseEnergy(TaskGroup_& tg_, WALKER_TYPES type, csrM& hij, 
                                 csrM& combined_U, csrM& combined_J, ComplexType E0);

  template<class csrM>
  Vector<size_t> find_occupied_pairs(WALKER_TYPES type, 
                      std::vector<csrM>& U, std::vector<csrM>& J);

  template<bool MP, bool REAL, class csrM, class IArr, class map_t>
  void addComponent(TaskGroup_& tg_, WALKER_TYPES type, PropagatorTypes ptype, 
                      csrM& U, csrM& J, std::vector<ModelComponent<MP,REAL>>& Hams, 
                      IArr& n2IJ, map_t& IJ2n);

  template<bool MP, bool REAL>
  HamiltonianOperations<MP> getHamiltonianOperations_impl(WALKER_TYPES type,
                                                         std::vector<PsiT_Matrix>& PsiT,
                                                         TaskGroup_& TGprop,   
                                                         TaskGroup_& TGwfn,    
                                                         hdf_archive& hdf_restart);
};

} // namespace afqmc
} // namespace sfqmc

#include "AFQMC/Hamiltonians/ModelHamOpsGenerator.icc"

#endif
