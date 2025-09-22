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


#ifndef SFQMC_AFQMC_FACTORIZEDSPARSEHAMILTONIAN_H
#define SFQMC_AFQMC_FACTORIZEDSPARSEHAMILTONIAN_H

#include <iostream>
#include <vector>
#include <map>
#include <fstream>

#include "hdf/hdf_archive.h"
#include "io/ptree/ptree_utilities.hpp"
#include "Utilities/app_loggers.h"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "Numerics/csr_blas.hpp"

#include "SparseMatrix/matrix_emplace_wrapper.hpp"
#include "SparseMatrix/csr_matrix_construct.hpp"
#include "AFQMC/Hamiltonians/rotateHamiltonian.hpp"
#include "AFQMC/HamiltonianOperations/HamiltonianOperations.hpp"

namespace sfqmc
{
namespace afqmc
{
class FactorizedSparseHamiltonian : public AFQMCInfo 
{
public:

  FactorizedSparseHamiltonian(AFQMCInfo const& info,
                              ptree pt_in,
                              TaskGroup_& tg_,
                              ComplexType nucE = 0,
                              ComplexType fzcE = 0)
      : AFQMCInfo(info),
        TG(tg_),
        NuclearCoulombEnergy(nucE),
        FrozenCoreEnergy(fzcE),
        fileName(""),
        cutoff_exx(1e-6),
        cutoff_cholesky(1e-6),
        factorizedHalfRotationType("DD"),
	n_reading_cores(-1),
	use_transpose(true),
        maximum_buffer_size(1024)
  {
    distribute_Ham = (TG.getNumberOfTGs() > 1);

    if (distribute_Ham)
      APP_ABORT(" Error: Distributed FactorizedHamiltonian not yet implemented.");

    ptree pt = interpret_inputs(pt_in);
    app_log(2, ""); 
    app_log(2,"FactorizedSparseHamiltonian input:");
    app_log(2, io::to_string(pt));
    app_log(2, ""); 
    name = pt.get<std::string>("name", name);
    fileName = pt.get<std::string>("filename");
    factorizedHalfRotationType = pt.get<std::string>("rotation_type");
    use_transpose = pt.get<bool>("use_transpose");
    cutoff_exx = pt.get<double>("cutoff_exx");
    cutoff_cholesky = pt.get<double>("cutoff_cholesky");
    maximum_buffer_size = pt.get<int>("buffer_size");
    n_reading_cores = pt.get<int>("num_io_cores");

  }

  ~FactorizedSparseHamiltonian() {}

  FactorizedSparseHamiltonian(FactorizedSparseHamiltonian const& other) = delete;
  FactorizedSparseHamiltonian(FactorizedSparseHamiltonian&& other)      = default;
  FactorizedSparseHamiltonian& operator=(FactorizedSparseHamiltonian const& other) = delete;
  FactorizedSparseHamiltonian& operator=(FactorizedSparseHamiltonian&& other) = delete;

  ComplexType getNuclearCoulombEnergy() const { return NuclearCoulombEnergy; }

  template<bool MP>
  HamiltonianOperations<MP> getHamiltonianOperations(WALKER_TYPES type,
                                                     std::vector<PsiT_Matrix>& PsiT,
                                                     TaskGroup_& TGprop,
                                                     TaskGroup_& TGwfn,
                                                     hdf_archive& hdf_restart);

  HamiltonianTypes getHamType()
  {
    return FactorizedSparse;
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string name, filename, rotation_type;
    bool use_transpose;
    double cutoff_exx, cutoff_cholesky;
    int buffer_size, num_io_cores;
    filename      = pt0.get<std::string>("filename");
    name          = pt0.get<std::string>("name", "ham0");
    rotation_type = pt0.get<std::string>("rotation_type", "DD");
    use_transpose = pt0.get<bool>("use_transpose", true);
    cutoff_exx      = pt0.get<double>("cutoff_exx", 1e-6);
    cutoff_cholesky = pt0.get<double>("cutoff_cholesky", 1e-6);
    buffer_size     = pt0.get<int>("buffer_size", 1024);
    num_io_cores    = pt0.get<int>("num_io_cores", -1);
    // validate inputs
#if defined(HAVE_MKL)
    if (rotation_type != "SS" && rotation_type != "DD" && rotation_type != "SD")
#else
    if (rotation_type != "DD" && rotation_type != "SD")
#endif
    {
      app_error(" Invalid parameter in Hamiltonian, rotation_type: {}", rotation_type);
      app_error(" Valid options: DD, SD (, and SS with MKL) ."); 
      APP_ABORT(" Invalid parameter in Hamiltonian, rotation_type ");
    }
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("filename", filename);
    pt1.put("rotation_type", rotation_type);
    pt1.put("use_transpose", use_transpose);
    pt1.put("cutoff_exx", cutoff_exx);
    pt1.put("cutoff_cholesky", cutoff_cholesky);
    pt1.put("buffer_size", buffer_size);
    pt1.put("num_io_cores", num_io_cores);
    std::unordered_set<std::string> pass_through_keys = {
      "system"
    };
    io::compare_known_keys("Factorized Sparse Hamiltonian",pt1, pt0,pass_through_keys);
    return pt1;
  }

protected:
  // for hamiltonian distribution
  TaskGroup_& TG;

  ComplexType NuclearCoulombEnergy;
  ComplexType FrozenCoreEnergy;

  std::string fileName;

  bool distribute_Ham;

  // options
  double cutoff_exx;
  double cutoff_cholesky;

  std::string factorizedHalfRotationType;
  int n_reading_cores;
  bool use_transpose;
  int maximum_buffer_size;

  /*
   * Definition:
   *  - LikType: data type of cholesky matrix, read from file 
   *  - LakType: data type of half-rotated cholesky matrix
   *  - V2XType: data type of 2-electron exchange integrals 
   * Rules: 
   *  - LikType: Determined from datatype in h5 file. 
   *  - If LikType = ComplexType, then LakType = V2XType = ComplexType
   *  - If LikType = RealType, then:
   *      - If all PsiT's are real (exactly zero complex part), then LakType = V2XType = RealType 
   *        otherwise LakType = V2XType = ComplexType
   *  - Since LakType = V2XType in all cases, I'm only templating to LakType
   */  
  template<bool MP, class LikType, class LakType, class PsiT_Type> 
  HamiltonianOperations<MP> getHamiltonianOperations_impl(WALKER_TYPES type,
                                                         std::vector<PsiT_Matrix>& PsiT,
                                                         std::vector<PsiT_Type>& PsiTsp,
                                                         TaskGroup_& TGprop,   
                                                         TaskGroup_& TGwfn,    
                                                         hdf_archive& hdf_restart);

  template<typename VType>
  mpi3_csr_matrix<VType> calculateHSPotentials(double cut,
                                               TaskGroup_& TGprop,
                                               mpi3_csr_matrix<VType>& Likn_bare);  

  template<typename CType, class PsiT_Mat, class csrMat>
  mpi3_csr_matrix<CType> halfRotatedHijkl(WALKER_TYPES type,
                                          TaskGroup_& TGHam,
                                          PsiT_Mat* Alpha,
                                          PsiT_Mat* Beta,
					  csrMat const& Likn_base,
                                          RealType const cut = 1e-6);

};

//#include "AFQMC/Hamiltonians/FactorizedSparseHamiltonian.icc"

} // namespace afqmc
} // namespace sfqmc

#endif
