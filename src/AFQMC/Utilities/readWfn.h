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

#ifndef AFQMC_READWFN_H
#define AFQMC_READWFN_H

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctype.h>

#include "hdf/hdf_archive.h"
#include "AFQMC/config.h"
#include "SparseMatrix/csr_matrix.hpp"
#include "SparseMatrix/csr_matrix_construct.hpp"
#include "AFQMC/Wavefunctions/Excitations.hpp"

namespace sfqmc
{
namespace afqmc
{

void read_ph_wavefunction_hdf(hdf_archive& dump,
                              std::vector<ComplexType>& ci_coeff,
                              std::vector<int>& occs,
                              int& ndets,
                              WALKER_TYPES walker_type,
                              boost::mpi3::shared_communicator& comm,
                              int NMO,
                              int NAEA,
                              int NAEB,
                              std::vector<PsiT_Matrix>& PsiT,
                              std::string& type);

ph_excitations<int, ComplexType> build_ph_struct(std::vector<ComplexType> ci_coeff,
                                                 boost::multi::array_ref<int, 2>& occs,
                                                 int ndets,
                                                 boost::mpi3::shared_communicator& comm,
                                                 int NMO,
                                                 int NAEA,
                                                 int NAEB);

void getCommonInput(hdf_archive& dump,
                    int NMO,
                    int NAEA,
                    int NAEB,
                    int& ndets_to_read,
                    std::vector<ComplexType>& ci,
                    WALKER_TYPES& walker_type,
                    bool root);

WALKER_TYPES getWalkerType(std::string filename);
WALKER_TYPES getWalkerType(std::string filename, std::string type);

std::string getWavefunctionType(std::string filename);
std::tuple<int,int,int,int> getWavefunctionDims(std::string filename);

} // namespace afqmc

} // namespace sfqmc

#endif
