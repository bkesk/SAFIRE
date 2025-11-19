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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctype.h>

#include "AFQMC/config.h"

#include "nda/h5.hpp"
//#include "SparseMatrix/csr_matrix.hpp"
//#include "SparseMatrix/csr_matrix_construct.hpp"
//#include "AFQMC/Wavefunctions/Excitations.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
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
*/

void getCommonInput(h5::group& g,
                    int NMO,
                    int NAEA,
                    int NAEB,
                    int& ndets_to_read,
                    nda::array<ComplexType,1>& ci,
                    WALKER_TYPES& walker_type);

WALKER_TYPES getWalkerType(std::string filename);
WALKER_TYPES getWalkerType(std::string filename, std::string type);

std::string getWavefunctionType(std::string filename);
std::tuple<int,int,int,int> getWavefunctionDims(std::string filename);


template<MEMORY_SPACE MEM>
auto read_nomsd_wavefunction(h5::group& grp,int ndets,
                            WALKER_TYPES walker_type, int NMO, int nup, int ndown)
{
  using csr = PsiT_Matrix<MEM>;
  long nspin = (walker_type == COLLINEAR ? 2 : 1);
  long npol = (walker_type == NONCOLLINEAR ? 2 : 1);
  long Mtot = npol*NMO;

  std::vector<int> dims(5);
  h5::h5_read(grp,"dims",dims);
  utils::check(NMO==dims[0], " Error in getCommonInput(): Inconsistent NMO . ");
  utils::check(nup == dims[1], " Error in getCommonInput(): Inconsistent  nup. ");
  utils::check(ndown==dims[2], " Error in getCommonInput(): Inconsistent  ndown. ");
  utils::check(walker_type==afqmc::initWALKER_TYPES(dims[3]),
               " Error in getCommonInput(): Inconsistent walker_type. ");
  utils::check(ndets <= dims[4], " Error in getCommonInput(): Inconsistent  ndets_to_read. ");

  // keep in on host at first
  nda::array<csr, 2> psi(ndets,nspin);
  long nel[] = {nup, ndown};

  for(int id=0, k=0; id<ndets; ++id) {
    for(int is=0; is<nspin; ++is, ++k) {
      h5::group pgrp = grp.open_group("PsiT_"+std::to_string(k));
      psi(id,is) = std::move(math::sparse::HDF2CSR<ComplexType,MEM,int,int>(pgrp));
      utils::check(psi(id,is).shape() == std::array<long,2>{nel[is],NMO}, "Shape mismatch");
    } 
  } 

  return psi;
}


} // namespace afqmc

} // namespace sfqmc

