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


#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctype.h>

#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "readWfn.h"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/sparse/sparse.hpp"

namespace sfqmc
{
namespace afqmc
{

std::tuple<int,int,int,int> getWavefunctionDims(std::string filename)
{
  h5::file file(filename,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  std::string name;
  if (wgrp.has_key("NOMSD")) {
    name = std::string("NOMSD");
  } else if (wgrp.has_key("PHMSD")) {
    name = std::string("PHMSD");
  } else {
    utils::check(false, "Error in getWavefunctionDims: Missing NOMSD/PHMSD block."); 
  }
  h5::group ngrp = wgrp.open_group(name);
  std::vector<int> dims(5);
  h5::h5_read(ngrp,"dims",dims);
  return std::make_tuple(dims[4],dims[0],dims[1],dims[2]);
}

std::string getWavefunctionType(std::string filename)
{
  std::string type;
  h5::file file(filename,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  if (wgrp.has_key("NOMSD")) {
    type = std::string("NOMSD");
  } else if (wgrp.has_key("PHMSD")) {
    type = std::string("PHMSD");
  } else {
    utils::check(false, "Error in getWavefunctionType: Missing NOMSD/PHMSD block."); 
  }
  return type;
}

WALKER_TYPES getWalkerType(std::string filename, std::string type)
{
  h5::file file(filename,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  utils::check(wgrp.has_key(type), "Error in getWavefunctionDims: Missing wfn type:{}",type);
  h5::group ngrp = wgrp.open_group(type);
  std::vector<int> Idata(5);
  h5::h5_read(ngrp,"dims",Idata);
  int wfn_type = Idata[3];
  if (wfn_type == 1)
    return CLOSED;
  else if (wfn_type == 2)
    return COLLINEAR;
  else if (wfn_type == 3)
    return NONCOLLINEAR;
  else if (wfn_type ==4)
    return FULLYPOLARIZED;
  else
    return UNDEFINED_WALKER_TYPE;
}

WALKER_TYPES getWalkerType(std::string filename)
{
  h5::file file(filename,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  std::string type;
  if (wgrp.has_key("NOMSD")) {
    type = std::string("NOMSD");
  } else if (wgrp.has_key("PHMSD")) {
    type = std::string("PHMSD");
  } else {
    utils::check(false, "Error in getWavefunctionType: Missing NOMSD/PHMSD block.");
  }
  h5::group ngrp = wgrp.open_group(type);
  std::vector<int> Idata(5);
  h5::h5_read(ngrp,"dims",Idata);
  int wfn_type = Idata[3];
  if (wfn_type == 1)
    return CLOSED;
  else if (wfn_type == 2)
    return COLLINEAR;
  else if (wfn_type == 3)
    return NONCOLLINEAR;
  else if (wfn_type ==4)
    return FULLYPOLARIZED;
  else
    return UNDEFINED_WALKER_TYPE;
}
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
                              std::string& type)
{
  int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
  RUNTIME_CHECK(walker_type != UNDEFINED_WALKER_TYPE, "");
  if (walker_type == CLOSED)
    APP_ABORT(" Error: walker_type==CLOSED not yet implemented in read_ph_wavefunction_hdf.");
  using Alloc = shared_allocator<ComplexType>;
  int NEL      = NAEA;
  bool mixed   = false;
  if (walker_type == COLLINEAR)
    NEL += NAEB;

  / *
   * type:
   *   - occ: All determinants are specified with occupation numbers
   *
   *   - 0: excitations out of a RHF reference
   *          NOTE: Does not mean perfect pairing, means excitations from a single reference
   *   - 1: excitations out of a UHF reference (not yet working)
   * /
  WALKER_TYPES wtype;
  getCommonInput(dump, NMO, NAEA, NAEB, ndets, ci_coeff, wtype, comm.root());
  if (wtype == CLOSED)
    APP_ABORT(" Error: walker_type==CLOSED not yet implemented for PHMSD Trial wavefunctions.");
  else if (wtype == NONCOLLINEAR)
    APP_ABORT(" Error: walker_type==NONCOLLINEAR not yet implemented for PHMSD Trial wavefunctions. Contact developers if you need this feature.");

  // limiting to this for now, kind of irrelevant until we find a FCI code that works in
  // a UHF basis
  if( wtype != walker_type )
    APP_ABORT(" Error: walker_type in wavefunction file differs from input file. ");

  int type_;
  if (!dump.readEntry(type_, "type"))
    APP_ABORT(" Error in read_ph_wavefunction_hdf: Problems reading type. ");
  if (type_ == 0) {
    type = "occ";
  } else if(type_ == 1) {
    type = "mixed";
    mixed = true;
  } else if(type_ == 2) {
    type = "mixed";
    mixed = true;
  } else {
    app_error(" Unknown type: {}",type_);
    APP_ABORT("Error in read_ph_wavefunction_hdf: Unknown value of dataset type.");
  }

  if (mixed)
  { // read reference
    PsiT.reserve( ( type_ == 2 ? 2 : 1) );

    if (dump.push(std::string("PsiT_") + std::to_string(0), false)<0)
      APP_ABORT(" Error in read_ph_wavefunction_hdf: Group PsiT not found. ");
    PsiT.emplace_back(csr_hdf5::HDF2CSR<PsiT_Matrix, Alloc>(dump, comm));
    // check size(0) when you know about active orbitals
    if( PsiT.back().size(1) != npol*NMO ) { 
      app_error(" Incorrect dimension of PsiT.size(1): {} ", PsiT.back().size(1));
      APP_ABORT(" Error: For PHMSD type=mixed, PsiT.size(1) must be npol*NMO.");
    }
    dump.pop();
    if (type_ == 2)
    {
      if(walker_type != COLLINEAR)
        APP_ABORT(" Error in read_ph_wavefunction_hdf: Inconsistent walker_type and wfn type.");
      if (dump.push(std::string("PsiT_") + std::to_string(1), false)<0)
        APP_ABORT(" Error in WavefunctionFactory: Group PsiT not found. ");
      PsiT.emplace_back(csr_hdf5::HDF2CSR<PsiT_Matrix, Alloc>(dump, comm));
      dump.pop();
    }
  }
  if (!dump.readEntry(occs, "occs"))
    APP_ABORT("Error reading occs array.");
  if( occs.size() < ndets*NEL )
    APP_ABORT(" Error in read_ph_wavefunction_hdf: occupation array too small." );
  comm.barrier();
}

ph_excitations<int, ComplexType> build_ph_struct(std::vector<ComplexType> ci_coeff,
                                                 boost::multi::array_ref<int, 2>& occs,
                                                 int ndets,
                                                 boost::mpi3::shared_communicator& comm,
                                                 int NMO,
                                                 int NAEA,
                                                 int NAEB)
{
  ComplexType ci;
  // count number of k-particle excitations
  // counts[0] has special meaning, it must be equal to NAEA+NAEB.
  std::vector<size_t> counts_alpha(NAEA + 1);
  std::vector<size_t> counts_beta(NAEB + 1);
  // ugly but need dynamic memory allocation
  std::vector<std::vector<int>> unique_alpha(NAEA + 1);
  std::vector<std::vector<int>> unique_beta(NAEB + 1);
  // reference configuration, taken as the first one right now
  std::vector<int> refa;
  std::vector<int> refb;
  // space to read configurations
  std::vector<int> confg;
  // space for excitation string identifying the current configuration
  std::vector<int> exct;
  // record file position to come back
  std::vector<int> Iwork; // work arrays for permutation calculation
  std::streampos start;
  if (comm.root())
  {
    confg.reserve(NAEA);
    Iwork.resize(2 * NAEA);
    exct.reserve(2 * NAEA);
    for (int i = 0; i < ndets; i++)
    {
      ci = ci_coeff[i];
      // alpha
      confg.clear();
      for (int k = 0, q = 0; k < NAEA; k++)
      {
        q = occs[i][k];
        if (q < 0 || q >= NMO)
          APP_ABORT("Error: Bad occupation number " + std::to_string(q) + " in determinant " + std::to_string(i) + " in wavefunction file. ");
        confg.emplace_back(q);
      }
      if (i == 0)
      {
        refa = confg;
      }
      else
      {
        int np = get_excitation_number(true, refa, confg, exct, ci, Iwork);
        push_excitation(exct, unique_alpha[np]);
      }
      if(NAEB==0) continue; // NONCOLLINEAR
      // beta
      confg.clear();
      for (int k = 0, q = 0; k < NAEB; k++)
      {
        q = occs[i][NAEA + k];
        if (q < NMO || q >= 2 * NMO)
          APP_ABORT("Error: Bad occupation number " + std::to_string(q) + " in determinant " + std::to_string(i) + " in wavefunction file. ");
        confg.emplace_back(q);
      }
      if (i == 0)
      {
        refb = confg;
      }
      else
      {
        int np = get_excitation_number(true, refb, confg, exct, ci, Iwork);
        push_excitation(exct, unique_beta[np]);
      }
    }
    // now that we have all unique configurations, count
    for (int i = 1; i <= NAEA; i++)
      counts_alpha[i] = unique_alpha[i].size();
    for (int i = 1; i <= NAEB; i++)
      counts_beta[i] = unique_beta[i].size();
  }
  comm.broadcast_n(counts_alpha.begin(), counts_alpha.size());
  if(NAEB>0) comm.broadcast_n(counts_beta.begin(), counts_beta.size());
  // using int for now, but should move to short later when everything works well
  // ph_struct stores the reference configuration on the index [0]
  ph_excitations<int, ComplexType> ph_struct(ndets, NAEA, NAEB, counts_alpha, counts_beta, shared_allocator<int>(comm));

  if (comm.root())
  {
    std::map<int, int> refa2loc;
    for (int i = 0; i < NAEA; i++)
      refa2loc[refa[i]] = i;
    std::map<int, int> refb2loc;
    for (int i = 0; i < NAEB; i++)
      refb2loc[refb[i]] = i;
    // add reference
    ph_struct.add_reference(refa, refb);
    // add unique configurations
    // alpha
    for (int n = 1; n < unique_alpha.size(); n++)
      for (std::vector<int>::iterator it = unique_alpha[n].begin(); it < unique_alpha[n].end(); it += (2 * n))
        ph_struct.add_alpha(n, it);
    // beta
    for (int n = 1; n < unique_beta.size(); n++)
      for (std::vector<int>::iterator it = unique_beta[n].begin(); it < unique_beta[n].end(); it += (2 * n))
        ph_struct.add_beta(n, it);
    // read configurations
    int alpha_index;
    int beta_index;
    int np;
    for (int i = 0; i < ndets; i++)
    {
      ci = ci_coeff[i];
      confg.clear();
      for (int k = 0, q = 0; k < NAEA; k++)
      {
        q = occs[i][k];
        if (q < 0 || q >= NMO)
          APP_ABORT("Error: Bad occupation number " + std::to_string(q) + " in determinant " + std::to_string(i) + " in wavefunction file. ");
        confg.emplace_back(q);
      }
      np = get_excitation_number(true, refa, confg, exct, ci, Iwork);
      alpha_index =
          ((np == 0) ? (0)
                     : (find_excitation(exct, unique_alpha[np]) + ph_struct.number_of_unique_smaller_than(np)[0]));
      if(NAEB==0) {
        ph_struct.add_configuration(alpha_index, 0, ci);
	continue;
      }
      confg.clear();
      for (int k = 0, q = 0; k < NAEB; k++)
      {
        q = occs[i][NAEA + k];
        if (q < NMO || q >= 2 * NMO)
          APP_ABORT("Error: Bad occupation number " + std::to_string(q) + " in determinant " + std::to_string(i) + " in wavefunction file. ");
        confg.emplace_back(q);
      }
      np = get_excitation_number(true, refb, confg, exct, ci, Iwork);
      beta_index =
          ((np == 0) ? (0) : (find_excitation(exct, unique_beta[np]) + ph_struct.number_of_unique_smaller_than(np)[1]));
      ph_struct.add_configuration(alpha_index, beta_index, ci);
    }
  }
  comm.barrier();
  return ph_struct;
}
*/

/*
 * Read trial wavefunction information from file.
 */
void getCommonInput(h5::group& grp,
                    int NMO,
                    int nup,
                    int ndown,
                    int& ndets_to_read,
                    nda::array<ComplexType,1>& ci,
                    WALKER_TYPES& walker_type)
{
  // check for consistency in parameters
  std::vector<int> dims(5);
  h5::h5_read(grp,"dims",dims);
  utils::check(NMO==dims[0], " Error in getCommonInput(): Inconsistent NMO . ");
  utils::check(nup == dims[1], " Error in getCommonInput(): Inconsistent  nup. ");
  utils::check(ndown==dims[2], " Error in getCommonInput(): Inconsistent  ndown. ");
  walker_type = afqmc::initWALKER_TYPES(dims[3]);
  if (ndets_to_read < 1)
    ndets_to_read = dims[4];
  app_log(1," - Number of determinants in trial wavefunction: {} ", ndets_to_read);
  utils::check(ndets_to_read <= dims[4], " Error in getCommonInput(): Inconsistent  ndets_to_read. ");
  ci.resize(ndets_to_read);
  nda::h5_read(grp,"ci_coeffs",ci);
  app_log(1," - Coefficient of first determinant: {} ", ci[0]);
}


} // namespace afqmc

} // namespace sfqmc
