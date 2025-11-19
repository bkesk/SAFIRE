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

#include <complex>
#include <random>
#include <algorithm>

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"

namespace sfqmc
{
namespace afqmc
{

template<typename T>
struct TEST_DATA
{
  int NMO, nup, ndown;
  T E0, E1, E2;
  T Xsum, Vsum;
};

inline int read_nmo_from_hdf(std::string fileName)
{
  int NMO(0);
  h5::file file(fileName,'r');
  h5::group grp(file);
  auto format = get_hamiltonian_format(grp);
  app_log(1, "Reading NMO from hamil file: {} of format {} ", fileName, format);
  if (format == "std")
  {
    utils::check(grp.has_subgroup("Hamiltonian"), "Missing Hamiltonian dataset.");
    h5::group hgrp = grp.open_group("Hamiltonian");
    std::vector<int> Idata(8);
    h5::h5_read(hgrp,"dims",Idata);
    NMO = Idata[3];
  } else if (format == "coqui") {
    utils::check(grp.has_subgroup("System"), "Missing Hamiltonian dataset.");
    h5::group hgrp = grp.open_group("System");
    h5::h5_read_attribute(hgrp,"number_of_bands",NMO);
  }
  return NMO;
}

inline std::tuple<int, int, int> read_info_from_wfn(std::string fileName, std::string type)
{
  app_log(1, "Reading info from wfn file: {} of type {} ", fileName, type);
  h5::file file(fileName,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  if(type == "any") {
    if( wgrp.has_key("NOMSD") )
      type = "NOMSD";
    else if( wgrp.has_key("PHMSD") )
      type = "PHMSD";
    else
      utils::check(false,"Missing NOMSD/PHMSD datasets in Wavefunction.");
  }
  h5::group mgrp = wgrp.open_group(type);
  std::vector<int> Idata(5);
  h5::h5_read(mgrp,"dims",Idata);
  return std::make_tuple(Idata[0], Idata[1], Idata[2]);
}

template<typename T>
TEST_DATA<T> read_test_results_from_hdf(std::string fileName, std::string wfn_type = "")
{
  h5::file file(fileName,'r');
  h5::group grp(file);
  int nmo = 0, nup = 0, ndn = 0;
  if (grp.has_key("Hamiltonian"))
  {
    h5::group hgrp = grp.open_group("Hamiltonian");
    std::vector<int> Idata(8);
    h5::h5_read(hgrp,"dims",Idata);
    nmo = Idata[3];
    nup = Idata[4];
    ndn = Idata[5];
  } else if(grp.has_key("System")) {
    h5::group sgrp = grp.open_group("System");
    h5::group bgrp = sgrp.open_group("BZ");
    int nkpts, nbnd;
    double nel;
    h5::h5_read_attribute(bgrp,"number_of_kpoints",nkpts);
    h5::h5_read_attribute(sgrp,"number_of_bands",nbnd);
    h5::h5_read_attribute(sgrp,"number_of_elec",nel);
    std::vector<int> dims(5);
    // values based on System.
    nup = int(nel/2.0)*nkpts;
    ndn = nup*nkpts; 
    nmo = nbnd*nkpts;
    // If Wavefunction is present, overwrite from data there
    if(grp.has_key("Wavefunction")) {
      h5::group wgrp = grp.open_group("Wavefunction");
      if(wgrp.has_key("NOMSD")) {
        h5::group ngrp = grp.open_group("NOMSD");
        std::vector<int> Idata(5);
        h5::h5_read(ngrp,"dims",Idata);
        nmo = dims[0];
        nup = dims[1];
        ndn = dims[2];
      } else if(wgrp.has_key("PHMSD")) {
        h5::group ngrp = grp.open_group("PHMSD");
        std::vector<int> Idata(5);
        h5::h5_read(ngrp,"dims",Idata);
        nmo = dims[0];
        nup = dims[1];
        ndn = dims[2];
      } 
    }   
  } else {
    utils::check(false," Error in read_test_results_from_hdf(): Invalid h5 format. ");
  }
  T E0(0), E1(0), E2(0), Xsum(0), Vsum(0);
  if (grp.has_key("TEST_RESULTS"))
  { 
    h5::group tgrp = grp.open_group("TEST_RESULTS");
    h5::h5_read(tgrp,wfn_type + "_E0",E0);
    h5::h5_read(tgrp,wfn_type + "_E1",E1);
    h5::h5_read(tgrp,wfn_type + "_E2",E2);
    h5::h5_read(tgrp,wfn_type + "_Xsum",Xsum);
    h5::h5_read(tgrp,wfn_type + "_Vsum",Vsum);
  }

  return TEST_DATA<T>{nmo, nup, ndn, E0, E1, E2, Xsum, Vsum};
}

/*
// Create a fake output hdf5 filename for unit tests.
inline std::string create_test_hdf(std::string& wfn_file, std::string& hamil_file)
{
  std::size_t startw   = wfn_file.find_last_of("\\/");
  std::size_t endw     = wfn_file.find_last_of(".");
  std::string wfn_base = wfn_file.substr(startw + 1, endw - startw - 1);

  std::size_t starth   = hamil_file.find_last_of("\\/");
  std::size_t endh     = hamil_file.find_last_of(".");
  std::string ham_base = hamil_file.substr(starth + 1, endh - starth - 1);

  return wfn_base + "_" + ham_base + ".h5";
}

// generate matrix of random integers between [a0, a0+range)
inline void fillRandomMatrix(std::vector<int>& vec, int range, int a=0)
{
  std::mt19937 generator(0);
  std::uniform_int_distribution<int> distribution(a, a+range-1);
  // avoid uninitialized warning
  [[maybe_unused]] int tmp = distribution(generator);
  for (int i = 0; i < vec.size(); i++)
  {
    int val  = distribution(generator);
    vec[i] = val;
  }
}

template<typename T>
void fillRandomMatrix(std::vector<T>& vec)
{
  std::mt19937 generator(0);
  std::normal_distribution<T> distribution(0.0, 1.0);
  // avoid uninitialized warning
  T tmp = distribution(generator);
  for (int i = 0; i < vec.size(); i++)
  {
    T val  = distribution(generator);
    vec[i] = val;
  }
}

template<typename T>
void fillRandomMatrix(std::vector<std::complex<T>>& vec)
{
  std::mt19937 generator(0);
  std::normal_distribution<T> distribution(0.0, 1.0);
  [[maybe_unused]] T tmp = distribution(generator);
  for (int i = 0; i < vec.size(); i++)
  {
    T re   = distribution(generator);
    T im   = distribution(generator);
    vec[i] = std::complex<T>(re, im);
  }
}
*/

} // namespace afqmc
} // namespace sfqmc

