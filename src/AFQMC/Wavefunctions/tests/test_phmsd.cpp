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

//#undef NDEBUG

#include "catch2/catch.hpp"

#include "config.h"
#include "IO/AppAbort.hpp"

#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/check.hpp"
#include "utilities/h5_utils.hpp"
#include "utilities/test_common.hpp"
#include "IO/app_loggers.h"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>
#include <algorithm>

#include "AFQMC/Wavefunctions/Excitations.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "numerics/sparse/sparse.hpp"

using std::complex;
using std::ifstream;
using std::string;
using std::real;
using std::imag;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void test_read_phmsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file)
{
  using sfqmc::utils::ARRAY_EQUAL;
  using nda::range;
  auto all = range::all;
  utils::check(utils::file_exists(hamil_file),
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);
  utils::check(utils::file_exists(wfn_file),
               " Wavefunction file not found: {}. \n Run unit test with --wfn /path/to/wfn.h5 ", wfn_file);

  // First strip path of filename.
  std::string base_name = wfn_file.substr(wfn_file.find_last_of("\\/") + 1);
  // Remove file extension.
  std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
  auto file_data       = read_test_results_from_hdf<ComplexType>(hamil_file, test_wfn);
  auto [NMO,nup,ndown] = read_info_from_wfn(wfn_file, "PHMSD");
  utils::check(NMO == file_data.NMO, "Incompatible NMO.");

  WALKER_TYPES type    = afqmc::getWalkerType(wfn_file, "PHMSD");
  int nspin            = (type == COLLINEAR) ? 2 : 1;
  int npol             = (type == NONCOLLINEAR) ? 2 : 1;
  int nel              = (type == COLLINEAR) ? nup+ndown : nup;

  h5::file file(wfn_file,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  h5::group ngrp = wgrp.open_group("PHMSD");
  int ndets_to_read = -1;
  std::string wfn_type;

  nda::array<int,2> occs;
  nda::array<ComplexType,1> coeffs;
  nda::array<PsiT_Matrix<HOST_MEMORY>, 1> PsiT_MO;
  read_ph_wavefunction_hdf(ngrp, coeffs, occs, ndets_to_read, type, NMO, nup, ndown, PsiT_MO, wfn_type); 

  ph_excitations<int, ComplexType> abij = build_ph_struct(coeffs, occs, ndets_to_read, NMO, nup, ndown);
  using std::get;
  auto cit = abij.configurations_begin();
  nda::array<int,1> configa(nup), configb(ndown);
  // Is it fortuitous that the order of determinants is the same?
  for (int nd = 0; nd < ndets_to_read; nd++, ++cit)
  {
    int alpha_ix = get<0>(*cit);
    int beta_ix  = get<1>(*cit);
    auto ci      = get<2>(*cit);
    abij.get_configuration(0, alpha_ix, configa);
    abij.get_configuration(1, beta_ix, configb);
    std::sort(configa.begin(), configa.end());
    std::sort(configb.begin(), configb.end());
    for (int i = 0; i < nup; i++)
    {
      REQUIRE(configa[i] == occs(nd,i));
    }
    for (int i = 0; i < ndown; i++)
    {
      REQUIRE(configb[i] == occs(nd,i + nup));
    }
    REQUIRE(std::abs(coeffs[nd]) == std::abs(ci));
  }
  // Check sign of permutation.
  REQUIRE(abij.number_of_configurations() == ndets_to_read);
}

template<class Mat>
void getSlaterMatrix(Mat&& SM, math::sparse::CSRMatrix auto&& Orbs, nda::MemoryVector auto&& occs, std::string orb_type)
{
  SM() = ComplexType(0.0);
  if( orb_type == "mixed" ) {
    auto row_begin = Orbs.row_begin();
    auto row_end = Orbs.row_end();
    auto vals = Orbs.values();
    auto cols = Orbs.columns();
    for (int r = 0; r < occs.extent(0); r++)
      for(int j=row_begin(occs(r)); j<row_end(occs(r)); ++j)
        SM(r,cols(j)) = vals(j); 
  } else {
    for (int r = 0; r < occs.extent(0); r++)
      SM(r,occs(r)) = ComplexType(1.0); 
  }
}

template<MEMORY_SPACE MEM>
void test_phmsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file)
{
  using sfqmc::utils::ARRAY_EQUAL;
  using nda::range;
  auto all = range::all;
  utils::check(utils::file_exists(hamil_file),
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);
  utils::check(utils::file_exists(wfn_file),
               " Wavefunction file not found: {}. \n Run unit test with --wfn /path/to/wfn.h5 ", wfn_file);

  // First strip path of filename.
  std::string base_name = wfn_file.substr(wfn_file.find_last_of("\\/") + 1);
  // Remove file extension.
  std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
  auto file_data       = read_test_results_from_hdf<ComplexType>(hamil_file, test_wfn);
  auto [NMO,nup,ndown] = read_info_from_wfn(wfn_file, "PHMSD");
  utils::check(NMO == file_data.NMO, "Incompatible NMO.");

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "PHMSD");
  int nspin         = (type == COLLINEAR) ? 2 : 1;
  int npol          = (type == NONCOLLINEAR) ? 2 : 1;
  int nel           = (type == COLLINEAR) ? nup+ndown : nup;
  int nwalk         = 1; 
  int ndets         = 100; 
  double dt         = 0.01;
  std::shared_ptr<utils::RandomGenerator_t> rng = std::make_shared<utils::RandomGenerator_t>();

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("rediag","no");
  wfn_pt.put("ndets_to_read",ndets);
  wfn_pt.put("algorithm",0);

  WavefunctionFactory WfnFac(InfoMap);
  WfnFac.push("wfn0", wfn_pt);
  Wavefunction& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  if(type == CLOSED) wlk_pt.put("walker_type","closed");
  else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
  else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
  else if (type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");

  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
  auto initial_guess = WfnFac.getInitialGuess("wfn0");
  REQUIRE(initial_guess.shape() == std::array<long,3>{nspin,npol*NMO,nup});

  // apply small unitary rotation to initial_guess
  // add different rotations to every walker to test routines
  {
    nda::array<ComplexType,3> rotated_initial_guess(nspin,npol*NMO,nup);
    nda::array<ComplexType,2> R = nda::rand(std::array<long,2>{npol*NMO,npol*NMO});
    nda::array<ComplexType,1> tau(npol*NMO);
    nda::lapack::geqrf(nda::transpose(R),tau);
    nda::lapack::gqr(nda::transpose(R),tau);
    for(int is=0; is<nspin; ++is)
      nda::blas::gemm(R,initial_guess(is,all,all),rotated_initial_guess(is,all,all));
    wset.resize(nwalk, rotated_initial_guess);
  }

  // 0. Get raw occupancies and coefficients from file.
  nda::array<PsiT_Matrix<HOST_MEMORY>, 1> PsiT_MO;
  nda::array<ComplexType,2> PsiA(nup,npol*NMO); 
  nda::array<ComplexType,2> PsiB(ndown,npol*NMO); 
  nda::array<ComplexType,1> coeffs;
  nda::array<int,2> occs;
  h5::file f(wfn_file,'r');
  h5::group g = h5::group(f).open_group("Wavefunction").open_group("PHMSD");
  std::string orb_type;
  read_ph_wavefunction_hdf(g, coeffs, occs, ndets, type, 
                                NMO, nup, ndown, PsiT_MO, orb_type);

  // 1. Overlap 
  ComplexType ovlp_sum = ComplexType(0.0);
  for (int idet = 0; idet < ndets; idet++)
  {
    // Construct slater matrix from given set of occupied orbitals.
    nda::array<ComplexType,1> ov(nwalk,ComplexType(0.0));
    getSlaterMatrix(PsiA, PsiT_MO(0), occs(idet,range(nup)),orb_type);
    det_ops::Log_Overlap(PsiA,wset.template SlaterMatrices<HOST_MEMORY>(Alpha),ov);
    if(type == COLLINEAR) {
      nda::array<int, 1> ob = occs(idet,range(nup,nup+ndown)) - NMO; 
      getSlaterMatrix(PsiB, PsiT_MO(1), ob,orb_type);
      det_ops::Log_Overlap(PsiB,wset.template SlaterMatrices<HOST_MEMORY>(Beta),ov);
    }
    ovlp_sum += std::conj(coeffs[idet]) * std::exp(ov(0));
  }
  wfn.Log_Overlap(wset);
  
  // log(ovlp_sum)
  ovlp_sum = std::log(ovlp_sum);

  // the phase can be off by 2*pi due to small round-off errors around 0, what to do???
  for (auto it = wset.begin(); it != wset.end(); ++it)
    REQUIRE(std::abs(it->get_property(OVLP)) == Approx(std::abs(ovlp_sum)));

  // 2. Green function
  nda::array<ComplexType,3> G(nwalk,nspin*npol*NMO,npol*NMO);
  nda::array<ComplexType,3> Gt(nwalk,nspin*npol*NMO,npol*NMO);
  G() = ComplexType(0.0);
  for (int idet = 0; idet < ndets; idet++)
  {
    nda::array<ComplexType,1> ov(nwalk,ComplexType(0.0));
    Gt() = ComplexType(0.0);
    getSlaterMatrix(PsiA, PsiT_MO(0), occs(idet,range(nup)),orb_type);
    det_ops::MixedDensityMatrix(PsiA,wset.template SlaterMatrices<HOST_MEMORY>(Alpha),Gt(all,range(npol*NMO),all),ov,false);
    if(type == COLLINEAR) {
      nda::array<int, 1> ob = occs(idet,range(nup,nup+ndown)) - NMO;
      getSlaterMatrix(PsiB, PsiT_MO(1), ob,orb_type);
      det_ops::MixedDensityMatrix(PsiB,wset.template SlaterMatrices<HOST_MEMORY>(Beta),Gt(all,range(npol*NMO,2*npol*NMO),all),ov,false);
    }
    for(int iw=0; iw<nwalk; ++iw)
      G(iw,all,all) += std::conj(coeffs[idet]) * std::exp(ov(iw) - ovlp_sum) * Gt(iw,all,all); 
  }
  auto Gt2d = nda::reshape(Gt,std::array<long,2>{nwalk,nspin*npol*NMO*npol*NMO});
  Gt2d() = ComplexType(0.0);
  wfn.MixedDensityMatrix(wset,Gt2d,false);
  ARRAY_EQUAL(G,Gt);

  wfn.Energy(wset);
  app_log(2, "Ov: {}  {}",wset[0].get_property(OVLP),ovlp_sum); 
  app_log(2, "Energy: E1:{}, EJ:{}, EXX:{}",wset[0].get_property(E1_),wset[0].get_property(EJ_),wset[0].get_property(EXX_));

  // vMF
  {
    memory::array<MEM,ComplexType,1> v(wfn.number_of_cholesky_vectors());
    wfn.vMF(v,dt);
  }

  // G_MF
  {
    auto gMF = wfn.G_MF();
  }

}

TEST_CASE("test_read_phmsd", "[test_read_phmsd]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  test_read_phmsd<HOST_MEMORY>(mpi,UTEST_HAMIL, UTEST_WFN);
}

TEST_CASE("test_phmsd", "[read_phmsd]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  test_phmsd<HOST_MEMORY>(mpi,UTEST_HAMIL, UTEST_WFN);
#if defined(ENABLE_DEVICE)
  test_phmsd<DEVICE_MEMORY>(mpi,UTEST_HAMIL, UTEST_WFN);
#endif
}

} // namespace sfqmc
