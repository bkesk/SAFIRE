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

#undef NDEBUG

#include "catch2/catch.hpp"

#include "config.h"
#include "IO/AppAbort.hpp"

#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>

#include "utilities/Timer.hpp"
#include "utilities/test_common.hpp"
#include "utilities/check.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/readWfn.cpp"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

#include "numerics/sparse/sparse.hpp"

using std::complex;
using std::ifstream;
using std::string;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void wfn_fac(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file, bool dense_trial)
{
  using sfqmc::utils::ARRAY_EQUAL;
  using sfqmc::utils::VALUE_EQUAL;
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
  test_wfn = test_wfn.substr(test_wfn.find('_') + 1);

  auto file_data       = read_test_results_from_hdf<ComplexType>(hamil_file, test_wfn);
  auto [NMO,nup,ndown] = read_info_from_wfn(wfn_file, "any");
  utils::check(NMO == file_data.NMO, "Incompatible NMO.");

  WALKER_TYPES type    = afqmc::getWalkerType(wfn_file, "any");
  int nspin            = (type == COLLINEAR or type == COLLINEAR_FT) ? 2 : 1;
  int npol             = (type == NONCOLLINEAR or type == NONCOLLINEAR_FT) ? 2 : 1;
  int nel              = (type == COLLINEAR or type == COLLINEAR_FT) ? nup+ndown : nup;  
  double dt(0.01);

  int ntau(0);
  if(type == COLLINEAR_FT or type == NONCOLLINEAR_FT){
    ntau = nup;
    nup = NMO;
    ndown = NMO;
  }

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, ntau}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt); 
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 11; // choose prime number to force non-trivial splits in shared routines
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  if(type == CLOSED) wlk_pt.put("walker_type","closed");
  else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
  else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
  else if(type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
  else if(type == COLLINEAR_FT) wlk_pt.put("walker_type","collinear-ft");
  else if(type == NONCOLLINEAR_FT) wlk_pt.put("walker_type","noncollinear-ft");

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("dense_trial",dense_trial);

  WavefunctionFactory<MEM> WfnFac(InfoMap);
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);

  //nwalk=nw;
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);

  if(type != COLLINEAR_FT and type != NONCOLLINEAR_FT)
  {
    auto initial_guess = WfnFac.getInitialGuess("wfn0"); 
    REQUIRE(initial_guess.shape() == std::array<long,3>{nspin,npol*NMO,nup});

    wset.resize(nwalk, initial_guess);
  }
  else
  {
    auto initial_guess_ft = WfnFac.getInitialGuess_ft("wfn0"); 
    REQUIRE(initial_guess_ft.shape() == std::array<long,4>{3,nspin,npol*NMO,NMO});

    wset.resize(nwalk, initial_guess_ft);
  }

  // Overlap
  //if(type != COLLINEAR_FT and type != NONCOLLINEAR_FT)
  wfn.Log_Overlap(wset);

  Watch Time;
  Time.reset();

  wfn.Energy(wset);

  if (std::abs(file_data.E0 + file_data.E1 + file_data.E2) > 1e-8)
  {
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      VALUE_EQUAL(it->get_property(E1_), file_data.E0 + file_data.E1);
      VALUE_EQUAL(it->get_property(EXX_) + it->get_property(EJ_), file_data.E2);
      VALUE_EQUAL(it->energy(), file_data.E0 + file_data.E1 + file_data.E2);
    }
  }
  else
  {
    app_log(1," E: {}", wset[0].energy()); 
    app_log(1," E0+E1: {}", wset[0].get_property(E1_));
    app_log(1," EJ: {}", wset[0].get_property(EJ_)); 
    app_log(1," EXX: {}", wset[0].get_property(EXX_));
  }
  
  nda::array<ComplexType,1> nMF(2*NMO); 
  // G_MF
  {
    auto gMF_d = wfn.G_MF();
    auto gMF = nda::to_host(gMF_d());
    nMF(nda::range(npol*NMO))= nda::diagonal(gMF()(0,nda::range::all,nda::range::all));
    if(type == COLLINEAR_FT)
      nMF(nda::range(npol*NMO,nspin*npol*NMO))= nda::diagonal(gMF()(1,nda::range::all,nda::range::all));
  }

  // vMF
  {
    bool natural_shift = true;
    memory::array<MEM,ComplexType,1> v(wfn.number_of_cholesky_vectors());
    // potentials must be initialized for discrete HS decomp.
    wfn.update_potentials(dt,nMF,v,natural_shift);
    wfn.vMF(v,dt);
  }

  Time.reset();
  memory::array<MEM,ComplexType,2> X(nwalk,wfn.number_of_cholesky_vectors());

  wfn.vbias(wset, X, dt);
  //std::cout<<"X = "<<X()<<std::endl;
  {
    auto X_h = nda::to_host(X);
    ComplexType Xsum = 0;
    if (std::abs(file_data.Xsum) > 1e-8)
    {
      for (int n = 0; n < nwalk; n++)
      {
        Xsum = nda::sum(X_h(n,all));
        VALUE_EQUAL(Xsum,file_data.Xsum);
      }
    }
    else
    {
      Xsum = nda::sum(X_h(0,all));
      ComplexType Xsum2 = 0;
      for (auto& v: X_h(0,all) )
        Xsum2 += ComplexType(0.5) * v * v; 
      app_log(1," Xsum: {}", Xsum);
      app_log(1," Xsum2 (EJ): {}", Xsum2 / dt);
    }
  }

  if (wfn.getHamType() == ModelHamiltonian) // only sparseP2 is used - denseP2 is hardcoded to never run!
  {

    auto vHS = wfn.vHS_sparse(X, dt); 
    utils::check(vHS.extent(0) == nspin, "Size mismatch");
    utils::check((vHS(0).shape() == std::array<long,2>{nwalk*npol*NMO,nwalk*npol*NMO}) and
                 (vHS(nspin-1).shape() == std::array<long,2>{nwalk*npol*NMO,nwalk*npol*NMO}),
                 "Size mismatch");        
    /*
        // Convert sparse matrices to dense CMatrix objects for easier manipulation
        CMatrix vHS_up_dense({vHS_up->size(0), vHS_up->size(1)}, alloc_);
        CMatrix vHS_down_dense({vHS_down->size(0), vHS_down->size(1)}, alloc_);
        
        // Initialize dense matrices to zero
        std::fill_n(vHS_up_dense.origin(), vHS_up_dense.num_elements(), ComplexType(0.0));
        std::fill_n(vHS_down_dense.origin(), vHS_down_dense.num_elements(), ComplexType(0.0));
        
        // Convert sparse to dense using correct sparse matrix API
        for (int row = 0; row < static_cast<int>(vHS_up->size(0)); ++row) {
          auto [nnz, vals, cols] = vHS_up->sparse_row(row);
          for (size_t i = 0; i < nnz; ++i) {
            vHS_up_dense[row][cols[i]] = vals[i];
          }
        }
        
        // For collinear systems, handle vHS_down if it's different from vHS_up
        // For noncollinear systems, vHS_up and vHS_down point to the same matrix
        if (vHS_up != vHS_down) {
          // COLLINEAR case: fill vHS_down_dense separately
          for (int row = 0; row < static_cast<int>(vHS_down->size(0)); ++row) {
            auto [nnz, vals, cols] = vHS_down->sparse_row(row);
            for (size_t i = 0; i < nnz; ++i) {
              vHS_down_dense[row][cols[i]] = vals[i];
            }
          }
        } else {
          // NONCOLLINEAR case: copy the same data
          std::copy_n(vHS_up_dense.origin(), vHS_up_dense.num_elements(), vHS_down_dense.origin());
        }
        
        ComplexType Vsum = 0;
        if (std::abs(file_data.Vsum) > 1e-8)
        {
          for (int n = 0; n < nwalk; n++)
          {
            Vsum = 0;
            if (wfn.transposed_vHS())
            {
              for (int i = 0; i < vHS_up_dense.size(1); i++)
                Vsum += vHS_up_dense[n][i];
              for (int i = 0; i < vHS_down_dense.size(1); i++)
                Vsum += vHS_down_dense[n][i];
            }
            else
            {
              for (int i = 0; i < vHS_up_dense.size(0); i++)
                Vsum += vHS_up_dense[i][n];
              for (int i = 0; i < vHS_down_dense.size(0); i++)
                Vsum += vHS_down_dense[i][n];
            }
            VALUE_EQUAL(Vsum,file_data.Vsum);
          }
        } else {
          Vsum = 0;
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS_up_dense.size(1); i++)
              Vsum += vHS_up_dense[0][i];
            for (int i = 0; i < vHS_down_dense.size(1); i++)
              Vsum += vHS_down_dense[0][i];
          }
          else
          {
            for (int i = 0; i < vHS_up_dense.size(0); i++)
              Vsum += vHS_up_dense[i][0];
            for (int i = 0; i < vHS_down_dense.size(0); i++)
              Vsum += vHS_down_dense[i][0];
          }
          app_log(1," Vsum: {}", ComplexType(Vsum));
        }
*/
  } else { // not a model Hamiltonian

    /*
    Time.reset();
    auto vHS_d = wfn.vHS(X, dt);
    auto vHS = nda::to_host(vHS_d);

    ComplexType Vsum = 0;
    if (std::abs(file_data.Vsum) > 1e-8)
    {
      for (int n = 0; n < nwalk; n++)
      {
        Vsum = nda::sum(vHS(all,n,all,all));
        VALUE_EQUAL(Vsum,file_data.Vsum);
      }
    }
    else
    {
      Vsum = nda::sum(vHS(all,0,all,all));
      app_log(1," Vsum: {}", Vsum);
    }
    */

  }
}

TEST_CASE("wfn_fac_sdet", "[wavefunction_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  wfn_fac<HOST_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN,true);
  wfn_fac<HOST_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN,false);
#if defined(ENABLE_DEVICE)
  wfn_fac<DEVICE_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN,true);
  wfn_fac<DEVICE_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN,false);
#endif
}

} // namespace sfqmc
