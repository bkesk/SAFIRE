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
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/Timer.hpp"
#include "utilities/test_common.hpp"
#include "utilities/check.hpp"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"
#include "numerics/sparse/sparse.hpp"

#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/readWfn.h" 

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

using std::cerr;
using std::complex;
using std::cout;
using std::endl;
using std::ifstream;
using std::setprecision;
using std::string;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void propg_fac(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file, bool dense_trial)
{
  using sfqmc::utils::ARRAY_EQUAL;
  using nda::range;
  auto all = range::all;
  utils::check(utils::file_exists(hamil_file),
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);
  utils::check(utils::file_exists(wfn_file),
               " Wavefunction file not found: {}. \n Run unit test with --wfn /path/to/wfn.h5 ", wfn_file);

  int NMO = read_nmo_from_hdf(hamil_file);
  auto[wfn_NMO,nup, ndown] = read_info_from_wfn(wfn_file,"any");
  utils::check(NMO == wfn_NMO, "Error: NMO != wfn_NMO.");
  WALKER_TYPES type         = getWalkerType(wfn_file);
  int nspin                 = (type == COLLINEAR or type == COLLINEAR_FT) ? 2 : 1;
  int npol                  = (type == NONCOLLINEAR or type == NONCOLLINEAR_FT) ? 2 : 1;

  int ntau = 0;
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
  ham_pt.put("shift_1body",true);
  //ham_pt.put("shift_1body",false);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 11; 
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(777));

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  if(type == CLOSED) wlk_pt.put("walker_type","closed");
  else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
  else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
  else if(type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
  else if(type == COLLINEAR_FT) wlk_pt.put("walker_type","collinear-ft");
  else if(type == NONCOLLINEAR_FT) wlk_pt.put("walker_type","noncollinear-ft");
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("dense_trial",dense_trial);

  WavefunctionFactory<MEM> WfnFac(InfoMap);
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);

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

  ptree prop_pt;
  prop_pt.put("name","prop0");
  prop_pt.put("system","info0");
  prop_pt.put("denseP2",true);
  bool readFields = true;
  prop_pt.put("readFields",readFields);

  PropagatorFactory<MEM> PropgFac(InfoMap);
  PropgFac.push("prop0", prop_pt);
  auto& prop = PropgFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  std::cout << setprecision(8);
  wfn.Energy(wset);
  {
    ComplexType eav = 0, ov = 0;
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      eav += it->get_property(WEIGHT) * (it->energy());
      ov += it->get_property(WEIGHT);
    }
    app_log(1," Initial Energy: {}", (eav / ov).real()); 
  }
  double tot_time = 0;
  RealType dt     = 0.01;
  RealType Eshift = std::abs(wset[0].get_property(OVLP));
  if(type != COLLINEAR_FT and type != NONCOLLINEAR_FT){
    for (int i = 0; i < 10; i++)
    {
      prop.Propagate(wset, Eshift, dt);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += it->get_property(WEIGHT) * (it->energy());
        ov += it->get_property(WEIGHT);
      }
      tot_time += dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
    for (int i = 0; i < 10; i++)
    {
      prop.Propagate(wset, Eshift, 2 * dt);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += it->get_property(WEIGHT) * (it->energy());
        ov += it->get_property(WEIGHT);
      }
      tot_time += 2 * dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
  } 
  else {

    //int ntau_test = 400; 
    //memory::array<MEM,ComplexType,2> X(ntau_test,NMO);
    //nda::array<ComplexType,2> X(ntau_test,NMO);

    //X() = ComplexType(0.0);

    //read_fields_from_hdf<ComplexType>("Aux_Fields.h5", X);

    //for(int i = 0; i < 5; ++i){
    //  for(int j = 0; j < NMO; ++j){
    //    std::cout<<X(i,j)<<" ";
    //  }
    //  std::cout<<std::endl;
    //}

    if(readFields) prop.read_fields_from_hdf("Aux_Fields.h5");

    dt = 0.099;
    int ntau_test = 10;
    ComplexType mu = 0.5;
    for(int i = 0; i < 3; i++)
    {
      prop.Propagate(wset, Eshift, dt, i+1);
      //std::cout<<"UR DR VR before stabilization\n";
      //auto URtmp = wset.UMatrices(Alpha);
      //auto DRtmp = wset.DMatrices(Alpha);
      //auto VRtmp = wset.VMatrices(Alpha);
      //for(int j = 0; j < NMO; ++j)
      //  for(int k = 0; k < NMO; ++k)
      //    std::cout<<URtmp(0,j,k)<<"  "<<DRtmp(0,k)<<"  "<<VRtmp(0,j,k)<<std::endl;
      prop.Orthogonalize(wset);
      //std::cout<<"UR DR VR after stabilization\n";
      //URtmp = wset.UMatrices(Alpha);
      //for(int j = 0; j < NMO; ++j)
      //  for(int k = 0; k < NMO; ++k)
      //    std::cout<<URtmp(0,j,k)<<"  "<<DRtmp(0,k)<<"  "<<VRtmp(0,j,k)<<std::endl;
      tot_time += dt;
      memory::buffered_array<MEM,ComplexType,1> n_avg(nwalk,ComplexType(0.0));
      ComplexType ntot_avg = 0;
      wfn.Energy(wset, n_avg, i+1);
      //for(int n = 0; n < nwalk; ++n) ntot_avg += n_avg(n);
      //std::cout<<"ntot_avg = "<<ntot_avg<<std::endl;
      ComplexType ehubb =0, eav = 0, ov = 0;
      int n = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        ehubb += it->get_property(WEIGHT) * (it->get_property(EJ_));
        eav += it->get_property(WEIGHT) * (it->energy());
        ntot_avg += it->get_property(WEIGHT) * n_avg(n);
        ov += it->get_property(WEIGHT);
        n++;
      }
      //app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      //app_log(1," -- {}  {}  {}",i,tot_time,((eav + mu*ntot_avg) / ov).real());
      app_log(1," -- {}  {}  {}  {}",i,tot_time,(ehubb / ov).real(), ((eav + mu*ntot_avg) / ov).real());
      //prop.Orthogonalize(wset);
      //tot_time += dt;
    }
  }
std::cout<<" setup: " <<AFQMCTimer.elapsed(setup_timer) <<std::endl;
  if(mpi->comm.root()) AFQMCTimer.print_all();
}

TEST_CASE("propg_fac", "[propagator_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  if (UTEST_HAMIL!="" and UTEST_WFN!="") {
    app_log(0,"Propagator factory unit testing. Running user provided test:");
    app_log(0," Hamiltonian: {}", UTEST_HAMIL);
    app_log(0," Wavefunction: {}", UTEST_WFN);
    propg_fac<HOST_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN,false);
    propg_fac<HOST_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN,true);
#if defined(ENABLE_DEVICE)
    propg_fac<DEVICE_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN,false);
    propg_fac<DEVICE_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN,true);
#endif
  } else {
    app_log(0,"Propagator factory unit testing. Running standard tests.");
    bool finiteT = true;
    auto files = utils::get_unit_tests_files(true,true,true,true,false,finiteT);
    for( auto f : files ) {
      try {
        propg_fac<HOST_MEMORY>(mpi,std::get<0>(f),std::get<1>(f),false);
      } catch (const sfqmc::AppAbortException& e) {
        FAIL_CHECK("APP_ABORT in propg_fac<HOST_MEMORY>(" << std::get<0>(f) << ", dense=false): " << e.what());
      }
      try {
        propg_fac<HOST_MEMORY>(mpi,std::get<0>(f),std::get<1>(f),true);
      } catch (const sfqmc::AppAbortException& e) {
        FAIL_CHECK("APP_ABORT in propg_fac<HOST_MEMORY>(" << std::get<0>(f) << ", dense=true): " << e.what());
      }
#if defined(ENABLE_DEVICE)
      try {
        propg_fac<DEVICE_MEMORY>(mpi,std::get<0>(f),std::get<1>(f),false);
      } catch (const sfqmc::AppAbortException& e) {
        FAIL_CHECK("APP_ABORT in propg_fac<DEVICE_MEMORY>(" << std::get<0>(f) << ", dense=false): " << e.what());
      }
      try {
        propg_fac<DEVICE_MEMORY>(mpi,std::get<0>(f),std::get<1>(f),true);
      } catch (const sfqmc::AppAbortException& e) {
        FAIL_CHECK("APP_ABORT in propg_fac<DEVICE_MEMORY>(" << std::get<0>(f) << ", dense=true): " << e.what());
      }
#endif
    }
  }
}


} // namespace sfqmc
