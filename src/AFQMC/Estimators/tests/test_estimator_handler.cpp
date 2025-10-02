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

#include "catch_amalgamated.hpp"

#include "config.h"
#include "Utilities/AppAbort.hpp"

#include "io/ptree/ptree_utilities.hpp"
#include "hdf/hdf_archive.h"

#include <stdio.h>
#include <string>

#include "AFQMC/config.h"
#include "SparseMatrix/tests/matrix_helpers.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/EstimatorHandler.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Estimators/BackPropagatedEstimator.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "Memory/buffer_managers.h"
#include "Memory/device_rng.hpp"

using std::complex;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<bool MP, class Allocator>
void measure_schedule(boost::mpi3::communicator& world)
{

int population_control_interval = 10;

// 1. setup:

//using pointer = typename Allocator::pointer;

if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
{
  app_log(1," Skipping ham_ops_basic_serial. Hamiltonian or wavefunction file not found. ");
  app_log(1," Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
}
else
{
  // Global Task Group
  afqmc::GlobalTaskGroup gTG(world); // make a debug task for this test!
  
  // set up the TG handler for this test
  int ncores = 1;
  auto& node(gTG.Node());
  #if defined(ENABLE_DEVICE)
      // check ncores 
      if(ncores != 1) {
          app_warning(" Warning: Only ncores=1 allowed in device build. Setting to 1.");
        ncores = 1;
      }
  #else
      ncores = std::max(std::min(ncores, node.size()), 1);
  #endif

  TaskGroupHandler TGHandler(gTG,ncores);

  hdf_archive dump;
  if (!dump.open(UTEST_HAMIL, H5F_ACC_RDONLY))
  {
    APP_ABORT(" Error opening integral file in SparseGeneralHamiltonian. ");
  }
  auto format = get_hamiltonian_format(dump,gTG.Global());
  dump.close();

  int NMO, wfn_NMO, NAEA, NAEB;
  NMO = read_nmo_from_hdf(UTEST_HAMIL,format);
  std::tie(wfn_NMO,NAEA, NAEB) = read_info_from_wfn(UTEST_WFN, "any");
  CHECK(NMO == wfn_NMO);

  utils::RandomGenerator_t rng;
  auto rng_dev = utils::make_device_rng(777);
  auto TG = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",UTEST_HAMIL);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(gTG, "ham0");

  WALKER_TYPES type                = afqmc::getWalkerType(UTEST_WFN);
  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("system","info0");
  if(type == CLOSED) wlk_pt.put("walker_type","closed");
  else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
  else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
  else if(type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
  WalkerSet wset(TG, wlk_pt, InfoMap["info0"], &rng);

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",UTEST_WFN);
  wfn_pt.put("dense_trial","yes");

  //Allocator alloc_(make_localTG_allocator<ComplexType>(TG));
  int nwalk = 1; // choose prime number to force non-trivial splits in shared routines
  WavefunctionFactory WfnFac(InfoMap, MP);
  WfnFac.push("wfn0", wfn_pt);
  Wavefunction& wfn = WfnFac.getWavefunction(TG, TG, "wfn0", type, &ham, 1e-6, nwalk);

  ptree prop_pt;
  prop_pt.put("name","prop0");
  prop_pt.put("system","info0");

  PropagatorFactory PropgFac(InfoMap, MP);
  PropgFac.push("prop0", prop_pt);
  Propagator& prop = PropgFac.getPropagator(TG, "prop0", wfn, &rng_dev);

  auto initial_guess = WfnFac.getInitialGuess("wfn0");
  REQUIRE(initial_guess.size(0) == 2);
  REQUIRE(initial_guess.size(1) == NMO);
  REQUIRE(initial_guess.size(2) == NAEA);
  wset.resize(nwalk, initial_guess[0], initial_guess[0]);
  //using EstimPtr = std::shared_ptr<EstimatorBase>;
  
  // number of steps to propagate
  int nStep = 200;

  // define / run test cases
  std::vector<ptree> cases;

  ptree test_case;

  test_case.put("name", "case1");
  test_case.put("meas1", 5);
  test_case.put("meas2", 5);
  test_case.put("gcd", 5*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();

  test_case.put("name", "case2");
  test_case.put("meas1", 5);
  test_case.put("meas2", 10);
  test_case.put("gcd", 5*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();

  test_case.put("name", "case3");
  test_case.put("meas1", 5);
  test_case.put("meas2", 20);
  test_case.put("gcd", 5*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();

  test_case.put("name", "case4");
  test_case.put("meas1", 20);
  test_case.put("meas2", 10);
  test_case.put("gcd", 10*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();
  
  test_case.put("name", "case5");
  test_case.put("meas1", 5);
  test_case.put("meas2", 100);
  test_case.put("gcd", 5*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();
  
  test_case.put("name", "case6");
  test_case.put("meas1", 5);
  test_case.put("meas2", 7);
  test_case.put("gcd", 1*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();
    
  test_case.put("name", "case7");
  test_case.put("meas1", 11);
  test_case.put("meas2", 7);
  test_case.put("gcd", 1*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();
  
  test_case.put("name", "case8");
  test_case.put("meas1", 47);
  test_case.put("meas2", 99);
  test_case.put("gcd", 1*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();
  
  test_case.put("name", "case9");
  test_case.put("meas1", 201);
  test_case.put("meas2", 7);
  test_case.put("gcd", 1*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();
  
  // test that we default properly
  test_case.put("name", "case10");
  test_case.put("meas1", 1);  // This is what energy estimator will actually use
  test_case.put("meas2", 7); // noncommensurate on purpose
  test_case.put("gcd", 1*population_control_interval);
  cases.push_back(test_case);
  test_case.clear();

  ptree one_rdm;
  one_rdm.put("name","one_rdm");

  for (auto test_ptree: cases)
  {
    //TODO update the PropertyTee for our test case(s) using new parameter names
    ptree est_pt_energy;
    est_pt_energy.put("name","energy");
    est_pt_energy.put("overwrite",true);

    
    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt_energy));

    ptree est_pt_bp;
    est_pt_bp.put("name","back_propagation");
    est_pt_bp.put("measure_interval_multiplier",test_ptree.get<int>("meas2"));
    est_pt_bp.put("equil_multiplier",0);
    est_pt_bp.put("bp_walker_ortho_interval",1);
    est_pt_bp.add_child("onerdm",one_rdm);

    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt_bp));

    ptree est_pt;
    est_pt.add_child("estimator",est_pt_energy);
    est_pt.add_child("estimator",est_pt_bp);
    est_pt.put("population_control_interval",population_control_interval);
    // Set the global measure_interval_multiplier that will be used by BasicEstimator and EnergyEstimator
    est_pt.put("measure_interval_multiplier",test_ptree.get<int>("meas1"));

    // to verify the ptree
    std::cout <<" Test case Ptree:  "<< std::endl;
    std::cout << io::to_string(est_pt) << std::endl;

    int measure_interval;
    //int estimator1_calls = 0;
    //int estimator2_calls = 0;
    int estimator_handler_querries = 0;
    {
      int nPopulation = 1;
      float dt = 0.01f;
      float total_time = 0.0f;
      double E1 = 0.0;
      EstimatorHandler estim0(TGHandler, InfoMap["info0"], "test_est_handler",
        est_pt, wset, WfnFac, wfn, prop,
                          type, HamFac, "ham0", dt);
    
      // set measurement intervals
      measure_interval = estim0.get_max_common_interval();
      std::cout << "Querrying estimator handler with interval " << measure_interval << " (commensurate with all measurement intervals)" << std::endl;

      /* Fake Driver Block */
      std::vector<ComplexType> dummyData;
      //wset.popControl(dummyData);

      for (int iStep = 0; iStep < nStep; ++iStep)
      {
        prop.Propagate(1, wset, E1, dt, false);
        total_time += dt;

        if (total_time < 1.0 || (iStep + 1) % nPopulation == 0 || iStep == 0)
        {
          AFQMCTimer.start(popcont_timer);
          wset.processWalkerData(dummyData);
          wset.popControl(); // make this a call to actual pop control
          AFQMCTimer.stop(popcont_timer);
          estim0.accumulate_step(total_time, wset, dummyData);
        }

        if ((iStep + 1) % measure_interval == 0 )
        {
          estim0.accumulate_block(total_time, wset);
          estim0.print(iStep + 1, total_time, E1, wset);
          estimator_handler_querries++;
        }
      }
    
    }
    int expected_estimator_querries = nStep / test_ptree.get<int>("gcd");
    // Energy estimator uses meas1 as the global measure_interval_multiplier
    int energy_interval = test_ptree.get<int>("meas1") * population_control_interval;
    int expected_measurments = nStep / energy_interval;
    // read results from "test_est_handler.scalar.dat"
    std::string filename = "test_est_handler.scalar.dat";
    std::ifstream in(filename.c_str());
    if (not in.good())
    {
      app_error(" Error opening file in test_est_handler.scalar.dat. \n");
      APP_ABORT("");
    }
    int line_count = -1; // first line is header
    std::string line;
    while (std::getline(in, line))
    {
      line_count++;
      cout << line << endl;
    }
    app_log(1, "\n[TESTS] Running test case: {} \n",test_ptree.get<std::string>("name","no name"));
    CHECK(measure_interval == test_ptree.get<int>("gcd"));
    CHECK(estimator_handler_querries == expected_estimator_querries);
    CHECK(line_count == expected_measurments);
    in.close();
  }
}
}

// 2. run cases.
TEST_CASE("measure_schedule", "[estimators]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_AFQMC_timer(); 
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);
  measure_schedule<false, Alloc>(world);
  measure_schedule<true, Alloc>(world);
  release_memory_managers();
}
}
