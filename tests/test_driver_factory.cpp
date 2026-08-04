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

#undef NDEBUG

#include "catch2/catch_test_macros.hpp"
  
#include "config.h"
#include "IO/app_loggers.h"
#include "AFQMC/parameters.hpp"
#include "utilities/Random.hpp"
#include "utilities/Timer.hpp"
#include "test_common.hpp"
#include "utilities/check.hpp"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"
#include "numerics/sparse/sparse.hpp"
  
#include "test_utils.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/Utilities/AFQMCTimer.h"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Drivers/DriverFactory.h"


extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void driver_factory_build(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file,
             WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE, bool finiteT = false)
{
  HamiltonianFactory HamFac;
  WalkerSetFactory<MEM> WSetFac;
  WavefunctionFactory<MEM> WfnFac{};
  PropagatorFactory<MEM> PropFac;
  DriverFactory<MEM> DriverFac(mpi, WSetFac, PropFac, WfnFac, HamFac);

  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  WfnFac.push("wfn0", WavefunctionParameters{.name = "wfn0", .filename = wfn_file});
  PropFac.push("prop0", PropagatorParameters{.name = "prop0"});

  WalkerSetParameters wlk_full{.name = "wlk0"};

  const WavefunctionParameters wfn_min{.filename = wfn_file};
  const HamiltonianParameters ham_min{.filename = hamil_file};
  WalkerSetParameters wlk_min{.max_weight = 4.0};

  // KE: Some special walker_types must match the wavefunction type;
  //     if an explicit walker type is provided to this test, use it!
  if (walker_type != UNDEFINED_WALKER_TYPE) {
    wlk_full.walker_type = walker_type;
    wlk_min.walker_type = walker_type;
  }

  WSetFac.push("wlk0", wlk_full);

  const PropagatorParameters prop_min{.hybrid = true};

  // Fix the seed so the test is reproducible.
  constexpr int test_seed = 463;

  const bool default_walker = (walker_type == UNDEFINED_WALKER_TYPE);

  ExecuteParameters exec{};

  if (default_walker) {
    // wfn only - this is invalid unless wfn file and hamil file are the same
    if (hamil_file == wfn_file) {
      exec = ExecuteParameters{.wavefunction = wfn_min, .seed = test_seed};
      app_log(0,"[driver_factory] TEST: wfn only (inline); walker_type={}", walkerTypeToString(walker_type));
      CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));
    }

    // wfn and ham
    exec = ExecuteParameters{.wavefunction = wfn_min, .hamiltonian = ham_min, .seed = test_seed};
    app_log(0,"[driver_factory] TEST: wfn+ham (inline); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));

    // wfn, ham, prop
    exec = ExecuteParameters{
        .wavefunction = wfn_min, .hamiltonian = ham_min, .propagator = prop_min, .seed = test_seed};
    app_log(0,"[driver_factory] TEST: wfn+ham+prop (inline); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));
  }

  // wfn, ham, prop, wlk
  exec = ExecuteParameters{.walker_set = wlk_min,
                           .wavefunction = wfn_min,
                           .hamiltonian = ham_min,
                           .propagator = prop_min,
                           .seed = test_seed};
  app_log(0,"[driver_fac] TEST: wfn+ham+prop+wlk (all inline); walker_type={}", walkerTypeToString(walker_type));
  if(finiteT)
    CHECK(DriverFac.executeDriver(DriverType::ftafqmc,"drv_test",0,exec));
  else
    CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));

  if (default_walker) {
    // external wfn
    if (hamil_file == wfn_file) {
      exec = ExecuteParameters{.wavefunction = std::string{"wfn0"}, .seed = test_seed};
      app_log(0,"[driver_factory] TEST: wfn only (external); walker_type={}", walkerTypeToString(walker_type));
      CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));
    }

    // wfn and ham
    exec = ExecuteParameters{
        .wavefunction = std::string{"wfn0"}, .hamiltonian = std::string{"ham0"}, .seed = test_seed};
    app_log(0,"[driver_factory] TEST: wfn+ham (external); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));

    // wfn, ham, prop
    exec = ExecuteParameters{.wavefunction = std::string{"wfn0"},
                             .hamiltonian = std::string{"ham0"},
                             .propagator = std::string{"prop0"},
                             .seed = test_seed};
    app_log(0,"[driver_factory] TEST: wfn+ham+prop (external); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));
  }

  // wfn, ham, prop, wlk (all external)
  exec = ExecuteParameters{.walker_set = std::string{"wlk0"},
                           .wavefunction = std::string{"wfn0"},
                           .hamiltonian = std::string{"ham0"},
                           .propagator = std::string{"prop0"},
                           .seed = test_seed};
  app_log(0,"[driver_fac] TEST: wfn+ham+prop+wlk (all external); walker_type={}", walkerTypeToString(walker_type));
  if(finiteT)
    CHECK(DriverFac.executeDriver(DriverType::ftafqmc,"drv_test",0,exec));
  else
    CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));

  // mixed external internal
  exec = ExecuteParameters{
      .walker_set = std::string{"wlk0"}, .wavefunction = wfn_min, .seed = test_seed};
  if (hamil_file == wfn_file) {
    app_log(0,"[driver_fac] TEST: wfn(inline)+wlk(external); walker_type={}", walkerTypeToString(walker_type));
    if(finiteT)
      CHECK(DriverFac.executeDriver(DriverType::ftafqmc,"drv_test",0,exec));
    else
      CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));
  }

  if (default_walker) {
    exec = ExecuteParameters{
        .wavefunction = wfn_min, .hamiltonian = std::string{"ham0"}, .seed = test_seed};
    app_log(0,"[driver_factory] TEST: wfn(inline)+ham(external); walker_type={}", walkerTypeToString(walker_type));
    CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));
  }

  exec = ExecuteParameters{.walker_set = std::string{"wlk0"},
                           .wavefunction = std::string{"wfn0"},
                           .hamiltonian = ham_min,
                           .seed = test_seed};
  app_log(0,"[driver_fac] TEST: wfn(external)+ham(inline)+wlk(external); walker_type={}", walkerTypeToString(walker_type));
  if(finiteT)
    CHECK(DriverFac.executeDriver(DriverType::ftafqmc,"drv_test",0,exec));
  else
    CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));

  exec = ExecuteParameters{.walker_set = wlk_min,
                           .wavefunction = std::string{"wfn0"},
                           .hamiltonian = ham_min,
                           .seed = test_seed};
  app_log(0,"[driver_fac] TEST: wfn(external)+ham(inline)+wlk(inline); walker_type={}", walkerTypeToString(walker_type));
  if(finiteT)
    CHECK(DriverFac.executeDriver(DriverType::ftafqmc,"drv_test",0,exec));
  else
    CHECK(DriverFac.executeDriver(DriverType::afqmc,"drv_test",0,exec));

  // many more possibilities (combinatorial...) Add any problematic ones if needed
}

TEST_CASE("driver_factory: build", "[driver_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  
  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES walker_type, bool finiteT) {
    driver_factory_build<MEM>(mpi, hamil_file, wfn_file, walker_type, finiteT);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::FINITE_T | TestFiles::ALL_SYSTEMS);
}


} // namespace sfqmc
