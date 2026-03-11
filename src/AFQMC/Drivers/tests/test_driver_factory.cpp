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
#include "AFQMC/Utilities/AFQMCTimer.h"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Drivers/DriverFactory.h"

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
void driver_fac(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file)
{
  using nda::range;
  auto all = range::all;
  utils::check(utils::file_exists(hamil_file),
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);
  utils::check(utils::file_exists(wfn_file),
               " Wavefunction file not found: {}. \n Run unit test with --wfn /path/to/wfn.h5 ", wfn_file);

  std::map<std::string, AFQMCInfo> InfoMap;
  HamiltonianFactory HamFac(InfoMap);
  WalkerSetFactory<MEM> WSetFac(InfoMap);
  WavefunctionFactory<MEM> WfnFac(InfoMap);
  PropagatorFactory<MEM> PropFac(InfoMap);
  DriverFactory<MEM> DriverFac(mpi, InfoMap, WSetFac, PropFac, WfnFac, HamFac);

  const auto[NMO, nup, ndown] = read_info_from_wfn(wfn_file,"any");
  AFQMCInfo info("sys0",NMO,nup,ndown);
  InfoMap.insert(std::pair<std::string, AFQMCInfo>(info.name, info));

  ptree ham_full;
  ham_full.put("name","ham0");
  ham_full.put("system","sys0");
  ham_full.put("filename",hamil_file);
  HamFac.push("ham0", ham_full);

  ptree wfn_full;
  wfn_full.put("name","wfn0");
  wfn_full.put("system","sys0");
  wfn_full.put("filename",wfn_file);
  WfnFac.push("wfn0", wfn_full);

  ptree wlk_full;
  wlk_full.put("name","wlk0");
  wlk_full.put("system","sys0");
  WSetFac.push("wlk0", wlk_full);

  ptree prop_full;
  prop_full.put("name","prop0");
  prop_full.put("system","sys0");
  PropFac.push("prop0", prop_full);

  ptree wfn_min;
  wfn_min.put("filename",wfn_file);

  ptree ham_min;
  ham_min.put("filename",hamil_file);  

  ptree wlk_min;
  wlk_min.put("max_weight","4.0");

  ptree prop_min;
  prop_min.put("hybrid","true");

  ptree exec;

  
  exec.put_child("wavefunction",wfn_min);
  // wfn only - this is invalid unless wfn file and hamil file are the same
  if (hamil_file == wfn_file)
    REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));
  
  // wfn and ham
  exec.put_child("hamiltonian",ham_min);
  REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // wfn, ham, prop
  exec.put_child("propagator",prop_min);
  REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // wfn, ham, prop, wlk
  exec.put_child("walker_set",wlk_min);
  REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // external wfn 
  exec.clear();
  exec.put("wavefunction","wfn0");
  if (hamil_file == wfn_file)
    REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // wfn and ham
  exec.put("hamiltonian","ham0");
  REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // wfn, ham, prop
  exec.put("propagator","prop0");
  REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // wfn, ham, prop, wlk
  exec.put("walker_set","wlk0");
  REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));

  // mixed external internal
  exec.clear();
  exec.put_child("wavefunction",wfn_min);
  exec.put("walker_set","wlk0");
  if (hamil_file == wfn_file)
    REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));  

  exec.clear();
  exec.put_child("wavefunction",wfn_min);
  exec.put("hamiltonian","ham0");
  REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));  

  exec.clear();
  exec.put("wavefunction","wfn0");
  exec.put_child("hamiltonian",ham_min);
  exec.put("walker_set","wlk0");
  REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));  

  exec.clear();
  exec.put("wavefunction","wfn0");
  exec.put_child("hamiltonian",ham_min);
  exec.put_child("walker_set",wlk_min);
  REQUIRE(DriverFac.executeDriver("afqmc","drv_test",0,exec));
 
  // many more possibilities (combinatorial...) Add any problematic ones if needed
}

TEST_CASE("driver_fac", "[driver_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  
  if (UTEST_HAMIL!="" and UTEST_WFN!="") {
    app_log(0,"Driver factory unit testing. Running user provided test:");
    app_log(0," Hamiltonian: {}", UTEST_HAMIL);
    app_log(0," Wavefunction: {}", UTEST_WFN);
    driver_fac<HOST_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);
#if defined(ENABLE_DEVICE)
    driver_fac<DEVICE_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);
#endif
   } else {
    app_log(0,"Driver factory unit testing. Running standard tests.");
    auto files = utils::molecule_unit_tests_files(true,true,true,true,false);
    for( auto f : files ) { 
      driver_fac<HOST_MEMORY>(mpi,std::get<0>(f),std::get<1>(f));
#if defined(ENABLE_DEVICE)
      driver_fac<DEVICE_MEMORY>(mpi,std::get<0>(f),std::get<1>(f));
#endif
    }
  }
}


} // namespace sfqmc
