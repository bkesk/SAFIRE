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
#include "AFQMC/config.h"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "IO/app_loggers.h"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "utilities/Timer.hpp"
//#include "AFQMC/Utilities/readWfn.h"
//#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"

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

inline std::tuple<int, int, int> read_info_from_hdf(std::string fileName)
{
  h5::file file(fileName,'r');
  h5::group grp(file);
  auto format = get_hamiltonian_format(grp);
  if(format=="std") {
    utils::check(grp.has_subgroup("Hamiltonian"), "Missing Hamiltonian dataset.");
    h5::group hgrp = grp.open_group("Hamiltonian");
    std::vector<int> Idata(8);
    h5::h5_read(hgrp,"dims",Idata);
    return std::make_tuple(Idata[3], Idata[4], Idata[5]);
  } else if(format=="coqui") {
    int NMO=0;
    utils::check(grp.has_subgroup("System"), "Missing Hamiltonian dataset.");
    h5::group hgrp = grp.open_group("System");
    h5::h5_read_attribute(hgrp,"number_of_bands",NMO);
    return std::make_tuple(NMO,1,1);
  }
  utils::check(false, "Invalid format.");
  return std::make_tuple(0,0,0); 
}

template<MEMORY_SPACE MEM>
void ham_factory(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                 std::string hamil_file) 
{
  utils::check(utils::file_exists(hamil_file), 
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);

  int NMO=0, nup=0, ndown=0;
  std::tie(NMO, nup, ndown) = read_info_from_hdf(hamil_file);
  REQUIRE(NMO > 0);
  REQUIRE(nup > 0);
  REQUIRE(nup >= ndown);

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",UTEST_HAMIL);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  [[maybe_unused]] Hamiltonian& ham_ = HamFac.getHamiltonian(mpi, "ham0");
}
/*
template<class Alloc>
void ham_generation_timing(boost::mpi3::communicator& world)
{
  if (not file_exists("./afqmc_timing.h5"))
  {
    APP_ABORT(" afqmc_timing.h5 file not found. ");
  }
  else
  {
    // Global Task Group
    afqmc::GlobalTaskGroup gTG(world);

    int NMO, nup, ndown;
    std::tie(NMO, nup, ndown) = read_info_from_hdf("./afqmc_timing.h5");

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));
    HamiltonianFactory HamFac(InfoMap);

    const char* xml_block = "<Hamiltonian name=\"ham0\" type=\"SparseGeneral\" info=\"info0\"> \
    <parameter name=\"filetype\">hdf5</parameter> \
    <parameter name=\"version\">new</parameter> \
    <parameter name=\"filename\">./afqmc_timing.h5</parameter> \
    <parameter name=\"cutoff_cholesky\">1e-5</parameter> \
  </Hamiltonian> \
";
    InputParser parser;
    ptree pt = parser.parse(xml_block, "xml");
    __app_log()__(1, "\nHamiltonian input:\n{}\n", to_string(pt));
    std::string ham_name("ham0");
    HamFac.push(ham_name, pt.get_child("Hamiltonian"));

    myTimer Timer_;
    Timer_.start("GenTest0");
    Hamiltonian& ham = HamFac.getHamiltonian(gTG, ham_name);
    Timer_.stop("GenTest0");
    app_log(1,"\n*********************************************************************");
    app_log(1," Time to create hamiltonian in ham_generation_timing_hdf: {}",
		  Timer_.total("GenTest0"));
    app_log(1,"*********************************************************************");
  }
}
*/
TEST_CASE("ham_factory", "[hamiltonian_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  ham_factory<HOST_MEMORY>(mpi,UTEST_HAMIL);
#if defined(ENABLE_DEVICE)
  ham_factory<DEVICE_MEMORY>(mpi,UTEST_HAMIL);
#endif
}
/*
TEST_CASE("ham_generation_timing_hdf", "[hamiltonian_factory]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  auto node = world.split_shared(world.rank());

  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  auto node   = world.split_shared(world.rank());
  using Alloc = shared_allocator<ComplexType>;
#endif

  ham_generation_timing<Alloc>(world);
}
*/
} // namespace sfqmc
