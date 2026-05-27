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

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "AFQMC/config.h"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "IO/app_loggers.h"
#include "test_common.hpp"
#include "utilities/mpi_context.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "utilities/Timer.hpp"
#include "test_utils.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "numerics/sparse/sparse.hpp"

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
void ham_factory(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                 std::string hamil_file) 
{
  utils::check(utils::file_exists(hamil_file), 
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);

  int NMO = read_nmo_from_hdf(hamil_file);
  CHECK(NMO > 0);
  int nup=1, ndown=1;

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  [[maybe_unused]] Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");
}

TEST_CASE("ham_factory", "[hamiltonian_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    ham_factory<MEM>(mpi, hamil_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::PHMSD | TestFiles::FINITE_T | TestFiles::ALL_SYSTEMS);
}


} // namespace sfqmc
