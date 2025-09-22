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

#include "catch_amalgamated.hpp"

#include "config.h"
#include "Utilities/AppAbort.hpp"

#include "hdf/hdf_archive.h"
#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "Utilities/app_loggers.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Utilities/myTimer.h"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"
#include "AFQMC/Utilities/test_utils.hpp"

#include "SparseMatrix/csr_matrix_construct.hpp"
#include "Numerics/ma_blas.hpp"

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

template<class Allocator>
void ham_factory(boost::mpi3::communicator& world)
{
  using pointer = device_ptr<ComplexType>;
  if (not file_exists(UTEST_HAMIL))
  {
    APP_ABORT(" Hamiltonian file not found. Run unit test with --hamil /path/to/hamil.h5.");
  }
  else
  {
    // Global Task Group
    GlobalTaskGroup gTG(world);

    int NMO, NAEA, NAEB;
    std::tie(NMO, NAEA, NAEB) = read_info_from_hdf(UTEST_HAMIL);
    REQUIRE(NAEA == NAEB);

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));
    HamiltonianFactory HamFac(InfoMap);
    std::string hamil_xml = "<Hamiltonian name=\"ham0\" info=\"info0\"> \
<parameter name=\"filetype\">hdf5</parameter> \
<parameter name=\"filename\">" +
        UTEST_HAMIL + "</parameter> \
<parameter name=\"cutoff_cholesky\">1e-5</parameter> \
</Hamiltonian> \
";
    InputParser parser;
    ptree pt = parser.parse(hamil_xml, "xml");
    __app_log()__(1, "\nHamiltonian input:\n{}\n", to_string(pt));
    std::string ham_name("ham0");
    HamFac.push(ham_name, pt.get_child("Hamiltonian"));

    Hamiltonian& ham_ = HamFac.getHamiltonian(gTG, ham_name);
  }
}

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

    int NMO, NAEA, NAEB;
    std::tie(NMO, NAEA, NAEB) = read_info_from_hdf("./afqmc_timing.h5");

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));
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
TEST_CASE("ham_factory", "[hamiltonian_factory]")
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

  ham_factory<Alloc>(world);
}

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

} // namespace sfqmc
