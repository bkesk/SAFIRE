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

#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include "config.h"
#include "arch/arch.h"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include<iostream>

namespace sfqmc::utils::detail {
  // gets allocated in utilities/test_common.hpp when requested, cleanup below before exiting main
  std::shared_ptr<sfqmc::utils::mpi_context_t<boost::mpi3::communicator>> __unit_test_mpi_context__ = nullptr;
}

// input files 
std::string UTEST_HAMIL, UTEST_WFN;

int main(int argc, char* argv[])
{
  boost::mpi3::environment env(argc, argv);
  auto world = boost::mpi3::environment::get_world_instance();

  int output_level=2, debug_level=2;
  if(const char* env_p = std::getenv("OUTPUT_LEVEL")) {
    output_level = std::atoi(env_p);
    if(output_level < 0) output_level=2;
    if(output_level > 5) output_level=2;
  }
  if(const char* env_p = std::getenv("DEBUG_LEVEL")) {
    debug_level = std::atoi(env_p);
    if(debug_level < 0) debug_level=2;
    if(debug_level > 5) debug_level=2;
  }
  sfqmc::arch::init(world.root(),output_level,debug_level);

  Catch::Session session;
  using namespace Catch::clara;
  // Build command line parser.
  auto cli = session.cli() |
      Opt(UTEST_HAMIL, "UTEST_HAMIL")["--hamil"]("Hamiltonian file to be used by unit test if applicable.") |
      Opt(UTEST_WFN, "UTEST_WFN")["--wfn"]("Wavefunction file to be used by unit test if applicable.");
  session.cli(cli);

  // Parse arguments.
  int parser_err = session.applyCommandLine(argc, argv);
  if( parser_err != 0 ) // Indicates a command line error
      return parser_err; 

  // Run the tests.
  int result = session.run(argc, argv);

  // cleanup mpi context 
  sfqmc::utils::detail::__unit_test_mpi_context__.reset();

  return result;
}
