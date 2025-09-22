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
#include "catch_amalgamated.hpp"

#include "config.h"

#include "mpi3/environment.hpp"
namespace mpi3 = boost::mpi3;

#include<iostream>

// input files 
std::string UTEST_HAMIL, UTEST_WFN;

int main(int argc, char* argv[])
{
  mpi3::environment env(argc, argv);
  Catch::Session session;
  using namespace Catch::Clara;
  // Build command line parser.
  auto cli = session.cli() |
      Opt(UTEST_HAMIL, "UTEST_HAMIL")["--hamil"]("Hamiltonian file to be used by unit test if applicable.") |
      Opt(UTEST_WFN, "UTEST_WFN")["--wfn"]("Wavefunction file to be used by unit test if applicable.");
  session.cli(cli);
  // Parse arguments.
  int parser_err = session.applyCommandLine(argc, argv);
  // Run the tests.
  int result = session.run(argc, argv);
  if (parser_err != 0)
  {
    return parser_err;
  }
  else
  {
    return result;
  }
}
