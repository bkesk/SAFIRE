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

#include <iostream>
#include <vector>
#include <stdexcept>
#include <stack>
#include "cxxopts.hpp"
#include "io/ptree/InputParser.hpp"
#include "Utilities/app_loggers.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "Utilities/app_version.h"
#include "Memory/arch.hpp"

#include "AFQMC/AFQMCFactory.h"


/*
 * pyscf : {
 *
 * }
 *
 * dice : {
 *
 * }
 *
 * qe : {
 *   qe driver input
 * }
 *
 * pw2posthf : {
 *  pw2posthf input
 * }
 *
 * mp2 : {
 *
 * }
 *
 * rpa : {
 *
 * }
 *
 * crpa : {
 *
 * }
 *
 * afqmc : {
 *   afqmc input file
 * }
 *
 * *** execution blocks are processed sequentially, so order is important.
 *     Communication between blocks occurs though appropriate hdf5 I/O. *** 
 */ 

/** @file safire.cpp
 */
int main(int argc, char** argv)
{
  mpi3::environment env(argc, argv);
  auto world = mpi3::environment::get_world_instance();
  bool root(world.root());

  // parse command line inputs
  std::vector<std::string> inputs;
  int output_level, debug_level; 
  //if (world.root()) // everyone parse input file (e.g. hdf can be read in parallel)
  { // parse command line inputs
    cxxopts::Options options(argv[0],"SAFIRE");
    options
      .positional_help("[optional args]")
      .show_positional_help();
    options.add_options()
      ("h,help", "print help message")
      ("v,version", "print version message")
      ("verbosity", "0, 1, 2, ...: higher means more", cxxopts::value<int>()->default_value("2"))
      ("debug", "0, 1, 2, ...: higher means more", cxxopts::value<int>()->default_value("0"))
      ("filenames", "input filenames", cxxopts::value<std::vector<std::string>>())
    ;
    options.parse_positional({"filenames"});
    auto args = options.parse(argc, argv);
    // record program options
    if (args.count("help"))
    {
      if(root)
        std::cout << options.help({"", "Group"}) << std::endl;
      mpi3::environment::finalize();
      exit(0);
    }
    output_level = args["verbosity"].as<int>();
    // check program options
    if (output_level < 0) 
    {
      std::cerr << "verbosity < 0: " << output_level << std::endl;
      mpi3::environment::finalize();
      exit(1);
    }
    debug_level = args["debug"].as<int>();
    if (debug_level < 0) 
    {
      std::cerr << "debug < 0: " << debug_level << std::endl;
      mpi3::environment::finalize();
      exit(1);
    }
    if (args.count("version"))
    {
      if(root){
        // we can't access help string, annoying so hard code this
        std::cerr << options.program() << " | SAFIRE" << std::endl;
        print_version(3);  // print everything
      }
      mpi3::environment::finalize();
      exit(0);
    }

    // input files are positional arguments
    int nfile = args.count("filenames");
    if (nfile < 1)
    {
      if(root)
        std::cout << "no input file given; exiting ..." << std::endl;
      mpi3::environment::finalize();
      exit(1);
    } else {
      inputs = args["filenames"].as<std::vector<std::string>>();
    }
  }

  // setup output loggers
  setup_loggers(world.root(), output_level, debug_level);

  std::string welcome(
    std::string("") + 
              "███████╗ █████╗ ███████╗██╗██████╗ ███████╗\n" + 
              "██╔════╝██╔══██╗██╔════╝██║██╔══██╗██╔════╝\n" + 
              "███████╗███████║█████╗  ██║██████╔╝█████╗  \n" + 
              "╚════██║██╔══██║██╔══╝  ██║██╔══██╗██╔══╝  \n" + 
              "███████║██║  ██║██║     ██║██║  ██║███████╗\n" + 
              "╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝\n" +
              "\n");
  app_log(1, welcome);      

  if(root)
    print_version(output_level);

  // !!!! assume a single input for now
  std::string myinput = inputs[0];
  InputParser parser;
  try {
    parser.read(myinput);
  } catch (std::exception const& e) {
    app_error("Error parsing input file. Check format.");
    mpi3::environment::finalize();
    exit(1);	
  }

  for(auto const& it : parser.get_root())
  { // go through all simulation requests
    std::string cname = it.first;
    if (cname == "afqmc") {
      ptree sim = it.second;
      sfqmc::afqmc::AFQMCFactory afqmc_fac("afqmc", world, sim);
    } else if(cname == "legacy_afqmc") { // TEMP: to test against
      ptree sim = it.second;
      sfqmc::afqmc::AFQMCFactory afqmc_fac("legacy_afqmc", world, sim);
    } else if(cname == "cs_afqmc" || cname == "csafqmc") {
      ptree sim = it.second;
      int n_groups = sim.get<int>("project.n_groups", 1);
      sfqmc::afqmc::AFQMCFactory afqmc_fac("csafqmc",world, 
						  sim,n_groups);
    } else {
      app_error("unknown calculation type: {} \n",cname.c_str());
      mpi3::environment::finalize();
      exit(1);	
    }
  } // simulations end

  return 0;
}

