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
#include "IO/ptree/InputParser.hpp"
#include "IO/app_loggers.h"
#include "IO/AppAbort.hpp"

#include "config.h"
#include "configuration.hpp"

#include "utilities/check.hpp"
#include "arch/arch.h"
#include "utilities/mpi_context.h"
#include "utilities/app_version.h"

#include "AFQMC/AFQMCFactory.h"

/*
 * *** execution blocks are processed sequentially, so order is important.
 *     Communication between blocks occurs though appropriate hdf5 I/O. *** 
 */ 

/** @file safire.cpp
 */
int main_impl(int argc, char** argv)
{
  using namespace sfqmc;
  auto world = mpi3::environment::get_world_instance();
  bool root(world.root());
  std::string compute="default";

  constexpr const char *welcome{
    "███████╗ █████╗ ███████╗██╗██████╗ ███████╗\n"
    "██╔════╝██╔══██╗██╔════╝██║██╔══██╗██╔════╝\n"
    "███████╗███████║█████╗  ██║██████╔╝█████╗  \n"
    "╚════██║██╔══██║██╔══╝  ██║██╔══██╗██╔══╝  \n"
    "███████║██║  ██║██║     ██║██║  ██║███████╗\n"
    "╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝\n"
    "\n"};

  // parse command line inputs
  std::vector<std::string> inputs;
  int output_level, debug_level; 

  // parse command line inputs
  cxxopts::Options options(argv[0],"SAFIRE");
  options
    .positional_help("[optional args]")
    .show_positional_help();
  options.add_options()
    ("h,help", "print help message")
    ("v,version", "print version message")
    ("verbosity", "0, 1, 2, 3: higher means more", cxxopts::value<int>()->default_value("2"))
    ("debug", "0, 1, 2, 3: higher means more", cxxopts::value<int>()->default_value("0"))
    ("compute", "where to execute: cpu, gpu, default", cxxopts::value<std::string>()->default_value("default"))
    ("filenames", "input filenames", cxxopts::value<std::vector<std::string>>())
  ;
  options.parse_positional({"filenames"});

  cxxopts::ParseResult args{};
  try {
    args = options.parse(argc, argv);
  } catch(const cxxopts::exceptions::exception &e) {
    throw AppAbortException{e.what()};
  }

  // record program options
  if (args.count("help"))
  {
    if(root)
      std::cout << options.help({"", "Group"}) << std::endl;
    return 0;
  }
  output_level = args["verbosity"].as<int>();
  // check program options
  if (output_level < 0) 
  {
    throw AppAbortException{std::format("verbosity ({}) has to be >= 0!", output_level)};
  }
  debug_level = args["debug"].as<int>();
  if (debug_level < 0) 
  {
    throw AppAbortException{std::format("debug ({}) has to be >= 0!", output_level)};
  }
  if (args.count("version"))
  {
    if(root) {
      std::cout << welcome;
      print_version(3);  // print everything
    }
    return 0;
  }
  compute = args["compute"].as<std::string>();
  if (compute != "cpu" and compute != "gpu" and compute != "default")
  {
    throw AppAbortException{std::format("Invalid compute: {} (allowed values: \"cpu\", \"gpu\", \"default\")", compute)};
  }

  // input files are positional arguments
  int nfile = args.count("filenames");
  if (nfile < 1)
  {
    throw AppAbortException{"no input file given; exiting ..."};
  } else {
    inputs = args["filenames"].as<std::vector<std::string>>();
  }

  // setup output loggers
  sfqmc::arch::init(root,output_level,debug_level);

  app_log(1, welcome);      

  if(root)
    print_version(output_level);

  // !!!! assume a single input for now
  std::string myinput = inputs[0];
  InputParser parser;
  try {
    parser.read(myinput);
  } catch (std::exception const& e) {
    throw AppAbortException("Error parsing input file. Check format.");
  }

  auto mpi = std::make_shared<utils::mpi_context_t<boost::mpi3::communicator>>(utils::make_mpi_context(world));

  for(auto const& it : parser.get_root())
  { // go through all simulation requests
    std::string cname = it.first;
    if (cname == "afqmc") {
      ptree sim = it.second;
#if defined(ENABLE_DEVICE)
      if(compute=="gpu" or compute=="default") { 
        sfqmc::arch::check_device_configuration();
        auto afqmc_fac = afqmc::AFQMCFactory<DEVICE_MEMORY>("afqmc", mpi, sim);
      } else 
#endif
        auto afqmc_fac = afqmc::AFQMCFactory<HOST_MEMORY>("afqmc", mpi, sim);
    } else if(cname == "cs_afqmc" || cname == "csafqmc") {
      ptree sim = it.second;
      int n_groups = sim.get<int>("project.n_groups", 1);
// need new strategy for n_group>1, need to add a new "global" communicator to the context.
#if defined(ENABLE_DEVICE)
      if(compute=="gpu" or compute=="default") { 
        sfqmc::arch::check_device_configuration();
        auto afqmc_fac = afqmc::AFQMCFactory<DEVICE_MEMORY>("csafqmc",mpi,sim,n_groups);
      } else 
#endif
        auto afqmc_fac = afqmc::AFQMCFactory<HOST_MEMORY>("csafqmc",mpi,sim,n_groups);
    } else {
      app_error("unknown calculation type: {} \n",cname.c_str());
      throw sfqmc::AppAbortException("APP_ABORT triggered");
    }
  } // simulations end
  return 0;
}

int main(int argc, char** argv) {
  mpi3::environment env(argc, argv);
  try {
    return main_impl(argc, argv);
  } catch (const sfqmc::AppAbortException& e) {
    if(env.world().root()) {
      std::cerr << fmt::format("Error: {}\n", e.what());
    }
    env.world().abort(1);
    return 1;
  } 
  // destructor of env calls MPI finalize
}
