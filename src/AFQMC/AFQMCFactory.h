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


// -*- C++ -*-
/**@file AFQMCFactory.h
 * @brief Top level class for AFQMC. Parses input and performs setup of classes.
 */

#ifndef SFQMC_AFQMCFACTORY_H
#define SFQMC_AFQMCFACTORY_H

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>

#include "config.h"
#include "mpi3/communicator.hpp"
#include "io/ptree/ptree_utilities.hpp"

#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Drivers/DriverFactory.h"
#include "Memory/buffer_managers.h"


namespace sfqmc
{
namespace afqmc
{
/**
 * @brief Factory class for AFQMC. Parses input, performs setup of classes, and executes the driver.
 *
 * @details The AFQMCFactory class is the top-level class for AFQMC. It parses the input file, performs the setup of classes, and executes the driver. 
 It contains instances of the following factories which are used to construct the objects used during AFQMC calculations:
  * - HamiltonianFactory HamFac
  * - WalkerSetFactory WSetFac
  * - WavefunctionFactory WfnFac
  * - PropagatorFactory PropFac
  * - DriverFactory DriverFac

  * It also instances of the following classes which handle MPI communication and task group management:
  * - GlobalTaskGroup gTG
  * - TaskGroupHandler TGHandler
 *
 * @param type std::string describing the type of Driver to be used. Valid choices are "afqmc", "legacy_afqmc", and "csafqmc".
  * @param comm_ boost::mpi3::communicator The MPI communicator.
  * @param pt boost::property_tree::ptree The property tree containing input file parameters
  * @param n_groups int The number of groups to be used in the task group.
 */
class AFQMCFactory
{
public:
  ///constructor
  AFQMCFactory(std::string type, boost::mpi3::communicator& comm_, 
	       const ptree pt, int n_groups=1);

  ///destructor
  ~AFQMCFactory(); 

private:

  // mixed or double precision 
  bool mixed_precision;

  // number of cores for TG parallelization
  int ncores;

  int m_series;
  std::string project_title;

  // global TG from which all TGs are constructed
  GlobalTaskGroup gTG;

  // object that manages the TGs. Must be placed here,
  // since it must be destroyed last
  TaskGroupHandler TGHandler;

  // container of AFQMCInfo objects
  std::map<std::string, AFQMCInfo> InfoMap;

  // Hamiltonian factory
  HamiltonianFactory HamFac;

  // WalkerHandler factory
  WalkerSetFactory WSetFac;

  // Wavefunction factoru
  WavefunctionFactory WfnFac;

  // Propagator factory
  PropagatorFactory PropFac;

  // driver factory
  DriverFactory DriverFac;

  //
  //  Traverse input tree and creates all non-executable objects.
  //  Created objects (pointers actually) are stored in maps based on name in xml block.
  //  Executable sections (drivers) are created with objects already exiting
  //  in the maps.
  //
  bool parse(const ptree pt);

  //
  //  Traverse input tree and creates executable sections, using objects created during parsing.
  //
  bool execute(std::string type, const ptree pt);
};
} // namespace afqmc
} // namespace sfqmc

#endif
