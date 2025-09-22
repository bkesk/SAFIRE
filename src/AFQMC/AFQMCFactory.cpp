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
/**@file AFQMCFactory.cpp
 * @brief Top level class for AFQMC. Parses input and performs setup of classes.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <complex>
#include <tuple>
#include <queue>
#include <algorithm>
#include "config.h"
#include "Utilities/AppAbort.hpp"

#include "Utilities/app_loggers.h"
#include "mpi3/shared_communicator.hpp"
#include "Utilities/Random.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMCFactory.h"
#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Drivers/DriverFactory.h"

namespace sfqmc
{

namespace afqmc
{

AFQMCFactory::AFQMCFactory(std::string type,
			   boost::mpi3::communicator& comm_, 
			   const ptree pt, 
			   int n_groups) 
      : mixed_precision(pt.get<bool>("project.mixed_precision", false)),
        ncores(pt.get<int>("project.ncores", 1)),
        m_series(pt.get<int>("project.series", 0)),
        project_title(pt.get<std::string>("project.id", "afqmc")),
        gTG(comm_,n_groups),
        TGHandler(gTG, -10),
        InfoMap(),
        HamFac(InfoMap),
        WSetFac(InfoMap),
        WfnFac(InfoMap, mixed_precision),
        PropFac(InfoMap, mixed_precision),
        DriverFac(ncores, gTG, TGHandler, InfoMap, mixed_precision,
                  WSetFac, PropFac, WfnFac, HamFac)
{
  auto& node(gTG.Node());
#if defined(ENABLE_DEVICE)
  // check ncores 
  if(ncores != 1) {
    app_warning(" Warning: Only ncores=1 allowed in device build. Setting to 1.");
    ncores = 1;
  }
#else
  ncores = std::max(std::min(ncores, node.size()), 1);
#endif

  app_log(1, " AFQMCFactory Project settings: ");
  app_log(1, "    -- mixed_precision: {} ", mixed_precision);
  app_log(1, "    -- ncores (local) : {} ", ncores);
  app_log(1, "    -- n_groups       : {} ", n_groups); 
  app_log(1, "    -- id             : {} ", project_title);
  app_log(1, "    -- series         : {} ", m_series);
  app_log(1, "    -- MPI tasks/node : {} ", node.size());
  app_log(1, "    -- MPI nodes      : {} ", gTG.getTotalNodes());
  app_log(1, "    -- MPI tasks      : {} \n\n", gTG.getTotalCores()*gTG.getTotalNodes());

  // move this to arch::INIT(), which should be called regardless of device!
#if defined(ENABLE_DEVICE)
  // initialize device
  int rank   = gTG.Global().rank();
  int nprocs = gTG.Global().size();
  auto iseed = utils::make_seed(gTG.Global());
  arch::INIT(node, (unsigned long long int)(iseed));
#endif
  // setup buffers manager
  boost::mpi3::shared_communicator local(node.split(node.rank() / ncores, node.rank()));
  HostBufferManager host_buffer(20uL * 1024uL * 1024uL);  // setup monostate
  DeviceBufferManager dev_buffer(20uL * 1024uL * 1024uL); // setup monostate
  LocalTGBufferManager local_buffer(local, 20uL * 1024uL * 1024uL);
  // all the way to here to arch::INIT();

  // parse input
  if(not parse(pt))
    APP_ABORT(" Error in AFQMCFactory: Problems parsing the input file. ");

  // execute 
  if(not execute(type,pt))  
    APP_ABORT(" Error in AFQMCFactory: Problems executing the input file. ");

  // move this to arch::FINALIZE();
  release_memory_managers();
}

AFQMCFactory::~AFQMCFactory() = default; 
    
bool AFQMCFactory::parse(const ptree pt_in)
{
  InfoMap.clear();

  // first look only for AFQMCInfo
  // Careful here, since all factories have a reference to this map
  // It must be built before any factory is used
  for(auto& it : pt_in)
  {
    std::string cname = it.first;
    ptree pt = it.second;
    if (cname == "system")
    {
      AFQMCInfo info;
      info.parse(pt);
      std::pair<std::map<std::string, AFQMCInfo>::iterator, bool> ret;
      ret = InfoMap.insert(std::pair<std::string, AFQMCInfo>(info.name, info));
      if (ret.second == false)
      {
        app_error("ERROR: AFQMCInfo already defined: {} ", info.name);
        app_error_flush();
        return false;
      }
    }
  }

  // now look for non-executable blocks
  for(auto& it : pt_in)
  {
    std::string cname = it.first;
    ptree pt = it.second;
    std::string oname = pt.get<std::string>("name", "");

    if (cname == "hamiltonian")
    {
      if (oname == "") APP_ABORT("hamiltonian outside execute block must be named");
      HamFac.push(oname, pt);
    }
    else if (cname == "wavefunction")
    {
      if (oname == "") APP_ABORT("wavefunction outside execute block must be named");
      WfnFac.push(oname, pt);
    }
    else if (cname == "walker_set")
    {
      if (oname == "") APP_ABORT("walker_set outside execute block must be named");
      WSetFac.push(oname, pt);
    }
    else if (cname == "propagator")
    {
      if (oname == "") APP_ABORT("propagator outside execute block must be named");
      PropFac.push(oname, pt);
    }
    else if(cname != "project" and cname != "execute") {
      app_warning("Ignoring unknown input block: {}", cname);
    }
  }

  return true;
}

bool AFQMCFactory::execute(std::string type, const ptree pt_in)
{
  char fileroot[256];

  for(auto& it : pt_in)
  {
    std::string cname = it.first;
    ptree pt = it.second;
    if (cname == "execute")
    {
      snprintf(fileroot, sizeof(fileroot), "%s.s%03d", project_title.c_str(), m_series);
      	
      // execute driver
      if (!DriverFac.executeDriver(type, std::string(fileroot), m_series, pt))
      {
        app_error("Error in DriverFactory::executeDriver::run()");
 	app_error_flush();
        return false;
      }

      m_series++;
    }
  }

  return true;
}

} // namespace afqmc

} // namespace sfqmc
