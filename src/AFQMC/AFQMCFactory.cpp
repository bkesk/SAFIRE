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
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"

#include "IO/app_loggers.h"
#include "AFQMC/AFQMCFactory.h"
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
                           std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
			   const ptree pt, int n_groups) 
      : m_series(pt.get<int>("project.series", 0)),
        project_title(pt.get<std::string>("project.id", "afqmc")),
        mpi(_mpi), 
        InfoMap(),
        HamFac(InfoMap),
        WSetFac(InfoMap),
        WfnFac(InfoMap),
        PropFac(InfoMap),
        DriverFac(mpi, InfoMap, WSetFac, PropFac, WfnFac, HamFac) 
{
  utils::check(n_groups==1, "finish!!!");
  app_log(1, " AFQMCFactory Project settings: ");
  app_log(1, "    -- id             : {} ", project_title);
  app_log(1, "    -- series         : {} ", m_series);
  app_log(1, "    -- n_groups       : {} ", n_groups);
  app_log(1, "    -- MPI tasks/node : {} ", mpi->node_comm.size());
  app_log(1, "    -- MPI nodes      : {} ", mpi->internode_comm.size());
  app_log(1, "    -- MPI tasks      : {} \n\n", mpi->comm.size()); 

  // parse input
  utils::check(parse(pt), " Error in AFQMCFactory: Problems parsing the input file. ");

  // execute 
  utils::check(execute(type,pt), "Error in AFQMCFactory: Problems executing the input file. ");
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
