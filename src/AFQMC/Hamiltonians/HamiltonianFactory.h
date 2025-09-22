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


#ifndef SFQMC_AFQMC_HAMILTONIANFACTORY_H
#define SFQMC_AFQMC_HAMILTONIANFACTORY_H

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include "io/ptree/ptree_utilities.hpp"
#include "Utilities/app_loggers.h"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"

namespace sfqmc
{
namespace afqmc
{
class HamiltonianFactory
{
public:
  HamiltonianFactory(std::map<std::string, AFQMCInfo>& info) : 
	InfoMap(info)
  {}

  ~HamiltonianFactory()
  {
    // delete Hamiltonian objects
    //for (auto it = hamiltonians.begin(); it != hamiltonians.end(); ++it)
    //  delete it->second;
  }

  bool is_constructed(const std::string& ID)
  {
    auto xml = hamBlocks.find(ID);
    if (xml == hamBlocks.end())
      APP_ABORT(" Error in WavefunctionFactory::is_constructed(string&): Missing xml block. ");
    auto ham = hamiltonians.find(ID);
    if (ham == hamiltonians.end())
      return false;
    else
      return true;
  }

  // returns a pointer to the base Hamiltonian class associated with a given ID
  Hamiltonian& getHamiltonian(GlobalTaskGroup& gTG, const std::string& ID)
  {
    auto xml = hamBlocks.find(ID);
    if (xml == hamBlocks.end())
      APP_ABORT("Error: Missing xml Block in HamiltonianFactory::getHamiltonian(string&). ");
    auto ham = hamiltonians.find(ID);
    if (ham == hamiltonians.end())
    {
      auto newham = hamiltonians.insert(std::make_pair(ID, buildHamiltonian(gTG, xml->second)));
      if (!newham.second)
        APP_ABORT(" Error: Problems inserting new hamiltonian in HamiltonianFactory::getHamiltonian(streing&). ");
      return (newham.first)->second;
    }
    else
      return ham->second;
  }

  // adds a xml block from which a Hamiltonian can be built
  void push(const std::string& ID, ptree pt)
  {
    auto xml = hamBlocks.find(ID);
    if (xml != hamBlocks.end())
      APP_ABORT("Error: Repeated Hamiltonian block in HamiltonianFactory. Hamiltonian names must be unique. ");
    hamBlocks.insert(std::make_pair(ID, pt));
  }

  ptree get_input(const std::string& ID) const
  {
    auto xml = hamBlocks.find(ID);
    if (xml == hamBlocks.end())
    {
      app_error("HamFac cannot find {}", ID);
      APP_ABORT("Error: failed to find Hamiltonian with above name.");
    }
    return xml->second;
  }

  // this routine allows you to modify the input block associated with ID 
  ptree& get_input(const std::string& ID)
  {
    auto xml = hamBlocks.find(ID);
    if (xml == hamBlocks.end())
    { 
      app_log(1,"failed to find {}", ID);
      APP_ABORT("Error: failed to find Hamiltonian with above name.");
    }
    return xml->second;
  }

protected:
  // reference to container of AFQMCInfo objects
  std::map<std::string, AFQMCInfo>& InfoMap;

  // keep ownership of the TGs in the Factory
  // this way you can reuse them if necessary
  // and you don't then need to worry about the semantics of mpi3::communicator
  std::map<int, TaskGroup_> TGMap;

  // generates a new Hamiltonian and returns the pointer to the base class
  Hamiltonian buildHamiltonian(GlobalTaskGroup& gTG, ptree pt)
  {
    std::string fham_type;
    fham_type = pt.get<std::string>("filetype", "hdf5");

    app_log(1,"\n****************************************************");
    app_log(1,"               Initializing Hamiltonian ");
    app_log(1,"\n****************************************************\n");
    app_log(2, " Hamiltonian Factory input: ");
    app_log(2, "{}", io::to_string(pt));

    if (fham_type == "hdf5")
      return fromHDF5(gTG, pt);
    else
    {
      app_error("Unknown Hamiltonian filetype in HamiltonianFactory::buildHamiltonian(): {}", 
		    fham_type);
      APP_ABORT(" Error: Unknown Hamiltonian filetype in HamiltonianFactory::buildHamiltonian(). ");
    }
    return Hamiltonian{};
  }

  Hamiltonian fromHDF5(GlobalTaskGroup& gTG, ptree pt);

  TaskGroup_& getTG(GlobalTaskGroup& gTG, int nTG)
  {
    if (gTG.getTotalNodes() % nTG != 0)
      APP_ABORT("Error: number_of_TGs must divide the total number of processors. \n\n");
    int nnodes = gTG.getTotalNodes() / nTG;
    auto t     = TGMap.find(nnodes);
    if (t == TGMap.end())
    {
      auto p = TGMap.insert(std::make_pair(nnodes,
                                           TaskGroup_(gTG, std::string("HamiltonianTG_") + std::to_string(nnodes),
                                                      nnodes, gTG.getTotalCores())));
      if (!p.second)
        APP_ABORT(" Error: Problems creating new hamiltonian TG in HamiltonianFactory::getTG(int). ");
      return (p.first)->second;
    }
    return t->second;
  }

  std::map<std::string, ptree> hamBlocks;

  std::map<std::string, Hamiltonian> hamiltonians;

};
} // namespace afqmc
} // namespace sfqmc

#endif
