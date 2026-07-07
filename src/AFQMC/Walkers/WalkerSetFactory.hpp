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

#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
class WalkerSetFactory
{
public:
  WalkerSetFactory(std::map<std::string, AFQMCInfo>& info) : InfoMap(info) {}

  ~WalkerSetFactory() {}

  bool is_constructed(const std::string& ID)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
      utils::check(false," Error in WalkerSetFactory::is_constructed(string&): Missing xml block. ");
    auto wlk = handlers.find(ID);
    if (wlk == handlers.end())
      return false;
    else
      return true;
  }

  auto& getWalkerSet(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, const std::string& ID, std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
      utils::check(false,"Error: Missing xml Block in WalkerSetFactory::getWalkerSet(string&). ");
    auto wlk = handlers.find(ID);
    if (wlk == handlers.end())
    {
      auto newwlk = handlers.insert(std::make_pair(ID, buildHandler(mpi, xml->second, rng)));
      if (!newwlk.second)
        utils::check(false," Error: Problems inserting new hamiltonian. ");
      return (newwlk.first)->second;
    }
    else
      return wlk->second;
  }

  ptree get_input(const std::string& ID) const
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
    {
      app_log(1,"WlkFac cannot find {}",ID);
      utils::check(false,"Error: failed to find walker_set with above name.");
      return ptree{};	
    }
    else
      return xml->second;
  }

  // this routine allows you to modify the input block associated with ID 
  ptree& get_input(const std::string& ID)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
    { 
      app_log(1,"failed to find {}", ID);
      utils::check(false,"Error: failed to find walker_set with above name.");
    }
    return xml->second;
  }

  // adds a xml block from which a WalkerSet can be built
  void push(const std::string& ID, ptree pt)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml != wlkBlocks.end())
      utils::check(false,"Error: Repeated WalkerSet block in WalkerSetFactory. WalkerSet names must be unique. ");
    wlkBlocks.insert(std::make_pair(ID, pt));
  }

protected:
  // reference to container of AFQMCInfo objects
  std::map<std::string, AFQMCInfo>& InfoMap;

  // generates a new WalkerSet and returns the pointer to the base class
  WalkerSet<MEM> buildHandler(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, ptree pt, std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng)
  {
    std::string type, info;
    info = pt.get<std::string>("system", "");
    utils::check(InfoMap.find(info) != InfoMap.end(), "ERROR: Undefined system: {}", info);

    auto& sysinfo = InfoMap[info];
    return WalkerSet<MEM>(WalkerSetBase<MEM>(mpi, pt, sysinfo.NMO, sysinfo.nup, sysinfo.ndown, rng));
  }

  std::map<std::string, ptree> wlkBlocks;

  std::map<std::string, WalkerSet<MEM>> handlers;
};
} // namespace afqmc
} // namespace sfqmc

