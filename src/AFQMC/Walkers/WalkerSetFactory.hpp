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
#include "io/ptree/ptree_utilities.hpp"
#include "Utilities/Random.hpp"
#include "Utilities/app_loggers.h"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
class WalkerSetFactory
{
public:
  WalkerSetFactory(std::map<std::string, AFQMCInfo>& info) : InfoMap(info) {}

  ~WalkerSetFactory() {}

  bool is_constructed(const std::string& ID)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
      APP_ABORT(" Error in WalkerSetFactory::is_constructed(string&): Missing xml block. ");
    auto wlk = handlers.find(ID);
    if (wlk == handlers.end())
      return false;
    else
      return true;
  }

  WalkerSet& getWalkerSet(TaskGroup_& TG, const std::string& ID, utils::RandomGenerator_t* rng)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml == wlkBlocks.end())
      APP_ABORT("Error: Missing xml Block in WalkerSetFactory::getWalkerSet(string&). ");
    auto wlk = handlers.find(ID);
    if (wlk == handlers.end())
    {
      auto newwlk = handlers.insert(std::make_pair(ID, buildHandler(TG, xml->second, rng)));
      if (!newwlk.second)
        APP_ABORT(" Error: Problems inserting new hamiltonian in WalkerSetFactory::getHandler(streing&). ");
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
      APP_ABORT("Error: failed to find walker_set with above name.");
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
      APP_ABORT("Error: failed to find walker_set with above name.");
    }
    return xml->second;
  }

  // adds a xml block from which a WalkerSet can be built
  void push(const std::string& ID, ptree pt)
  {
    auto xml = wlkBlocks.find(ID);
    if (xml != wlkBlocks.end())
      APP_ABORT("Error: Repeated WalkerSet block in WalkerSetFactory. WalkerSet names must be unique. ");
    wlkBlocks.insert(std::make_pair(ID, pt));
  }

protected:
  // reference to container of AFQMCInfo objects
  std::map<std::string, AFQMCInfo>& InfoMap;

  // generates a new WalkerSet and returns the pointer to the base class
  WalkerSet buildHandler(TaskGroup_& TG, ptree pt, utils::RandomGenerator_t* rng)
  {
    std::string type, info;
    type = pt.get<std::string>("type", "shared");
    info = pt.get<std::string>("system", "");
    if (InfoMap.find(info) == InfoMap.end())
    {
      app_error("ERROR: Undefined system: {}", info);
      APP_ABORT("");
    }

    // keep like this until you have another choice and a variant framework in place
    if (type != "shared")
      APP_ABORT(" Error: Unknown WalkerSet type in WalkerSetFactory::buildHandler(). ");

    return WalkerSet(TG, pt, InfoMap[info], rng);
  }

  std::map<std::string, ptree> wlkBlocks;

  std::map<std::string, WalkerSet> handlers;
};
} // namespace afqmc
} // namespace sfqmc

