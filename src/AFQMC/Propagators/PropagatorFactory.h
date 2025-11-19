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
#include "utilities/Random.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/Propagator.hpp"

namespace sfqmc
{
namespace afqmc
{
class PropagatorFactory
{
public:
  PropagatorFactory(std::map<std::string, AFQMCInfo>& info) : 
	InfoMap(info)
  {}

  ~PropagatorFactory() {}

  bool is_constructed(const std::string& ID)
  {
    auto xml = propBlocks.find(ID);
    if (xml == propBlocks.end())
      utils::check(false," Error in WavefunctionFactory::is_constructed(string&): Missing xml block. ");
    auto p0 = propagators.find(ID);
    if (p0 == propagators.end())
      return false;
    else
      return true;
  }

  // returns a pointer to the base Propagator class associated with a given ID
  auto& getPropagator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, const std::string& ID, Wavefunction& wfn, std::shared_ptr<utils::DeviceRandomGenerator_t> rng)
  {
    auto xml = propBlocks.find(ID);
    if (xml == propBlocks.end())
      utils::check(false," Error in PropagatorFactory::getPropagator(string&): Missing xml block. ");
    auto p0 = propagators.find(ID);
    if (p0 == propagators.end())
    {
      auto newp = propagators.insert(std::make_pair(ID, buildPropagator(mpi, xml->second, wfn, rng)));
      if (not newp.second)
        utils::check(false," Error: Problems building new propagator in PropagatorFactory::getPropagator(string&). ");
      return (newp.first)->second;
    }
    else
      return p0->second;
  }

  ptree get_input(const std::string& ID) const
  {
    auto xml = propBlocks.find(ID);
    if (xml == propBlocks.end())
    {
      app_log(1,"HamFac cannot find {}", ID);
      utils::check(false,"Error: failed to find Hamiltonian with above name.");
    }
    return xml->second;
  }

  // this routine allows you to modify the input block associated with ID 
  ptree& get_input(const std::string& ID)
  {
    auto xml = propBlocks.find(ID);
    if (xml == propBlocks.end())
    { 
      app_log(1,"failed to find {}", ID);
      utils::check(false,"Error: failed to find propagator with above name.");
    }
    return xml->second;
  }

  // adds a xml block from which a Propagator can be built
  void push(const std::string& ID, ptree pt)
  {
    auto xml = propBlocks.find(ID);
    if (xml != propBlocks.end())
      utils::check(false,"Error: Repeated Propagator block in PropagatorFactory. Propagator names must be unique. ");
    propBlocks.insert(std::make_pair(ID, pt));
  }

protected:
  // reference to container of AFQMCInfo objects
  std::map<std::string, AFQMCInfo>& InfoMap;

  // generates a new Propagator and returns the pointer to the base class
  Propagator buildPropagator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, ptree pt, Wavefunction& wfn, std::shared_ptr<utils::DeviceRandomGenerator_t> rng)
  {
    std::string compute  = pt.get<std::string>("compute", memory::default_compute);
    
    app_log(1,"\n****************************************************");
    app_log(1,"               Initializing Propagator ");
    app_log(1,"\n****************************************************");
    if (compute == "cpu")
      return buildAFQMCPropagator<HOST_MEMORY>(mpi, pt, wfn, rng);
#if defined(ENABLE_DEVICE)
    else if(compute == "gpu")
      return buildAFQMCPropagator<DEVICE_MEMORY>(mpi, pt, wfn, rng);
#endif
    return Propagator{};
  }

  template<MEMORY_SPACE MEM>
  Propagator buildAFQMCPropagator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, ptree pt, Wavefunction& wfn, std::shared_ptr<utils::DeviceRandomGenerator_t> r);

  std::map<std::string, ptree> propBlocks;

  std::map<std::string, Propagator> propagators;
};


} // namespace afqmc

} // namespace sfqmc

