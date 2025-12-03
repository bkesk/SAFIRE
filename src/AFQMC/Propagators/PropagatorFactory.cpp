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


#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>

#include "config.h"
#include "utilities/check.hpp"

#include "AFQMC/config.h"
#include "PropagatorFactory.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
Propagator PropagatorFactory::buildAFQMCPropagator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, ptree pt, Wavefunction& wfn, std::shared_ptr<utils::DeviceRandomGenerator_t> rng)
{
  std::string info = pt.get<std::string>("system", "");

  utils::check(InfoMap.find(info) != InfoMap.end(),"ERROR: Undefined system in PropagatorFactory. ");
  AFQMCInfo& AFinfo = InfoMap[info];

//  if( wfn.getHamType() == ModelHamiltonian ) {
//    return Propagator(AFQMCModelPropagator<MEM>(AFinfo, pt, mpi, wfn, rng)); 
//    return Propagator{};
//  } else {  
    return Propagator(AFQMCBasePropagator<MEM>(AFinfo, pt, mpi, wfn, rng));
//  }
  return Propagator{};
}

template Propagator 
PropagatorFactory::buildAFQMCPropagator<HOST_MEMORY>(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>>,ptree,Wavefunction&,std::shared_ptr<utils::DeviceRandomGenerator_t>);

#if defined(ENABLE_DEVICE)
template Propagator 
PropagatorFactory::buildAFQMCPropagator<DEVICE_MEMORY>(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>>,ptree,Wavefunction&,std::shared_ptr<utils::DeviceRandomGenerator_t>);
#endif


} // namespace afqmc

} // namespace sfqmc
