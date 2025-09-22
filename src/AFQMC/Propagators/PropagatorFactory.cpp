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
#include "Utilities/AppAbort.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "PropagatorFactory.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"

namespace sfqmc
{
namespace afqmc
{
Propagator PropagatorFactory::buildAFQMCPropagator(TaskGroup_& TG,
                                                   ptree pt,
                                                   Wavefunction& wfn,
                                                   utils::DeviceRandomGenerator_t* rng)
{
  // allocator for local memory

  std::string info = pt.get<std::string>("system", "");

  if (InfoMap.find(info) == InfoMap.end())
    APP_ABORT("ERROR: Undefined system in PropagatorFactory. ");
  AFQMCInfo& AFinfo = InfoMap[info];

  // Add spin_dependent here based on type in HamOps
  if( wfn.getHamType() == ModelHamiltonian ) {
    // move the call to get FieldTypes to inside propagator constructor
    boost::multi::array<int, 1> FieldTypes(iextensions<1u>{wfn.local_number_of_cholesky_vectors()});
    wfn.getFieldTypes(FieldTypes);
    if( TG.getNGroupsPerTG() > 1 ) 
      APP_ABORT(" Error: nnodes > 1 not allowed with model Hamiltonians. ");
    if(mixed_precision) {
      return Propagator(AFQMCModelPropagator<true>(AFinfo, pt, TG, wfn, std::move(FieldTypes), rng)); 
    } else {
      return Propagator(AFQMCModelPropagator<false>(AFinfo, pt, TG, wfn, std::move(FieldTypes), rng)); 
    }
  } else {  
    if (TG.getNGroupsPerTG() == 1)
      if(mixed_precision) {
        return Propagator(AFQMCBasePropagator<true>(AFinfo, pt, TG, wfn, rng));
      } else {	
        return Propagator(AFQMCBasePropagator<false>(AFinfo, pt, TG, wfn, rng));
      }
    else
    {
      if (wfn.distribution_over_cholesky_vectors()) {
        // use specialized distributed algorithm for case
        // when vbias doesn't need reduction over TG
        if(mixed_precision) {
          return Propagator(AFQMCDistributedPropagatorDistCV<true>(AFinfo, pt, TG, wfn, rng));
        } else {
          return Propagator(AFQMCDistributedPropagatorDistCV<false>(AFinfo, pt, TG, wfn, rng));
	}
      } else {
        if(mixed_precision) {
          return Propagator(AFQMCDistributedPropagator<true>(AFinfo, pt, TG, wfn, rng));
        } else {
          return Propagator(AFQMCDistributedPropagator<false>(AFinfo, pt, TG, wfn, rng));
	}
      }
    }
  }
}


} // namespace afqmc

} // namespace sfqmc
