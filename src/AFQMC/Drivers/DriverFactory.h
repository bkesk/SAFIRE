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

#include "utilities/mpi_context.h"

#include "AFQMC/parameters.hpp"

#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM>
class DriverFactory
{
  using communicator = boost::mpi3::communicator;

public:
  DriverFactory(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                WalkerSetFactory<MEM>& wsetfac_,
                PropagatorFactory<MEM>& pfac_,
                WavefunctionFactory<MEM>& wfnfac_,
                HamiltonianFactory& hfac)
      : mpi(_mpi),
        WSetFac(wsetfac_),
        PropFac(pfac_),
        HamFac(hfac),
        WfnFac(wfnfac_)
  { }

  ~DriverFactory() {}

  bool executeDriver(DriverType type, std::string title, int m_series, const ExecuteParameters& exec);

private:
  bool executeAFQMCDriver(std::string title, int m_series, const ExecuteParameters& exec);
  bool executeFTAFQMCDriver(std::string title, int m_series, const ExecuteParameters& exec);
  bool executeCSAFQMCDriver(std::string title, int m_series, const ExecuteParameters& exec);

  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  // WalkerHandler factory
  WalkerSetFactory<MEM>& WSetFac;

  // Propagator factory
  PropagatorFactory<MEM>& PropFac;

  // Hamiltonian factory
  HamiltonianFactory& HamFac;

  // Wavefunction factory
  WavefunctionFactory<MEM>& WfnFac;

  int unique_id = 0;

  // Returns the name associated with the wavefunction block of an execute block. It can be
  // either the name of a previously registered block or a full (possibly nameless)
  // declaration. After the successful return of this routine (e.g. wfn_name), we can assume
  // that WfnFac.get_input(wfn_name) exists and that it contains a non-empty filename.
  std::string get_wavefunction_id(const ExecuteParameters& exec);

  std::tuple<std::string,std::string,std::string,std::string>
    get_component_ids(const ExecuteParameters& exec);

  // Resolves a sub-block of an execute block to the name of a block registered in the given
  // factory:
  // 1. A string references a block that must already be registered, otherwise the code aborts.
  // 2. A full declaration is registered under its own name, or, if it does not have one, under
  //    a generated unique name.
  // 3. An absent block registers `fallback` under a generated unique name.
  template<class Params, class Factory>
  std::string resolve_or_push(std::string_view key, const std::optional<utils::BlockRef<Params>>& block,
                              Factory& fac, Params fallback)
  {
    if(block) {
      if(const auto* name = std::get_if<std::string>(&*block)) {
        // retrieve the name to make sure it exists (aborts if not)
        fac.get_input(*name);
        return *name;
      }
      Params params = std::get<Params>(*block);
      if(params.name.empty()) {
        params.name = std::format("{}_unique_id_{}", key, ++unique_id);
      }
      std::string name = params.name;
      fac.push(name, std::move(params));
      return name;
    }

    fallback.name = std::format("{}_unique_id_{}", key, ++unique_id);
    std::string name = fallback.name;
    fac.push(name, std::move(fallback));
    return name;
  }

};

} // namespace afqmc

} // namespace sfqmc

