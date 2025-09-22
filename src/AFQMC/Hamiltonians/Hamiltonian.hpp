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

#ifndef SFQMC_AFQMC_HAMILTONIAN_HPP
#define SFQMC_AFQMC_HAMILTONIAN_HPP

#include <fstream>

#include "AFQMC/config.h"
#include "boost/variant.hpp"

#include "AFQMC/Hamiltonians/FactorizedSparseHamiltonian.h"
#include "AFQMC/Hamiltonians/ModelHamOpsGenerator.h"
#include "AFQMC/Hamiltonians/THCHamiltonian.h"
#include "AFQMC/Hamiltonians/KPFactorizedHamiltonian.h"
#include "AFQMC/Hamiltonians/RealDenseHamiltonian.h"
#include "AFQMC/Hamiltonians/RealDenseHamiltonian_v2.h"
#include "AFQMC/HamiltonianOperations/HamiltonianOperations.hpp"

namespace sfqmc
{
namespace afqmc
{
namespace dummy
{
/*
 * Empty class to avoid need for default constructed Hamiltonians.
 * Throws is any visitor is called.
 */
class dummy_Hamiltonian
{
public:
  dummy_Hamiltonian(){ APP_ABORT(" Error: Reached default constructor of dummy_Hamiltonian. "); };

  ComplexType getNuclearCoulombEnergy() const
  {
    throw std::runtime_error("calling visitor on dummy object");
    return 0;
  }

  template<bool MP, class... Args>
  HamiltonianOperations<MP> getHamiltonianOperations([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy object");
    return HamiltonianOperations<MP>{};
  }

  HamiltonianTypes getHamType()
  { 
    throw std::runtime_error("calling visitor on dummy object");
    return UNKNOWN; 
  }
};
} // namespace dummy

class Hamiltonian : public boost::variant<dummy::dummy_Hamiltonian 
                                          ,FactorizedSparseHamiltonian
                                          ,THCHamiltonian 
                                          ,ModelHamOpsGenerator 
                                          ,KPFactorizedHamiltonian
                                          ,RealDenseHamiltonian
                                          ,RealDenseHamiltonian_v2
					>
{

public:
  Hamiltonian() = default; 
  explicit Hamiltonian(THCHamiltonian&& other) : variant(std::move(other)) {}
  explicit Hamiltonian(ModelHamOpsGenerator&& other) : variant(std::move(other)) {}
  explicit Hamiltonian(FactorizedSparseHamiltonian&& other) : variant(std::move(other)) {}
  explicit Hamiltonian(KPFactorizedHamiltonian&& other) : variant(std::move(other)) {}
  explicit Hamiltonian(RealDenseHamiltonian&& other) : variant(std::move(other)) {}
  explicit Hamiltonian(RealDenseHamiltonian_v2&& other) : variant(std::move(other)) {}

  explicit Hamiltonian(THCHamiltonian const& other)              = delete;
  explicit Hamiltonian(ModelHamOpsGenerator const& other)              = delete;
  explicit Hamiltonian(FactorizedSparseHamiltonian const& other) = delete;
  explicit Hamiltonian(KPFactorizedHamiltonian const& other) = delete;
  explicit Hamiltonian(RealDenseHamiltonian const& other)    = delete;
  explicit Hamiltonian(RealDenseHamiltonian_v2 const& other) = delete;

  Hamiltonian(Hamiltonian const& other) = delete;
  Hamiltonian(Hamiltonian&& other)      = default;

  Hamiltonian& operator=(Hamiltonian const& other) = delete;
  Hamiltonian& operator=(Hamiltonian&& other) = default;

  ComplexType getNuclearCoulombEnergy()
  {
    return boost::apply_visitor([&](auto&& a) { return a.getNuclearCoulombEnergy(); }, *this);
  }

  template<bool MP, class... Args>
  HamiltonianOperations<MP> getHamiltonianOperations(Args&&... args)
  {
      return boost::apply_visitor([&](auto&& a) { 
	return a.template getHamiltonianOperations<MP>(std::forward<Args>(args)...); }, *this);
  }

  HamiltonianTypes getHamType()
  {
    return boost::apply_visitor([&](auto&& a) { return a.getHamType(); }, *this);
  }
};

} // namespace afqmc

} // namespace sfqmc

#endif
