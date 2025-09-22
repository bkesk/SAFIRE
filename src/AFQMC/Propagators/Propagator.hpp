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

#ifndef SFQMC_AFQMC_PROPAGATOR_HPP
#define SFQMC_AFQMC_PROPAGATOR_HPP

#include <fstream>

#include "AFQMC/config.h"
#include "boost/variant.hpp"

#include "AFQMC/Propagators/AFQMCBasePropagator.h"
#include "AFQMC/Propagators/AFQMCModelPropagator.h"
#include "AFQMC/Propagators/AFQMCDistributedPropagatorDistCV.h"
#include "AFQMC/Propagators/AFQMCDistributedPropagator.h"

namespace sfqmc
{
namespace afqmc
{
namespace dummy
{
/*
 * Empty class to avoid need for default constructed Propagators.
 * Throws is any visitor is called. 
 */
class dummy_Propagator
{
public:
  dummy_Propagator(){};

  template<class WlkSet>
  void Propagate([[maybe_unused]] int steps, [[maybe_unused]] WlkSet& wset, [[maybe_unused]]  RealType& E1, [[maybe_unused]] RealType dt, [[maybe_unused]]  int fix_bias = 1)
  {
    throw std::runtime_error("calling visitor on dummy object");
  }

  template<class... Args>
  void BackPropagate([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy object");
  }

  template<class... Args>
  void Orthogonalize([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy object");
  }

  template<class... Args>
  void PropagateOperators([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy object");
  }

  bool hybrid_propagation()
  {
    throw std::runtime_error("calling visitor on dummy_Propagator object");
    return false;
  }

  bool free_propagation()
  {
    throw std::runtime_error("calling visitor on dummy_Propagator object");
    return false;
  }

  int global_number_of_cholesky_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_Propagator object");
    return 0;
  }

  void generateP1(double, WALKER_TYPES) { throw std::runtime_error("calling visitor on dummy_Propagator object"); }

  void set_rng_block_size(int) { throw std::runtime_error("calling visitor on dummy_Propagator object"); }

};
} // namespace dummy

class Propagator : public boost::variant<dummy::dummy_Propagator,
                                         AFQMCBasePropagator<true>,
                                         AFQMCBasePropagator<false>,
                                         AFQMCDistributedPropagatorDistCV<true>,
                                         AFQMCDistributedPropagatorDistCV<false>,
                                         AFQMCDistributedPropagator<true>,
                                         AFQMCDistributedPropagator<false>,
                                         AFQMCModelPropagator<true>, 
                                         AFQMCModelPropagator<false>
                                        >
{
public:
  Propagator() { APP_ABORT(" Error: Reached default constructor of Propagator. "); }
  explicit Propagator(AFQMCBasePropagator<true>&& other) : variant(std::move(other)) {}
  explicit Propagator(AFQMCBasePropagator<true> const& other) = delete;

  explicit Propagator(AFQMCBasePropagator<false>&& other) : variant(std::move(other)) {}
  explicit Propagator(AFQMCBasePropagator<false> const& other) = delete;

  explicit Propagator(AFQMCDistributedPropagatorDistCV<true>&& other) : variant(std::move(other)) {}
  explicit Propagator(AFQMCDistributedPropagatorDistCV<true> const& other) = delete;

  explicit Propagator(AFQMCDistributedPropagatorDistCV<false>&& other) : variant(std::move(other)) {}
  explicit Propagator(AFQMCDistributedPropagatorDistCV<false> const& other) = delete;

  explicit Propagator(AFQMCDistributedPropagator<true>&& other) : variant(std::move(other)) {}
  explicit Propagator(AFQMCDistributedPropagator<true> const& other) = delete;

  explicit Propagator(AFQMCDistributedPropagator<false>&& other) : variant(std::move(other)) {}
  explicit Propagator(AFQMCDistributedPropagator<false> const& other) = delete;

  explicit Propagator(AFQMCModelPropagator<true>&& other) : variant(std::move(other)) {}
  explicit Propagator(AFQMCModelPropagator<true> const& other) = delete;

  explicit Propagator(AFQMCModelPropagator<false>&& other) : variant(std::move(other)) {}
  explicit Propagator(AFQMCModelPropagator<false> const& other) = delete;

  Propagator(Propagator const& other) = delete;
  Propagator(Propagator&& other)      = default;

  Propagator& operator=(Propagator const& other) = delete;
  Propagator& operator=(Propagator&& other) = default;

  template<class... Args>
  void Propagate(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.Propagate(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void BackPropagate(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.BackPropagate(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void PropagateOperators(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.PropagateOperators(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void generateP1(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.generateP1(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void Orthogonalize(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.Orthogonalize(std::forward<Args>(args)...); }, *this);
  }

  bool hybrid_propagation()
  {
    return boost::apply_visitor([&](auto&& a) { return a.hybrid_propagation(); }, *this);
  }

  bool free_propagation()
  {
    return boost::apply_visitor([&](auto&& a) { return a.free_propagation(); }, *this);
  }

  int global_number_of_cholesky_vectors() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.global_number_of_cholesky_vectors(); }, *this);
  }

  template<class... Args>
  void set_rng_block_size(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.set_rng_block_size(std::forward<Args>(args)...); }, *this);
  }
};

} // namespace afqmc

} // namespace sfqmc

#endif
