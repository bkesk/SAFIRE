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

#include <variant>
#include <complex>

#include "configuration.hpp"
#include "utilities/macros.hpp"
#include "AFQMC/Walkers/WalkerSetBase.h"

namespace sfqmc
{
namespace afqmc
{

// MAM: Make variant with memory types
//using WalkerSet = WalkerSetBase<HOST_MEMORY>;

class WalkerSet
{

  public:

    WalkerSet() = default;

    explicit WalkerSet(WalkerSetBase<HOST_MEMORY> const& arg) : var(arg) {}
    explicit WalkerSet(WalkerSetBase<HOST_MEMORY> && arg) : var(std::move(arg)) {}

    WalkerSet& operator=(WalkerSetBase<HOST_MEMORY> const& arg) { var = arg; return *this; }
    WalkerSet& operator=(WalkerSetBase<HOST_MEMORY> && arg) { var = std::move(arg); return *this; }

#if defined(ENABLE_DEVICE)   

    explicit WalkerSet(WalkerSetBase<DEVICE_MEMORY> const& arg) : var(arg) {}
    explicit WalkerSet(WalkerSetBase<DEVICE_MEMORY> && arg) : var(std::move(arg)) {}

    WalkerSet& operator=(WalkerSetBase<DEVICE_MEMORY> const& arg) { var = arg; return *this; }
    WalkerSet& operator=(WalkerSetBase<DEVICE_MEMORY> && arg) { var = std::move(arg); return *this; }

#endif

    ~WalkerSet() = default;
    WalkerSet(WalkerSet const&) = default;
    WalkerSet(WalkerSet &&) = default;
    WalkerSet& operator=(WalkerSet const&) = default;
    WalkerSet& operator=(WalkerSet &&) = default;

    static ptree interpret_inputs(const ptree pt0) {
      // assuming that input/ptree interpretation does not depend on memory space
      return WalkerSetBase<HOST_MEMORY>::interpret_inputs(pt0);
    }

    // Using macros to simplify the implementation. 
    // Look at WalkerSetBase for description of routines. 
    VISITOR(get_memory_space,var,const)
    VISITOR(size,var,const)
    VISITOR(capacity,var,const)
    VISITOR(NumBackProp,var,const)
    VISITOR(NumCholVecs,var,const)
    VISITOR(HistoryBufferLength,var,const)
    VISITOR(getBPPos,var,const)
    VISITOR(getHistoryPos,var,const)
    VISITOR(clean,var,)
    VISITOR(getWalkerType,var,const)
    VISITOR(get_target_population,var,const)
    VISITOR(get_global_target_population,var,const)
    VISITOR(walker_dims,var,const)
    VISITOR(GlobalPopulation,var,const)
    VISITOR(GlobalWeight,var,const)
    VISITOR(get_mpi,var,const)
    VISITOR(single_walker_memory_usage,var,const)
    VISITOR(single_walker_size,var,const)
    VISITOR(single_walker_bp_memory_usage,var,const)
    VISITOR(single_walker_bp_size,var,const)
    VISITOR(population_control_parameters,var,const)
    VISITOR(walkerSizeIO,var,const)
    VISITOR(getLogOverlapFactor,var,const)
    VISITOR(getRNG,var,)
    VISITOR(begin,var,)  
    VISITOR(begin,var,const)  
    VISITOR(end,var,)  
    VISITOR(end,var,const)  

    VOID_VISITOR(advanceBPPos,var,)
    VOID_VISITOR(advanceHistoryPos,var,)
    VOID_VISITOR(popControl,var,)
    VOID_VISITOR(scaleWeightsByOverlap,var,)
    VOID_VISITOR(resetWeights,var,)

    VOID_VISITOR_ARGS(resize,var,)
    VOID_VISITOR_ARGS(setBPPos,var,)
    VOID_VISITOR_ARGS(setHistoryPos,var,)
    VOID_VISITOR_ARGS(reserve,var,)
    VOID_VISITOR_ARGS(resize_bp,var,)
    VOID_VISITOR_ARGS(benchmark,var,)
    VOID_VISITOR_ARGS(processWalkerData,var,)
    VOID_VISITOR_ARGS(scaleWeight,var,)
    VOID_VISITOR_ARGS(popControl,var,)
    VOID_VISITOR_ARGS(copyToIO,var,const)
    VOID_VISITOR_ARGS(copyFromIO,var,)
    VOID_VISITOR_ARGS(getProperty,var,const)
    VOID_VISITOR_ARGS(setProperty,var,)
    VOID_VISITOR_ARGS(storeFields,var,)
    VOID_VISITOR_ARGS(adjustLogOverlapFactor,var,)
    VOID_VISITOR_ARGS(loadBalance,var,)

    auto operator[](int i) { 
      return std::visit( [&](auto&& v) { return v[i]; }, var ); 
    } 
    auto operator[](int i) const { 
      return std::visit( [&](auto&& v) { return v[i]; }, var ); 
    } 

    template<MEMORY_SPACE M>
    auto SlaterMatrices( SpinTypes s ) { 
      return std::visit( [&](auto&& v) { return  v.template SlaterMatrices<M>(s); }, var ); 
    } 
    template<MEMORY_SPACE M>
    auto SlaterMatrices( SpinTypes s ) const { 
      return std::visit( [&](auto&& v) { return  v.template SlaterMatrices<M>(s); }, var ); 
    } 
    template<MEMORY_SPACE M>
    auto SlaterMatricesN( SpinTypes s ) { 
      return std::visit( [&](auto&& v) { return  v.template SlaterMatricesN<M>(s); }, var ); 
    } 
    template<MEMORY_SPACE M>
    auto SlaterMatricesN( SpinTypes s ) const { 
      return std::visit( [&](auto&& v) { return  v.template SlaterMatricesN<M>(s); }, var ); 
    } 
    template<MEMORY_SPACE M>
    auto SlaterMatricesAux( SpinTypes s ) { 
      return std::visit( [&](auto&& v) { return  v.template SlaterMatricesAux<M>(s); }, var ); 
    } 
    template<MEMORY_SPACE M>
    auto SlaterMatricesAux( SpinTypes s ) const { 
      return std::visit( [&](auto&& v) { return  v.template SlaterMatricesAux<M>(s); }, var ); 
    } 

    template<MEMORY_SPACE M>
    auto getFields() { 
      return std::visit( [&](auto&& v) { return  v.template getFields<M>(); }, var ); 
    } 
    template<MEMORY_SPACE M>
    auto getFields(int ip) { 
      return std::visit( [&](auto&& v) { return  v.template getFields<M>(ip); }, var ); 
    } 
    template<MEMORY_SPACE M>
    auto getWeightFactors() { 
      return std::visit( [&](auto&& v) { return  v.template getWeightFactors<M>(); }, var ); 
    } 
    template<MEMORY_SPACE M>
    auto getWeightHistory() { 
      return std::visit( [&](auto&& v) { return  v.template getWeightHistory<M>(); }, var ); 
    } 
    
  private:

#if defined(ENABLE_DEVICE)   
    std::variant<WalkerSetBase<HOST_MEMORY>,WalkerSetBase<DEVICE_MEMORY>> var;
#else
    std::variant<WalkerSetBase<HOST_MEMORY>> var;
#endif

};

// MAM: move to factory or utils file, this will remain a template even if we instantiate above
template<MEMORY_SPACE _M_>
inline decltype(auto) make_WalkerSet(
                std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi_,
                ptree pt,
                AFQMCInfo& info,
                std::shared_ptr<utils::RandomGenerator_t> r)
{
#if defined(ENABLE_DEVICE)   
  static_assert(_M_ == DEVICE_MEMORY or _M_ == HOST_MEMORY, "Memory space mismatch.");
#else
  static_assert(_M_ == HOST_MEMORY, "Memory space mismatch.");
#endif
  return WalkerSet( WalkerSetBase<_M_>(_mpi_,pt,info,r) );
}

} // namespace afqmc

} // namespace sfqmc

