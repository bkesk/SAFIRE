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

#ifndef SFQMC_AFQMC_SHAREDWALKERSET_H
#define SFQMC_AFQMC_SHAREDWALKERSET_H

#include <random>
#include <type_traits>
#include <memory>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "Utilities/Random.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/taskgroup.h"

#include "AFQMC/Walkers/Walkers.hpp"
#include "AFQMC/Walkers/WalkerControl.hpp"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "AFQMC/Walkers/WalkerSetBase.h"

namespace sfqmc
{
namespace afqmc
{
/*
 * Class that contains and handles walkers.
 * Implements communication, load balancing, and I/O operations.   
 * Walkers are always accessed through the handler.
 */
class SharedWalkerSet : public WalkerSetBase<shared_allocator<ComplexType>, ComplexType*>
{
public:
  using Base = WalkerSetBase<shared_allocator<ComplexType>, ComplexType*>;

  /// constructor
  SharedWalkerSet(afqmc::TaskGroup_& tg_, ptree pt, AFQMCInfo& info, utils::RandomGenerator_t* r)
      : Base(tg_,
             pt,
             info,
             r,
             shared_allocator<ComplexType>{tg_.TG_local()},
             shared_allocator<ComplexType>{tg_.TG_local()})
  {}

  /// destructor
  ~SharedWalkerSet() {}

  SharedWalkerSet(SharedWalkerSet const& other) = delete;
  SharedWalkerSet(SharedWalkerSet&& other)      = default;
  SharedWalkerSet& operator=(SharedWalkerSet const& other) = delete;
  SharedWalkerSet& operator=(SharedWalkerSet&& other) = delete;
};

} // namespace afqmc

} // namespace sfqmc

//#include "AFQMC/Walkers/SharedWalkerSet.icc"

#endif
