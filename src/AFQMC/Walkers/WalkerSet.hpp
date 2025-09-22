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

#ifndef SFQMC_AFQMC_WALKERSET_HPP
#define SFQMC_AFQMC_WALKERSET_HPP

#include "AFQMC/Walkers/SharedWalkerSet.hpp"
#include "AFQMC/Walkers/SerialWalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
using WalkerSet = SerialWalkerSet;
#else
using WalkerSet = SharedWalkerSet;
#endif

} // namespace afqmc

} // namespace sfqmc

#endif
