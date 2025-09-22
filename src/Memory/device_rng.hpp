/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef MEMORY_DEVICE_RNG_HPP
#define MEMORY_DEVICE_RNG_HPP

#ifdef ENABLE_CUDA
#include "Memory/device_pointers.hpp"
#include "Memory/CUDA/cuda_arch.h"
#include "Numerics/device_kernels.hpp"
#elif ENABLE_HIP
#include "Memory/device_pointers.hpp"
#include "Memory/HIP/hip_arch.h"
#include "Numerics/device_kernels.hpp"
#endif
#include "Utilities/Random.hpp"

namespace utils 
{

#ifdef ENABLE_CUDA
using DeviceRandomGenerator_t = curandGenerator_t;
inline DeviceRandomGenerator_t make_device_rng(RandomGenerator_t::result_type iseed)
{ 
  unsigned long long int v(iseed);
  return arch::make_device_rng(v);
}
#else
using DeviceRandomGenerator_t = std::mt19937;
inline DeviceRandomGenerator_t make_device_rng(RandomGenerator_t::result_type iseed)
{ 
  return DeviceRandomGenerator_t{iseed};
}
#endif

#if defined(ENABLE_DEVICE)

template<class T>
void sampleGaussianFields_n(device::device_pointer<T> V, int n, DeviceRandomGenerator_t& rng)
{
  kernels::sampleGaussianRNG(raw_pointer_cast(V), n, rng);
}

template<class T>
void sampleUniformFields_n(device::device_pointer<T> V, int n, DeviceRandomGenerator_t& rng)
{
  kernels::sampleUniformRNG(raw_pointer_cast(V), n, rng);
}

inline std::vector<RandomGenerator_t::result_type> save(DeviceRandomGenerator_t& rng)
{
  std::vector<RandomGenerator_t::result_type> state;
  std::stringstream str;
//  str << rng;
  std::copy(std::istream_iterator<RandomGenerator_t::result_type>(str),
            std::istream_iterator<RandomGenerator_t::result_type>(),
            std::back_inserter(state));
  return state;
}

inline void load(DeviceRandomGenerator_t& rng,
                 std::vector<RandomGenerator_t::result_type>& state)
{
  std::stringstream str;
  std::copy(state.begin(), state.end(),
            std::ostream_iterator<RandomGenerator_t::result_type>(str, " "));
//  str >> rng;
}

#endif

}

#endif
