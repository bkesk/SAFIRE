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

#pragma once

#include <ctime>
#include <vector>
#include <random>
#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "arch/arch.h"
#include "nda/nda.hpp"
#include "numerics/nda_functions.hpp"

#if defined(ENABLE_DEVICE)
#include "curand.h"
#endif

namespace sfqmc {
namespace utils
{

/*
using RandomGenerator_t = std::mt19937;
#if defined(ENABLE_CUDA)
using DeviceRandomGenerator_t = curandGenerator_t;
inline DeviceRandomGenerator_t make_device_rng(RandomGenerator_t::result_type iseed)
{
  unsigned long long int v(iseed);
  return ::sfqmc::cuda::make_device_rng(v);
}
#else
using DeviceRandomGenerator_t = std::mt19937;
inline DeviceRandomGenerator_t make_device_rng(RandomGenerator_t::result_type iseed)
{
  return DeviceRandomGenerator_t{iseed};
}
#endif
*/

#if defined(ENABLE_DEVICE)
template<MEMORY_SPACE MEM = HOST_MEMORY>
using RandomGenerator_t = std::conditional_t<MEM==DEVICE_MEMORY,curandGenerator_t,std::mt19937>; 
#else
template<MEMORY_SPACE MEM = HOST_MEMORY>
using RandomGenerator_t = std::mt19937; 
#endif

template<MEMORY_SPACE MEM>
auto make_rng(RandomGenerator_t<>::result_type iseed)
{
  if constexpr (MEM==HOST_MEMORY) {
    return RandomGenerator_t<>(iseed);
  } else {
    unsigned long long int v(iseed);
    return ::sfqmc::cuda::make_device_rng(v);
  }
}

// Return Nth primer number
template<typename UInt>
UInt get_prime(UInt N)
{
  utils::check(not(N < 1),"N must be positive, provided N = {}", N);
  if(N==UInt(1)) return UInt(1);
  if(N==UInt(2)) return UInt(2);
  if(N==UInt(3)) return UInt(3);
  N-=UInt(3);
  std::vector<UInt> primes;
  primes.reserve(4096);
  primes.push_back(3);
  UInt largest = 3;
  while (N) { 
    largest += 2; 
    bool is_prime = true;
    for (int j = 0; j < primes.size(); j++) { 
      if (largest % primes[j] == 0) { 
        is_prime = false;
        break;
      }
      else if (primes[j] * primes[j] > largest) { 
        break;
      }
    }
    if (is_prime) { 
      primes.push_back(largest);
      N--;
    }
  }  
  return largest;
}

inline typename RandomGenerator_t<HOST_MEMORY>::result_type make_seed(boost::mpi3::communicator& comm)
{
  using result_type = typename RandomGenerator_t<HOST_MEMORY>::result_type;
  result_type baseoffset;
  if (comm.root())
    baseoffset = static_cast<int>(static_cast<result_type>(std::time(0)) % 1024);
  comm.broadcast_value(baseoffset);
  baseoffset += result_type(comm.rank());
  return get_prime<result_type>(baseoffset); 
}

inline typename RandomGenerator_t<HOST_MEMORY>::result_type split_seed(int seed, boost::mpi3::communicator& comm)
{
  /*
  Splits the given seed accross the given comm to guarantee that each MPI rank has a unique, but  
    reproducible seed
  */
  using result_type = typename RandomGenerator_t<HOST_MEMORY>::result_type;
  result_type baseoffset = seed;
  baseoffset += result_type(comm.rank());
  return get_prime<result_type>(baseoffset);
} 

template<nda::MemoryVector Vec>
requires( nda::mem::on_host<Vec> )
void sampleUniformFields(Vec && V, RandomGenerator_t<HOST_MEMORY>& rng)
{
  using T = nda::get_value_t<decltype(V)>;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  for(long i=0; i<V.extent(0); i++)
    V(i) = T(distribution(rng));
}

#if defined(ENABLE_DEVICE)
template<nda::MemoryVector Vec>
requires( nda::mem::on_device<Vec> )
void sampleUniformFields(Vec && V, RandomGenerator_t<DEVICE_MEMORY>& rng)
{
  using T = nda::get_value_t<Vec>;
  static_assert(std::is_same_v<T,double> or std::is_same_v<T,std::complex<double>>,
                "sampleUniformFields: Type not implemented.");
  if constexpr (std::is_same_v<T,double>) {
    cuda::curand_check(curandGenerateUniformDouble(rng, reinterpret_cast<double*>(V.data()), V.size()),
                         "curandGenerateUniformDouble");
  } else if constexpr (std::is_same_v<T,std::complex<double>>) {
    cuda::curand_check(curandGenerateUniformDouble(rng, reinterpret_cast<double*>(V.data()), 2 * V.size()),
                         "curandGenerateUniformDouble");
    nda::zero_imag(V);
  } 
  arch::synchronize_if_set();
}
#endif

inline std::vector<RandomGenerator_t<HOST_MEMORY>::result_type> save(RandomGenerator_t<HOST_MEMORY>& rng) 
{
  std::vector<RandomGenerator_t<HOST_MEMORY>::result_type> state;
  std::stringstream str;
  str << rng;
  std::copy(std::istream_iterator<RandomGenerator_t<HOST_MEMORY>::result_type>(str), 
	    std::istream_iterator<RandomGenerator_t<HOST_MEMORY>::result_type>(), 
	    std::back_inserter(state));  
  return state;
}

inline void load(RandomGenerator_t<HOST_MEMORY>& rng, 
		 std::vector<RandomGenerator_t<HOST_MEMORY>::result_type>& state) 
{
  std::stringstream str;
  std::copy(state.begin(), state.end(),
	    std::ostream_iterator<RandomGenerator_t<HOST_MEMORY>::result_type>(str, " "));
  str >> rng;
}

}
}

