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
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "arch/arch.h"

#if defined(ENABLE_CUDA)

#endif

namespace sfqmc {
namespace utils
{

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

inline typename RandomGenerator_t::result_type make_seed(boost::mpi3::communicator& comm)
{
  using result_type = typename RandomGenerator_t::result_type;
  result_type baseoffset;
  if (comm.root())
    baseoffset = static_cast<int>(static_cast<result_type>(std::time(0)) % 1024);
  comm.broadcast_value(baseoffset);
  baseoffset += result_type(comm.rank());
  return get_prime<result_type>(baseoffset); 
}

inline typename RandomGenerator_t::result_type split_seed(int seed, boost::mpi3::communicator& comm)
{
  /*
  Splits the given seed accross the given comm to guarantee that each MPI rank has a unique, but  
    reproducible seed
  */
  using result_type = typename RandomGenerator_t::result_type;
  result_type baseoffset = seed;
  baseoffset += result_type(comm.rank());
  return get_prime<result_type>(baseoffset);
} 

template<class Vec,
         typename = typename std::enable_if_t<std::decay<Vec>::type::dimensionality == 1>>
void sampleGaussianFields(Vec&& V, RandomGenerator_t& rng)
{
  std::normal_distribution<double> distribution(0.0,1.0);
  for(auto& v: V) v = distribution(rng);
}

template<class Mat,
         typename = typename std::enable_if_t<(std::decay<Mat>::type::dimensionality > 1)>,
         typename = void>
void sampleGaussianFields(Mat&& M, RandomGenerator_t& rng)
{
  for (int i = 0, iend = M.size(0); i < iend; ++i)
    sampleGaussianFields(M[i], rng);
}

template<class T>
void sampleGaussianFields_n(T* V, int n, RandomGenerator_t& rng)
{
  using real_t = typename sfqmc::afqmc::remove_complex<T>::type;
  std::normal_distribution<real_t> distribution(0.0, 1.0);  
  for(int i=0; i<n; i++)
    V[i] = T(distribution(rng));
}

template<class T>
void sampleUniformFields_n(T* V, int n, RandomGenerator_t& rng)
{
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  for(int i=0; i<n; i++)
    V[i] = T(distribution(rng));
}

inline std::vector<RandomGenerator_t::result_type> save(RandomGenerator_t& rng) 
{
  std::vector<RandomGenerator_t::result_type> state;
  std::stringstream str;
  str << rng;
  std::copy(std::istream_iterator<RandomGenerator_t::result_type>(str), 
	    std::istream_iterator<RandomGenerator_t::result_type>(), 
	    std::back_inserter(state));  
  return state;
}

inline void load(RandomGenerator_t& rng, 
		 std::vector<RandomGenerator_t::result_type>& state) 
{
  std::stringstream str;
  std::copy(state.begin(), state.end(),
	    std::ostream_iterator<RandomGenerator_t::result_type>(str, " "));
  str >> rng;
}

}
}

