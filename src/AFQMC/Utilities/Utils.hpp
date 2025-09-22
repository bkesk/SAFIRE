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

#ifndef AFQMC_UTILITIES_UTILS_HPP
#define AFQMC_UTILITIES_UTILS_HPP

#include <numeric>
#include <stack>
#include <iostream>
#include <fstream>
#include <complex>
#include <list>
#include <algorithm>

#include "mpi.h"

#include "config.0.h"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "Utilities/app_loggers.h"
#include "Numerics/detail/utilities.hpp"

#ifdef ENABLE_CUDA
#include "Memory/device_pointers.hpp"
#include "Memory/CUDA/cuda_arch.h"
#include "Numerics/device_kernels.hpp"
#include "cuda_runtime.h"
#elif ENABLE_HIP
#include "Memory/device_pointers.hpp"
#include "Memory/HIP/hip_arch.h"
#include "Numerics/device_kernels.hpp"
#include "hip/hip_runtime.h"
#endif

namespace sfqmc
{
namespace afqmc
{

inline bool file_exists(const std::string& name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

template<class Iter, class Compare>
void parallel_inplace_merge(int np, int rk, Iter* beg, Iter* mid, Iter* end, MPI_Comm comm, Compare comp)
{
  if (np == 1)
  {
    std::inplace_merge(beg, mid, end, comp);
    return;
  }

  MPI_Barrier(comm);

  Iter *p1, *p2;
  if (std::distance(beg, mid) >= std::distance(mid, end))
  {
    p1      = beg + std::distance(beg, mid) / 2;
    auto it = std::lower_bound(mid, end, *p1, comp);
    p2      = &(*it);
  }
  else
  {
    p2      = mid + std::distance(mid, end) / 2;
    auto it = std::lower_bound(beg, mid, *p2, comp);
    p1      = &(*it);
  }

  MPI_Barrier(comm);
  if (rk == 0)
    std::rotate(p1, mid, p2);
  MPI_Barrier(comm);

  mid = p1 + std::distance(mid, p2);

  if (rk < np / 2)
    parallel_inplace_merge<Iter, Compare>(np / 2, rk, beg, p1, mid, comm, comp);
  else
    parallel_inplace_merge<Iter, Compare>(np / 2, rk - np / 2, mid, p2, end, comm, comp);
}

// Simple circular buffer object.
// Assumed to be of fixed size and should not be resized / reallocated.
// Access elements of the buffer through member function `current'.
template<class T>
class CircularBuffer
{
public:
  // default constructor.
  CircularBuffer() : nelem(0), head(0), buffer(0){};
  // construct 2D buffer. Useful for indexing buffer of vectors or objects.
  CircularBuffer(int nelements, int ncols)
  {
    buffer.resize(nelements, T(ncols));
    nelem = nelements;
    head  = 0;
  }
  // construct 1D buffer.
  CircularBuffer(int nelements)
  {
    buffer.resize(nelements);
    nelem = nelements;
    head  = 0;
  }
  // copy
  CircularBuffer& operator=(const CircularBuffer& cb)
  {
    if (this == &cb)
    {
      return *this;
    }
    else
    {
      nelem  = cb.nelem;
      head   = cb.head;
      buffer = cb.buffer;
    }
  }
  // destructor
  ~CircularBuffer(){};
  // get pointer to current entry (defined by head) in buffer.
  T* current() { return &buffer[head]; }
  // forward traversal.
  void increment() { head = (head + 1) % nelem; }
  // backward traversal.
  void decrement() { head = (head - 1 + nelem) % nelem; }
  // reset head to original location.
  void reset() { head = 0; }

private:
  // number of elements in buffer.
  int nelem;
  // current entry in buffer.
  int head;
  std::vector<T> buffer;
};

template<typename IType, typename integer>
void balance_partition_ordered_set(integer N, IType const* indx, std::vector<IType>& subsets)
{
  int64_t avg = 0;

  if (*(indx + N) == 0)
    exit(1);

  IType nsets = subsets.size() - 1;
  IType i0    = 0;
  IType iN    = N;
  while (*(indx + i0) == *(indx + i0 + 1))
    i0++;
  while (*(indx + iN - 1) == *(indx + iN))
    iN--;
  avg = static_cast<int64_t>(*(indx + iN)) - static_cast<int64_t>(*(indx + i0));
  avg /= nsets;

  // no c++14 :-(
  //    template<class Iter>
  //    auto partition = [=] (IType i0, IType iN, int n, Iter vals) {
  auto partition = [=](IType i0_, IType iN_, int n, typename std::list<IType>::iterator vals) {
    // finds optimal position for subsets[i]
    auto step = [=](IType i0__, IType iN__, IType& ik) {
      IType imin  = ik;
      ik          = i0__ + 1;
      double v1   = double(std::abs(static_cast<int64_t>(*(indx + ik)) - static_cast<int64_t>(*(indx + i0__)) - avg));
      double v2   = double(std::abs(static_cast<int64_t>(*(indx + iN__)) - static_cast<int64_t>(*(indx + ik)) - avg));
      double vmin = v1 * v1 + v2 * v2;
      for (int k = i0__ + 2, kend = iN__; k < kend; k++)
      {
        v1       = double(std::abs(static_cast<int64_t>(*(indx + k)) - static_cast<int64_t>(*(indx + i0__)) - avg));
        v2       = double(std::abs(static_cast<int64_t>(*(indx + iN__)) - static_cast<int64_t>(*(indx + k)) - avg));
        double v = v1 * v1 + v2 * v2;
        if (v < vmin)
        {
          vmin = v;
          ik   = k;
        }
      }
      return ik != imin;
    };

    if (n == 2)
    {
      *vals = i0_ + 1;
      step(i0_, iN_, *vals);
      return;
    }

    std::vector<IType> set(n + 1);
    set[0] = i0;
    set[n] = iN;
    for (int i = n - 1; i >= 1; i--)
      set[i] = iN + i - n;
    bool changed;
    do
    {
      changed = false;
      for (IType i = 1; i < n; i++)
        changed |= step(set[i - 1], set[i + 1], set[i]);
    } while (changed);

    std::copy_n(set.begin() + 1, n - 1, vals);

    return;
  };

  // dummy factorization
  std::stack<IType> factors;
  IType n0 = nsets;
  for (IType i = 2; i <= nsets; i++)
  {
    while (n0 % i == 0)
    {
      factors.push(i);
      n0 /= i;
    }
    if (n0 == 1)
      break;
  }
  RUNTIME_CHECK(n0 == 1, "");

  std::list<IType> sets;
  sets.push_back(i0);
  sets.push_back(iN);

  while (factors.size() > 0)
  {
    auto ns = factors.top();
    factors.pop();

    // divide all current partitions into ns sub-partitions
    typename std::list<IType>::iterator it = sets.begin();
    it++;
    for (; it != sets.end(); it++)
    {
      typename std::list<IType>::iterator its = it;
      its--;
      auto i_0 = *its;
      its     = sets.insert(it, std::size_t(ns - 1), i_0 + 1);
      partition(i_0, *it, ns, its);
    }
  }

  typename std::list<IType>::iterator it    = sets.begin();
  typename std::vector<IType>::iterator itv = subsets.begin();
  for (; itv < subsets.end(); itv++, it++)
    *itv = *it;

  return;
}

template<typename IType>
void balance_partition_ordered_set(std::vector<IType> const& indx, std::vector<IType>& subsets)
{
  if (indx.size() < 2 || subsets.size() < 2)
    return;
  balance_partition_ordered_set(indx.size() - 1, indx.data(), subsets);
}

template<typename IType>
void balance_partition_ordered_set_wcounts(std::vector<IType> const& counts, std::vector<IType>& subsets)
{
  if (counts.size() == 0 || subsets.size() < 2)
    return;
  subsets.resize(counts.size() + 1);
  std::vector<IType> indx(counts.size() + 1);
  IType cnt = 0;
  auto it   = indx.begin();
  *it++     = 0;
  for (auto& v : counts)
    *it++ = (cnt += v);
  balance_partition_ordered_set(counts.size(), indx.data(), subsets);
}

template<class Vec,
         typename VType,
	 typename = typename std::enable_if_t<Vec::dimensionality==1>
        >
void variance_based_truncation(Vec&& v, VType scale_) 
{
  auto trunc = [](auto v_, auto m, auto s) {
    if( v_ > m+s )
      return m+s; 
    else if( v_ < m-s )
      return m-s; 
    else
      return v_;
  };
  // truncate real and complex parts independently for now, modify if needed later
  using Type = typename std::decay_t<Vec>::element;
  using RType = typename remove_complex<Type>::type;  // or decltype(v[0])
  RType av(0.0),var(0.0), rscl(scale_);
  int nz=v.size();
  for(int i=0; i<nz; i++) av += ma::real(v[i]);
  av /= nz;
  for(int i=0; i<nz; i++) var += (ma::real(v[i]) - av)*(ma::real(v[i]) - av);
  var /= nz;
  RType sig = rscl*std::sqrt(var);
  // in case large deviations exist, which can dignificantly modify the variance,
  // I'm recalculating the variance with the requested truncation
  RType av1(0.0),var1(0.0);
  for(int i=0; i<nz; i++) av1 += (trunc(ma::real(v[i]),av,sig)); 
  av1 /= nz;
  for(int i=0; i<nz; i++) var1 += (trunc(ma::real(v[i]),av,sig) - av1)*(trunc(ma::real(v[i]),av,sig) - av1);
  var1 /= nz;
  RType sig1 = rscl*std::sqrt(var1);
  for(int i=0; i<nz; i++) {
//    RType t = std::abs(ma::real(v[i]) - trunc(ma::real(v[i]),av1,sig1));
//    if(t > 1e-4) {
//      app_log(1,"Truncating v[i]: {}, tr[v[i]]: {}, av: {}, av1: {}, sig: {}, sig1:{} ",ma::real(v[i]),trunc(ma::real(v[i]),av1,sig1),av,av1,sig,sig1);
//      print=true;
//    }
    v[i] = trunc(ma::real(v[i]),av1,sig1);
  }
//  if(print)
//    for(int i=0; i<nz; i++)
//      app_log(1,"{} {}",i,ma::real(v[i]));
}

#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/resource.h>
#endif

inline size_t freemem()
{
#ifdef __linux__
  struct sysinfo si;
  sysinfo(&si);
  si.freeram += si.bufferram;
  return si.freeram >> 20;
#else
  return 0;
#endif
}

inline void memory_report()
{
  app_log(2, "\n --> CPU Memory Available: {} \n ", freemem());
#ifdef ENABLE_CUDA
  size_t free_, tot_;
  cudaMemGetInfo(&free_, &tot_);
  app_log(2, " --> GPU Memory Available: {},  Total in MB: {} ", 
		  free_ / 1024.0 / 1024.0, tot_ / 1024.0 / 1024.0 ); 
#elif ENABLE_HIP
  size_t free_, tot_;
  hipMemGetInfo(&free_, &tot_);
  app_log(2, " --> GPU Memory Available: {},  Total in MB: {} ", 
		  free_ / 1024.0 / 1024.0, tot_ / 1024.0 / 1024.0 ); 
#endif
}

} // namespace afqmc

} // namespace sfqmc

#endif
