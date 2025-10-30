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

#include <numeric>
#include <stack>
#include <iostream>
#include <fstream>
#include <complex>
#include <list>
#include <algorithm>

#include "config.0.h"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "IO/app_loggers.h"

#ifdef ENABLE_CUDA
#include "cuda_runtime.h"
#elif ENABLE_HIP
#include "hip/hip_runtime.h"
#endif

namespace sfqmc
{
namespace afqmc
{

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
  if constexpr (std::is_same_v<Type,RType>) { // real case
    for(int i=0; i<nz; i++) av += v[i];
    av /= nz;
    for(int i=0; i<nz; i++) var += (v[i] - av)*(v[i] - av);
    var /= nz;
    RType sig = rscl*std::sqrt(var);
    // in case large deviations exist, which can dignificantly modify the variance,
    // I'm recalculating the variance with the requested truncation
    RType av1(0.0),var1(0.0);
    for(int i=0; i<nz; i++) av1 += (trunc(v[i],av,sig)); 
    av1 /= nz;
    for(int i=0; i<nz; i++) var1 += (trunc(v[i],av,sig) - av1)*(trunc(v[i],av,sig) - av1);
    var1 /= nz;
    RType sig1 = rscl*std::sqrt(var1);
    for(int i=0; i<nz; i++) {
      v[i] = trunc(v[i],av1,sig1);
    }
  } else {
    for(int i=0; i<nz; i++) av += std::real(v[i]);
    av /= nz;
    for(int i=0; i<nz; i++) var += (std::real(v[i]) - av)*(std::real(v[i]) - av);
    var /= nz;
    RType sig = rscl*std::sqrt(var);
    // in case large deviations exist, which can dignificantly modify the variance,
    // I'm recalculating the variance with the requested truncation
    RType av1(0.0),var1(0.0);
    for(int i=0; i<nz; i++) av1 += (trunc(std::real(v[i]),av,sig));
    av1 /= nz;
    for(int i=0; i<nz; i++) var1 += (trunc(std::real(v[i]),av,sig) - av1)*(trunc(std::real(v[i]),av,sig) - av1);
    var1 /= nz;
    RType sig1 = rscl*std::sqrt(var1);
    for(int i=0; i<nz; i++) {
      v[i] = trunc(std::real(v[i]),av1,sig1);
    }
  }
}

} // namespace afqmc

} // namespace sfqmc

