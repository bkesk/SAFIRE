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

#ifndef CONSTRUCT_X_H
#define CONSTRUCT_X_H

#include <cassert>
#include <complex>

namespace kernels
{
void construct_X(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 std::complex<double> const* vMF,
                 std::complex<double> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 std::complex<double>* X);
void construct_X(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 std::complex<double> const* vMF,
                 std::complex<float> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 std::complex<float>* X);
void construct_X(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 std::complex<double> const* vMF,
                 std::complex<float> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 std::complex<double>* X);
void construct_X(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 std::complex<double> const* vMF,
                 std::complex<double> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 std::complex<float>* X);


} // namespace kernels

#endif
