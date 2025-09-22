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

#ifndef BATCHED_DOT_WABN_WBAN_H
#define BATCHED_DOT_WABN_WBAN_H

#include <cassert>
#include <complex>
#include "Numerics/detail/HIP/Kernels/hip_settings.h"

namespace kernels
{
void batched_dot_wabn_wban(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<double> const* alpha,
                           std::complex<double> const* Tab,
                           std::complex<double>* y,
                           int incy);
void batched_dot_wabn_wban(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<float> const* alpha,
                           std::complex<float> const* Tab,
                           std::complex<float>* y,
                           int incy);
void batched_dot_wabn_wban(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<float> const* alpha,
                           std::complex<float> const* Tab,
                           std::complex<double>* y,
                           int incy);

void batched_dot_wanb_wbna(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<double> const* alpha,
                           std::complex<double> const* Tab,
                           std::complex<double>* y,
                           int incy);
void batched_dot_wanb_wbna(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<float> const* alpha,
                           std::complex<float> const* Tab,
                           std::complex<float>* y,
                           int incy);
void batched_dot_wanb_wbna(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<float> const* alpha,
                           std::complex<float> const* Tab,
                           std::complex<double>* y,
                           int incy);

} // namespace kernels

#endif
