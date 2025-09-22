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

#ifndef VBIAS_FROM_V1_H
#define VBIAS_FROM_V1_H

#include <cassert>
#include <complex>
#include "Numerics/detail/HIP/Kernels/hip_settings.h"

namespace kernels
{
void vbias_from_v1(int nwalk,
                   int nkpts,
                   int nchol_max,
                   int* Qsym,
                   int* kminus,
                   int* ncholpQ,
                   int* ncholpQ0,
                   std::complex<double> const alpha,
                   std::complex<double> const* v1,
                   std::complex<double>* vb);
void vbias_from_v1(int nwalk,
                   int nkpts,
                   int nchol_max,
                   int* Qsym,
                   int* kminus,
                   int* ncholpQ,
                   int* ncholpQ0,
                   std::complex<float> const alpha,
                   std::complex<float> const* v1,
                   std::complex<float>* vb);
void vbias_from_v1(int nwalk,
                   int nkpts,
                   int nchol_max,
                   int* Qsym,
                   int* kminus,
                   int* ncholpQ,
                   int* ncholpQ0,
                   std::complex<double> const alpha,
                   std::complex<float> const* v1,
                   std::complex<double>* vb);
} // namespace kernels

#endif
