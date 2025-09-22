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

#include <complex>

namespace kernels
{
// +=
void op_plus(double* x, double inc);
void op_plus(float* x, float inc);
void op_plus(std::complex<double>* x, std::complex<double> inc);
void op_plus(std::complex<float>* x, std::complex<float> inc);

// -=
void op_minus(double* x, double inc);
void op_minus(float* x, float inc);
void op_minus(std::complex<double>* x, std::complex<double> inc);
void op_minus(std::complex<float>* x, std::complex<float> inc);

// *=
void op_times(double* x, double inc);
void op_times(float* x, float inc);
void op_times(std::complex<double>* x, std::complex<double> inc);
void op_times(std::complex<float>* x, std::complex<float> inc);

// /=
void op_div(double* x, double inc);
void op_div(float* x, float inc);
void op_div(std::complex<double>* x, std::complex<double> inc);
void op_div(std::complex<float>* x, std::complex<float> inc);

} // namespace kernels
