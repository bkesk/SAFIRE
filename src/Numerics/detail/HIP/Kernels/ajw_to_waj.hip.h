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

#ifndef AJW_TO_WAJ_H
#define AJW_TO_WAJ_H

#include <complex>

namespace kernels
{
void ajw_to_waj(int na, int nj, int nw, int inca, double const* A, double* B);
void ajw_to_waj(int na, int nj, int nw, int inca, float const* A, float* B);
void ajw_to_waj(int na, int nj, int nw, int inca, std::complex<double> const* A, std::complex<double>* B);
void ajw_to_waj(int na, int nj, int nw, int inca, std::complex<float> const* A, std::complex<float>* B);


void transpose_wabn_to_wban(int nwalk,
                            int na,
                            int nb,
                            int nchol,
                            std::complex<double> const* Tab,
                            std::complex<double>* Tba);
void transpose_wabn_to_wban(int nwalk,
                            int na,
                            int nb,
                            int nchol,
                            std::complex<float> const* Tab,
                            std::complex<float>* Tba);
void transpose_wabn_to_wban(int nwalk,
                            int na,
                            int nb,
                            int nchol,
                            std::complex<double> const* Tab,
                            std::complex<float>* Tba);
void transpose_wabn_to_wban(int nwalk,
                            int na,
                            int nb,
                            int nchol,
                            std::complex<float> const* Tab,
                            std::complex<double>* Tba);

} // namespace kernels
#endif
