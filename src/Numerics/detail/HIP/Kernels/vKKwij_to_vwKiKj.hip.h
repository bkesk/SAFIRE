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

#ifndef VKKWIJ_TO_VWKIKJ_H
#define VKKWIJ_TO_VWKIKJ_H

#include <complex>

namespace kernels
{
void vKKwij_to_vwKiKj(int nwalk,
                      int nkpts,
                      int nmo_max,
                      int nmo_tot,
                      int* kk,
                      int* nmo,
                      int* nmo0,
                      double const* A,
                      double* B);
void vKKwij_to_vwKiKj(int nwalk,
                      int nkpts,
                      int nmo_max,
                      int nmo_tot,
                      int* kk,
                      int* nmo,
                      int* nmo0,
                      float const* A,
                      float* B);
void vKKwij_to_vwKiKj(int nwalk,
                      int nkpts,
                      int nmo_max,
                      int nmo_tot,
                      int* kk,
                      int* nmo,
                      int* nmo0,
                      float const* A,
                      double* B);
void vKKwij_to_vwKiKj(int nwalk,
                      int nkpts,
                      int nmo_max,
                      int nmo_tot,
                      int* kk,
                      int* nmo,
                      int* nmo0,
                      std::complex<double> const* A,
                      std::complex<double>* B);
void vKKwij_to_vwKiKj(int nwalk,
                      int nkpts,
                      int nmo_max,
                      int nmo_tot,
                      int* kk,
                      int* nmo,
                      int* nmo0,
                      std::complex<float> const* A,
                      std::complex<float>* B);
void vKKwij_to_vwKiKj(int nwalk,
                      int nkpts,
                      int nmo_max,
                      int nmo_tot,
                      int* kk,
                      int* nmo,
                      int* nmo0,
                      std::complex<float> const* A,
                      std::complex<double>* B);
} // namespace kernels

#endif
