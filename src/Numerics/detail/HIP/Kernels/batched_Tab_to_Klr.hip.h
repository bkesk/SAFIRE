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

#ifndef BATCHED_TAB_TO_KLR_H
#define BATCHED_TAB_TO_KLR_H

#include <cassert>
#include <complex>

namespace kernels
{
void batched_Tab_to_Klr(int nterms,
                        int nwalk,
                        int nocc,
                        int nchol_max,
                        int nchol_tot,
                        int ncholQ,
                        int ncholQ0,
                        int* kdiag,
                        std::complex<float> const* Tab,
                        std::complex<float>* Kl,
                        std::complex<float>* Kr);

void batched_Tab_to_Klr(int nterms,
                        int nwalk,
                        int nocc,
                        int nchol_max,
                        int nchol_tot,
                        int ncholQ,
                        int ncholQ0,
                        int* kdiag,
                        std::complex<double> const* Tab,
                        std::complex<double>* Kl,
                        std::complex<double>* Kr);

void batched_Tanb_to_Klr(int nterms,
                         int nwalk,
                         int nocc,
                         int nchol_max,
                         int nchol_tot,
                         int ncholQ,
                         int ncholQ0,
                         int* kdiag,
                         std::complex<float> const* Tab,
                         std::complex<float>* Kl,
                         std::complex<float>* Kr);

void batched_Tanb_to_Klr(int nterms,
                         int nwalk,
                         int nocc,
                         int nchol_max,
                         int nchol_tot,
                         int ncholQ,
                         int ncholQ0,
                         int* kdiag,
                         std::complex<double> const* Tab,
                         std::complex<double>* Kl,
                         std::complex<double>* Kr);

} // namespace kernels

#endif
