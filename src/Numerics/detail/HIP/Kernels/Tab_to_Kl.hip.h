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

#ifndef TAB_TO_KL_H
#define TAB_TO_KL_H

#include <cassert>
#include <complex>

namespace kernels
{
void Tab_to_Kl(int nwalk, int nocc, int nchol, std::complex<float> const* Tab, std::complex<float>* Kl);

void Tab_to_Kl(int nwalk, int nocc, int nchol, std::complex<double> const* Tab, std::complex<double>* Kl);

void Tanb_to_Kl(int nwalk, int nocc, int nchol, int nchol_tot, std::complex<float> const* Tab, std::complex<float>* Kl);

void Tanb_to_Kl(int nwalk,
                int nocc,
                int nchol,
                int nchol_tot,
                std::complex<double> const* Tab,
                std::complex<double>* Kl);

} // namespace kernels

#endif
