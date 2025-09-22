////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
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
                        int ncholQ,
                        int* kdiag,
                        std::complex<float> const* Tab,
                        std::complex<float>* Kl,
                        int ldkl,
                        std::complex<float>* Kr,
                        int ldkr);

void batched_Tab_to_Klr(int nterms,
                        int nwalk,
                        int nocc,
                        int nchol_max,
                        int ncholQ,
                        int* kdiag,
                        std::complex<double> const* Tab,
                        std::complex<double>* Kl,
                        int ldkl,
                        std::complex<double>* Kr,
                        int ldkr);

void batched_Tanb_to_Klr(int nterms,
                         int nwalk,
                         int nocc,
                         int nchol_max,
                         int ncholQ,
                         int* kdiag,
                         std::complex<float> const* Tab,
                         std::complex<float>* Kl,
                         int ldkl,
                         std::complex<float>* Kr,
                         int ldkr);

void batched_Tanb_to_Klr(int nterms,
                         int nwalk,
                         int nocc,
                         int nchol_max,
                         int ncholQ,
                         int* kdiag,
                         std::complex<double> const* Tab,
                         std::complex<double>* Kl,
                         int ldkl,
                         std::complex<double>* Kr,
                         int ldkr);

} // namespace kernels

#endif
