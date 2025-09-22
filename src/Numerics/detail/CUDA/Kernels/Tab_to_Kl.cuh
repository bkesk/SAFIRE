////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef TAB_TO_KL_H
#define TAB_TO_KL_H

#include <cassert>
#include <complex>

namespace kernels
{
void Tab_to_Kl(int nwalk, int nocc, int nchol, std::complex<float> const* Tab, std::complex<float>* Kl);

void Tab_to_Kl(int nwalk, int nocc, int nchol, std::complex<double> const* Tab, std::complex<double>* Kl);

void Tanb_to_Kl(int nwalk, int nocc, int nchol, int nact, std::complex<float> const* Tab, std::complex<float>* Kl, int ldkl);

void Tanb_to_Kl(int nwalk,
                int nocc,
                int nchol,
                int nact,
                std::complex<double> const* Tab,
                std::complex<double>* Kl,
		int ldkl);

} // namespace kernels

#endif
