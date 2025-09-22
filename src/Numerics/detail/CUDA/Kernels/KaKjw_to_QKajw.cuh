////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef KAKJW_to_QKAJW_H
#define KAKJW_to_QKAJW_H

#include <cassert>
#include <complex>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"

namespace kernels
{
template<typename T1, typename T2>
void KaKjw_to_QKajw(int nwalk,
                    int nkpts,
                    int npol,
                    int nmo_max,
                    int nmo_tot,
                    int nocc_max,
                    int* nmo,
                    int* nmo0,
                    int* nocc,
                    int* nocc0,
                    int* QKtok2,
                    T1 const* A,
                    T2* B);
template<typename T1, typename T2>
void KaKjw_to_QKajw(int nwalk,
                    int nkpts,
                    int npol,
                    int nmo_max,
                    int nmo_tot,
                    int nocc_max,
                    int* nmo,
                    int* nmo0,
                    int* nocc,
                    int* nocc0,
                    int* QKtok2,
                    std::complex<T1> const* A,
                    std::complex<T2>* B);
} // namespace kernels

#endif
