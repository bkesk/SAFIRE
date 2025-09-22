////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef VKKWIJ_TO_VWKIKJ_H
#define VKKWIJ_TO_VWKIKJ_H

#include <complex>

namespace kernels
{
template<typename T1, typename T2>
void vKKwij_to_vwKiKj(int nwalk,
                      int nkpts,
                      int nmo_max,
                      int nmo_tot,
                      int* kk,
                      int* nmo,
                      int* nmo0,
                      T1 const* A,
                      T2* B);
template<typename T1, typename T2>
void vKKwij_to_vwKiKj(int nwalk,
                      int nkpts,
                      int nmo_max,
                      int nmo_tot,
                      int* kk,
                      int* nmo,
                      int* nmo0,
                      std::complex<T1> const* A,
                      std::complex<T2>* B);
} // namespace kernels

#endif
