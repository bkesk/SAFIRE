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

#ifndef NUMERICS_TENSOR_OPERATIONS_CUDA_HPP
#define NUMERICS_TENSOR_OPERATIONS_CUDA_HPP

#include "Numerics/detail/CUDA/Kernels/AGiwj_BGjwi_CG.cuh"

namespace ma
{

using sfqmc::afqmc::is_device_array;

template<typename T, typename Q>
void KaKjw_to_KKwaj(int nwalk,
                    int nkpts,
                    int npol,
                    int nmo_max,
                    int nmo_tot,
                    int nocc_max,
                    device::device_pointer<int> nmo,
                    device::device_pointer<int> nmo0,
                    device::device_pointer<int> nocc,
                    device::device_pointer<int> nocc0,
                    device::device_pointer<Q> A,
                    device::device_pointer<T> B,
		    device_cuda_backend)
{ 
  kernels::KaKjw_to_KKwaj(nwalk, nkpts, npol, nmo_max, nmo_tot, nocc_max, raw_pointer_cast(nmo), raw_pointer_cast(nmo0),                    
                          raw_pointer_cast(nocc), raw_pointer_cast(nocc0), raw_pointer_cast(A), raw_pointer_cast(B));
}

template<typename T, typename Q>
void KaKjw_to_QKajw(int nwalk,
                    int nkpts,
                    int npol,
                    int nmo_max,
                    int nmo_tot,
                    int nocc_max,
                    device::device_pointer<int> nmo,
                    device::device_pointer<int> nmo0,
                    device::device_pointer<int> nocc,
                    device::device_pointer<int> nocc0,
                    device::device_pointer<int> QKtok2,
                    device::device_pointer<Q> A,
                    device::device_pointer<T> B,
		    device_cuda_backend)
{
  kernels::KaKjw_to_QKajw(nwalk, nkpts, npol, nmo_max, nmo_tot, nocc_max, raw_pointer_cast(nmo), raw_pointer_cast(nmo0),
                          raw_pointer_cast(nocc), raw_pointer_cast(nocc0), raw_pointer_cast(QKtok2), raw_pointer_cast(A), raw_pointer_cast(B));
}

template<typename T, typename Q>
void vKKwij_to_vwKiKj(int nwalk,
                      int nkpts,
                      int nmo_max,
                      int nmo_tot,
                      device::device_pointer<int> kk,
                      device::device_pointer<int> nmo,
                      device::device_pointer<int> nmo0,
                      device::device_pointer<Q> A,
                      device::device_pointer<T> B,
		      device_cuda_backend)
{
  kernels::vKKwij_to_vwKiKj(nwalk, nkpts, nmo_max, nmo_tot, raw_pointer_cast(kk), raw_pointer_cast(nmo), raw_pointer_cast(nmo0),
                            raw_pointer_cast(A), raw_pointer_cast(B));
}

template<typename T, typename Q>
void transpose_wabn_to_wban(int nwalk, int na, int nb, int nchol, device::device_pointer<T> Tab, device::device_pointer<Q> Tba, device_cuda_backend)
{
  kernels::transpose_wabn_to_wban(nwalk, na, nb, nchol, raw_pointer_cast(Tab), raw_pointer_cast(Tba));
}

// C[n][w] = sum_a Aup[ I[n] ][a] B[w][a][ J[n] ]
// where I[n] = n2IJ[n]/M
//       J[n] = n2IJ[n]%M 
template<typename T, typename Q1, typename Q2>
void getGIJ_impl(int nw, int nIJ, int nspin, int M, int nel_a, int nel_b,
        device::device_pointer<Q1> Aup, int ldau, device::device_pointer<Q1> Adn, int ldad,
	device::device_pointer<Q2> B, int ldb, long strideB,
        device::device_pointer<T> C, int ldc, device::device_pointer<size_t const> n2IJ, device_cuda_backend)
{
  kernels::getGIJ_impl(nw,nIJ,nspin,M,nel_a,nel_b,raw_pointer_cast(Aup),ldau,raw_pointer_cast(Adn),ldad,
                       raw_pointer_cast(B), ldb, strideB, raw_pointer_cast(C), ldc, raw_pointer_cast(n2IJ));
}

// C[g] = sum_n,w,i,j Xw[w] * A[n][g][i][w][j] * B[n][g][j][w][i]
template<class MatX, class MatA, class MatB, class MatC,
    typename = std::enable_if_t< is_device_array<std::decay_t<MatX>>::value >,
    typename = std::enable_if_t< is_device_array<std::decay_t<MatA>>::value >,
    typename = std::enable_if_t< is_device_array<std::decay_t<MatB>>::value >,
    typename = std::enable_if_t< is_device_array<std::decay_t<MatC>>::value >,
    typename = void
>
inline static void AGiwj_BGjwi_CG(MatX const& Xw, MatA const& AGiwj, MatB const& BGjwi, MatC&& CG, device_cuda_backend)
{
  static_assert(std::decay_t<MatA>::dimensionality == 5, "Wrong dimensionality.");
  static_assert(std::decay_t<MatB>::dimensionality == 5, "Wrong dimensionality.");
  static_assert(std::decay_t<MatX>::dimensionality == 1, "Wrong dimensionality.");
  static_assert(std::decay_t<MatC>::dimensionality == 1, "Wrong dimensionality.");

  RUNTIME_CHECK(AGiwj.size(0) == BGjwi.size(0), "");
  RUNTIME_CHECK(AGiwj.size(1) == BGjwi.size(1), "");
  RUNTIME_CHECK(AGiwj.size(2) == BGjwi.size(4), "");
  RUNTIME_CHECK(AGiwj.size(3) == BGjwi.size(3), "");
  RUNTIME_CHECK(AGiwj.size(4) == BGjwi.size(2), "");
  RUNTIME_CHECK(Xw.size(0) == AGiwj.size(3), "");
  RUNTIME_CHECK(CG.size(0) == AGiwj.size(1), "");

  kernels::AGiwj_BGjwi_CG_impl(AGiwj.size(0), AGiwj.size(1), AGiwj.size(2), AGiwj.size(3), 
        AGiwj.size(4),
        raw_pointer_cast(Xw.origin()), raw_pointer_cast(AGiwj.origin()), raw_pointer_cast(BGjwi.origin()),
        raw_pointer_cast(CG.origin()));
}

}

#endif
