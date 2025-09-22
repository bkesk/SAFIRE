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

#ifndef NUMERICS_TENSOR_OPERATIONS_HPP
#define NUMERICS_TENSOR_OPERATIONS_HPP

#include "Numerics/detail/tensor_operations.hpp"


namespace ma
{

template<class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 4>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 4>
        >
void transpose_wabn_to_wban(MA1 && A, MA2 && B)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == B.size(2), "");
  RUNTIME_CHECK(A.size(2) == B.size(1), "");
  RUNTIME_CHECK(A.size(3) == B.size(3), "");
  RUNTIME_CHECK(A.stride(0) == A.size(1)*A.size(2)*A.size(3), "");
  RUNTIME_CHECK(A.stride(1) == A.size(2)*A.size(3), "");
  RUNTIME_CHECK(A.stride(2) == A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == 1, "");
  RUNTIME_CHECK(B.stride(0) == B.size(1)*B.size(2)*B.size(3), "");
  RUNTIME_CHECK(B.stride(1) == B.size(2)*B.size(3), "");
  RUNTIME_CHECK(B.stride(2) == B.size(3), "");
  RUNTIME_CHECK(B.stride(3) == 1, "");
  ma::transpose_wabn_to_wban(A.size(0), A.size(1), A.size(2), A.size(3), 
		  pointer_dispatch(A.origin()), pointer_dispatch(B.origin()), select_backend<MA2>());
}

template<class MatA,
         class MatB,
         class MatC,
         class MatD,
	 class IVec,
         typename = std::enable_if_t<std::decay_t<MatA>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MatB>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MatC>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MatD>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<IVec>::dimensionality == 1>
        >
void getGIJ(MatA const& Gc, MatB&& GIJ, MatC const& PsiCA, MatD const& PsiCB, IVec const& n2IJ)
{ 
  int nspin  = 2; 
  int NMO    = PsiCA.size(0);
  int nwalk  = Gc.size(0);
  int nel_a  = PsiCA.size(1);
  int nel_b  = PsiCB.size(1);
  int nIJ = n2IJ.size();
  RUNTIME_CHECK(PsiCB.size(0) == NMO, "");
  RUNTIME_CHECK(Gc.size(1)    == (nel_a+nel_b), "");
  RUNTIME_CHECK(Gc.size(2)    == NMO, "");
  RUNTIME_CHECK(GIJ.size(0)   == nIJ, "");
  RUNTIME_CHECK(GIJ.size(1)   == nwalk, "");
  RUNTIME_CHECK(Gc.stride(2) == 1, "");
  RUNTIME_CHECK(GIJ.stride(1) == 1, "");
  RUNTIME_CHECK(PsiCA.stride(1) == 1, "");
  RUNTIME_CHECK(PsiCB.stride(1) == 1, "");

  ma::getGIJ_impl(nwalk, nIJ, nspin, NMO, nel_a, nel_b,
            pointer_dispatch(PsiCA.origin()), PsiCA.stride(0), 
	    pointer_dispatch(PsiCB.origin()), PsiCB.stride(0),
            pointer_dispatch(Gc.origin()), Gc.stride(1), Gc.stride(0), 
            pointer_dispatch(GIJ.origin()), GIJ.stride(0),
	    pointer_dispatch(n2IJ.origin()),
	    select_backend<MatB>());
}

template<class MatA,
         class MatB,
         class MatC,
         class IVec,
         typename = std::enable_if_t<std::decay_t<MatA>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MatB>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MatC>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<IVec>::dimensionality == 1>
        >
void getGIJ(MatA const& Gc, MatB&& GIJ, MatC const& PsiCA, IVec const& n2IJ)
{
  int nspin  = 1;
  int NMO    = PsiCA.size(0);
  int nwalk  = Gc.size(0);
  int nel_a  = PsiCA.size(1);
  int nIJ = n2IJ.size();
  RUNTIME_CHECK(Gc.size(1)    == nel_a, "");
  RUNTIME_CHECK(Gc.size(2)    == NMO, "");
  RUNTIME_CHECK(GIJ.size(0)   == nIJ, "");
  RUNTIME_CHECK(GIJ.size(1)   == nwalk, "");
  RUNTIME_CHECK(Gc.stride(2) == 1, "");
  RUNTIME_CHECK(GIJ.stride(1) == 1, "");
  RUNTIME_CHECK(PsiCA.stride(1) == 1, "");

  ma::getGIJ_impl(nwalk, nIJ, nspin, NMO, nel_a, 0,
            pointer_dispatch(PsiCA.origin()), PsiCA.stride(0),
            pointer_dispatch(PsiCA.origin()), PsiCA.stride(0),
            pointer_dispatch(Gc.origin()), Gc.stride(1), Gc.stride(0),
            pointer_dispatch(GIJ.origin()), GIJ.stride(0),
            pointer_dispatch(n2IJ.origin()),
            select_backend<MatB>());
}

template<class MatX, class MatA, class MatB, class MatC>
void AGiwj_BGjwi_CG(MatX const& Xw, MatA const& AGiwj, MatB const& BGjwi, MatC&& CG)
{
  ma::AGiwj_BGjwi_CG(Xw, AGiwj, BGjwi, std::forward<MatC>(CG), select_backend<MatX>());
}
 

/*******
 *  Simple wrappers to hide dispatch mechanism.
 *  No MA interface for these.
 *******/
 
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
void KaKjw_to_KKwaj(int nwalk, int nkpts, int npol, int nmo_max, int nmo_tot, int nocc_max,
                    T1 nopk, T2 nopk0, T3 nelpk, T4 nelpk0, T5 A, T6 B)
{
  ma::KaKjw_to_KKwaj(nwalk,nkpts,npol,nmo_max,nmo_tot,nocc_max, 
	pointer_dispatch(nopk), pointer_dispatch(nopk0), pointer_dispatch(nelpk), pointer_dispatch(nelpk0), 
	pointer_dispatch(A), pointer_dispatch(B), typename ma_dispatch<T6>::backend{});
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
void KaKjw_to_QKajw(int nwalk, int nkpts, int npol, int nmo_max, int nmo_tot, int nocc_max,
                    T1 nmo, T2 nmo0, T3 nocc, T4 nocc0, T5 QKtok2, T6 A, T7 B)
{
  ma::KaKjw_to_QKajw(nwalk,nkpts,npol,nmo_max,nmo_tot,nocc_max, 
	pointer_dispatch(nmo), pointer_dispatch(nmo0), pointer_dispatch(nocc), pointer_dispatch(nocc0), 
	pointer_dispatch(QKtok2), pointer_dispatch(A), pointer_dispatch(B), typename ma_dispatch<T7>::backend{});
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
void vKKwij_to_vwKiKj(int nwalk, int nkpts, int nmo_max, int nmo_tot, T1 kk, T2 nopk, T3 nopk0, T4 A, T5 B)
{
  ma::vKKwij_to_vwKiKj(nwalk,nkpts,nmo_max,nmo_tot,pointer_dispatch(kk), 
        pointer_dispatch(nopk), pointer_dispatch(nopk0), pointer_dispatch(A), pointer_dispatch(B), 
	typename ma_dispatch<T5>::backend{});
}

} // namespace ma
  
#endif
