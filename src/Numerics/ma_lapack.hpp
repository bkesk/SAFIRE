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

#ifndef MA_LAPACK_HPP
#define MA_LAPACK_HPP

#include "AFQMC/Utilities/type_conversion.hpp"
#include "Numerics/detail/lapack.hpp"
#include "multi/array_ref.hpp"
#include <cassert>

namespace ma
{

// MAM: Use buffer allocators to allocate directly at the lower level.
//      Remove workspace routines and eliminate interface with buffers.
//      Also add Strided versions through regular interface, specialized for 3D arrays

template<class MultiArray2D>
int getrf_optimal_workspace_size(MultiArray2D&& A)
{
  RUNTIME_CHECK(A.stride(0) > 0, "");
  RUNTIME_CHECK(A.stride(1) == 1, "");

  int res;
  ma::getrf_bufferSize(A.size(1), A.size(0), A.origin(), A.stride(0), res,
		       select_backend<MultiArray2D>());
  return res;
}

template<class MultiArray2D, class Array1D, class Buffer,
	typename = std::enable_if_t<std::decay_t<MultiArray2D>::dimensionality == 2>
	>
MultiArray2D&& getrf(MultiArray2D&& m, Array1D& pivot, Buffer&& WORK)
{
  RUNTIME_CHECK(m.stride(0) >= std::max(std::size_t(1), std::size_t(m.size(1))), "");
  RUNTIME_CHECK(m.stride(1) == 1, "");
  RUNTIME_CHECK(pivot.size() >= std::min(m.size(1), m.size(0) + 1), "");

  int status = -1;
  ma::getrf(m.size(1), m.size(0), m.origin(), m.stride(0), pivot.data(), status,
        WORK.data(), select_backend<MultiArray2D>());
  //assert(status==0);
  return std::forward<MultiArray2D>(m);
}

template<typename T1, typename T2, typename T3>
void getrfBatched(int n, T1* A, int lda, T2 PIV, T3 INFO, int batchSize)
{
  ma::getrfBatched(n, A, lda, PIV, INFO, batchSize, typename ma_dispatch<T1>::backend{}); 
}

/*
template<class MultiArray2D, class Array1D, 
        typename = std::enable_if_t<MultiArray2D::dimensionality == 2>
        >
MultiArray2D&& getrf(MultiArray2D&& m, Array1D& pivot)
{ 
  // use StaticVector for buffer space with appropriate buffer allocator (use mapping function)
  // call getrf(m, pivot, buffer)
}

template<class MultiArray3D, class Array1D, class Buffer, 
        typename = std::enable_if_t<MultiArray3D::dimensionality == 3>,
        typename = void
        >
MultiArray3D&& getrf(MultiArray3D&& m, Array1D& pivot, Buffer&& WORK)
{ 
}

template<class MultiArray3D, class Array1D, 
        typename = std::enable_if_t<MultiArray3D::dimensionality == 3>,
	typename = void
        >
MultiArray3D&& getrf(MultiArray3D&& m, Array1D& pivot)
{ 
  // same as 2D but with m.size(0)* more buffer space
}
*/

template<class MultiArray2D>
int getri_optimal_workspace_size(MultiArray2D&& A)
{
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(A.size(0) == A.size(1), "");
  int lwork = -1;
  ma::getri_bufferSize(A.size(0), A.origin(), A.stride(0), lwork, select_backend<MultiArray2D>());
  return lwork;
}

template<class MultiArray2D, class MultiArray1D, class Buffer>
MultiArray2D&& getri(MultiArray2D&& A, MultiArray1D const& IPIV, Buffer&& WORK)
{
  //	assert(A.stride(0) > std::max(std::size_t(1), A.size(1)));
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(IPIV.size() >= size_t(A.size(0)), "");
  RUNTIME_CHECK(WORK.size() >= std::max(std::size_t(1), size_t(A.size(0))), "");

  int status = -1;
  ma::getri(A.size(0), A.origin(), A.stride(0), IPIV.data(),
        WORK.data(), WORK.size(), status, select_backend<MultiArray2D>());
  RUNTIME_CHECK(status == 0, "");
  return std::forward<MultiArray2D>(A);
}

template<typename T1, typename T2, typename T3, typename T4>
void getriBatched(int n, T1* A, int lda, T2 PIV, T3* Ainv, int ldc, T4 INFO, int batchSize)
{
  ma::getriBatched(n, A, lda, PIV, Ainv, ldc, INFO, batchSize, typename ma_dispatch<T1>::backend{});
}

template<class MultiArray2D>
int geqrf_optimal_workspace_size(MultiArray2D&& A)
{
  RUNTIME_CHECK(A.stride(0) > 0, "");
  RUNTIME_CHECK(A.stride(1) == 1, "");

  int res;
  ma::geqrf_bufferSize(A.size(1), A.size(0), A.origin(), A.stride(0), res, 
	   	select_backend<MultiArray2D>());
  return res;
}

template<typename T1, typename T2, typename T3>
inline void matinvBatched(int n, T1* A, int lda, T2* Ainv, int ldainv, T3 info, int batchSize)
{
  ma::matinvBatched(n, A, lda, Ainv, ldainv, info, batchSize, typename ma_dispatch<T1>::backend{});
}

template<class MultiArray2D, class Array1D, class Buffer,
	 typename = typename std::enable_if_t<std::decay_t<MultiArray2D>::dimensionality == 2>>
MultiArray2D&& geqrf(MultiArray2D&& A, Array1D&& TAU, Buffer&& WORK)
{
  // why was this here???
  //assert(A.stride(0) > std::max(std::size_t(1), A.size(0)));
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(TAU.stride(0) == 1, "");
  RUNTIME_CHECK(TAU.size() >= std::max(std::size_t(1), size_t(std::min(A.size(0), A.size(1)))), "");
  RUNTIME_CHECK(WORK.size() >= std::max(std::size_t(1), size_t(A.size(0))), "");

  int status = -1;
  ma::geqrf(A.size(1), A.size(0), A.origin(), A.stride(0), TAU.origin(),
        WORK.origin(), WORK.size(), status, select_backend<MultiArray2D>());
  RUNTIME_CHECK(status == 0, "");
  return std::forward<MultiArray2D>(A);
}

template<class MultiArray3D, class Array2D,
         typename = typename std::enable_if_t<std::decay_t<MultiArray3D>::dimensionality == 3>,
         typename = typename std::enable_if_t<std::decay_t<Array2D>::dimensionality == 2>
	>
MultiArray3D&& geqrf(MultiArray3D&& A, Array2D&& TAU)
{
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(A.size(0) <= TAU.size(0), "");
  RUNTIME_CHECK(TAU.stride(1) == 1, "");
  RUNTIME_CHECK(TAU.size(1) >= std::max(std::size_t(1), size_t(std::min(A.size(1), A.size(2)))), "");
  int Nb = A.size(0); 
  using sfqmc::afqmc::DeviceBufferManager;
  using sfqmc::afqmc::HostBufferManager;
  DeviceBufferManager buffer_manager;
  auto alloc(buffer_manager.get_generator().template get_allocator<int>());
  auto ptr = alloc.allocate(Nb);
  ma::geqrfStrided(A.size(2), A.size(1), A.origin(), A.stride(1), A.stride(0), 
		   TAU.origin(), TAU.stride(0), ptr, A.size(0), select_backend<MultiArray3D>());

// should I check all the time?
#ifndef NDEBUG
  HostBufferManager hbuffer_manager;
  auto halloc(hbuffer_manager.get_generator().template get_allocator<int>());
  auto hptr = halloc.allocate(Nb);
  using std::copy_n;
  copy_n(ptr, Nb, hptr);
  for(int i=0; i<Nb; i++)
    RUNTIME_CHECK(hptr[i] == 0, "");
  halloc.deallocate(hptr, Nb);
#endif 

  alloc.deallocate(ptr, Nb);
  return std::forward<MultiArray3D>(A);
}

template<class MultiArray2D>
int gelqf_optimal_workspace_size(MultiArray2D&& A)
{
  RUNTIME_CHECK(A.stride(0) > 0, "");
  RUNTIME_CHECK(A.stride(1) == 1, "");

  int res;
  ma::gelqf_bufferSize(A.size(1), A.size(0), A.origin(), A.stride(0), res,
		select_backend<MultiArray2D>());
  return res;
}

template<class MultiArray2D, class Array1D, class Buffer>
MultiArray2D&& gelqf(MultiArray2D&& A, Array1D&& TAU, Buffer&& WORK)
{
  RUNTIME_CHECK(A.stride(1) > 0, "");
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(TAU.stride(0) == 1, "");
  RUNTIME_CHECK(TAU.size() >= std::max(std::size_t(1), size_t(std::min(A.size(0), A.size(1)))), "");
  RUNTIME_CHECK(WORK.size() >= std::max(std::size_t(1), size_t(A.size(1))), "");

  int status = -1;
  ma::gelqf(A.size(1), A.size(0), A.origin(), A.stride(0), TAU.data(),
        WORK.data(), WORK.size(), status, select_backend<MultiArray2D>());
  RUNTIME_CHECK(status == 0, "");
  return std::forward<MultiArray2D>(A);
}


template<class MultiArray2D>
int gqr_optimal_workspace_size(MultiArray2D&& A)
{
  RUNTIME_CHECK(A.stride(0) > 0, "");
  RUNTIME_CHECK(A.stride(1) == 1, "");

  int res;
  ma::gqr_bufferSize(A.size(1), A.size(0), std::max(std::size_t(1), size_t(std::min(A.size(0), A.size(1)))),
                 A.origin(), A.stride(0), res, select_backend<MultiArray2D>());
  return res;
}

template<class MultiArray2D, class Array1D, class Buffer,
         typename = typename std::enable_if_t<std::decay_t<MultiArray2D>::dimensionality == 2>,
         typename = typename std::enable_if_t<std::decay_t<Array1D>::dimensionality == 1>,
         typename = typename std::enable_if_t<std::decay_t<Buffer>::dimensionality == 1>
	 >
MultiArray2D&& gqr(MultiArray2D&& A, Array1D&& TAU, Buffer&& WORK)
{
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(TAU.stride(0) == 1, "");
  RUNTIME_CHECK(TAU.size() >= std::max(std::size_t(1), size_t(std::min(A.size(0), A.size(1)))), "");
  RUNTIME_CHECK(WORK.size() >= std::max(std::size_t(1), size_t(A.size(0))), "");

  int status = -1;
  ma::gqr(A.size(1), A.size(0), std::max(std::size_t(1), size_t(std::min(A.size(0), A.size(1)))),
      A.origin(), A.stride(0), TAU.origin(), WORK.origin(),
      WORK.size(), status, select_backend<MultiArray2D>());
  RUNTIME_CHECK(status == 0, "");
  return std::forward<MultiArray2D>(A);
}

template<class MultiArray3D, class Array2D, class Buffer2D,
         typename = typename std::enable_if_t<std::decay_t<MultiArray3D>::dimensionality == 3>,
         typename = typename std::enable_if_t<std::decay_t<Array2D>::dimensionality == 2>,
         typename = typename std::enable_if_t<std::decay_t<Buffer2D>::dimensionality == 2>,
	 typename = void 
         >
MultiArray3D&& gqr(MultiArray3D&& A, Array2D&& TAU, Buffer2D&& WORK)
{
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(TAU.stride(1) == 1, "");
  RUNTIME_CHECK(A.size(0) <= TAU.size(0), "");
  RUNTIME_CHECK(A.size(0) <= WORK.size(0), "");
  RUNTIME_CHECK(TAU.size(1) >= std::max(std::size_t(1), size_t(std::min(A.size(1), A.size(2)))), "");
  RUNTIME_CHECK(WORK.size(1) >= std::max(std::size_t(1), size_t(A.size(1))), "");
  RUNTIME_CHECK(WORK.size(1) == WORK.stride(0), "");
  int Nb = A.size(0); 
  using sfqmc::afqmc::DeviceBufferManager;
  using sfqmc::afqmc::HostBufferManager;
  DeviceBufferManager buffer_manager;
  auto alloc(buffer_manager.get_generator().template get_allocator<int>());
  auto ptr = alloc.allocate(Nb);

  ma::gqrStrided(A.size(2), A.size(1), std::max(std::size_t(1), size_t(std::min(A.size(1), A.size(2)))),
      A.origin(), A.stride(1), A.stride(0), TAU.origin(), TAU.stride(0), WORK.origin(),
      WORK.size(1), ptr, Nb, select_backend<MultiArray3D>());

// should I check all the time?
#ifndef NDEBUG
  HostBufferManager hbuffer_manager;
  auto halloc(hbuffer_manager.get_generator().template get_allocator<int>());
  auto hptr = halloc.allocate(Nb);
  using std::copy_n;
  copy_n(ptr, Nb, hptr);
  for(int i=0; i<Nb; i++)
    RUNTIME_CHECK(hptr[i] == 0, "");
  halloc.deallocate(hptr, Nb);
#endif
  
  alloc.deallocate(ptr, Nb);
  return std::forward<MultiArray3D>(A);
}

template<class MultiArray2D>
int glq_optimal_workspace_size(MultiArray2D&& A)
{
  RUNTIME_CHECK(A.stride(0) > 0, "");
  RUNTIME_CHECK(A.stride(1) == 1, "");

  int res;
  ma::glq_bufferSize(A.size(1), A.size(0), std::max(std::size_t(1), size_t(std::min(A.size(0), A.size(1)))),
                 A.origin(), A.stride(0), res, select_backend<MultiArray2D>());
  return res;
}

template<class MultiArray2D, class Array1D, class Buffer>
MultiArray2D&& glq(MultiArray2D&& A, Array1D&& TAU, Buffer&& WORK)
{
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(TAU.stride(0) == 1, "");
  RUNTIME_CHECK(TAU.size() >= std::max(std::size_t(1), size_t(std::min(A.size(0), A.size(1)))), "");
  RUNTIME_CHECK(WORK.size() >= std::max(std::size_t(1), size_t(A.size(1))), "");

  int status = -1;
  ma::glq(A.size(1), A.size(0), std::max(std::size_t(1), size_t(std::min(A.size(0), A.size(1)))),
      A.origin(), A.stride(0), TAU.data(), WORK.data(),
      WORK.size(), status, select_backend<MultiArray2D>());
  RUNTIME_CHECK(status == 0, "");
  return std::forward<MultiArray2D>(A);
}

template<class MultiArray2D, typename = typename std::enable_if_t<MultiArray2D::dimensionality == 2>>
MultiArray2D&& potrf(MultiArray2D&& A)
{
  RUNTIME_CHECK(A.size(0) == A.size(1), "");
  int INFO;
  ma::potrf('U', A.size(0), A.origin(), A.stride(0), INFO, select_backend<MultiArray2D>());
  if (INFO != 0)
    throw std::runtime_error(" error in ma::potrf: Error code != 0");
}

template<class MultiArray2D>
int gesvd_optimal_workspace_size(MultiArray2D&& A)
{
  RUNTIME_CHECK(A.stride(0) > 0, "");
  RUNTIME_CHECK(A.stride(1) == 1, "");

  int res;
  ma::gesvd_bufferSize(A.size(1), A.size(0), A.origin(), res, select_backend<MultiArray2D>());
  return res;
}

template<class MultiArray2D, class Array1D, class MultiArray2DU, class MultiArray2DV, class Buffer, class RBuffer>
MultiArray2D&& gesvd(char jobU,
                     char jobVT,
                     MultiArray2D&& A,
                     Array1D&& S,
                     MultiArray2DU&& U,
                     MultiArray2DV&& VT,
                     Buffer&& WORK,
                     RBuffer&& RWORK)
{
  RUNTIME_CHECK(A.stride(1) > 0, "");
  RUNTIME_CHECK(A.stride(1) == 1, "");

  // in C: A = U * S * VT
  // in F: At = (U * S * VT)t = VTt * S * Ut
  // so I need to switch U <--> VT when calling fortran interface
  int status = -1;
  ma::gesvd(jobVT, jobU, A.size(1), A.size(0), A.origin(), A.stride(0), S.origin(),
        VT.origin(), VT.stride(0), // !!!
        U.origin(), U.stride(0),   // !!!
        WORK.data(), WORK.size(), RWORK.origin(), status,
	select_backend<MultiArray2D>());
  RUNTIME_CHECK(status == 0, "");
  return std::forward<MultiArray2D>(A);
}

template<class MultiArray1D,
         class MultiArray2D,
         typename = typename std::enable_if_t<MultiArray1D::dimensionality == 1>,
         typename = typename std::enable_if_t<MultiArray2D::dimensionality == 2>>
std::pair<MultiArray1D, MultiArray2D> symEig(MultiArray2D const& A)
{
  using Type       = typename MultiArray2D::element;
  using RealType   = typename sfqmc::afqmc::remove_complex<Type>::value_type;
  using extensions = typename boost::multi::layout_t<1u>::extensions_type;
  RUNTIME_CHECK(A.size(0) == A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(A.size(0) > 0, "");
  int N   = A.size(0);
  int LDA = A.stride(0);

  MultiArray1D eigVal(extensions{N});
  MultiArray2D eigVec({N, N});
  MultiArray2D A_({N, N});
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      A_[i][j] = ma::conj(A[i][j]);
  char JOBZ('V');
  char RANGE('A');
  char UPLO_param('U');
  RealType VL     = 0;
  RealType VU     = 0;
  int IL          = 0;
  int IU          = 0;
  RealType ABSTOL = 0; //DLAMCH( 'Safe minimum' );
  int M;               // output: total number of eigenvalues found
  std::vector<int> ISUPPZ(2 * N);
  std::vector<Type> WORK(1); // set with workspace query
  int LWORK = -1;
  std::vector<RealType> RWORK(1); // set with workspace query
  int LRWORK = -1;
  std::vector<int> IWORK(1);
  int LIWORK = -1;
  int INFO;

  ma::hevr(JOBZ, RANGE, UPLO_param, N, A_.origin(), LDA, VL, VU, IL, IU, ABSTOL, M,
       eigVal.origin(), eigVec.origin(), N, ISUPPZ.data(),
       WORK.data(), LWORK, RWORK.data(), LRWORK, IWORK.data(),
       LIWORK, INFO, select_backend<MultiArray2D>());

  LWORK = int(real(WORK[0]));
  WORK.resize(LWORK);
  LRWORK = int(RWORK[0]);
  RWORK.resize(LRWORK);
  LIWORK = int(IWORK[0]);
  IWORK.resize(LIWORK);

  ma::hevr(JOBZ, RANGE, UPLO_param, N, A_.origin(), LDA, VL, VU, IL, IU, ABSTOL, M,
       eigVal.origin(), eigVec.origin(), N, ISUPPZ.data(),
       WORK.data(), LWORK, RWORK.data(), LRWORK, IWORK.data(),
       LIWORK, INFO, select_backend<MultiArray2D>());
  if (INFO != 0)
    throw std::runtime_error(" error in ma::eig: Error code != 0");
  if (M != N)
    throw std::runtime_error(" error in ma::eig: Not enough eigenvalues");
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++)
      std::swap(eigVec[i][j], eigVec[j][i]);

  return std::pair<MultiArray1D, MultiArray2D>{eigVal, eigVec};
}

// Careful!!!!
// This routine modifies the original matrix.
template<class MultiArray1D,
         class MultiArray2D,
         class MultiArray2DA,
         typename = typename std::enable_if_t<MultiArray1D::dimensionality == 1>,
         typename = typename std::enable_if_t<MultiArray2DA::dimensionality == 2>,
         typename = typename std::enable_if_t<MultiArray2D::dimensionality == 2>>
std::pair<MultiArray1D, MultiArray2D> symEigSelect(MultiArray2DA& A, int neig)
{
  using Type   = typename MultiArray2D::element;
  using TypeA  = typename MultiArray2DA::element;
  static_assert(std::is_same<Type, TypeA>::value, "Wrong types.");
  using RealType   = typename sfqmc::afqmc::remove_complex<Type>::value_type;
  using extensions = typename boost::multi::layout_t<1u>::extensions_type;
  RUNTIME_CHECK(A.size(0) == A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(A.size(0) > 0, "");
  int N   = A.size(0);
  int LDA = A.stride(0);

  MultiArray1D eigVal(extensions{neig});
  MultiArray2D eigVec({neig, N});
  // Transpose A to avoid using more memory
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++)
    {
      using std::swap;
      swap(A[i][j], A[j][i]);
    }

  char JOBZ('V');
  char RANGE('I');
  char UPLO_param('U');
  RealType VL     = 0;
  RealType VU     = 0;
  int IL          = 1;
  int IU          = neig;
  RealType ABSTOL = 0; //DLAMCH( 'Safe minimum' );
  int M;               // output: total number of eigenvalues found
  std::vector<int> ISUPPZ(2 * N);
  std::vector<Type> WORK(1); // set with workspace query
  int LWORK = -1;
  std::vector<RealType> RWORK(1); // set with workspace query
  int LRWORK = -1;
  std::vector<int> IWORK(1);
  int LIWORK = -1;
  int INFO;

  ma::hevr(JOBZ, RANGE, UPLO_param, N, A.origin(), LDA, VL, VU, IL, IU, ABSTOL, M,
       eigVal.origin(), eigVec.origin(), N, ISUPPZ.data(),
       WORK.data(), LWORK, RWORK.data(), LRWORK, IWORK.data(),
       LIWORK, INFO, select_backend<MultiArray2D>());

  LWORK = int(real(WORK[0]));
  WORK.resize(LWORK);
  LRWORK = int(RWORK[0]);
  RWORK.resize(LRWORK);
  LIWORK = int(IWORK[0]);
  IWORK.resize(LIWORK);

  ma::hevr(JOBZ, RANGE, UPLO_param, N, A.origin(), LDA, VL, VU, IL, IU, ABSTOL, M,
       eigVal.origin(), eigVec.origin(), N, ISUPPZ.data(),
       WORK.data(), LWORK, RWORK.data(), LRWORK, IWORK.data(),
       LIWORK, INFO, select_backend<MultiArray2D>());
  if (INFO != 0)
    throw std::runtime_error(" error in ma::eig: Error code != 0");
  if (M != neig)
    throw std::runtime_error(" error in ma::eig: Not enough eigenvalues");

  return std::pair<MultiArray1D, MultiArray2D>{eigVal, eigVec};
}

// Careful!!!!
// This routine modifies the original matrix.
template<class MultiArray1D,
         class MultiArray2D,
         class MultiArray2DA,
         class MultiArray2DB,
         typename = typename std::enable_if_t<MultiArray1D::dimensionality == 1>,
         typename = typename std::enable_if_t<MultiArray2DA::dimensionality == 2>,
         typename = typename std::enable_if_t<MultiArray2DB::dimensionality == 2>,
         typename = typename std::enable_if_t<MultiArray2D::dimensionality == 2>>
std::pair<MultiArray1D, MultiArray2D> genEigSelect(MultiArray2DA& A, MultiArray2DB& S, int neig, int itype = 1)
{
  using Type   = typename MultiArray2D::element;
  using TypeA  = typename MultiArray2DA::element;
  using TypeB  = typename MultiArray2DB::element;
  static_assert(std::is_same<Type, TypeA>::value, "Wrong types.");
  static_assert(std::is_same<TypeA, TypeB>::value, "Wrong types.");
  using RealType   = typename sfqmc::afqmc::remove_complex<Type>::value_type;
  using extensions = typename boost::multi::layout_t<1u>::extensions_type;
  RUNTIME_CHECK(A.size(0) == A.size(1), "");
  RUNTIME_CHECK(A.size(0) == S.size(0), "");
  RUNTIME_CHECK(S.size(0) == S.size(1), "");
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(A.size(0) > 0, "");
  RUNTIME_CHECK(S.stride(1) == 1, "");
  RUNTIME_CHECK(S.size(0) > 0, "");
  int N   = A.size(0);
  int LDA = A.stride(0);
  int LDS = S.stride(0);

  MultiArray1D eigVal(extensions{neig});
  MultiArray2D eigVec({neig, N});
  // Transpose A to avoid using more memory
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++)
    {
      using std::swap;
      swap(A[i][j], A[j][i]);
      swap(S[i][j], S[j][i]);
    }

  char JOBZ('V');
  char RANGE('I');
  char UPLO_param('U');
  RealType VL     = 0;
  RealType VU     = 0;
  int IL          = 1;
  int IU          = neig;
  RealType ABSTOL = 0;       //DLAMCH( 'Safe minimum' );
  int M;                     // output: total number of eigenvalues found
  std::vector<Type> WORK(1); // set with workspace query
  int LWORK = -1;
  std::vector<RealType> RWORK(7 * N); // set with workspace query
  std::vector<int> IWORK(5 * N);
  std::vector<int> IFAIL(N);
  int INFO;

  ma::gvx(itype, JOBZ, RANGE, UPLO_param, N, A.origin(), LDA, S.origin(), LDS, VL, VU, IL, IU,
      ABSTOL, M, eigVal.origin(), eigVec.origin(), N, WORK.data(),
      LWORK, RWORK.data(), IWORK.data(), IFAIL.data(), INFO, select_backend<MultiArray2D>());

  LWORK = int(real(WORK[0]));
  WORK.resize(LWORK);

  ma::gvx(itype, JOBZ, RANGE, UPLO_param, N, A.origin(), LDA, S.origin(), LDS, VL, VU, IL, IU,
      ABSTOL, M, eigVal.origin(), eigVec.origin(), N, WORK.data(),
      LWORK, RWORK.data(), IWORK.data(), IFAIL.data(), INFO, select_backend<MultiArray2D>());
  if (INFO != 0)
    throw std::runtime_error(" error in ma::eig: Error code != 0");
  if (M != neig)
    throw std::runtime_error(" error in ma::eig: Not enough eigenvalues");

  return std::pair<MultiArray1D, MultiArray2D>{eigVal, eigVec};
}

} // namespace ma

#endif
