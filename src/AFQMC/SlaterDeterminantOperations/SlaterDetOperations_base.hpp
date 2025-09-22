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

#ifndef SFQMC_AFQMC_SLATERDETOPERATIONS_BASE_HPP
#define SFQMC_AFQMC_SLATERDETOPERATIONS_BASE_HPP

#include <fstream>

#include "AFQMC/config.h"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "Numerics/ma_operations.hpp"
#include "Numerics/csr_blas.hpp"
#include "AFQMC/SlaterDeterminantOperations/mixed_density_matrix.hpp"
#include "AFQMC/SlaterDeterminantOperations/apply_expM.hpp"


namespace sfqmc
{
namespace afqmc
{
template<typename Type, class BufferManager>
class SlaterDetOperations_base
{
public:
  using T = Type;
  using R = typename remove_complex<T>::value_type;

  using buffer_type_I = typename BufferManager::template allocator_t<int>;
  using buffer_type_R = typename BufferManager::template allocator_t<R>;
  using buffer_type_T = typename BufferManager::template allocator_t<T>;

  using IVector = boost::multi::static_array<int, 1, buffer_type_I>;
  using IMatrix = boost::multi::static_array<int, 2, buffer_type_I>;
  using RVector = boost::multi::static_array<R, 1, buffer_type_R>;
  using TVector = boost::multi::static_array<T, 1, buffer_type_T>;
  using TMatrix = boost::multi::static_array<T, 2, buffer_type_T>;
  using TTensor = boost::multi::static_array<T, 3, buffer_type_T>;

  SlaterDetOperations_base(BufferManager b) : buffer_manager(b) {}

  SlaterDetOperations_base(int NMO, int NAEA, BufferManager b) : buffer_manager(b)
  {
    // only used to determine size of WORK

    TMatrix TNN({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TNM({NAEA, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TMN({NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());

    // reserve enough space in lapack's work array
    // Make sure it is large enough for:
    // 1. getri( TNN )
    //  2. geqrf( TNM )
    work_size = std::max(ma::getri_optimal_workspace_size(TNN), ma::geqrf_optimal_workspace_size(TNM));
    //  3. gqr( TNM )
    work_size = std::max(work_size, ma::gqr_optimal_workspace_size(TNM));
    //  4. gelqf( TMN )
    work_size = std::max(work_size, ma::gelqf_optimal_workspace_size(TMN));
    //  5. glq( TMN )
    work_size = std::max(work_size, ma::glq_optimal_workspace_size(TMN));
    //  6. trf( TNN )
    work_size = std::max(work_size, ma::getrf_optimal_workspace_size(TNN));
    //  7. svd( TNN )
    work_size = std::max(work_size, ma::gesvd_optimal_workspace_size(TNN));
    work_size = std::max(work_size, NMO);
  }

  ~SlaterDetOperations_base() {}

  SlaterDetOperations_base(const SlaterDetOperations_base& other) = delete;
  SlaterDetOperations_base(SlaterDetOperations_base&& other)      = default;
  SlaterDetOperations_base& operator=(const SlaterDetOperations_base& other) = delete;
  SlaterDetOperations_base& operator=(SlaterDetOperations_base&& other) = default;

  template<class MatA, class MatB, class MatC>
  T MixedDensityMatrix(const MatA& hermA, const MatB& B, MatC&& C, T LogOverlapFactor, bool compact, bool herm = true)
  {
    int NMO  = (herm ? hermA.size(1) : hermA.size(0));
    int NAEA = (herm ? hermA.size(0) : hermA.size(1));
    TMatrix TNN({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TNM({NAEA, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::base::MixedDensityMatrix<T>(hermA, B, std::forward<MatC>(C), LogOverlapFactor,
                                                                    TNN, TNM, IWORK, WORK, compact, herm);
  }

  template<class MatA, class MatC>
  T MixedDensityMatrix(const MatA& A, MatC&& C, T LogOverlapFactor, bool compact = false)
  {
    int NMO  = A.size(0);
    int NAEA = A.size(1);
    TMatrix TNN({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TNM({NAEA, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::base::MixedDensityMatrix<T>(A, A, std::forward<MatC>(C), LogOverlapFactor, TNN,
                                                                    TNM, IWORK, WORK, compact, false);
  }

  template<class MatA, class MatB3D, class MatC3D, class TVec,
           typename = typename std::enable_if_t<MatA::dimensionality == 2 or
                                                MatA::dimensionality == -2>,
           typename = typename std::enable_if_t<std::decay_t<MatB3D>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<MatC3D>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<TVec>::dimensionality == 1>
          >
  // A[N, M]/[M,N]  (depending on herm)
  // B[nwalk, M, N]
  // C[nwalk, M, N]/[nwalk, M, M]  (depending on compact)
  void BatchedMixedDensityMatrix(MatA const& hermA,
                                 MatB3D&& B,
                                 MatC3D&& C,
                                 T LogOverlapFactor,
                                 TVec&& ovlp,
                                 bool compact = false,
                                 bool herm    = true)
  {
    int M    = (herm ? hermA.size(1) : hermA.size(0));
    int N    = (herm ? hermA.size(0) : hermA.size(1));
    int nbatch = B.size(0);
    RUNTIME_CHECK(C.size(0) == nbatch, "");
    RUNTIME_CHECK(ovlp.size(0) == nbatch, "");
    int n1 = nbatch, n2 = N, n3 = M;
    if (compact)
    {
      n1 = n2 = n3 = 0;
    }
    TTensor TNN3D({nbatch, N, N}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor TNM3D({n1, n2, n3}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{nbatch * (M + 1)},
                  buffer_manager.get_generator().template get_allocator<int>());
    SlaterDeterminantOperations::batched::MixedDensityMatrix(hermA, std::forward<MatB3D>(B),
                    std::forward<MatC3D>(C), LogOverlapFactor, std::forward<TVec>(ovlp),
                    TNN3D, TNM3D, IWORK, compact, herm);
  }

  template<class MatA, class MatB, class MatC>
  T MixedDensityMatrix_noHerm(const MatA& A,
                              const MatB& B,
                              MatC&& C,
                              T LogOverlapFactor,
                              bool compact = false,
                              bool useSVD  = false)
  {
    int NMO  = A.size(0);
    int NAEA = A.size(1);
    if (useSVD)
    {
      TMatrix TNN1({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
      TMatrix TNN2({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
      TMatrix TNM({NAEA, NMO}, buffer_manager.get_generator().template get_allocator<T>());
      TMatrix TMN({NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
      TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
      RVector RWORK(iextensions<1u>{6 * NAEA + 1}, buffer_manager.get_generator().template get_allocator<R>());
      IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
      return SlaterDeterminantOperations::base::MixedDensityMatrix_noHerm_wSVD<T>(A, B, std::forward<MatC>(C),
                                                                                  LogOverlapFactor, RWORK, TNN1, TNN2,
                                                                                  TMN, TNM, IWORK, WORK, compact);
    }
    else
    {
      TMatrix TNN({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
      TMatrix TNM({NAEA, NMO}, buffer_manager.get_generator().template get_allocator<T>());
      TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
      IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
      return SlaterDeterminantOperations::base::MixedDensityMatrix_noHerm<T>(A, B, std::forward<MatC>(C),
                                                                             LogOverlapFactor, TNN, TNM, IWORK, WORK,
                                                                             compact);
    }
  }

  template<class Iptr, class MatA, class MatB, class MatC, class MatQ>
  T MixedDensityMatrixForWoodbury(const MatA& hermA,
                                  const MatB& B,
                                  MatC&& C,
                                  T LogOverlapFactor,
                                  Iptr ref,
                                  MatQ&& QQ0,
                                  bool compact = false)
  {
    int Nact = hermA.size(0);
    int NEL  = B.size(1);
    int NMO  = B.size(0);
    RUNTIME_CHECK(hermA.size(1) == B.size(0), "");
    RUNTIME_CHECK(QQ0.size(0) == Nact, "");
    RUNTIME_CHECK(QQ0.size(1) == NEL, "");
    TMatrix TNN({NEL, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TAB({Nact, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TNM({NEL, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::base::MixedDensityMatrixForWoodbury<T>(hermA, B, std::forward<MatC>(C),
                                                                               LogOverlapFactor,
                                                                               std::forward<MatQ>(QQ0), ref, TNN, TAB,
                                                                               TNM, IWORK, WORK, compact);
  }

  template<class Iptr, class MatA, class Mat3DB, class Mat3DC, class TVec, class Mat3DQ,
           typename = typename std::enable_if_t<MatA::dimensionality == 2 or
                                                MatA::dimensionality == -2>,
           typename = typename std::enable_if_t<std::decay_t<Mat3DB>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<Mat3DC>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<Mat3DQ>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<TVec>::dimensionality == 1>
          >
  void BatchedMixedDensityMatrixForWoodbury(MatA const& hermA,
                                     Mat3DB const& B,
                                     Mat3DC&& C,
                                     T LogOverlapFactor,
                                     TVec && ovlp,
                                     Iptr ref,
                                     Mat3DQ&& QQ0,
                                     bool compact = false)
  {
    int Nact = hermA.size(0);
    int nb   = B.size(0);
    int NMO  = B.size(1);
    int NEL  = B.size(2);
    RUNTIME_CHECK(hermA.size(1) == B.size(1), "");
    RUNTIME_CHECK(C.size(0) == nb, "");
    RUNTIME_CHECK(QQ0.size(0) == nb, "");
    RUNTIME_CHECK(QQ0.size(1) == Nact, "");
    RUNTIME_CHECK(QQ0.size(2) == NEL, "");
    int s = ( compact ? 0 : 1);
    TTensor TNN({nb, NEL, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor TAB({nb, Nact, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor TNM({s*nb, s*NEL, s*NMO}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{nb*(NMO+1)}, 
                                 buffer_manager.get_generator().template get_allocator<int>());
    SlaterDeterminantOperations::batched::MixedDensityMatrixForWoodbury<T>(hermA, B, 
                std::forward<Mat3DC>(C), LogOverlapFactor, std::forward<TVec>(ovlp),
                std::forward<Mat3DQ>(QQ0), ref, TNN, TAB, TNM, IWORK, compact);
  }

  template<class Iptr, class MatA, class MatB, class MatC>
  T MixedDensityMatrixFromConfiguration(MatA const& hermA,
                                        MatB const& B,
                                        MatC&& C,
                                        T LogOverlapFactor,
                                        Iptr ref,
                                        bool compact = false)
  {
    int Nact = hermA.size(0);
    int NEL  = B.size(1);
    int NMO  = B.size(0);
    RUNTIME_CHECK(hermA.size(1) == B.size(0), "");
    TMatrix TNN({NEL, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TAB({Nact, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TNM({NEL, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::base::MixedDensityMatrixFromConfiguration<T>(hermA, B, std::forward<MatC>(C),
                                                                                     LogOverlapFactor, ref, TNN, TAB,
                                                                                     TNM, IWORK, WORK, compact);
  }

  template<class Iptr, class MatA, class Mat3DB, class Mat3DC, class TVec,
           typename = typename std::enable_if_t<MatA::dimensionality == 2 or
                                                MatA::dimensionality == -2>,
           typename = typename std::enable_if_t<std::decay_t<Mat3DB>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<Mat3DC>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<TVec>::dimensionality == 1>
          >
  void BatchedMixedDensityMatrixFromConfiguration(MatA const& hermA,
                                        Mat3DB const& B,
                                        Mat3DC&& C,
                                        T LogOverlapFactor,
                                        TVec && ovlp,
                                        Iptr ref,
                                        bool compact = false)
  {
    int Nact = hermA.size(0);
    int nb   = B.size(0);
    int NMO  = B.size(1);
    int NEL  = B.size(2);
    RUNTIME_CHECK(hermA.size(1) == B.size(1), "");
    RUNTIME_CHECK(C.size(0) >= nb, "");
    int scl = ( compact ? 0 : 1 );
    TTensor TNN({nb, NEL, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor TAB({nb, Nact, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor TNM({scl*nb, scl*NEL, scl*NMO}, 
                                  buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{nb * (NMO + 1)}, 
                                  buffer_manager.get_generator().template get_allocator<int>());
    SlaterDeterminantOperations::batched::MixedDensityMatrixFromConfiguration<T>(hermA, B, 
        std::forward<Mat3DC>(C), LogOverlapFactor, std::forward<TVec>(ovlp), ref, 
        TNN, TAB, TNM, IWORK, compact);
  }

  template<class MatA3D, class MatB3D, class MatC3D, class TVec,
           typename = typename std::enable_if_t<std::decay_t<MatA3D>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<MatB3D>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<MatC3D>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<TVec>::dimensionality == 1>
          >
  // A[nwalk, M, N]
  // B[nwalk, M, N]
  // C[nwalk, Nc, M], Nc = compact ? N, M; 
  void BatchedDensityMatrix(MatA3D&& A,
                            MatB3D&& B,
                            MatC3D&& C,
                            T LogOverlapFactor,
                            TVec&& ovlp,
                            bool compact = false)
  {
    int M    = A.size(1);
    int N    = A.size(2);
    int nbatch = A.size(0);
    RUNTIME_CHECK(B.size(0) == nbatch, "");
    RUNTIME_CHECK(C.size(0) == nbatch, "");
    RUNTIME_CHECK(ovlp.size(0) == nbatch, "");
    TTensor TNN3D({nbatch, N, N}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor TNM3D({nbatch, N, M}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{nbatch * (M + 1)},
                  buffer_manager.get_generator().template get_allocator<int>());
    SlaterDeterminantOperations::batched::DensityMatrix(std::forward<MatA3D>(A), 
                    std::forward<MatB3D>(B), std::forward<MatC3D>(C), LogOverlapFactor, 
                    std::forward<TVec>(ovlp), TNN3D, TNM3D, IWORK, compact);
  }

  template<class MatA>
  T Overlap(const MatA& A, T LogOverlapFactor)
  {
    int NAEA = A.size(1);
    TMatrix TNN({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TNN2({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NAEA + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::base::Overlap<T>(A, A, LogOverlapFactor, TNN, IWORK, TNN2.elements(), false);
  }

  template<class MatA, class MatB>
  T Overlap(const MatA& hermA, const MatB& B, T LogOverlapFactor, bool herm = true)
  {
    int NAEA = (herm ? hermA.size(0) : hermA.size(1));
    TMatrix TNN({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TNN2({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NAEA + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::base::Overlap<T>(hermA, B, LogOverlapFactor, TNN, IWORK, TNN2.elements(), herm);
  }

  template<class MatA, class MatB>
  T Overlap_noHerm(const MatA& A, const MatB& B, T LogOverlapFactor)
  {
    int NAEA = A.size(1);
    TMatrix TNN({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TNN2({NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NAEA + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::base::Overlap<T>(A, B, LogOverlapFactor, TNN, IWORK, TNN2.elements(), false);
  }

  // routines for PHMSD
  template<typename Iptr, class MatA, class MatB, class MatC>
  T OverlapForWoodbury(const MatA& hermA, const MatB& B, T LogOverlapFactor, Iptr ref, MatC&& QQ0)
  {
    int Nact = hermA.size(0);
    int NEL  = B.size(1);
    RUNTIME_CHECK(hermA.size(1) == B.size(0), "");
    RUNTIME_CHECK(QQ0.size(0) == Nact, "");
    RUNTIME_CHECK(QQ0.size(1) == NEL, "");
    TMatrix TNN({NEL, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix TMN({Nact, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{Nact + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::base::OverlapForWoodbury<T>(hermA, B, LogOverlapFactor, std::forward<MatC>(QQ0),
                                                                    ref, TNN, TMN, IWORK, WORK);
  }

  template<typename Iptr, class MatA, class Mat3DB, class Mat3DC, class TVec,
           typename = typename std::enable_if_t<MatA::dimensionality == 2 or
                                                MatA::dimensionality == -2>,
           typename = typename std::enable_if_t<std::decay_t<Mat3DB>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<Mat3DC>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<TVec>::dimensionality == 1>
          >
  void BatchedOverlapForWoodbury(MatA const& hermA, Mat3DB const& B, T LogOverlapFactor, TVec ovlp,
                                 Iptr ref, Mat3DC&& QQ0)
  {
    int Nact = hermA.size(0);
    int nb   = B.size(0); 
    int NEL  = B.size(2);
    RUNTIME_CHECK(hermA.size(1) == B.size(1), "");
    RUNTIME_CHECK(QQ0.size(0) == nb, "");
    RUNTIME_CHECK(QQ0.size(1) == Nact, "");
    RUNTIME_CHECK(QQ0.size(2) == NEL, "");
    TTensor TNN({nb, NEL, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor TMN({nb, Nact, NEL}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{nb*(Nact+1)}, 
                                 buffer_manager.get_generator().template get_allocator<int>());
    SlaterDeterminantOperations::batched::OverlapForWoodbury<T>(hermA, B, LogOverlapFactor, 
            std::forward<TVec>(ovlp), std::forward<Mat3DC>(QQ0), ref, TNN, TMN, IWORK);
  }

  template<class MatA, class Mat3DB, class TVec,
           typename = typename std::enable_if_t<MatA::dimensionality == 2 or
                                                MatA::dimensionality == -2>,
           typename = typename std::enable_if_t<std::decay_t<Mat3DB>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<TVec>::dimensionality == 1>
          >
  void BatchedOverlap(MatA const& hermA,
                      Mat3DB&& B,
                      T LogOverlapFactor,
                      TVec&& ovlp,
                      bool herm = true)
  {
    int NMO    = (herm ? hermA.size(1) : hermA.size(0));
    int NAEA   = (herm ? hermA.size(0) : hermA.size(1));
    int nbatch = B.size();
    RUNTIME_CHECK(ovlp.size() == nbatch, "");
    TTensor TNN3D({nbatch, NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{nbatch * (NMO + 1)}, buffer_manager.get_generator().template get_allocator<int>());
    SlaterDeterminantOperations::batched::Overlap(hermA, std::forward<Mat3DB>(B), LogOverlapFactor,
            std::forward<TVec>(ovlp), TNN3D, IWORK, herm);
  }

  template<class Mat3DA, class Mat3DB, class TVec,
           typename = typename std::enable_if_t<std::decay_t<Mat3DA>::dimensionality == 3>, 
           typename = typename std::enable_if_t<std::decay_t<Mat3DB>::dimensionality == 3>,
           typename = typename std::enable_if_t<std::decay_t<TVec>::dimensionality == 1>,
           typename = void 
          >
  void BatchedOverlap(Mat3DA&& A,
                      Mat3DB&& B,
                      T LogOverlapFactor,
                      TVec&& ovlp)
  {
    int nbatch = A.size(0);
    int NMO    = A.size(1); 
    int NAEA   = A.size(2); 
    RUNTIME_CHECK(B.size(0) == nbatch, "");
    RUNTIME_CHECK(ovlp.size() == nbatch, "");
    TTensor TNN3D({nbatch, NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{nbatch * (NMO + 1)}, buffer_manager.get_generator().template get_allocator<int>());
    SlaterDeterminantOperations::batched::Overlap(std::forward<Mat3DA>(A), std::forward<Mat3DB>(B), 
            LogOverlapFactor, std::forward<TVec>(ovlp), TNN3D, IWORK);
  }

  template<class Mat, class MatP1, class MatV>
  void Propagate(Mat&& A, const MatP1& P1, const MatV& V, int order = 6, char TA = 'N', bool noncollinear = false)
  {
    int npol = noncollinear ? 2 : 1;
    int NMO  = A.size(0);
    int NAEA = A.size(1);
    int M    = NMO / npol;
    RUNTIME_CHECK(NMO % npol == 0, "");
    RUNTIME_CHECK(P1.size(0) == NMO, "");
    RUNTIME_CHECK(P1.size(1) == NMO, "");
    TMatrix TMN({NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix T1({M, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix T2({M, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    using ma::H;
    using ma::T;
    if (TA == 'H' || TA == 'h')
    {
      ma::product(ma::H(P1), std::forward<Mat>(A), TMN);
      for (int p = 0; p < npol; ++p)
        SlaterDeterminantOperations::apply_expM(V, TMN.sliced(p * M, (p + 1) * M), T1, T2, order, TA);
      ma::product(ma::H(P1), TMN, std::forward<Mat>(A));
    }
    else if (TA == 'T' || TA == 't')
    {
      ma::product(ma::T(P1), std::forward<Mat>(A), TMN);
      for (int p = 0; p < npol; ++p)
        SlaterDeterminantOperations::apply_expM(V, TMN.sliced(p * M, (p + 1) * M), T1, T2, order, TA);
      ma::product(ma::T(P1), TMN, std::forward<Mat>(A));
    }
    else
    {
      ma::product(P1, std::forward<Mat>(A), TMN);
      for (int p = 0; p < npol; ++p)
        SlaterDeterminantOperations::apply_expM(V, TMN.sliced(p * M, (p + 1) * M), T1, T2, order);
      ma::product(P1, TMN, std::forward<Mat>(A));
    }
  }

  // Special case for noncollinear walkers with spin-dependent V 
  // SMup[nbatch][NMO][NAEA]
  // SMdn[nbatch][NMO][NAEB]
  // P1up/dn[NMO][NMO]
  // Vup/Vdn[M][M], where M=NMO/npol (no spin off-diagonal interaction) 
  template<class Mat, class MatP1, class MatV1, class MatV2>
  void Propagate(Mat&& A, const MatP1& P1, const MatV1& V1, const MatV2& V2, int order = 6, char TA = 'N')
  {
    int npol = 2;
    int NMO  = A.size(0);
    int NAEA = A.size(1);
    int M    = NMO / npol;
    RUNTIME_CHECK(NMO % npol == 0, "");
    RUNTIME_CHECK(P1.size(0) == NMO, "");
    RUNTIME_CHECK(P1.size(1) == NMO, "");
    TMatrix TMN({NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix T1({M, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix T2({M, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    using ma::H;
    using ma::T;
    if (TA == 'H' || TA == 'h')
    {
      ma::product(ma::H(P1), std::forward<Mat>(A), TMN);
      SlaterDeterminantOperations::apply_expM(V1, TMN.sliced(0, M), T1, T2, order, TA);
      SlaterDeterminantOperations::apply_expM(V2, TMN.sliced(M, 2*M), T1, T2, order, TA);
      ma::product(ma::H(P1), TMN, std::forward<Mat>(A));
    }
    else if (TA == 'T' || TA == 't')
    {
      ma::product(ma::T(P1), std::forward<Mat>(A), TMN);
      SlaterDeterminantOperations::apply_expM(V1, TMN.sliced(0, M), T1, T2, order, TA);
      SlaterDeterminantOperations::apply_expM(V2, TMN.sliced(M, 2*M), T1, T2, order, TA);
      ma::product(ma::T(P1), TMN, std::forward<Mat>(A));
    }
    else
    {
      ma::product(P1, std::forward<Mat>(A), TMN);
      SlaterDeterminantOperations::apply_expM(V1, TMN.sliced(0, M), T1, T2, order);
      SlaterDeterminantOperations::apply_expM(V2, TMN.sliced(M, 2*M), T1, T2, order);
      ma::product(P1, TMN, std::forward<Mat>(A));
    }
  }

  // Suitable for collinear/closed walkers only -> Maybe fullypolarized as well!
  // SM[nbatch][NMO][NAEA]
  // P1[NMO][NMO]
  // V[NMO][NMO]
  template<class Array3D, class MatP1, class MatV,
           typename = typename std::enable_if_t<std::decay_t<Array3D>::dimensionality == 3>
          >
  void BatchedPropagate(Array3D && SM,
                        MatP1 const& P1,
                        MatV const& V,
                        int order = 6,
                        char TA   = 'N')

  { 
    int nbatch = SM.size(0); 
    if( nbatch <= 0 ) return; 
    int NMO    = SM.size(1);
    int NAEA   = SM.size(2);
    RUNTIME_CHECK(P1.size(0) == NMO, "");
    RUNTIME_CHECK(P1.size(1) == NMO, "");
    TTensor TMN({nbatch, NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor T1({nbatch, NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor T2({nbatch, NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    // Apply P1
    if( TA == 'H' ) 
      ma::productStridedBatched(ma::H(P1), SM, TMN);
    else if( TA == 'T' )
      ma::productStridedBatched(ma::T(P1), SM, TMN);
    else
      ma::productStridedBatched(P1, SM, TMN);
    // Apply exp(i*V)  
    SlaterDeterminantOperations::apply_expM(V, TMN, T1, T2, order, TA);
    // Apply P1
    if( TA == 'H' )
      ma::productStridedBatched(ma::H(P1), TMN, SM);
    else if( TA == 'T' )
      ma::productStridedBatched(ma::T(P1), TMN, SM);
    else
      ma::productStridedBatched(P1, TMN, SM);
  }

  // Special case for noncollinear walkers with spin-dependent V 
  // SMup[nbatch][NMO][NAEA]
  // SMdn[nbatch][NMO][NAEB]
  // P1up/dn[NMO][NMO]
  // Vup/Vdn[M][M], where M=NMO/npol (no spin off-diagonal interaction) 
  template<class Array3D, class MatP1, class MatV,
           typename = typename std::enable_if_t<std::decay_t<Array3D>::dimensionality == 3>
          >
  void BatchedPropagate(Array3D && SM,
                        MatP1 const& P1,
                        MatV const& Vup,
                        MatV const& Vdn,
                        int order = 6,
                        char TA   = 'N')

  {
    int nbatch = SM.size(0);
    if( nbatch <= 0 ) return;
    int NMO    = SM.size(1);
    int NAEA   = SM.size(2);
    int npol = 2; 
    int M      = NMO / npol;
    RUNTIME_CHECK(NMO % npol == 0, "");
    RUNTIME_CHECK(P1.size(0) == NMO, "");
    RUNTIME_CHECK(P1.size(1) == NMO, "");
    TTensor TMN({nbatch, NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor T1({nbatch, NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor T2({nbatch, NMO, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    // Apply P1
    if( TA == 'H' )
      ma::productStridedBatched(ma::H(P1), SM, TMN);
    else if( TA == 'T' )
      ma::productStridedBatched(ma::T(P1), SM, TMN);
    else
      ma::productStridedBatched(P1, SM, TMN);
    // Apply exp(i*V) 
    // treat 2 polarizations as separate elements in the batch
    using TTensor_ref = boost::multi::array_ref<T, 3, decltype(TMN.origin())>;
    TTensor_ref TMN_(TMN.origin(), {npol * nbatch, M, NAEA});
    TTensor_ref T1_(T1.origin(), {npol * nbatch, M, NAEA});
    TTensor_ref T2_(T2.origin(), {npol * nbatch, M, NAEA});
    SlaterDeterminantOperations::apply_expM_noncollinear(Vup, Vdn, TMN_, T1_, T2_,
                                                                  order, TA);
    // Apply P1
    if( TA == 'H' )
      ma::productStridedBatched(ma::H(P1), TMN, SM);
    else if( TA == 'T' )
      ma::productStridedBatched(ma::T(P1), TMN, SM);
    else
      ma::productStridedBatched(P1, TMN, SM);
  }

  // need to check if this is equivalent to QR!!!
  template<class Mat>
  T Orthogonalize(Mat&& A, T LogOverlapFactor)
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    // QR on the transpose
    int NMO  = A.size(0);
    int NAEA = A.size(1);
    TMatrix AT({NAEA, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TVector scl(iextensions<1u>{NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TVector TAU(iextensions<1u>{NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
    ma::transpose(A, AT);
    ma::geqrf(AT, TAU, WORK);
    using ma::determinant_from_geqrf;
    using ma::scale_columns;
    T res = determinant_from_geqrf(AT.size(0), AT.origin(), AT.stride(0), scl.origin(), LogOverlapFactor);
    ma::gqr(AT, TAU, WORK);
    ma::transpose(AT, A);
    scale_columns(A.size(0), A.size(1), A.origin(), A.stride(0), scl.origin());
#else
    int NMO = A.size(0);
    TVector TAU(iextensions<1u>{NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
    ma::gelqf(std::forward<Mat>(A), TAU, WORK);
    T res(0.0);
    for (int i = 0; i < A.size(1); i++)
    {
      if (real(A[i][i]) < 0)
        IWORK[i] = -1;
      else
        IWORK[i] = 1;
      res += std::log(T(IWORK[i]) * A[i][i]);
    }
    res -= LogOverlapFactor;
    ma::glq(std::forward<Mat>(A), TAU, WORK);
    for (int i = 0; i < A.size(0); ++i)
      for (int j = 0; j < A.size(1); ++j)
        A[i][j] *= T(IWORK[j]);
#endif
    return res;
  }

  template<class Mat3D, class PTR,
           typename = typename std::enable_if_t<std::decay_t<Mat3D>::dimensionality == 3>
          >
  void BatchedOrthogonalize(Mat3D&& A, T LogOverlapFactor, PTR log_detR)
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    // QR on the transpose
    if (A.size(0) == 0)
      return;
    int nbatch = A.size(0);
    int NMO    = A.size(1);
    int NAEA   = A.size(2);
    TTensor AT({nbatch, NAEA, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix T_({nbatch, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix scl({nbatch, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    int sz = ma::gqr_optimal_workspace_size(AT[0]);
    TMatrix WORK({nbatch, sz}, buffer_manager.get_generator().template get_allocator<T>());
    for (int i = 0; i < nbatch; i++)
      ma::transpose(A[i], AT[i]);
    // careful, expects fortran order
    ma::geqrf(AT, T_); 
    using ma::determinant_from_geqrf;
    using ma::scale_columns;
    for (int i = 0; i < nbatch; i++)
      *(log_detR + i) = determinant_from_geqrf(NAEA, AT[i].origin(), NMO, scl[i].origin(), LogOverlapFactor);
    ma::gqr(AT, T_, WORK);
    for (int i = 0; i < nbatch; i++)
      ma::transpose(AT[i], A[i]);
//    using ma::term_by_term_matrix_vector_strided;
//    term_by_term_matrix_vector_strided(ma::TOp_MUL, 1, NMO, NAEA, 
//            A.origin(), A.stride(1), A.stride(0), T(1.0), scl.origin(), 1, scl.stride(0), nbatch );
    ma::elementwise(ma::TOp_MUL, 1, scl(scl.extension(0),{0,NAEA}), A);
//    for (int i = 0; i < nbatch; i++)
//      scale_columns(NMO, NAEA, (A[i]).origin(), (A[i]).stride(0), scl[i].origin());
#else
    int nw = A.size(0);
    for (int i = 0; i < nw; i++)
      *(log_detR + i) = Orthogonalize(A[i], LogOverlapFactor);
#endif
  }

  template<class Mat3D,
           typename = typename std::enable_if_t<std::decay_t<Mat3D>::dimensionality == 3>
          >
  void BatchedOrthogonalize(Mat3D&& A, T LogOverlapFactor)
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    // QR on the transpose
    if (A.size(0) == 0)
      return;
    int nbatch = A.size(0);
    int NMO    = A.size(1);
    int NAEA   = A.size(2);
    TTensor AT({nbatch, NAEA, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix T_({nbatch, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    TMatrix scl({nbatch, NMO}, buffer_manager.get_generator().template get_allocator<T>());
    int sz = ma::gqr_optimal_workspace_size(AT[0]);
    TMatrix WORK({nbatch, sz}, buffer_manager.get_generator().template get_allocator<T>());
    for (int i = 0; i < nbatch; i++)
      ma::transpose(A[i], AT[i]);
    // careful, expects fortran order
    ma::geqrf(AT, T_);
    using ma::determinant_from_geqrf;
    using ma::scale_columns;
    for (int i = 0; i < nbatch; i++)
      determinant_from_geqrf(NAEA, AT[i].origin(), NMO, scl[i].origin());
    ma::gqr(AT, T_, WORK);
    for (int i = 0; i < nbatch; i++)
      ma::transpose(AT[i], A[i]);
//    using ma::term_by_term_matrix_vector_strided;
//    term_by_term_matrix_vector_strided(ma::TOp_MUL, 1, NMO, NAEA, 
//            A.origin(), A.stride(1), A.stride(0), T(1.0), scl.origin(), 1, scl.stride(0), nbatch );
    ma::elementwise(ma::TOp_MUL, 1, scl(scl.extension(0),{0,NAEA}), A);
#else
    int nw = A.size(0);
    for (int i = 0; i < nw; i++)
      Orthogonalize(A[i], LogOverlapFactor);
#endif
  }

protected:
  BufferManager buffer_manager;

  // keeping this to avoid having to know which routines are called in the lower level
  //  and let's me use static arrays
  int work_size = 0;
};

} // namespace afqmc

} // namespace sfqmc

#endif
