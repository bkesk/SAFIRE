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

#ifndef SFQMC_AFQMC_SLATERDETOPERATIONS_SERIAL_HPP
#define SFQMC_AFQMC_SLATERDETOPERATIONS_SERIAL_HPP

#include <fstream>

#include "AFQMC/config.h"
#include "Numerics/ma_operations.hpp"
#include "Numerics/csr_blas.hpp"
#include "AFQMC/SlaterDeterminantOperations/mixed_density_matrix.hpp"

#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations_base.hpp"

#include "mpi3/shared_communicator.hpp"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "Memory/buffer_managers.h"

namespace sfqmc
{
namespace afqmc
{
// Implementation that doesn't use shared memory
// This version is designed for GPU and/or threading with OpenMP/libraries
template<class Type, class BufferManager>
class SlaterDetOperations_serial : public SlaterDetOperations_base<Type, BufferManager>
{
public:
  using communicator = boost::mpi3::shared_communicator;
  using Base         = SlaterDetOperations_base<Type, BufferManager>;

  using T             = typename Base::T;
  using buffer_type_T = typename Base::buffer_type_T;
  using IVector       = typename Base::IVector;
  using TVector       = typename Base::TVector;
  using TMatrix       = typename Base::TMatrix;
  using TTensor       = typename Base::TTensor; 

  using Base::MixedDensityMatrix;
  using Base::MixedDensityMatrix_noHerm;
  using Base::MixedDensityMatrixForWoodbury;
  using Base::MixedDensityMatrixFromConfiguration;
  using Base::Orthogonalize;
  using Base::BatchedOrthogonalize;
  using Base::Overlap;
  using Base::Overlap_noHerm;
  using Base::OverlapForWoodbury;
  using Base::Propagate;
  using Base::BatchedOverlap;
  using Base::BatchedMixedDensityMatrix;
  using Base::BatchedDensityMatrix;
  using Base::BatchedPropagate;
  using Base::BatchedOverlapForWoodbury;
  using Base::BatchedMixedDensityMatrixForWoodbury;
  using Base::BatchedMixedDensityMatrixFromConfiguration;

  SlaterDetOperations_serial(BufferManager b = {}) : SlaterDetOperations_base<Type, BufferManager>(b) {}

  SlaterDetOperations_serial(int NMO, int NAEA, BufferManager b)
      : SlaterDetOperations_base<Type, BufferManager>(NMO, NAEA, b)
  {}

  ~SlaterDetOperations_serial() {}

  SlaterDetOperations_serial(const SlaterDetOperations_serial& other) = delete;
  SlaterDetOperations_serial(SlaterDetOperations_serial&& other)      = default;
  SlaterDetOperations_serial& operator=(const SlaterDetOperations_serial& other) = delete;
  SlaterDetOperations_serial& operator=(SlaterDetOperations_serial&& other) = default;

  // C must live in shared memory for this routine to work as expected
  template<class MatA, class MatB, class MatC>
  T MixedDensityMatrix(const MatA& hermA,
                       const MatB& B,
                       MatC&& C,
                       T LogOverlapFactor,
                       [[maybe_unused]] communicator& comm,
                       bool compact = false,
                       [[maybe_unused]] bool herm    = true)
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    APP_ABORT(" Error: SlaterDetOperations_serial should not be here. ");
#endif
    return Base::MixedDensityMatrix(hermA, B, C, LogOverlapFactor, compact);
  }

  template<class Iptr, class MatA, class MatB, class MatC, class MatQ>
  T MixedDensityMatrixForWoodbury(const MatA& hermA,
                                  const MatB& B,
                                  MatC&& C,
                                  T LogOverlapFactor,
                                  Iptr ref,
                                  MatQ&& QQ0,
                                  [[maybe_unused]] communicator& comm,
                                  bool compact = false)
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    APP_ABORT(" Error: SlaterDetOperations_serial should not be here. ");
#endif
    return Base::MixedDensityMatrixForWoodbury(hermA, B, std::forward<MatC>(C), LogOverlapFactor, ref,
                                               std::forward<MatQ>(QQ0), compact);
  }

  template<class MatA, class MatB>
  T Overlap(const MatA& hermA, 
            const MatB& B,
            T LogOverlapFactor,
            [[maybe_unused]] communicator& comm,
            [[maybe_unused]] bool herm = true)
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    APP_ABORT(" Error: SlaterDetOperations_serial should not be here. ");
#endif
    return Base::Overlap(hermA, B, LogOverlapFactor);
  }

  template<typename Iptr, class MatA, class MatB, class MatC>
  T OverlapForWoodbury(const MatA& hermA,
                       const MatB& B,
                       T LogOverlapFactor,
                       Iptr ref,
                       MatC&& QQ0,
                       [[maybe_unused]] communicator& comm)
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    APP_ABORT(" Error: SlaterDetOperations_serial should not be here. ");
#endif
    return Base::OverlapForWoodbury(hermA, B, LogOverlapFactor, ref, std::forward<MatC>(QQ0));
  }

  template<class Mat, class MatP1, class MatV>
  void Propagate(Mat&& A,
                 const MatP1& P1,
                 const MatV& V,
                 [[maybe_unused]] communicator& comm,
                 int order         = 6,
                 char TA           = 'N',
                 bool noncollinear = false)
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    APP_ABORT(" Error: SlaterDetOperations_serial should not be here. ");
#endif
    Base::Propagate(std::forward<Mat>(A), P1, V, order, TA, noncollinear);
  }

  template<class Mat, class MatP1, class MatV1, class MatV2>
  void Propagate(Mat&& A,
                 const MatP1& P1,
                 const MatV1& V1,
                 const MatV2& V2,
                 [[maybe_unused]] communicator& comm,
                 int order         = 6,
                 char TA           = 'N')
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    APP_ABORT(" Error: SlaterDetOperations_serial should not be here. ");
#endif
    Base::Propagate(std::forward<Mat>(A), P1, V1, V2, order, TA);
  }

  // C[nwalk, M, N]
  template<class MatA, class MatB, class MatC, class TVec>
  void BatchedMixedDensityMatrix(std::vector<MatA>& hermA,
                                 std::vector<MatB>& Bi,
                                 MatC&& C,
                                 T LogOverlapFactor,
                                 TVec&& ovlp,
                                 bool compact = false,
                                 bool herm    = true)
  {
    if (Bi.size() == 0)
      return;
    static_assert((pointedType<MatA>::dimensionality == 2 or pointedType<MatA>::dimensionality == -2),
                  "Wrong dimensionality");
    static_assert(pointedType<MatB>::dimensionality == 2, "Wrong dimensionality");
    static_assert(std::decay<MatC>::type::dimensionality == 3, "Wrong dimensionality");
    static_assert(std::decay<TVec>::type::dimensionality == 1, "Wrong dimensionality");
    int NMO    = (herm ? (*hermA[0]).size(1) : (*hermA[0]).size(0));
    int NAEA   = (herm ? (*hermA[0]).size(0) : (*hermA[0]).size(1));
    int nbatch = Bi.size();
    RUNTIME_CHECK(C.size(0) == nbatch, "");
    RUNTIME_CHECK(ovlp.size() == nbatch, "");
    int n1 = nbatch, n2 = NAEA, n3 = NMO;
    if (compact)
    {
      n1 = n2 = n3 = 0;
    }
    TTensor TNN3D({nbatch, NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    TTensor TNM3D({n1, n2, n3}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{nbatch * (NMO + 1)}, buffer_manager.get_generator().template get_allocator<int>());
    SlaterDeterminantOperations::batched::MixedDensityMatrix(hermA, Bi, std::forward<MatC>(C), LogOverlapFactor,
                                                             std::forward<TVec>(ovlp), TNN3D, TNM3D, IWORK, compact,
                                                             herm);
  }

  template<class MatA, class MatB, class TVec>
  void BatchedOverlap(std::vector<MatA>& hermA,
                      std::vector<MatB>& Bi,
                      T LogOverlapFactor,
                      TVec&& ovlp,
                      bool herm = true)
  {
    static_assert((pointedType<MatA>::dimensionality == 2 or pointedType<MatA>::dimensionality == -2),
                  "Wrong dimensionality");
    static_assert(pointedType<MatB>::dimensionality == 2, "Wrong dimensionality");
    if (Bi.size() == 0)
      return;
    RUNTIME_CHECK(hermA.size() > 0, "");
    static_assert(std::decay<TVec>::type::dimensionality == 1, "Wrong dimensionality");
    int NMO    = (herm ? (*hermA[0]).size(1) : (*hermA[0]).size(0));
    int NAEA   = (herm ? (*hermA[0]).size(0) : (*hermA[0]).size(1));
    int nbatch = Bi.size();
    RUNTIME_CHECK(ovlp.size() == nbatch, "");
    TTensor TNN3D({nbatch, NAEA, NAEA}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{nbatch * (NMO + 1)}, buffer_manager.get_generator().template get_allocator<int>());
    SlaterDeterminantOperations::batched::Overlap(hermA, Bi, LogOverlapFactor, std::forward<TVec>(ovlp), TNN3D, IWORK,
                                                  herm);
  }

protected:
  using Base::buffer_manager;
  using Base::work_size;
};

} // namespace afqmc

} // namespace sfqmc

#endif
