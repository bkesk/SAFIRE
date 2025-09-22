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

#ifndef SFQMC_AFQMC_SLATERDETOPERATIONS_SHARED_HPP
#define SFQMC_AFQMC_SLATERDETOPERATIONS_SHARED_HPP

#include <fstream>

#include "AFQMC/config.h"
#include "Numerics/ma_operations.hpp"
#include "Numerics/csr_blas.hpp"
#include "AFQMC/SlaterDeterminantOperations/mixed_density_matrix.hpp"
#include "AFQMC/SlaterDeterminantOperations/apply_expM.hpp"

#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations_base.hpp"

#include "mpi3/shared_communicator.hpp"
#include "Memory/buffer_managers.h"

namespace sfqmc
{
namespace afqmc
{
template<class T = ComplexType>
class SlaterDetOperations_shared : public SlaterDetOperations_base<T, HostBufferManager>
{
public:
  using Base         = SlaterDetOperations_base<T, HostBufferManager>;
  using communicator = boost::mpi3::shared_communicator;
  using shmTVector   = boost::multi::array<T, 1, shared_allocator<T>>;

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
  using IVector = typename Base::IVector;
  using TVector = typename Base::TVector;

  SlaterDetOperations_shared() : SlaterDetOperations_base<T, HostBufferManager>(HostBufferManager{}), SM_TMats(nullptr)
  {}

  SlaterDetOperations_shared(int NMO, int NAEA)
      : SlaterDetOperations_base<T, HostBufferManager>(NMO, NAEA, HostBufferManager{}), SM_TMats(nullptr)
  {}

  ~SlaterDetOperations_shared() {}

  SlaterDetOperations_shared(const SlaterDetOperations_shared& other) = delete;
  SlaterDetOperations_shared(SlaterDetOperations_shared&& other)      = default;
  SlaterDetOperations_shared& operator=(const SlaterDetOperations_shared& other) = delete;
  SlaterDetOperations_shared& operator=(SlaterDetOperations_shared&& other) = default;

  // C must live in shared memory for this routine to work as expected
  template<class MatA, class MatB, class MatC>
  T MixedDensityMatrix(const MatA& hermA,
                       const MatB& B,
                       MatC&& C,
                       T LogOverlapFactor,
                       communicator& comm,
                       bool compact = false,
                       bool herm    = true)
  {
    int NMO  = (herm ? hermA.size(1) : hermA.size(0));
    int NAEA = (herm ? hermA.size(0) : hermA.size(1));
    set_shm_buffer(comm, NAEA * (NAEA + NMO));
    RUNTIME_CHECK(SM_TMats->num_elements() >= NAEA * (NAEA + NMO), "");
    boost::multi::array_ref<T, 2> TNN(raw_pointer_cast(SM_TMats->origin()), {NAEA, NAEA});
    boost::multi::array_ref<T, 2> TNM(raw_pointer_cast(SM_TMats->origin()) + NAEA * NAEA, {NAEA, NMO});
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::shm::MixedDensityMatrix<T>(hermA, B, std::forward<MatC>(C), LogOverlapFactor,
                                                                   TNN, TNM, IWORK, WORK, comm, compact, herm);
  }

  template<class Iptr, class MatA, class MatB, class MatC, class MatQ>
  T MixedDensityMatrixForWoodbury(const MatA& hermA,
                                  const MatB& B,
                                  MatC&& C,
                                  T LogOverlapFactor,
                                  Iptr ref,
                                  MatQ&& QQ0,
                                  communicator& comm,
                                  bool compact = false)
  {
    int Nact = hermA.size(0);
    int NEL  = B.size(1);
    int NMO  = B.size(0);
    RUNTIME_CHECK(hermA.size(1) == B.size(0), "");
    RUNTIME_CHECK(QQ0.size(0) == Nact, "");
    RUNTIME_CHECK(QQ0.size(1) == NEL, "");

    set_shm_buffer(comm, NEL * (NEL + Nact + NMO));
    RUNTIME_CHECK(SM_TMats->num_elements() >= NEL * (NEL + Nact + NMO), "");
    size_t cnt = 0;
    boost::multi::array_ref<T, 2> TNN(raw_pointer_cast(SM_TMats->origin()), {NEL, NEL});
    cnt += TNN.num_elements();
    boost::multi::array_ref<T, 2> TAB(raw_pointer_cast(SM_TMats->origin()) + cnt, {Nact, NEL});
    cnt += TAB.num_elements();
    boost::multi::array_ref<T, 2> TNM(raw_pointer_cast(SM_TMats->origin()) + cnt, {NEL, NMO});
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{NMO + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::shm::MixedDensityMatrixForWoodbury<T>(hermA, B, std::forward<MatC>(C),
                                                                              LogOverlapFactor, std::forward<MatQ>(QQ0),
                                                                              ref, TNN, TAB, TNM, IWORK, WORK, comm,
                                                                              compact);
  }

  template<class MatA, class MatB>
  T Overlap(const MatA& hermA, const MatB& B, T LogOverlapFactor, communicator& comm, bool herm = true)
  {
    int NAEA = (herm ? hermA.size(0) : hermA.size(1));
    set_shm_buffer(comm, 2 * NAEA * NAEA);
    RUNTIME_CHECK(SM_TMats->num_elements() >= 2 * NAEA * NAEA, "");
    boost::multi::array_ref<T, 2> TNN(raw_pointer_cast(SM_TMats->origin()), {NAEA, NAEA});
    boost::multi::array_ref<T, 2> TNN2(raw_pointer_cast(SM_TMats->origin()) + NAEA * NAEA, {NAEA, NAEA});
    IVector IWORK(iextensions<1u>{NAEA + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::shm::Overlap<T>(hermA, B, LogOverlapFactor, TNN, IWORK, TNN2.elements(), comm, herm);
  }

  template<typename Iptr, class MatA, class MatB, class MatC>
  T OverlapForWoodbury(const MatA& hermA,
                       const MatB& B,
                       T LogOverlapFactor,
                       Iptr ref,
                       MatC&& QQ0,
                       communicator& comm)
  {
    int Nact = hermA.size(0);
    int NEL  = B.size(1);
    RUNTIME_CHECK(hermA.size(1) == B.size(0), "");
    RUNTIME_CHECK(QQ0.size(0) == Nact, "");
    RUNTIME_CHECK(QQ0.size(1) == NEL, "");
    set_shm_buffer(comm, NEL * (Nact + NEL));
    RUNTIME_CHECK(SM_TMats->num_elements() >= NEL * (Nact + NEL), "");
    boost::multi::array_ref<T, 2> TNN(raw_pointer_cast(SM_TMats->origin()), {NEL, NEL});
    boost::multi::array_ref<T, 2> TMN(raw_pointer_cast(SM_TMats->origin()) + NEL * NEL, {Nact, NEL});
    TVector WORK(iextensions<1u>{work_size}, buffer_manager.get_generator().template get_allocator<T>());
    IVector IWORK(iextensions<1u>{Nact + 1}, buffer_manager.get_generator().template get_allocator<int>());
    return SlaterDeterminantOperations::shm::OverlapForWoodbury<T>(hermA, B, LogOverlapFactor, std::forward<MatC>(QQ0),
                                                                   ref, TNN, TMN, IWORK, WORK, comm);
  }

  template<class Mat, class MatP1, class MatV>
  void Propagate(Mat&& A,
                 const MatP1& P1,
                 const MatV& V,
                 communicator& comm,
                 int order         = 6,
                 char TA           = 'N',
                 bool noncollinear = false)
  {
    int npol = noncollinear ? 2 : 1;
    int NMO  = A.size(0);
    int NAEA = A.size(1);
    int M    = NMO / npol;
    RUNTIME_CHECK(NMO % npol == 0, "");
    RUNTIME_CHECK(P1.size(0) == NMO, "");
    RUNTIME_CHECK(P1.size(1) == NMO, "");
    RUNTIME_CHECK(V.size(0) == M, "");
    RUNTIME_CHECK(V.size(1) == M, "");
    set_shm_buffer(comm, NAEA * (NMO + 2 * M));
    RUNTIME_CHECK(SM_TMats->num_elements() >= NAEA * (NMO + 2 * M), "");
    boost::multi::array_ref<T, 2> T0(raw_pointer_cast(SM_TMats->origin()), {NMO, NAEA});
    boost::multi::array_ref<T, 2> T1(raw_pointer_cast(T0.origin()) + T0.num_elements(), {M, NAEA});
    boost::multi::array_ref<T, 2> T2(raw_pointer_cast(T1.origin()) + T1.num_elements(), {M, NAEA});
    if (comm.root())
    {
      if (TA == 'H' || TA == 'h')
        ma::product(ma::H(P1), std::forward<Mat>(A), T0);
      else if (TA == 'T' || TA == 't')
        ma::product(ma::T(P1), std::forward<Mat>(A), T0);
      else
        ma::product(P1, std::forward<Mat>(A), T0);
    }
    comm.barrier();
    for (int p = 0; p < npol; ++p)
      SlaterDeterminantOperations::apply_expM(V, T0.sliced(p * M, (p + 1) * M), T1, T2, comm, order, TA);
    comm.barrier();
    if (comm.root())
    {
      if (TA == 'H' || TA == 'h')
        ma::product(ma::H(P1), T0, std::forward<Mat>(A));
      else if (TA == 'T' || TA == 't')
        ma::product(ma::T(P1), T0, std::forward<Mat>(A));
      else
        ma::product(P1, T0, std::forward<Mat>(A));
    }
    comm.barrier();
  }

  template<class Mat, class MatP1, class MatV1, class MatV2>
  void Propagate(Mat&& A,
                 const MatP1& P1,
                 const MatV1& V1,
                 const MatV2& V2,
                 communicator& comm,
                 int order         = 6,
                 char TA           = 'N')
  {
    int npol = 2; 
    int NMO  = A.size(0);
    int NAEA = A.size(1);
    int M    = NMO / npol;
    RUNTIME_CHECK(NMO % npol == 0, "");
    RUNTIME_CHECK(P1.size(0) == NMO, "");
    RUNTIME_CHECK(P1.size(1) == NMO, "");
    RUNTIME_CHECK(V1.size(0) == M, "");
    RUNTIME_CHECK(V1.size(1) == M, "");
    RUNTIME_CHECK(V2.size(0) == M, "");
    RUNTIME_CHECK(V2.size(1) == M, "");
    set_shm_buffer(comm, NAEA * (NMO + 2 * M));
    RUNTIME_CHECK(SM_TMats->num_elements() >= NAEA * (NMO + 2 * M), "");
    boost::multi::array_ref<T, 2> T0(raw_pointer_cast(SM_TMats->origin()), {NMO, NAEA});
    boost::multi::array_ref<T, 2> T1(raw_pointer_cast(T0.origin()) + T0.num_elements(), {M, NAEA});
    boost::multi::array_ref<T, 2> T2(raw_pointer_cast(T1.origin()) + T1.num_elements(), {M, NAEA});
    if (comm.root())
    {
      if (TA == 'H' || TA == 'h')
        ma::product(ma::H(P1), std::forward<Mat>(A), T0);
      else if (TA == 'T' || TA == 't')
        ma::product(ma::T(P1), std::forward<Mat>(A), T0);
      else
        ma::product(P1, std::forward<Mat>(A), T0);
    }
    comm.barrier();
    SlaterDeterminantOperations::apply_expM(V1, T0.sliced(0, M), T1, T2, comm, order, TA);
    SlaterDeterminantOperations::apply_expM(V2, T0.sliced(M, 2*M), T1, T2, comm, order, TA);
    comm.barrier();
    if (comm.root())
    {
      if (TA == 'H' || TA == 'h')
        ma::product(ma::H(P1), T0, std::forward<Mat>(A));
      else if (TA == 'T' || TA == 't')
        ma::product(ma::T(P1), T0, std::forward<Mat>(A));
      else
        ma::product(P1, T0, std::forward<Mat>(A));
    }
    comm.barrier();
  }

  template<class... Args>
  void BatchedOverlap([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: Batched routines not compatible with SlaterDetOperations_shared::BatchedOverlap ");
  }

  template<class... Args>
  void BatchedPropagate([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: Batched routines not compatible with SlaterDetOperations_shared::BatchedPropagate ");
  }

  // C[nwalk, M, N]
  template<class... Args>
  void BatchedMixedDensityMatrix([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: Batched routines not compatible with SlaterDetOperations_shared::BatchedMixedDensityMatrix ");
  }

  template<class... Args>
  void BatchedOrthogonalize([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: Batched routines not compatible with SlaterDetOperations_shared::BatchedOrthogonalize ");
  }

protected:
  using Base::buffer_manager;
  using Base::work_size;

  // shm temporary matrices
  std::unique_ptr<shmTVector> SM_TMats;

  void set_shm_buffer(communicator& comm, size_t N)
  {
    if (SM_TMats == nullptr || SM_TMats->get_allocator() != shared_allocator<T>{comm})
    {
      SM_TMats = std::move(std::make_unique<shmTVector>(iextensions<1u>{N}, shared_allocator<T>{comm}));
    }
    else if (SM_TMats->num_elements() < N)
      SM_TMats = std::move(std::make_unique<shmTVector>(iextensions<1u>{N}, shared_allocator<T>{comm}));
  }
};

} // namespace afqmc

} // namespace sfqmc

#endif
