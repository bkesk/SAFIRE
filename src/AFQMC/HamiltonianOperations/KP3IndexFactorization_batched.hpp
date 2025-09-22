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

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_KP3INDEXFACTORIZATION_BATCHED_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_KP3INDEXFACTORIZATION_BATCHED_HPP

#include <vector>
#include <type_traits>
#include <random>
#include <algorithm>

#include "Utilities/AppAbort.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "Numerics/ma_operations.hpp"

#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/Utils.hpp"
#include "Numerics/batched_operations.hpp"
#include "Numerics/tensor_operations.hpp"


namespace sfqmc
{
namespace afqmc
{
// testing the use of dynamic data transfer during execution to reduce memory in GPU
// when an approach is found, integrate in original class through additional template parameter

template<bool SP, class LQKankMatrix>
class KP3IndexFactorization_batched
{
  using SPComplexType    = typename to_working_precision<SP,ComplexType>::type;
  using SPRealType       = typename to_working_precision<SP,RealType>::type;

  static_assert( std::is_same<SPComplexType,typename LQKankMatrix::element_type>::value ,"Inconsistent types.");

  // allocators
  using Allocator          = device_allocator<ComplexType>;
  using SpAllocator        = device_allocator<SPComplexType>;
  using BAllocator         = device_allocator<bool>;
  using IAllocator         = device_allocator<int>;
  using Allocator_shared   = node_allocator<ComplexType>;
  using SpAllocator_shared = node_allocator<SPComplexType>;
  using IAllocator_shared  = node_allocator<int>;

  using device_alloc_type  = DeviceBufferManager::template allocator_t<SPComplexType>;
  using device_alloc_Itype = DeviceBufferManager::template allocator_t<int>;

  // type defs
  using pointer                 = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer           = typename std::allocator_traits<Allocator>::const_pointer;
  using sp_pointer              = typename std::allocator_traits<SpAllocator>::pointer;
  using const_sp_pointer        = typename std::allocator_traits<SpAllocator>::const_pointer;
  using pointer_shared          = typename std::allocator_traits<Allocator_shared>::pointer;
  using const_pointer_shared    = typename std::allocator_traits<Allocator_shared>::const_pointer;
  using sp_pointer_shared       = typename std::allocator_traits<SpAllocator_shared>::pointer;
  using const_sp_pointer_shared = typename std::allocator_traits<SpAllocator_shared>::const_pointer;

  using stdIVector = Vector<int>;

  using IVector    = Vector<int, IAllocator>;
  using BoolMatrix = Matrix<bool, BAllocator>;
  using CVector    = Vector<ComplexType, Allocator>;
  using IMatrix    = Matrix<int, IAllocator>;
  using CMatrix    = Matrix<ComplexType, Allocator>;
  using C3Tensor   = Array<ComplexType, 3, Allocator>;

  using SpVector  = Vector_<SpAllocator>;
  using SpMatrix  = Matrix<SPComplexType, SpAllocator>;
  using Sp3Tensor = Array<SPComplexType, 3, SpAllocator>;

  using CVector_ref   = Vector_ref<ComplexType, pointer>;
  using CMatrix_ref   = Matrix_ref<ComplexType, pointer>;
  using C3Tensor_ref  = Array_ref<ComplexType, 3, pointer>;
  using C4Tensor_ref  = Array_ref<ComplexType, 4, pointer>;
  using C3Tensor_cref = Array_ref<ComplexType const, 3, const_pointer>;

  using SpVector_ref  = Vector_ref<SPComplexType, sp_pointer>;
  using SpMatrix_ref  = Matrix_ref<SPComplexType, sp_pointer>;
  using Sp3Tensor_ref = Array_ref<SPComplexType, 3, sp_pointer>;
  using Sp4Tensor_ref = Array_ref<SPComplexType, 4, sp_pointer>;
  using Sp5Tensor_ref = Array_ref<SPComplexType, 5, sp_pointer>;

  using StaticIVector = StaticVector<int, device_alloc_Itype>;
  using StaticSpVector  = StaticVector<SPComplexType, device_alloc_type>;
  using StaticSpMatrix  = StaticMatrix<SPComplexType, device_alloc_type>;
  using Static3Tensor = StaticArray<SPComplexType, 3, device_alloc_type>;
  using Static4Tensor = StaticArray<SPComplexType, 4, device_alloc_type>;
  using Static5Tensor = StaticArray<SPComplexType, 5, device_alloc_type>;

  using shmCVector  = ComplexVector<Allocator_shared>;
  using shmCMatrix  = ComplexMatrix<Allocator_shared>;
  using shmIMatrix  = IntegerMatrix<IAllocator_shared>;
  using shmC3Tensor = Complex3Tensor<Allocator_shared>;

  using mpi3C3Tensor = Array<ComplexType, 3, shared_allocator<ComplexType>>;

  using shmSpVector  = Vector<SPComplexType, SpAllocator_shared>;
  using shmSpMatrix  = Matrix<SPComplexType, SpAllocator_shared>;
  using shmSp3Tensor = Array<SPComplexType, 3, SpAllocator_shared>;

public:
  static const HamiltonianTypes HamOpType = KPFactorized;
  HamiltonianTypes getHamType() const { return HamOpType; }

  // NOTE: careful with nocc_max, not consistently defined!!!

  // since arrays can be in host, can't assume that types are consistent
  template<class shmCMatrix_, class shmSpMatrix_>
  KP3IndexFactorization_batched(WALKER_TYPES type,
                                afqmc::TaskGroup_& tg_,
                                stdIVector&& nopk_,
                                stdIVector&& ncholpQ_,
                                stdIVector&& kminus_,
                                boost::multi::array<int, 2>&& nelpk_,
                                boost::multi::array<int, 2>&& QKToK2_,
                                mpi3C3Tensor&& hij_,
                                shmCMatrix_&& h1,
                                std::vector<shmSpMatrix_>&& vik,
                                std::vector<shmSpMatrix_>&& vak,
                                std::vector<shmSpMatrix_>&& vakn,
                                std::vector<shmSpMatrix_>&& vbl,
                                std::vector<shmSpMatrix_>&& vbln,
                                stdIVector&& qqm_,
                                mpi3C3Tensor&& vn0_,
                                std::vector<RealType>&& gQ_,
                                int nsampleQ_,
                                ComplexType e0_,
                                Allocator const& alloc_,
                                int cv0,
                                int gncv,
                                int bf_size = 4096)
      : TG(tg_),
        allocator_(alloc_),
        sp_allocator_(alloc_),
        device_buffer_manager(),
        walker_type(type),
        global_nCV(gncv),
        global_origin(cv0),
        default_buffer_size_in_MB(bf_size),
        last_nw(-1),
        E0(e0_),
        H1(std::move(hij_)),
        haj(std::move(h1)),
        nopk(std::move(nopk_)),
        ncholpQ(std::move(ncholpQ_)),
        kminus(std::move(kminus_)),
        nelpk(std::move(nelpk_)),
        QKToK2(std::move(QKToK2_)),
        LQKikn(std::move(move_vector<shmSpMatrix>(std::move(vik)))),
        //LQKank(std::move(move_vector<LQKankMatrix>(std::move(vak),TG.Node()))),
        LQKank(std::move(move_vector<LQKankMatrix>(std::move(vak)))),
        //needs_copy(true),
        needs_copy(not std::is_same<decltype(ma::pointer_dispatch(LQKank[0].origin())), sp_pointer>::value),
        LQKakn(std::move(move_vector<shmSpMatrix>(std::move(vakn)))),
        LQKbnl(std::move(move_vector<shmSpMatrix>(std::move(vbl)))),
        LQKbln(std::move(move_vector<shmSpMatrix>(std::move(vbln)))),
        Qmap(std::move(qqm_)),
        Q2vbias(Qmap.size()),
        vn0(std::move(vn0_)),
        nsampleQ(nsampleQ_),
        gQ(std::move(gQ_)),
        Qwn({1, 1}),
        generator(),
        distribution(gQ.begin(), gQ.end()),
        KKTransID({nopk.size(), nopk.size()}, IAllocator{allocator_}),
        dev_nopk(nopk),
        dev_i0pk(typename IVector::extensions_type{nopk.size()}, IAllocator{allocator_}),
        dev_kminus(kminus),
        dev_ncholpQ(ncholpQ),
        dev_Q2vbias(typename IVector::extensions_type{nopk.size()}, IAllocator{allocator_}),
        dev_Qmap(Qmap),
        dev_nelpk(nelpk),
        dev_a0pk(typename IMatrix::extensions_type{nelpk.size(0), nelpk.size(1)}, IAllocator{allocator_}),
        dev_QKToK2(QKToK2),
        EQ(nopk.size() + 2)
  {
    using std::copy_n;
    using std::fill_n;
    nocc_max = *std::max_element(nelpk.origin(), nelpk.origin() + nelpk.num_elements());
    fill_n(EQ.data(), EQ.size(), 0);
    int nkpts = nopk.size();
    // Defines behavior over Q vector:
    //   <0: Ignore (handled by another TG)
    //    0: Calculate, without rho^+ contribution
    //   >0: Calculate, with rho^+ contribution. LQKbln data located at Qmap[Q]-1
    number_of_symmetric_Q = 0;
    number_of_Q_points    = 0;
    local_nCV             = 0;
    std::fill_n(Q2vbias.origin(), nkpts, -1);
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Q > kminus[Q])
      {
        if (Qmap[kminus[Q]] == 0)
        {
          RUNTIME_CHECK(Qmap[Q] == 0, "");
          Q2vbias[Q] = 2 * local_nCV;
          local_nCV += ncholpQ[Q];
        }
        else
        {
          RUNTIME_CHECK(Qmap[kminus[Q]] < 0, "");
          RUNTIME_CHECK(Qmap[Q] < 0, "");
        }
      }
      else if (Qmap[Q] >= 0)
      {
        Q2vbias[Q] = 2 * local_nCV;
        local_nCV += ncholpQ[Q];
        if (Qmap[Q] > 0)
          number_of_symmetric_Q++;
      }
    }
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Qmap[Q] >= 0)
        number_of_Q_points++;
      if (Qmap[Q] > 0)
      {
        RUNTIME_CHECK(Q == kminus[Q], "");
        RUNTIME_CHECK(Qmap[Q] <= number_of_symmetric_Q, "");
      }
    }
    copy_n(Q2vbias.data(), nkpts, dev_Q2vbias.origin());
    // setup dev integer arrays
    std::vector<int> i0(nkpts);
    // dev_nopk
    i0[0] = 0;
    for (int i = 1; i < nkpts; i++)
      i0[i] = i0[i - 1] + nopk[i - 1];
    copy_n(i0.data(), nkpts, dev_i0pk.origin());
    // dev_nelpk
    for (int n = 0; n < nelpk.size(0); n++)
    {
      i0[0] = 0;
      for (int i = 1; i < nkpts; i++)
        i0[i] = i0[i - 1] + nelpk[n][i - 1];
      copy_n(i0.data(), nkpts, dev_a0pk[n].origin());
      if (walker_type == COLLINEAR)
      {
        i0[0] = 0;
        for (int i = 1; i < nkpts; i++)
          i0[i] = i0[i - 1] + nelpk[n][nkpts + i - 1];
        copy_n(i0.data(), nkpts, dev_a0pk[n].origin() + nkpts);
      }
    }
    // setup copy/transpose tags
    // 1: copy from [Ki][Kj] without rho^+ term
    // 2: transpose from [Ki][Kj] without rho^+ term
    // 3: ignore
    // -P: copy from [Ki][Kj] and transpose from [nkpts+P-1][]
    boost::multi::array<int, 2> KKid({nkpts, nkpts});
    std::fill_n(KKid.origin(), KKid.num_elements(), 3); // ignore everything by default
    for (int Q = 0; Q < nkpts; ++Q)
    { // momentum conservation index
      if (Qmap[Q] < 0)
        continue;
      if (Qmap[Q] > 0)
      { // both rho and rho^+
        RUNTIME_CHECK(Q == kminus[Q], "");
        for (int K = 0; K < nkpts; ++K)
        { // K is the index of the kpoint pair of (i,k)
          int QK      = QKToK2[Q][K];
          KKid[K][QK] = -Qmap[Q];
        }
      }
      else if (Q <= kminus[Q])
      {
        // since Qmap[Q]==0 here, Q==kminus[Q] means a hermitian L_ik
        for (int K = 0; K < nkpts; ++K)
        { // K is the index of the kpoint pair of (i,k)
          int QK      = QKToK2[Q][K];
          KKid[K][QK] = 1;
        }
      }
      else if (Q > kminus[Q])
      { // use L(-Q)(ki)*
        for (int K = 0; K < nkpts; ++K)
        { // K is the index of the kpoint pair of (i,k)
          int QK      = QKToK2[Q][K];
          KKid[K][QK] = 2;
        }
      }
    }
    copy_n(KKid.origin(), KKid.num_elements(), KKTransID.origin());

    long memank = 0;
    if (needs_copy)
      for (auto& v : LQKank)
        memank = std::max(memank, 2 * v.num_elements());
    else
      for (auto& v : LQKank)
        memank += v.num_elements();

    // report memory usage
    size_t likn(0), lakn(0), lbln(0);
    for (auto& v : LQKikn)
      likn += v.num_elements();
    for (auto& v : LQKakn)
      lakn += v.num_elements();
    for (auto& v : LQKbln)
      lbln += v.num_elements();
    for (auto& v : LQKbnl)
      lbln += v.num_elements();
    app_log(1,"****************************************************************** ");
    if (needs_copy)
      app_log(1,"  Using out of core storage of LQKakn ");
    else
      app_log(1,"  Using device storage of LQKakn ");
    app_log(1,"  Static memory usage by KP3IndexFactorization_batched (node 0 in MB) ");
    app_log(1,"    L[Q][K][ikn]: {}", double(likn * sizeof(SPComplexType)) / 1024.0 / 1024.0);
    app_log(1,"    L[Q][K][akn]: {}", double((lakn + memank) * sizeof(SPComplexType)) /1024.0/1024.0);
    app_log(1,"    L[Q][K][bln]: {}", double(lbln * sizeof(SPComplexType)) / 1024.0 / 1024.0); 
    memory_report();
  }

  ~KP3IndexFactorization_batched() {}

  KP3IndexFactorization_batched(const KP3IndexFactorization_batched& other) = delete;
  KP3IndexFactorization_batched& operator=(const KP3IndexFactorization_batched& other) = delete;
  KP3IndexFactorization_batched(KP3IndexFactorization_batched&& other)                 = default;
  KP3IndexFactorization_batched& operator=(KP3IndexFactorization_batched&& other) = default;

  // must have the same signature as shared classes, so keeping it with std::allocator
  // NOTE: THIS SHOULD USE mpi3::shm!!!
  boost::multi::array<ComplexType, 2> getOneBodyPropagatorMatrix(TaskGroup_& TG_, double dt,
                                                                 boost::multi::array<ComplexType, 1> const& vMF)
  {
    int nkpts = nopk.size();
    int NMO   = std::accumulate(nopk.begin(), nopk.end(), 0);
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;

    RUNTIME_CHECK((H1.size(0) == nkpts) || (H1.size(0) == 2*nkpts), "");
    if( H1.size(0) == 2*nkpts )
      RUNTIME_CHECK(walker_type == COLLINEAR, "");
    int K0 = (H1.size(0) == 2*nkpts) ? nkpts : 0;

    CVector vMF_(vMF);
    CVector P0D(iextensions<1u>{NMO * NMO});
    ma::fill(P0D, ComplexType(0));
    vHS(vMF_, P0D, dt);
    if (TG_.TG().size() > 1)
      TG_.TG().all_reduce_in_place_n(raw_pointer_cast(P0D.origin()), P0D.num_elements(), std::plus<>());

    boost::multi::array<ComplexType, 2> P0({NMO, NMO});
    copy_n(P0D.origin(), NMO * NMO, P0.origin());

    boost::multi::array<ComplexType, 2> P1({nspin * npol * NMO, npol * NMO});
    std::fill_n(P1.origin(), P1.num_elements(), ComplexType(0.0));

    // add spin-dependent H1
    for (int K = 0, nk0 = 0; K < nkpts; ++K)
    {
      for (int i = 0, I = nk0; i < nopk[K]; i++, I++)
      {
        for (int p = 0; p < npol; ++p)
          P1[p * NMO + I][p * NMO + I] += dt * H1[K][p * nopk[K] + i][p * nopk[K] + i];
        for (int j = i + 1, J = I + 1; j < nopk[K]; j++, J++)
        {
          for (int p = 0; p < npol; ++p)
          {
            P1[p * NMO + I][p * NMO + J] += dt * H1[K][p * nopk[K] + i][p * nopk[K] + j];
            P1[p * NMO + J][p * NMO + I] += dt * H1[K][p * nopk[K] + j][p * nopk[K] + i];
          }
        }
        if (walker_type == COLLINEAR)
        { 
          P1[NMO + I][I] += dt * H1[K+K0][i][i];
          for (int j = i + 1, J = I + 1; j < nopk[K]; j++, J++)
          { 
            P1[NMO + I][J] += dt * H1[K+K0][i][j];
            P1[NMO + J][I] += dt * H1[K+K0][j][i];
          }
        }
        if (walker_type == NONCOLLINEAR)
        {
          // offdiagonal piece
          for (int j = 0, J = nk0; j < nopk[K]; j++, J++)
          {
            P1[I][NMO + J] += dt * H1[K][i][nopk[K] + j];
            P1[NMO + J][I] += dt * H1[K][nopk[K] + j][i];
          }
        }
      }
      nk0 += nopk[K];
    }

    // add P0 (diagonal in spin), already has factor of dt
    for (int p = 0; p < npol; ++p)
      for (int I = 0; I < NMO; I++)
        for (int J = 0; J < NMO; J++)
          P1[p * NMO + I][p * NMO + J] += P0[I][J];
    if (walker_type == COLLINEAR)
    { 
      for (int I = 0; I < NMO; I++)
        for (int J = 0; J < NMO; J++)
          P1[NMO + I][J] += P0[I][J];
    }

    // add vn0 (diagonal in spin)
    for (int K = 0, nk0 = 0; K < nkpts; ++K)
    {
      for (int i = 0, I = nk0; i < nopk[K]; i++, I++)
      {
        for (int p = 0; p < npol; ++p)
          P1[p * NMO + I][p * NMO + I] += dt * vn0[K][i][i];
        for (int j = i + 1, J = I + 1; j < nopk[K]; j++, J++)
        {
          for (int p = 0; p < npol; ++p)
          {
            P1[p * NMO + I][p * NMO + J] += dt * vn0[K][i][j];
            P1[p * NMO + J][p * NMO + I] += dt * vn0[K][j][i];
          }
        }
        if (walker_type == COLLINEAR)
        { 
          P1[NMO + I][I] += dt * vn0[K][i][i];
          for (int j = i + 1, J = I + 1; j < nopk[K]; j++, J++)
          { 
            P1[NMO + I][J] += dt * vn0[K][i][j];
            P1[NMO + J][I] += dt * vn0[K][j][i];
          }
        }
      }
      nk0 += nopk[K];
    }

    using ma::conj;
    // symmetrize
    for (int I = 0; I < npol * NMO; I++)
    {
      for (int J = I + 1; J < npol * NMO; J++)
      {
        // This is really cutoff dependent!!!
        if (std::abs(P1[I][J] - ma::conj(P1[J][I])) * 2.0 > 1e-5)
        {
          app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 is not hermitian. ");
          app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",I,J,P1[I][J],P1[J][I]);
        }
        P1[I][J] = 0.5 * (P1[I][J] + ma::conj(P1[J][I]));
        P1[J][I] = ma::conj(P1[I][J]);
      }
    }
    if (walker_type == COLLINEAR)
    {
      for (int I = 0; I < NMO; I++)
      {
        for (int J = I + 1; J < NMO; J++)
        {
        // This is really cutoff dependent!!!
          if (std::abs(P1[NMO+I][J] - ma::conj(P1[NMO+J][I])) * 2.0 > 1e-5)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (beta) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",I,J,P1[NMO+I][J],P1[NMO+J][I]);
          }
          P1[NMO+I][J] = 0.5 * (P1[NMO+I][J] + ma::conj(P1[NMO+J][I]));
          P1[NMO+J][I] = ma::conj(P1[NMO+I][J]);
        }
      }
    }

    return P1;
  }

  template<class TVec>
  void getFieldTypes(TVec&& v) {
    int localnvc = local_number_of_cholesky_vectors();
    RUNTIME_CHECK(v.size() == localnvc, "");
    using std::fill_n;
    fill_n( v.origin(), v.size(), ContinuousChargePropagator );
  }

  template<class Mat, class MatB>
  void energy(Mat&& E,
              MatB const& Gc,
              int nd,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    if (nsampleQ > 0)
      energy_sampleQ(E, Gc, nd, addH1, addEJ, addEXX);
    else
      energy_exact(E, Gc, nd, addH1, addEJ, addEXX);
  }

  template<class Mat, class MatB>
  void energy_exact(Mat&& E,
                    MatB const& Gc,
                    int nd,
                    bool addH1  = true,
                    bool addEJ  = true,
                    bool addEXX = true)
  {
    using std::copy_n;
    using std::fill_n;
    int nkpts = nopk.size();
    RUNTIME_CHECK(E.size(1) >= 3, "");
    RUNTIME_CHECK(nd >= 0 && nd < nelpk.size(), "");

    int nwalk     = Gc.size(1);
    int nspin     = (walker_type == COLLINEAR ? 2 : 1);
    int npol      = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nmo_tot   = std::accumulate(nopk.begin(), nopk.end(), 0);
    int nmo_max   = *std::max_element(nopk.begin(), nopk.end());
    int nocca_tot = std::accumulate(nelpk[nd].begin(), nelpk[nd].begin() + nkpts, 0);
    int nchol_max = *std::max_element(ncholpQ.begin(), ncholpQ.end());
    int noccb_tot = 0;
    if (walker_type == COLLINEAR)
      noccb_tot = std::accumulate(nelpk[nd].begin() + nkpts, nelpk[nd].begin() + 2 * nkpts, 0);
    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error in AFQMC/HamiltonianOperations/sparse_matrix_energy::calculate_energy(). Incorrect matrix "
                "dimensions ");

    // take from BufferManager.
    //      long default_buffer_size_in_MB(4L*1024L);
    long batch_size(0);
    if (addEXX)
    {
      long Bytes = long(default_buffer_size_in_MB) * 1024L * 1024L;
      Bytes /= size_t(nwalk * nocc_max * nocc_max * nchol_max * sizeof(SPComplexType));
      long bz0 = std::max(2L, Bytes);
      // batch_size includes the factor of 2 from Q/Qm pair
      batch_size = std::min(bz0, long(2 * number_of_Q_points * nkpts));
      // make sure batch_size is even
      batch_size = batch_size - (batch_size % 2L);
      RUNTIME_CHECK(batch_size % 2L == 0, "");
    }

    long Knr = 0, Knc = 0;
    if (addEJ)
    {
      Knr = nwalk;
      Knc = local_nCV;
    }
    StaticSpMatrix Kl({Knr, Knc}, device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    StaticSpMatrix Kr({Knr, Knc}, device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    fill_n(Kr.origin(), Knr * Knc, SPComplexType(0.0));
    fill_n(Kl.origin(), Knr * Knc, SPComplexType(0.0));

    for (int n = 0; n < nwalk; n++)
      fill_n(E[n].origin(), 3, ComplexType(0.));

    RUNTIME_CHECK(Gc.num_elements() == nwalk * (nocca_tot + noccb_tot) * npol * nmo_tot, "");
    C3Tensor_cref G3Da(make_device_ptr(Gc.origin()), {nocca_tot * npol, nmo_tot, nwalk});
    C3Tensor_cref G3Db(make_device_ptr(Gc.origin()) + G3Da.num_elements() * (nspin - 1), {noccb_tot, nmo_tot, nwalk});

    // later on, rewrite routine to loop over spins, to avoid storage of both spin
    // components simultaneously
    Static4Tensor GKK({nspin, nkpts, nkpts, nwalk * npol * nmo_max * nocc_max},
                      device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    GKaKjw_to_GKKwaj(G3Da, GKK[0], nelpk[nd].sliced(0, nkpts), dev_nelpk[nd], dev_a0pk[nd]);
    if (walker_type == COLLINEAR)
      GKaKjw_to_GKKwaj(G3Db, GKK[1], nelpk[nd].sliced(nkpts, 2 * nkpts), dev_nelpk[nd].sliced(nkpts, 2 * nkpts),
                       dev_a0pk[nd].sliced(nkpts, 2 * nkpts));
    // one-body contribution
    // haj[ndet*nkpts][nocc*nmo]
    // not parallelized for now, since it would require customization of Wfn
    if (addH1)
    {
      for (int n = 0; n < nwalk; n++)
        fill_n(E[n].origin(), 1, ComplexType(E0));
      // must use Gc since GKK is is SP
      int na = 0, nk = 0, nb = 0;
      for (int K = 0; K < nkpts; ++K)
      {
	if constexpr (SP) {
          int ni(nopk[K]);
          C3Tensor_ref haj_K(make_device_ptr(haj[nd * nkpts + K].origin()), {nocc_max, npol, nmo_max});
          for (int a = 0; a < nelpk[nd][K]; ++a)
            for (int pol = 0; pol < npol; ++pol)
              ma::product(ComplexType(1.), ma::T(G3Da[(na + a) * npol + pol].sliced(nk, nk + ni)),
                          haj_K[a][pol].sliced(0,ni), ComplexType(1.), E({0, nwalk}, 0));
          na += nelpk[nd][K];
          if (walker_type == COLLINEAR)
          {
            boost::multi::array_ref<ComplexType, 2, pointer> haj_Kb(haj_K.origin() + haj_K.num_elements(),
                                                                    {nocc_max, nmo_max});
            for (int b = 0; b < nelpk[nd][nkpts + K]; ++b)
              ma::product(ComplexType(1.), ma::T(G3Db[nb + b].sliced(nk, nk + ni)), haj_Kb[b].sliced(0, ni),
                          ComplexType(1.), E({0, nwalk}, 0));
            nb += nelpk[nd][nkpts + K];
          }
          nk += ni;
        } else {
          nk = nopk[K];
          {
            na = nelpk[nd][K];
            CVector_ref haj_K(make_device_ptr(haj[nd * nkpts + K].origin()), {nocc_max * npol * nmo_max});
            SpMatrix_ref Gaj(GKK[0][K][K].origin(), {nwalk, nocc_max * npol * nmo_max});
            if(na>0) ma::product(ComplexType(1.), Gaj, haj_K, ComplexType(1.), E({0, nwalk}, 0));
          }
          if (walker_type == COLLINEAR)
          {
            na = nelpk[nd][nkpts + K];
            CVector_ref haj_K(make_device_ptr(haj[nd * nkpts + K].origin()) + nocc_max * nmo_max, {nocc_max * nmo_max});
            SpMatrix_ref Gaj(GKK[1][K][K].origin(), {nwalk, nocc_max * nmo_max});
            if(na>0) ma::product(ComplexType(1.), Gaj, haj_K, ComplexType(1.), E({0, nwalk}, 0));
          }
	}
      }
    }

    // move calculation of H1 here
    // NOTE: For CLOSED/NONCOLLINEAR, can do all walkers simultaneously to improve perf. of GEMM
    //       Not sure how to do it for COLLINEAR.
    if (addEXX)
    {
      int batch_cnt(0);
      using ma::gemmBatched;
      std::vector<sp_pointer> Aarray;
      std::vector<sp_pointer> Barray;
      std::vector<sp_pointer> Carray;
      Aarray.reserve(batch_size);
      Barray.reserve(batch_size);
      Carray.reserve(batch_size);
      std::vector<SPComplexType> scl_factors;
      scl_factors.reserve(batch_size);
      std::vector<int> kdiag;
      kdiag.reserve(batch_size);

      StaticIVector IMats(iextensions<1u>{batch_size},
                          device_buffer_manager.get_generator().template get_allocator<int>());
      fill_n(IMats.origin(), IMats.num_elements(), 0);
      StaticSpVector dev_scl_factors(iextensions<1u>{batch_size},
                                   device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      Static5Tensor T1({batch_size, nwalk, nocc_max, nocc_max, nchol_max},
                       device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      SPRealType scl = (walker_type == CLOSED ? 2.0 : 1.0);

      // I WANT C++17!!!!!!
      long mem_ank(0);
      if (needs_copy)
        mem_ank = nkpts * nocc_max * nchol_max * npol * nmo_max;
      StaticSpVector LBuff(iextensions<1u>{2 * mem_ank},
                         device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      sp_pointer LQptr(nullptr), LQmptr(nullptr);
      if (needs_copy)
      {
        // data will be copied here
        LQptr  = LBuff.origin();
        LQmptr = LBuff.origin() + mem_ank;
      }

      for (int spin = 0; spin < nspin; ++spin)
      {
        for (int Q = 0; Q < nkpts; ++Q)
        {
          if (Qmap[Q] < 0)
            continue;
          int Qm      = kminus[Q];

          // simple implementation for now
          Aarray.clear();
          Barray.clear();
          Carray.clear();
          scl_factors.clear();
          kdiag.clear();
          batch_cnt = 0;

          // choose source of data depending on whether data needs to be copied or not
          if (!needs_copy)
          {
            // set to local array origin
            LQptr  = make_device_ptr(LQKank[nd * nspin * nkpts + spin * nkpts + Q].origin());
            LQmptr = make_device_ptr(LQKank[nd * nspin * nkpts + spin * nkpts + Qm].origin());
          }

          SpMatrix_ref LQ(LQptr, LQKank[nd * nspin * nkpts + spin * nkpts + Q].extensions());
          SpMatrix_ref LQm(LQmptr, LQKank[nd * nspin * nkpts + spin * nkpts + Qm].extensions());

          if (needs_copy)
          {
            copy_n(raw_pointer_cast(LQKank[nd * nspin * nkpts + spin * nkpts + Q].origin()), LQ.num_elements(), LQ.origin());
            if (Q != Qm)
              copy_n(raw_pointer_cast(LQKank[nd * nspin * nkpts + spin * nkpts + Qm].origin()), LQm.num_elements(),
                     LQm.origin());
          }

          for (int Ka = 0; Ka < nkpts; ++Ka)
          {
            int K0 = ((Qmap[Q] > 0) ? 0 : Ka);
            for (int Kb = K0; Kb < nkpts; ++Kb)
            {
              int Kl_ = QKToK2[Qm][Kb];
              int Kk  = QKToK2[Q][Ka];

              if (addEJ && Ka == Kb)
                kdiag.push_back(batch_cnt);

              if (Qmap[Q] > 0)
                Aarray.push_back(sp_pointer(
                    LQKbnl[nd * nspin * number_of_symmetric_Q + spin * number_of_symmetric_Q + Qmap[Q] - 1][Kb]
                        .origin()));
              else
                Aarray.push_back(sp_pointer(LQm[Kb].origin()));

              Barray.push_back(GKK[spin][Ka][Kl_].origin());
              Carray.push_back(T1[batch_cnt++].origin());
              Aarray.push_back(sp_pointer(LQ[Ka].origin()));
              Barray.push_back(GKK[spin][Kb][Kk].origin());
              Carray.push_back(T1[batch_cnt++].origin());

              if (Qmap[Q] > 0 || Ka == Kb)
                scl_factors.push_back(SPComplexType(SPRealType(-scl * 0.5)));
              else
                scl_factors.push_back(SPComplexType(-scl));

              if (batch_cnt >= batch_size)
              {
                gemmBatched('T', 'N', nocc_max * nchol_max, nwalk * nocc_max, npol * nmo_max, SPComplexType(1.0),
                            Aarray.data(), npol * nmo_max, Barray.data(), npol * nmo_max, SPComplexType(0.0),
                            Carray.data(), nocc_max * nchol_max, Aarray.size());

                copy_n(scl_factors.data(), scl_factors.size(), dev_scl_factors.origin());
                ma::Apwabn_Apwban_Bw(dev_scl_factors.sliced(0, scl_factors.size()), 
				     T1.sliced(0, batch_cnt), 
				     E.rotated()[1].unrotated());

                if (addEJ)
                {
                  int nc0 = Q2vbias[Q] / 2; //std::accumulate(ncholpQ.begin(),ncholpQ.begin()+Q,0);
                  copy_n(kdiag.data(), kdiag.size(), IMats.origin());
		  // Kwn += Tpwaan
		  ma::Apwaan_Bwn(IMats.sliced(0l,long(kdiag.size())), 
			T1.sliced(0, batch_cnt), 
			Kl.rotated().sliced(nc0,nc0+ncholpQ[Q]).unrotated(),
			Kr.rotated().sliced(nc0,nc0+ncholpQ[Q]).unrotated());
                }

                // reset
                Aarray.clear();
                Barray.clear();
                Carray.clear();
                scl_factors.clear();
                kdiag.clear();
                batch_cnt = 0;
              }
            }
          }

          if (batch_cnt > 0)
          {
            gemmBatched('T', 'N', nocc_max * nchol_max, nwalk * nocc_max, npol * nmo_max, SPComplexType(1.0),
                        Aarray.data(), npol * nmo_max, Barray.data(), npol * nmo_max, SPComplexType(0.0), Carray.data(),
                        nocc_max * nchol_max, Aarray.size());

            copy_n(scl_factors.data(), scl_factors.size(), dev_scl_factors.origin());
            ma::Apwabn_Apwban_Bw(dev_scl_factors.sliced(0, scl_factors.size()), 
				 T1.sliced(0, batch_cnt), 
				 E.rotated()[1].unrotated());

            if (addEJ)
            {
              int nc0 = Q2vbias[Q] / 2; //std::accumulate(ncholpQ.begin(),ncholpQ.begin()+Q,0);
              copy_n(kdiag.data(), kdiag.size(), IMats.origin());
              // Kwn += Tpwaan
              ma::Apwaan_Bwn(IMats.sliced(0l,long(kdiag.size())), 
			T1.sliced(0, batch_cnt), 
			Kl.rotated().sliced(nc0,nc0+ncholpQ[Q]).unrotated(),
			Kr.rotated().sliced(nc0,nc0+ncholpQ[Q]).unrotated());
            }
          }
        } // Q
      }   // COLLINEAR
    }

    if (addEJ)
    {
      if (not addEXX)
      {
        // calculate Kr
        APP_ABORT(" Error: Finish addEJ and not addEXX");
      }
      SPRealType scl = (walker_type == CLOSED ? 2.0 : 1.0);
      using ma::adotpby;
      for (int n = 0; n < nwalk; ++n)
      {
        adotpby(SPComplexType(SPRealType(0.5 * scl * scl)), Kl[n], Kr[n], ComplexType(0.0), E[n].origin() + 2);
      }
    }
  }

  template<class Mat, class MatB>
  void energy_sampleQ([[maybe_unused]] Mat&& E,
                      [[maybe_unused]] MatB const& Gc,
                      [[maybe_unused]] int nd,
                      [[maybe_unused]] bool addH1  = true,
                      [[maybe_unused]] bool addEJ  = true,
                      [[maybe_unused]] bool addEXX = true)
  {
    APP_ABORT("Error: energy_sampleQ not yet implemented in batched routine.");
    /*
      using std::fill_n;
      int nkpts = nopk.size(); 
      RUNTIME_CHECK(E.size(1)>=3, "");
      RUNTIME_CHECK(nd >= 0 && nd < nelpk.size(), "");  

      int nwalk = Gc.size(1);
      int nspin = (walker_type==COLLINEAR?2:1);
      int nmo_tot = std::accumulate(nopk.begin(),nopk.end(),0);
      int nmo_max = *std::max_element(nopk.begin(),nopk.end());
      int nocca_tot = std::accumulate(nelpk[nd].begin(),nelpk[nd].begin()+nkpts,0);
      int nocca_max = *std::max_element(nelpk[nd].begin(),nelpk[nd].begin()+nkpts);
      int nchol_max = *std::max_element(ncholpQ.begin(),ncholpQ.end());
      int noccb_tot = 0;
      if(walker_type==COLLINEAR) noccb_tot = std::accumulate(nelpk[nd].begin()+nkpts,
                                      nelpk[nd].begin()+2*nkpts,0);
      if(E.size(0) != nwalk || E.size(1) < 3)
        APP_ABORT(" Error in AFQMC/HamiltonianOperations/sparse_matrix_energy::calculate_energy(). Incorrect matrix dimensions ");

      size_t mem_needs(nwalk*nkpts*nkpts*nspin*nocca_max*nmo_max);
      size_t cnt(0);  
      if(addEJ) { 
#if defined(ENABLE_MIXED_PRECISION)
        mem_needs += 2*nwalk*local_nCV;
#else
        if(not getKr) mem_needs += nwalk*local_nCV;
        if(not getKl) mem_needs += nwalk*local_nCV;
#endif
      }
      set_buffer(mem_needs);

      // messy
      sp_pointer Krptr(nullptr), Klptr(nullptr);
      long Knr=0, Knc=0;
      if(addEJ) {
        Knr=nwalk;
        Knc=local_nCV;
        cnt=0;
COMPLETE
        {
          Klptr = BTMats.origin()+cnt;
          cnt += nwalk*local_nCV;
        }
        fill_n(Krptr,Knr*Knc,SPComplexType(0.0));
        fill_n(Klptr,Knr*Knc,SPComplexType(0.0));
      } else if(getKr or getKl) {
        APP_ABORT(" Error: Kr and/or Kl can only be calculated with addEJ=true.");
      }
      SpMatrix_ref Kl(Klptr,{Knr,Knc});
      SpMatrix_ref Kr(Krptr,{Knr,Knc});

      for(int n=0; n<nwalk; n++) 
        fill_n(E[n].origin(),3,ComplexType(0.));

      RUNTIME_CHECK(Gc.num_elements() == nwalk*(nocca_tot+noccb_tot)*nmo_tot, "");
      C3Tensor_cref G3Da(make_device_ptr(Gc.origin()),{nocca_tot,nmo_tot,nwalk} );
      C3Tensor_cref G3Db(make_device_ptr(Gc.origin())+G3Da.num_elements()*(nspin-1),
                            {noccb_tot,nmo_tot,nwalk} );

      Sp4Tensor_ref GKK(BTMats.origin()+cnt,
                        {nspin,nkpts,nkpts,nwalk*nmo_max*nocca_max});
      cnt+=GKK.num_elements();
      GKaKjw_to_GKKwaj(G3Da,GKK[0],nelpk[nd].sliced(0,nkpts),dev_nelpk[nd],dev_a0pk[nd]);
      if(walker_type==COLLINEAR)  
        GKaKjw_to_GKKwaj(G3Db,GKK[1],nelpk[nd].sliced(nkpts,2*nkpts),
                                     dev_nelpk[nd].sliced(nkpts,2*nkpts),
                                     dev_a0pk[nd].sliced(nkpts,2*nkpts));

      // one-body contribution
      // haj[ndet*nkpts][nocc*nmo]
      // not parallelized for now, since it would require customization of Wfn 
      if(addH1) {
        // must use Gc since GKK is is SP
        int na=0, nk=0, nb=0;
        for(int n=0; n<nwalk; n++)
          E[n][0] = E0;  
        for(int K=0; K<nkpts; ++K) {
#if defined(ENABLE_MIXED_PRECISION) 
          CMatrix_ref haj_K(make_device_ptr(haj[nd*nkpts+K].origin()),{nocc_max,nmo_max});
          for(int a=0; a<nelpk[nd][K]; ++a)
            ma::product(ComplexType(1.),ma::T(G3Da[na+a].sliced(nk,nk+nopk[K])),
                                        haj_K[a].sliced(0,nopk[K]),
                        ComplexType(1.),E({0,nwalk},0));
          na+=nelpk[nd][K];
          if(walker_type==COLLINEAR) {
            boost::multi::array_ref<ComplexType,2,pointer> haj_Kb(haj_K.origin()+haj_K.num_elements(),
                                                      {nocc_max,nmo_max});
            for(int b=0; b<nelpk[nd][nkpts+K]; ++b)
              ma::product(ComplexType(1.),ma::T(G3Db[nb+b].sliced(nk,nk+nopk[K])),
                                        haj_Kb[b].sliced(0,nopk[K]),
                        ComplexType(1.),E({0,nwalk},0));
            nb+=nelpk[nd][nkpts+K];
          }
          nk+=nopk[K];
#else
          nk = nopk[K];
          {
            na = nelpk[nd][K];
            CVector_ref haj_K(make_device_ptr(haj[nd*nkpts+K].origin()),{nocc_max*nmo_max});
            SpMatrix_ref Gaj(GKK[0][K][K].origin(),{nwalk,nocc_max*nmo_max});
            ma::product(ComplexType(1.),Gaj,haj_K,ComplexType(1.),E({0,nwalk},0));
          }
          if(walker_type==COLLINEAR) {
            na = nelpk[nd][nkpts+K];
            CVector_ref haj_K(make_device_ptr(haj[nd*nkpts+K].origin())+nocc_max*nmo_max,{nocc_max*nmo_max});
            SpMatrix_ref Gaj(GKK[1][K][K].origin(),{nwalk,nocc_max*nmo_max});
            ma::product(ComplexType(1.),Gaj,haj_K,ComplexType(1.),E({0,nwalk},0));
          }
#endif
        }
      }

      // move calculation of H1 here	
      // NOTE: For CLOSED/NONCOLLINEAR, can do all walkers simultaneously to improve perf. of GEMM
      //       Not sure how to do it for COLLINEAR.
      if(addEXX) {  

        if(Qwn.size(0) != nwalk || Qwn.size(1) != nsampleQ)
          Qwn = std::move(boost::multi::array<int,2>({nwalk,nsampleQ})); 
        {
          for(int n=0; n<nwalk; ++n) 
            for(int nQ=0; nQ<nsampleQ; ++nQ) {
              Qwn[n][nQ] = distribution(generator);
              RealType drand = distribution(generator);
              RealType s(0.0);
              bool found=false;
              for(int Q=0; Q<nkpts; Q++) {
                s += gQ[Q];
                if( drand < s ) {
                  Qwn[n][nQ] = Q;
                  found=true;
                  break;
                }
              } 
              if(not found) 
                APP_ABORT(" Error: sampleQ Qwn. ");  
            }
        }
        size_t local_memory_needs = 2*nocca_max*nocca_max*nchol_max; 
        if(TMats.num_elements() < local_memory_needs) { 
          TMats = std::move(SpVector(iextensions<1u>{local_memory_needs})); 
          using std::fill_n;
          fill_n(TMats.origin(),TMats.num_elements(),SPComplexType(0.0));
        }
        size_t local_cnt=0; 
        RealType scl = (walker_type==CLOSED?2.0:1.0);
        size_t nqk=1;  
        for(int n=0; n<nwalk; ++n) {
          for(int nQ=0; nQ<nsampleQ; ++nQ) {
            int Q = Qwn[n][nQ];
            for(int Ka=0; Ka<nkpts; ++Ka) {
              for(int Kb=0; Kb<nkpts; ++Kb) {
                { 
                  int nchol = ncholpQ[Q];
                  int Qm = kminus[Q];
                  int Kl = QKToK2[Qm][Kb];
                  int Kk = QKToK2[Q][Ka];
                  int nl = nopk[Kl];
                  int nb = nelpk[nd][Kb];
                  int na = nelpk[nd][Ka];
                  int nk = nopk[Kk];

                  SpMatrix_ref Gal(GKK[0][Ka][Kl].origin()+n*na*nl,{na,nl});
                  SpMatrix_ref Gbk(GKK[0][Kb][Kk].origin()+n*nb*nk,{nb,nk});
                  SpMatrix_ref Lank(sp_pointer(LQKank[nd*nspin*nkpts+Q][Ka].origin()),
                                                 {na*nchol,nk});
                  auto bnl_ptr(sp_pointer(LQKank[nd*nspin*nkpts+Qm][Kb].origin()));
                  if( Q == Qm ) bnl_ptr = sp_pointer(LQKbnl[nd*nspin*number_of_symmetric_Q+Qmap[Q]-1][Kb].origin());
                  SpMatrix_ref Lbnl(bnl_ptr,{nb*nchol,nl});

                  SpMatrix_ref Tban(TMats.origin()+local_cnt,{nb,na*nchol});
                  Sp3Tensor_ref T3Dban(TMats.origin()+local_cnt,{nb,na,nchol});
                  SpMatrix_ref Tabn(Tban.origin()+Tban.num_elements(),{na,nb*nchol});
                  Sp3Tensor_ref T3Dabn(Tban.origin()+Tban.num_elements(),{na,nb,nchol});

                  ma::product(Gal,ma::T(Lbnl),Tabn);
                  ma::product(Gbk,ma::T(Lank),Tban);

                  SPComplexType E_(0.0);
                  for(int a=0; a<na; ++a)
                    for(int b=0; b<nb; ++b)
                      E_ += ma::dot(T3Dabn[a][b],T3Dban[b][a]);
                  E[n][1] -= scl*0.5*static_cast<ComplexType>(E_)/gQ[Q]/double(nsampleQ);

                } // if

                if(walker_type==COLLINEAR) {

                  { 
                    int nchol = ncholpQ[Q];
                    int Qm = kminus[Q];
                    int Kl = QKToK2[Qm][Kb];
                    int Kk = QKToK2[Q][Ka];
                    int nl = nopk[Kl];
                    int nb = nelpk[nd][nkpts+Kb];
                    int na = nelpk[nd][nkpts+Ka];
                    int nk = nopk[Kk];

                    SpMatrix_ref Gal(GKK[1][Ka][Kl].origin()+n*na*nl,{na,nl});
                    SpMatrix_ref Gbk(GKK[1][Kb][Kk].origin()+n*nb*nk,{nb,nk});
                    SpMatrix_ref Lank(sp_pointer(LQKank[(nd*nspin+1)*nkpts+Q][Ka].origin()),
                                                 {na*nchol,nk});
                    auto bnl_ptr(sp_pointer(LQKank[nd*nspin*nkpts+Qm][Kb].origin()));
                    if( Q == Qm ) bnl_ptr = sp_pointer(LQKbnl[(nd*nspin+1)*number_of_symmetric_Q+
                                                                Qmap[Q]-1][Kb].origin());
                    SpMatrix_ref Lbnl(bnl_ptr,{nb*nchol,nl});

                    SpMatrix_ref Tban(TMats.origin()+local_cnt,{nb,na*nchol});
                    Sp3Tensor_ref T3Dban(TMats.origin()+local_cnt,{nb,na,nchol});
                    SpMatrix_ref Tabn(Tban.origin()+Tban.num_elements(),{na,nb*nchol});
                    Sp3Tensor_ref T3Dabn(Tban.origin()+Tban.num_elements(),{na,nb,nchol});
  
                    ma::product(Gal,ma::T(Lbnl),Tabn);
                    ma::product(Gbk,ma::T(Lank),Tban);
  
                    SPComplexType E_(0.0);
                    for(int a=0; a<na; ++a)
                      for(int b=0; b<nb; ++b)
                        E_ += ma::dot(T3Dabn[a][b],T3Dban[b][a]);
                    E[n][1] -= scl*0.5*static_cast<ComplexType>(E_)/gQ[Q]/double(nsampleQ);

                  } // if
                } // COLLINEAR 
              } // Kb 
            } // Ka
          } // nQ
        } // n 
      }  

      if(addEJ) {
        size_t local_memory_needs = 2*nchol_max*nwalk; 
        if(TMats.num_elements() < local_memory_needs) { 
          TMats = std::move(SpVector(iextensions<1u>{local_memory_needs}));
          using std::fill_n;
          fill_n(TMats.origin(),TMats.num_elements(),SPComplexType(0.0));
        }
        cnt=0; 
        SpMatrix_ref Kr_local(TMats.origin(),{nwalk,nchol_max}); 
        cnt+=Kr_local.num_elements();
        SpMatrix_ref Kl_local(TMats.origin()+cnt,{nwalk,nchol_max}); 
        cnt+=Kl_local.num_elements();
        fill_n(Kr_local.origin(),Kr_local.num_elements(),SPComplexType(0.0));
        fill_n(Kl_local.origin(),Kl_local.num_elements(),SPComplexType(0.0));
        size_t nqk=1;  
        for(int Q=0; Q<nkpts; ++Q) {
          bool haveKE=false;
          for(int Ka=0; Ka<nkpts; ++Ka) {
            { 
              haveKE=true;
              int nchol = ncholpQ[Q];
              int Qm = kminus[Q];
              int Kl = QKToK2[Qm][Ka];
              int Kk = QKToK2[Q][Ka];
              int nl = nopk[Kl];
              int na = nelpk[nd][Ka];
              int nk = nopk[Kk];

              Sp3Tensor_ref Gwal(GKK[0][Ka][Kl].origin(),{nwalk,na,nl});
              Sp3Tensor_ref Gwbk(GKK[0][Ka][Kk].origin(),{nwalk,na,nk});
              Sp3Tensor_ref Lank(sp_pointer(LQKank[nd*nspin*nkpts+Q][Ka].origin()),
                                                 {na,nchol,nk});
              auto bnl_ptr(sp_pointer(LQKank[nd*nspin*nkpts+Qm][Ka].origin()));
              if( Q == Qm ) bnl_ptr = sp_pointer(LQKbnl[nd*nspin*number_of_symmetric_Q+Qmap[Q]-1][Ka].origin());
              Sp3Tensor_ref Lbnl(bnl_ptr,{na,nchol,nl});

              // Twan = sum_l G[w][a][l] L[a][n][l]
              for(int n=0; n<nwalk; ++n) 
                for(int a=0; a<na; ++a)  
                  ma::product(SPComplexType(1.0),Lbnl[a],Gwal[n][a],
                              SPComplexType(1.0),Kl_local[n]);
              for(int n=0; n<nwalk; ++n) 
                for(int a=0; a<na; ++a)  
                  ma::product(SPComplexType(1.0),Lank[a],Gwbk[n][a],
                              SPComplexType(1.0),Kr_local[n]);
            } // if

            if(walker_type==COLLINEAR) {

              { 
                haveKE=true;
                int nchol = ncholpQ[Q];
                int Qm = kminus[Q];
                int Kl = QKToK2[Qm][Ka];
                int Kk = QKToK2[Q][Ka];
                int nl = nopk[Kl];
                int na = nelpk[nd][nkpts+Ka];
                int nk = nopk[Kk];

                Sp3Tensor_ref Gwal(GKK[1][Ka][Kl].origin(),{nwalk,na,nl});
                Sp3Tensor_ref Gwbk(GKK[1][Ka][Kk].origin(),{nwalk,na,nk});
                Sp3Tensor_ref Lank(sp_pointer(LQKank[(nd*nspin+1)*nkpts+Q][Ka].origin()),
                                                 {na,nchol,nk});
                auto bnl_ptr(sp_pointer(LQKank[(nd*nspin+1)*nkpts+Qm][Ka].origin()));
                if( Q == Qm ) bnl_ptr = sp_pointer(LQKbnl[(nd*nspin+1)*number_of_symmetric_Q+Qmap[Q]-1][Ka].origin());
                Sp3Tensor_ref Lbnl(bnl_ptr,{na,nchol,nl});

                // Twan = sum_l G[w][a][l] L[a][n][l]
                for(int n=0; n<nwalk; ++n)
                  for(int a=0; a<na; ++a)  
                    ma::product(SPComplexType(1.0),Lbnl[a],Gwal[n][a],
                                SPComplexType(1.0),Kl_local[n]);
                for(int n=0; n<nwalk; ++n)
                  for(int a=0; a<na; ++a)  
                    ma::product(SPComplexType(1.0),Lank[a],Gwbk[n][a],
                                SPComplexType(1.0),Kr_local[n]);

              } // if
            } // COLLINEAR
          } // Ka
          if(haveKE) {
            int nc0 = Q2vbias[Q]/2; //std::accumulate(ncholpQ.begin(),ncholpQ.begin()+Q,0);  
            using ma::axpy;
            for(int n=0; n<nwalk; n++) {
              axpy(SPComplexType(1.0),Kr_local[n].sliced(0,ncholpQ[Q]),
                                        Kr[n].sliced(nc0,nc0+ncholpQ[Q])); 
              axpy(SPComplexType(1.0),Kl_local[n].sliced(0,ncholpQ[Q]),
                                        Kl[n].sliced(nc0,nc0+ncholpQ[Q])); 
            }
          } // to release the lock
          if(haveKE) { 
            fill_n(Kr_local.origin(),Kr_local.num_elements(),SPComplexType(0.0));
            fill_n(Kl_local.origin(),Kl_local.num_elements(),SPComplexType(0.0));
          }  
        } // Q
        nqk=0;  
        RealType scl = (walker_type==CLOSED?2.0:1.0);
        for(int n=0; n<nwalk; ++n) {
          for(int Q=0; Q<nkpts; ++Q) {      // momentum conservation index   
            {
              int nc0 = Q2vbias[Q]/2; //std::accumulate(ncholpQ.begin(),ncholpQ.begin()+Q,0);
              E[n][2] += 0.5*scl*scl*static_cast<ComplexType>(ma::dot(Kl[n]({nc0,nc0+ncholpQ[Q]}),
                                            Kr[n]({nc0,nc0+ncholpQ[Q]})));  
            }
          }
        }
      }
*/
  }

  template<class Mat, class MatB, class MatC>
  void energy(SpinTypes spin_component,
              Mat&& E,
              MatB const& Gc,
              int nd,
              MatC&& EJn,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    using std::copy_n;
    using std::fill_n;
    int nkpts = nopk.size();
    RUNTIME_CHECK(E.size(1) >= 3, "");
    RUNTIME_CHECK(nd >= 0 && nd < nelpk.size(), "");

    int nwalk     = Gc.size(1);
    int ispin = ( spin_component == Alpha ? 0 : 1);
    int nspin     = (walker_type == COLLINEAR ? 2 : 1);
    int npol      = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nmo_tot   = std::accumulate(nopk.begin(), nopk.end(), 0);
    int nmo_max   = *std::max_element(nopk.begin(), nopk.end());
    int nocca_tot = std::accumulate(nelpk[nd].begin(), nelpk[nd].begin() + nkpts, 0);
    int nchol_max = *std::max_element(ncholpQ.begin(), ncholpQ.end());
    int noccb_tot = 0;
    if (walker_type == COLLINEAR)
      noccb_tot = std::accumulate(nelpk[nd].begin() + nkpts, nelpk[nd].begin() + 2 * nkpts, 0);
    int nocc_tot[2] = {nocca_tot,noccb_tot};
    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error in AFQMC/HamiltonianOperations/sparse_matrix_energy::calculate_energy(). Incorrect matrix "
                "dimensions ");

    if (addEJ)
    {
      RUNTIME_CHECK(EJn.size(0) == nwalk, "");
      RUNTIME_CHECK(EJn.size(1) == number_of_ke_vectors(), "");
    }

    // take from BufferManager.
    //      long default_buffer_size_in_MB(4L*1024L);
    long batch_size(0);
    if (addEXX)
    {
      long Bytes = long(default_buffer_size_in_MB) * 1024L * 1024L;
      Bytes /= size_t(nwalk * nocc_max * nocc_max * nchol_max * sizeof(SPComplexType));
      long bz0 = std::max(2L, Bytes);
      // batch_size includes the factor of 2 from Q/Qm pair
      batch_size = std::min(bz0, long(2 * number_of_Q_points * nkpts));
      // make sure batch_size is even
      batch_size = batch_size - (batch_size % 2L);
      RUNTIME_CHECK(batch_size % 2L == 0, "");
    }

    long Knr = 0, Knc = 0;
    if (addEJ)
    {
      Knr = nwalk;
      Knc = local_nCV;
    }
    StaticSpMatrix Kl({Knr, Knc}, device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    StaticSpMatrix Kr({Knr, Knc}, device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    fill_n(Kr.origin(), Knr * Knc, SPComplexType(0.0));
    fill_n(Kl.origin(), Knr * Knc, SPComplexType(0.0));

    for (int n = 0; n < nwalk; n++)
      fill_n(E[n].origin(), 3, ComplexType(0.));

    RUNTIME_CHECK(Gc.num_elements() == nwalk * nocc_tot[ispin] * npol * nmo_tot, "");
    C3Tensor_cref G3D(make_device_ptr(Gc.origin()), {nocc_tot[ispin] * npol, nmo_tot, nwalk});

    // later on, rewrite routine to loop over spins, to avoid storage of both spin
    // components simultaneously
    Static3Tensor GKK({nkpts, nkpts, nwalk * npol * nmo_max * nocc_max},
                      device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    if( spin_component == Alpha ) 
      GKaKjw_to_GKKwaj(G3D, GKK, nelpk[nd].sliced(0, nkpts), dev_nelpk[nd], dev_a0pk[nd]);
    else
      GKaKjw_to_GKKwaj(G3D, GKK, nelpk[nd].sliced(nkpts, 2 * nkpts), 
		dev_nelpk[nd].sliced(nkpts, 2 * nkpts), dev_a0pk[nd].sliced(nkpts, 2 * nkpts));
    // one-body contribution
    // haj[ndet*nkpts][nocc*nmo]
    // not parallelized for now, since it would require customization of Wfn
    if (addH1)
    {
      if(spin_component == Alpha)
        for (int n = 0; n < nwalk; n++)
          fill_n(E[n].origin(), 1, ComplexType(E0));
      // must use Gc since GKK is is SP
      int na = 0, nk = 0, nb = 0;
      for (int K = 0; K < nkpts; ++K)
      {
	if constexpr (SP) {
          int ni(nopk[K]);
          C3Tensor_ref haj_K(make_device_ptr(haj[nd * nkpts + K].origin()), {nocc_max, npol, nmo_max});
	  if(spin_component == Alpha) { 
            for (int a = 0; a < nelpk[nd][K]; ++a)
              for (int pol = 0; pol < npol; ++pol)
                ma::product(ComplexType(1.), ma::T(G3D[(na + a) * npol + pol].sliced(nk, nk + ni)),
                          haj_K[a][pol].sliced(0,ni), ComplexType(1.), E({0, nwalk}, 0));
            na += nelpk[nd][K];
          } else {
            boost::multi::array_ref<ComplexType, 2, pointer> haj_Kb(haj_K.origin() + haj_K.num_elements(),
                                                                    {nocc_max, nmo_max});
            for (int b = 0; b < nelpk[nd][nkpts + K]; ++b)
              ma::product(ComplexType(1.), ma::T(G3D[nb + b].sliced(nk, nk + ni)), haj_Kb[b].sliced(0, ni),
                          ComplexType(1.), E({0, nwalk}, 0));
            nb += nelpk[nd][nkpts + K];
          }
          nk += ni;
        } else {
          nk = nopk[K];
	  if(spin_component == Alpha) {
            na = nelpk[nd][K];
            CVector_ref haj_K(make_device_ptr(haj[nd * nkpts + K].origin()), {nocc_max * npol * nmo_max});
            SpMatrix_ref Gaj(GKK[K][K].origin(), {nwalk, nocc_max * npol * nmo_max});
            if(na>0) ma::product(ComplexType(1.), Gaj, haj_K, ComplexType(1.), E({0, nwalk}, 0));
          } else {
            na = nelpk[nd][nkpts + K];
            CVector_ref haj_K(make_device_ptr(haj[nd * nkpts + K].origin()) + nocc_max * nmo_max, 
			       {nocc_max * nmo_max});
            SpMatrix_ref Gaj(GKK[K][K].origin(), {nwalk, nocc_max * nmo_max});
            if(na>0) ma::product(ComplexType(1.), Gaj, haj_K, ComplexType(1.), E({0, nwalk}, 0));
          }
	}
      }
    }

    // move calculation of H1 here
    // NOTE: For CLOSED/NONCOLLINEAR, can do all walkers simultaneously to improve perf. of GEMM
    //       Not sure how to do it for COLLINEAR.
    if (addEXX)
    {
      int batch_cnt(0);
      using ma::gemmBatched;
      std::vector<sp_pointer> Aarray;
      std::vector<sp_pointer> Barray;
      std::vector<sp_pointer> Carray;
      Aarray.reserve(batch_size);
      Barray.reserve(batch_size);
      Carray.reserve(batch_size);
      std::vector<SPComplexType> scl_factors;
      scl_factors.reserve(batch_size);
      std::vector<int> kdiag;
      kdiag.reserve(batch_size);

      StaticIVector IMats(iextensions<1u>{batch_size},
                          device_buffer_manager.get_generator().template get_allocator<int>());
      fill_n(IMats.origin(), IMats.num_elements(), 0);
      StaticSpVector dev_scl_factors(iextensions<1u>{batch_size},
                                   device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      Static5Tensor T1({batch_size, nwalk, nocc_max, nocc_max, nchol_max},
                       device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      SPRealType scl = (walker_type == CLOSED ? 2.0 : 1.0);

      // I WANT C++17!!!!!!
      long mem_ank(0);
      if (needs_copy)
        mem_ank = nkpts * nocc_max * nchol_max * npol * nmo_max;
      StaticSpVector LBuff(iextensions<1u>{2 * mem_ank},
                         device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      sp_pointer LQptr(nullptr), LQmptr(nullptr);
      if (needs_copy)
      {
        // data will be copied here
        LQptr  = LBuff.origin();
        LQmptr = LBuff.origin() + mem_ank;
      }

      {
        for (int Q = 0; Q < nkpts; ++Q)
        {
          if (Qmap[Q] < 0)
            continue;
          int Qm      = kminus[Q];

          // simple implementation for now
          Aarray.clear();
          Barray.clear();
          Carray.clear();
          scl_factors.clear();
          kdiag.clear();
          batch_cnt = 0;

          // choose source of data depending on whether data needs to be copied or not
          if (!needs_copy)
          {
            // set to local array origin
            LQptr  = make_device_ptr(LQKank[nd * nspin * nkpts + ispin * nkpts + Q].origin());
            LQmptr = make_device_ptr(LQKank[nd * nspin * nkpts + ispin * nkpts + Qm].origin());
          }

          SpMatrix_ref LQ(LQptr, LQKank[nd * nspin * nkpts + ispin * nkpts + Q].extensions());
          SpMatrix_ref LQm(LQmptr, LQKank[nd * nspin * nkpts + ispin * nkpts + Qm].extensions());

          if (needs_copy)
          {
            copy_n(raw_pointer_cast(LQKank[nd * nspin * nkpts + ispin * nkpts + Q].origin()), LQ.num_elements(), LQ.origin());
            if (Q != Qm)
              copy_n(raw_pointer_cast(LQKank[nd * nspin * nkpts + ispin * nkpts + Qm].origin()), LQm.num_elements(),
                     LQm.origin());
          }

          for (int Ka = 0; Ka < nkpts; ++Ka)
          {
            int K0 = ((Qmap[Q] > 0) ? 0 : Ka);
            for (int Kb = K0; Kb < nkpts; ++Kb)
            {
              int Kl_ = QKToK2[Qm][Kb];
              int Kk  = QKToK2[Q][Ka];

              if (addEJ && Ka == Kb)
                kdiag.push_back(batch_cnt);

              if (Qmap[Q] > 0)
                Aarray.push_back(sp_pointer(
                    LQKbnl[nd * nspin * number_of_symmetric_Q + ispin * number_of_symmetric_Q + Qmap[Q] - 1][Kb]
                        .origin()));
              else
                Aarray.push_back(sp_pointer(LQm[Kb].origin()));

              Barray.push_back(GKK[Ka][Kl_].origin());
              Carray.push_back(T1[batch_cnt++].origin());
              Aarray.push_back(sp_pointer(LQ[Ka].origin()));
              Barray.push_back(GKK[Kb][Kk].origin());
              Carray.push_back(T1[batch_cnt++].origin());

              if (Qmap[Q] > 0 || Ka == Kb)
                scl_factors.push_back(SPComplexType(SPRealType(-scl * 0.5)));
              else
                scl_factors.push_back(SPComplexType(-scl));

              if (batch_cnt >= batch_size)
              {
                gemmBatched('T', 'N', nocc_max * nchol_max, nwalk * nocc_max, npol * nmo_max, SPComplexType(1.0),
                            Aarray.data(), npol * nmo_max, Barray.data(), npol * nmo_max, SPComplexType(0.0),
                            Carray.data(), nocc_max * nchol_max, Aarray.size());

                copy_n(scl_factors.data(), scl_factors.size(), dev_scl_factors.origin());
                ma::Apwabn_Apwban_Bw(dev_scl_factors.sliced(0, scl_factors.size()), T1, E.rotated()[1].unrotated());

                if (addEJ)
                {
                  int nc0 = Q2vbias[Q] / 2; //std::accumulate(ncholpQ.begin(),ncholpQ.begin()+Q,0);
                  copy_n(kdiag.data(), kdiag.size(), IMats.origin());
		  // Kwn += Tpwaan
		  ma::Apwaan_Bwn(IMats.sliced(0l,long(kdiag.size())), T1, 
			Kl.rotated().sliced(nc0,nc0+ncholpQ[Q]).unrotated(),
			Kr.rotated().sliced(nc0,nc0+ncholpQ[Q]).unrotated());
                }

                // reset
                Aarray.clear();
                Barray.clear();
                Carray.clear();
                scl_factors.clear();
                kdiag.clear();
                batch_cnt = 0;
              }
            }
          }

          if (batch_cnt > 0)
          {
            gemmBatched('T', 'N', nocc_max * nchol_max, nwalk * nocc_max, npol * nmo_max, SPComplexType(1.0),
                        Aarray.data(), npol * nmo_max, Barray.data(), npol * nmo_max, SPComplexType(0.0), Carray.data(),
                        nocc_max * nchol_max, Aarray.size());

            copy_n(scl_factors.data(), scl_factors.size(), dev_scl_factors.origin());
            ma::Apwabn_Apwban_Bw(dev_scl_factors.sliced(0, scl_factors.size()), T1, E.rotated()[1].unrotated());

            if (addEJ)
            {
              int nc0 = Q2vbias[Q] / 2; //std::accumulate(ncholpQ.begin(),ncholpQ.begin()+Q,0);
              copy_n(kdiag.data(), kdiag.size(), IMats.origin());
              // Kwn += Tpwaan
              ma::Apwaan_Bwn(IMats.sliced(0l,long(kdiag.size())), T1, 
			Kl.rotated().sliced(nc0,nc0+ncholpQ[Q]).unrotated(),
			Kr.rotated().sliced(nc0,nc0+ncholpQ[Q]).unrotated());
            }
          }
        } // Q
      }   // COLLINEAR
    }

    if (addEJ)
    {
      if (not addEXX)
      {
        // calculate Kr
        APP_ABORT(" Error: Finish addEJ and not addEXX");
      }
      using ma::adotpby;
      for (int n = 0; n < nwalk; ++n)
      {
        adotpby(SPComplexType(0.5), Kl[n], Kr[n], ComplexType(0.0), E[n].origin() + 2);
      }
      using EJType = typename std::decay_t<MatC>::element_type;
      if constexpr  (not std::is_same_v<EJType,SPComplexType>) {
        if(spin_component == Alpha) {
          // store as Kl,Kn
	  ma::copy_n_cast(Kl,EJn(multi::ALL,{0,local_nCV}));
	  ma::copy_n_cast(Kr,EJn(multi::ALL,{local_nCV,2*local_nCV}));
        } else {
          // store as Kn,Kl
	  ma::copy_n_cast(Kr,EJn(multi::ALL,{0,local_nCV}));
	  ma::copy_n_cast(Kl,EJn(multi::ALL,{local_nCV,2*local_nCV}));
        }
      } else {
        if(spin_component == Alpha) {
          // store as Kl,Kr
	  ma::copy(Kl,EJn(multi::ALL,{0,local_nCV}));
	  ma::copy(Kr,EJn(multi::ALL,{local_nCV,2*local_nCV}));
        } else {
          // store as Kr,Kl
	  ma::copy(Kr,EJn(multi::ALL,{0,local_nCV}));
	  ma::copy(Kl,EJn(multi::ALL,{local_nCV,2*local_nCV}));
        }
      }
      // gets multiplied by 2 in PHMSD, so undo it here!	
      RealType factor = 1.0/std::sqrt(2.0);
      ma::scal(typename EJType::value_type(factor),EJn);
    }
  }

  template<class... Args>
  void fast_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: fast_energy not implemented in KP3IndexFactorization_batched. ");
  }

  template<class... Args>
  void ph_reference_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_reference_energy not implemented yet. ");
  }

  template<class... Args>
  void ph_excited_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_excited_energy not implemented yet. ");
  }

  template<
      class MatA,
      class MatB,
      typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 1)>,
      typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 1)>,
      typename = void>
  void vHS(MatA& X, MatB&& v, double dt, double a = 1., double c = 0.)
  {
    using BType = typename std::decay<MatB>::type::element;
    using AType = typename std::decay<MatA>::type::element;
    boost::multi::array_ref<AType, 2, decltype(X.origin())> X_(X.origin(), {X.size(0), 1});
    boost::multi::array_ref<BType, 2, decltype(v.origin())> v_(v.origin(), {1, v.size(0)});
    return vHS(X_, v_, dt, a, c);
  }

  template<
      class MatA,
      class MatB,
      typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
      typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>
      >
  void vHS(MatA& X, MatB&& v, double dt, double a = 1., double c = 0.)
  {
    // scale a by sqrt(dt)
    a *= std::sqrt(dt);
    int nkpts = nopk.size();
    int nwalk = X.size(1);
    RUNTIME_CHECK(v.size(0) == nwalk, "");
    int nmo_tot   = std::accumulate(nopk.begin(), nopk.end(), 0);
    int nmo_max   = *std::max_element(nopk.begin(), nopk.end());
    int nchol_max = *std::max_element(ncholpQ.begin(), ncholpQ.end());
    RUNTIME_CHECK(X.num_elements() == nwalk * 2 * local_nCV, "");
    RUNTIME_CHECK(v.num_elements() == nwalk * nmo_tot * nmo_tot, "");
    SPComplexType halfa(SPRealType(0.5 * a), 0.0);
    SPComplexType minusimhalfa(0.0, SPRealType(-0.5 * a));
    SPComplexType imhalfa(0.0, SPRealType(0.5 * a));

    Static3Tensor vKK({nkpts + number_of_symmetric_Q, nkpts, nwalk * nmo_max * nmo_max},
		device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    fill_n(vKK.origin(), vKK.num_elements(), SPComplexType(0.0));
    Static4Tensor XQnw({nkpts, 2, nchol_max, nwalk},
		device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    fill_n(XQnw.origin(), XQnw.num_elements(), SPComplexType(0.0));

    // "rotate" X
    //  XIJ = 0.5*a*(Xn+ -i*Xn-), XJI = 0.5*a*(Xn+ +i*Xn-)
    if constexpr (not std::is_same<std::decay_t<typename MatA::element_type>,SPComplexType>::value) {
      StaticSpMatrix Xdev(X.extensions(), 
		device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      copy_n_cast(ma::pointer_dispatch(X.origin()), X.num_elements(), Xdev.origin());
      for (int Q = 0; Q < nkpts; ++Q)
      {
        if (Qmap[Q] < 0)
          continue;
        int nq = Q2vbias[Q];
        auto&& Xp(Xdev.sliced(nq, nq + ncholpQ[Q]));
        auto&& Xm(Xdev.sliced(nq + ncholpQ[Q], nq + 2 * ncholpQ[Q]));
        ma::add(halfa, Xp, minusimhalfa, Xm, XQnw[Q][0].sliced(0, ncholpQ[Q]));
        ma::add(halfa, Xp, imhalfa, Xm, XQnw[Q][1].sliced(0, ncholpQ[Q]));
        nq += 2 * ncholpQ[Q];
      }
    } else {
      for (int Q = 0; Q < nkpts; ++Q)
      {
        if (Qmap[Q] < 0)
          continue;
        int nq = Q2vbias[Q];
        auto&& Xp(X.sliced(nq, nq + ncholpQ[Q]));
        auto&& Xm(X.sliced(nq + ncholpQ[Q], nq + 2 * ncholpQ[Q]));
        ma::add(halfa, Xp, minusimhalfa, Xm, XQnw[Q][0].sliced(0, ncholpQ[Q]));
        ma::add(halfa, Xp, imhalfa, Xm, XQnw[Q][1].sliced(0, ncholpQ[Q]));
        nq += 2 * ncholpQ[Q];
      }
    }

    //  then combine Q/(-Q) pieces
    //  X(Q)np = (X(Q)np + X(-Q)nm)
    for (int Q = 0; Q < nkpts; ++Q)
    {
      if (Qmap[Q] == 0)
      {
        int Qm = kminus[Q];
        ma::axpy(SPComplexType(1.0), XQnw[Qm][1], XQnw[Q][0]);
      }
    }
    {
      // assuming contiguous
      ma::scal(SPRealType(c), v);
    }

    int nmo_max2 = nmo_max * nmo_max;
    using ma::gemmBatched;
    std::vector<sp_pointer> Aarray;
    std::vector<sp_pointer> Barray;
    std::vector<sp_pointer> Carray;
    Aarray.reserve(nkpts * nkpts);
    Barray.reserve(nkpts * nkpts);
    Carray.reserve(nkpts * nkpts);
    for (int Q = 0; Q < nkpts; ++Q)
    { // momentum conservation index
      if (Qmap[Q] < 0)
        continue;
      // v[nw][i(in K)][k(in Q(K))] += sum_n LQK[i][k][n] X[Q][0][n][nw]
      if (Q <= kminus[Q])
      {
        for (int K = 0; K < nkpts; ++K)
        { // K is the index of the kpoint pair of (i,k)
          int QK = QKToK2[Q][K];
          Aarray.push_back(sp_pointer(LQKikn[Q][K].origin()));
          Barray.push_back(XQnw[Q][0].origin());
          Carray.push_back(vKK[K][QK].origin());
        }
      }
    }
    // C: v = T(X) * T(Lik) --> F: T(Lik) * T(X) = v
    gemmBatched('T', 'T', nmo_max2, nwalk, nchol_max, SPComplexType(1.0), Aarray.data(), nchol_max, Barray.data(),
                nwalk, SPComplexType(0.0), Carray.data(), nmo_max2, Aarray.size());


    Aarray.clear();
    Barray.clear();
    Carray.clear();
    for (int Q = 0; Q < nkpts; ++Q)
    { // momentum conservation index
      if (Qmap[Q] < 0)
        continue;
      // v[nw][i(in K)][k(in Q(K))] += sum_n LQK[i][k][n] X[Q][0][n][nw]
      if (Q > kminus[Q])
      { // use L(-Q)(ki)*
        for (int K = 0; K < nkpts; ++K)
        { // K is the index of the kpoint pair of (i,k)
          int QK = QKToK2[Q][K];
          Aarray.push_back(sp_pointer(LQKikn[kminus[Q]][QK].origin()));
          Barray.push_back(XQnw[Q][0].origin());
          Carray.push_back(vKK[K][QK].origin());
        }
      }
      else if (Qmap[Q] > 0)
      { // rho(Q)^+ term
        for (int K = 0; K < nkpts; ++K)
        { // K is the index of the kpoint pair of (i,k)
          int QK = QKToK2[Q][K];
          Aarray.push_back(sp_pointer(LQKikn[Q][K].origin()));
          Barray.push_back(XQnw[Q][1].origin());
          Carray.push_back(vKK[nkpts + Qmap[Q] - 1][QK].origin());
        }
      }
    }
    // C: v = T(X) * T(Lik) --> F: T(Lik) * T(X) = v
    gemmBatched('C', 'T', nmo_max2, nwalk, nchol_max, SPComplexType(1.0), Aarray.data(), nchol_max, Barray.data(),
                nwalk, SPComplexType(0.0), Carray.data(), nmo_max2, Aarray.size());


    using vType = typename std::decay<MatB>::type::element;
    boost::multi::array_ref<vType, 3, decltype(make_device_ptr(v.origin()))> v3D(make_device_ptr(v.origin()),
                                                                                 {nwalk, nmo_tot, nmo_tot});
    vKKwij_to_vwKiKj(vKK, v3D);
    // do I need to "rotate" back, can be done if necessary
  }

  template<
      class MatA,
      class MatB,
      typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 1)>,
      typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 1)>,
      typename = void>
  void vbias(const MatA& G, MatB&& v, double dt, double a = 1., double c = 0., int k = 0)
  {
    using BType = typename std::decay<MatB>::type::element;
    using AType = typename std::decay<MatA>::type::element;
    boost::multi::array_ref<BType, 2, decltype(v.origin())> v_(v.origin(), {v.size(0), 1});
    boost::multi::array_ref<AType const, 2, decltype(G.origin())> G_(G.origin(), {G.size(0), 1});
    return vbias(G_, v_, dt, a, c, k);
  }

  template<
      class MatA,
      class MatB,
      typename = std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
      typename = std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>
      >
  void vbias(const MatA& G, MatB&& v, double dt, double a = 1., double c = 0., int nd = 0)
  {
    using ma::gemmBatched;

    // scale a by sqrt(dt)
    a *= std::sqrt(dt);
    int nkpts = nopk.size();
    RUNTIME_CHECK(nd >= 0 && nd < nelpk.size(), "");
    int nwalk = G.size(1);
    RUNTIME_CHECK(v.size(0) == 2 * local_nCV, "");
    RUNTIME_CHECK(v.size(1) == nwalk, "");
    int nspin     = (walker_type == COLLINEAR ? 2 : 1);
    int npol      = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nmo_tot   = std::accumulate(nopk.begin(), nopk.end(), 0);
    int nmo_max   = *std::max_element(nopk.begin(), nopk.end());
    int nocca_tot = std::accumulate(nelpk[nd].begin(), nelpk[nd].begin() + nkpts, 0);
    int nocca_max = *std::max_element(nelpk[nd].begin(), nelpk[nd].begin() + nkpts);
    [[maybe_unused]] int noccb_max = nocca_max;
    int nchol_max = *std::max_element(ncholpQ.begin(), ncholpQ.end());
    int noccb_tot = 0;
    if (walker_type == COLLINEAR)
    {
      noccb_tot = std::accumulate(nelpk[nd].begin() + nkpts, nelpk[nd].begin() + 2 * nkpts, 0);
      noccb_max = *std::max_element(nelpk[nd].begin() + nkpts, nelpk[nd].begin() + 2 * nkpts);
    }
    RealType scl = (walker_type == CLOSED ? 2.0 : 1.0);
    SPComplexType halfa(SPRealType(0.5 * a * scl), 0.0);
    SPComplexType minusimhalfa(0.0, SPRealType(-0.5 * a * scl));
    SPComplexType imhalfa(0.0, SPRealType(0.5 * a * scl));

    RUNTIME_CHECK(G.num_elements() == nwalk * (nocca_tot + noccb_tot) * npol * nmo_tot, "");
    // MAM: use reshape when available, then no need to deal with types
    using GType = typename std::decay<MatA>::type::element;
    boost::multi::array_ref<GType const, 3, decltype(make_device_ptr(G.origin()))> G3Da(make_device_ptr(G.origin()),
                                                                                        {nocca_tot * npol, nmo_tot,
                                                                                         nwalk});
    boost::multi::array_ref<GType const, 3, decltype(make_device_ptr(G.origin()))> G3Db(make_device_ptr(G.origin()) +
                                                                                            G3Da.num_elements() *
                                                                                                (nspin - 1),
                                                                                        {noccb_tot, nmo_tot, nwalk});

    // assuming contiguous
    ma::scal(c, v);

    for (int spin = 0; spin < nspin; spin++)
    {
      Static3Tensor v1({nkpts + number_of_symmetric_Q, nchol_max, nwalk},
                       device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      Static3Tensor GQ({nkpts, nkpts * nocc_max * npol * nmo_max, nwalk},
                       device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      fill_n(v1.origin(), v1.num_elements(), SPComplexType(0.0));
      fill_n(GQ.origin(), GQ.num_elements(), SPComplexType(0.0));

      if (spin == 0)
        GKaKjw_to_GQKajw(G3Da, GQ, nelpk[nd], dev_nelpk[nd], dev_a0pk[nd]);
      else
        GKaKjw_to_GQKajw(G3Db, GQ, nelpk[nd].sliced(nkpts, 2 * nkpts), dev_nelpk[nd].sliced(nkpts, 2 * nkpts),
                         dev_a0pk[nd].sliced(nkpts, 2 * nkpts));

      // can use productStridedBatched if LQKakn is changed to a 3Tensor array
      int Kak = nkpts * nocc_max * npol * nmo_max;
      std::vector<sp_pointer> Aarray;
      std::vector<sp_pointer> Barray;
      std::vector<sp_pointer> Carray;
      Aarray.reserve(nkpts + number_of_symmetric_Q);
      Barray.reserve(nkpts + number_of_symmetric_Q);
      Carray.reserve(nkpts + number_of_symmetric_Q);
      for (int Q = 0; Q < nkpts; ++Q)
      { // momentum conservation index
        if (Qmap[Q] < 0)
          continue;
        // v_[Q][n][w] = sum_Kak LQ[Kak][n]*G[Q][Kak][w]
        //             F: -->   G[Kak][w] * LQ[Kak][n]
        Aarray.push_back(GQ[Q].origin());
        Barray.push_back(sp_pointer(LQKakn[nd * nspin * nkpts + spin * nkpts + Q].origin()));
        Carray.push_back(v1[Q].origin());
        if (Qmap[Q] > 0)
        {
          Aarray.push_back(GQ[Q].origin());
          Barray.push_back(sp_pointer(
              LQKbln[nd * nspin * number_of_symmetric_Q + spin * number_of_symmetric_Q + Qmap[Q] - 1].origin()));
          Carray.push_back(v1[nkpts + Qmap[Q] - 1].origin());
        }
      }
      gemmBatched('N', 'T', nwalk, nchol_max, Kak, SPComplexType(1.0), Aarray.data(), nwalk, Barray.data(), nchol_max,
                  SPComplexType(0.0), Carray.data(), nwalk, Aarray.size());
      // optimize later, right now it adds contributions from Q's not assigned
      vbias_from_v1(halfa, v1, v);
    }
  }

  template<class Mat, class MatB>
  void generalizedFockMatrix([[maybe_unused]] Mat&& G, [[maybe_unused]] MatB&& Fp, [[maybe_unused]] MatB&& Fm)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

  bool distribution_over_cholesky_vectors() const { return true; }
  int number_of_ke_vectors() const { return 2 * local_nCV; }
  int local_number_of_cholesky_vectors() const { return 2 * local_nCV; }
  int global_number_of_cholesky_vectors() const { return global_nCV; }
  int global_origin_cholesky_vector() const { return global_origin; }

  // transpose=true means G[nwalk][ik], false means G[ik][nwalk]
  bool transposed_G_for_vbias() const { return false; }
  bool transposed_G_for_E() const { return false; }
  // transpose=true means vHS[nwalk][ik], false means vHS[ik][nwalk]
  bool transposed_vHS() const { return true; }

  bool fast_ph_energy() const { return false; }
  bool spin_dependent_vHS() const { return false; }

  boost::multi::array<ComplexType, 2> getHSPotentials() { return boost::multi::array<ComplexType, 2>{}; }

private:
  int nocc_max;

  afqmc::TaskGroup_& TG;

  Allocator allocator_;
  SpAllocator sp_allocator_;
  DeviceBufferManager device_buffer_manager;

  WALKER_TYPES walker_type;

  int global_nCV;
  int local_nCV;
  int global_origin;

  int default_buffer_size_in_MB;
  int last_nw;

  ComplexType E0;

  // bare one body hamiltonian
  mpi3C3Tensor H1;

  // (potentially half rotated) one body hamiltonian
  shmCMatrix haj;
  //std::vector<shmCVector> haj;

  // number of orbitals per k-point
  boost::multi::array<int, 1> nopk;

  // number of cholesky vectors per Q-point
  boost::multi::array<int, 1> ncholpQ;

  // position of (-K) in kp-list for every K
  boost::multi::array<int, 1> kminus;

  // number of electrons per k-point
  // nelpk[ndet][nspin*nkpts]
  //shmIMatrix nelpk;
  boost::multi::array<int, 2> nelpk;

  // maps (Q,K) --> k2
  //shmIMatrix QKToK2;
  boost::multi::array<int, 2> QKToK2;

  //Cholesky Tensor Lik[Q][nk][i][k][n]
  std::vector<shmSpMatrix> LQKikn;

  // half-tranformed Cholesky tensor
  std::vector<LQKankMatrix> LQKank;
  const bool needs_copy;

  // half-tranformed Cholesky tensor
  std::vector<shmSpMatrix> LQKakn;

  // half-tranformed Cholesky tensor
  std::vector<shmSpMatrix> LQKbnl;

  // half-tranformed Cholesky tensor
  std::vector<shmSpMatrix> LQKbln;

  // number of Q vectors that satisfy Q==-Q
  int number_of_symmetric_Q;

  // number of Q points assigned to this task
  int number_of_Q_points;

  // Defines behavior over Q vector:
  //   <0: Ignore (handled by another TG)
  //    0: Calculate, without rho^+ contribution
  //   >0: Calculate, with rho^+ contribution. LQKbln data located at Qmap[Q]-1
  stdIVector Qmap;

  // maps Q (only for those with Qmap >=0) to the corresponding sector in vbias
  stdIVector Q2vbias;

  // one-body piece of Hamiltonian factorization
  mpi3C3Tensor vn0;

  int nsampleQ;
  std::vector<RealType> gQ;
  boost::multi::array<int, 2> Qwn;
  std::default_random_engine generator;
  std::discrete_distribution<int> distribution;

  IMatrix KKTransID;
  IVector dev_nopk;
  IVector dev_i0pk;
  IVector dev_kminus;
  IVector dev_ncholpQ;
  IVector dev_Q2vbias;
  IVector dev_Qmap;
  IMatrix dev_nelpk;
  IMatrix dev_a0pk;
  IMatrix dev_QKToK2;

  //    std::vector<std::unique_ptr<shared_mutex>> mutex;

  //    boost::multi::array<ComplexType,3> Qave;
  //    int cntQave=0;
  std::vector<ComplexType> EQ;
  //    std::default_random_engine generator;
  //    std::uniform_real_distribution<RealType> distribution(RealType(0.0),Realtype(1.0));

  template<class MatA, class MatB, class IVec, class IVec2>
  void GKaKjw_to_GKKwaj(MatA const& GKaKj, MatB&& GKKaj, [[maybe_unused]] IVec&& nocc, IVec2&& dev_no, IVec2&& dev_a0)
  {
    int npol    = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nmo_max = *std::max_element(nopk.begin(), nopk.end());
    //      int nocc_max = *std::max_element(nocc.begin(),nocc.end());
    int nmo_tot = GKaKj.size(1);
    int nwalk   = GKaKj.size(2);
    int nkpts   = nopk.size();
    RUNTIME_CHECK(GKKaj.num_elements() >= nkpts * nkpts * nwalk * nocc_max * npol * nmo_max, "");
    RUNTIME_CHECK(GKaKj.stride(0) == GKaKj.size(1)*GKaKj.size(2), "");
    RUNTIME_CHECK(GKaKj.stride(1) == GKaKj.size(2), "");
    RUNTIME_CHECK(GKaKj.stride(2) == 1, "");

    using ma::KaKjw_to_KKwaj;
    KaKjw_to_KKwaj(nwalk, nkpts, npol, nmo_max, nmo_tot, nocc_max, dev_nopk.origin(), dev_i0pk.origin(),
                   dev_no.origin(), dev_a0.origin(), GKaKj.origin(), GKKaj.origin());
  }

  template<class MatA, class MatB, class IVec, class IVec2>
  void GKaKjw_to_GQKajw(MatA const& GKaKj, MatB&& GQKaj, [[maybe_unused]] IVec&& nocc, IVec2&& dev_no, IVec2&& dev_a0)
  {
    int npol    = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nmo_max = *std::max_element(nopk.begin(), nopk.end());
    //      int nocc_max = *std::max_element(nocc.begin(),nocc.end());
    int nmo_tot = GKaKj.size(1);
    int nwalk   = GKaKj.size(2);
    int nkpts   = nopk.size();
    RUNTIME_CHECK(GQKaj.num_elements() >= nkpts * nkpts * nwalk * nocc_max * npol * nmo_max, "");
    RUNTIME_CHECK(GKaKj.stride(0) == GKaKj.size(1)*GKaKj.size(2), "");
    RUNTIME_CHECK(GKaKj.stride(1) == GKaKj.size(2), "");
    RUNTIME_CHECK(GKaKj.stride(2) == 1, "");

    using ma::KaKjw_to_QKajw;
    KaKjw_to_QKajw(nwalk, nkpts, npol, nmo_max, nmo_tot, nocc_max, dev_nopk.origin(), dev_i0pk.origin(),
                   dev_no.origin(), dev_a0.origin(), dev_QKToK2.origin(), GKaKj.origin(), GQKaj.origin());
  }


  /*
     *   vKiKj({nwalk,nmo_tot,nmo_tot});
     *   vKK({nkpts,nkpts,nwalk*nmo_max*nmo_max} );
     */
  template<class MatA, class MatB>
  void vKKwij_to_vwKiKj(MatA const& vKK, MatB&& vKiKj)
  {
    int nmo_max = *std::max_element(nopk.begin(), nopk.end());
    int nwalk   = vKiKj.size(0);
    int nmo_tot = vKiKj.size(1);
    int nkpts   = nopk.size();
    RUNTIME_CHECK(vKiKj.stride(0) == vKiKj.size(1)*vKiKj.size(2), "");
    RUNTIME_CHECK(vKiKj.stride(1) == vKiKj.size(2), "");
    RUNTIME_CHECK(vKiKj.stride(2) == 1, "");

    using ma::vKKwij_to_vwKiKj;
    vKKwij_to_vwKiKj(nwalk, nkpts, nmo_max, nmo_tot, KKTransID.origin(), dev_nopk.origin(), dev_i0pk.origin(),
                     vKK.origin(), vKiKj.origin());
  }

  template<class MatA, class MatB>
  void vbias_from_v1(ComplexType a, MatA const& v1, MatB&& vbias)
  {
    using BType   = typename std::decay<MatB>::type::element;
    int nwalk     = vbias.size(1);
    int nkpts     = nopk.size();
    int nchol_max = *std::max_element(ncholpQ.begin(), ncholpQ.end());

    using ma::vbias_from_v1;
    // using make_device_ptr(vbias.origin()) to catch errors here
    vbias_from_v1(nwalk, nkpts, nchol_max, dev_Qmap.origin(), dev_kminus.origin(), dev_ncholpQ.origin(),
                  dev_Q2vbias.origin(), static_cast<BType>(a), v1.origin(),
                  make_device_ptr(vbias.origin()));
  }
};

} // namespace afqmc

} // namespace sfqmc

#endif
