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

#ifndef SFQMC_AFQMC_FULL2RDM_HPP
#define SFQMC_AFQMC_FULL2RDM_HPP

#include "AFQMC/config.h"
#include <vector>
#include <string>
#include <iostream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Numerics/ma_operations.hpp"
#include "Memory/buffer_managers.h"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged 2 RDM.
 * The resulting RDM will be [3*spin][i][k][j][l]  
 * where x:2 for NONCOLLINEAR and 1 for everything else.
 * For collinear, the spin ordering is (a,a,a,a), (a,a,b,b), (b,b,b,b) 
 */
class full2rdm : public AFQMCInfo
{
  // allocators
  using Allocator        = device_allocator<ComplexType>;
  using Allocator_shared = node_allocator<ComplexType>;

  // type defs
  using pointer              = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer        = typename std::allocator_traits<Allocator>::const_pointer;
  using pointer_shared       = typename std::allocator_traits<Allocator_shared>::pointer;
  using const_pointer_shared = typename std::allocator_traits<Allocator_shared>::const_pointer;

  using CVector_ref    = boost::multi::array_ref<ComplexType, 1, pointer>;
  using CMatrix_ref    = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CVector        = boost::multi::array<ComplexType, 1, Allocator>;
  using CMatrix        = boost::multi::array<ComplexType, 2, Allocator>;
  using sharedCMatrix  = boost::multi::array<ComplexType, 2, Allocator_shared>;
  using stdCVector_ref = boost::multi::array_ref<ComplexType, 1>;
  using stdCMatrix_ref = boost::multi::array_ref<ComplexType, 2>;
  using stdCVector     = boost::multi::array<ComplexType, 1>;
  using stdCMatrix     = boost::multi::array<ComplexType, 2>;
  using stdIMatrix     = boost::multi::array<int, 2>;
  using mpi3CVector    = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3IMatrix    = boost::multi::array<int, 2, shared_allocator<int>>;
  using mpi3CMatrix    = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using mpi3CTensor    = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;
  using mpi3C4Tensor   = boost::multi::array<ComplexType, 4, shared_allocator<ComplexType>>;

  using stack_alloc_type = DeviceBufferManager::template allocator_t<ComplexType>;
  using StaticVector     = boost::multi::static_array<ComplexType, 1, stack_alloc_type>;
  using StaticMatrix     = boost::multi::static_array<ComplexType, 2, stack_alloc_type>;

public:
  full2rdm(afqmc::TaskGroup_& tg_, AFQMCInfo& info, ptree pt, WALKER_TYPES wlk, int nave_ = 1, int bsize = 1)
      : AFQMCInfo(info),
        TG(tg_),
        walker_type(wlk),
        writer(false),
        block_size(bsize),
        nave(nave_),
        apply_rotation(false),
        XRot({0, 0}, make_node_allocator<ComplexType>(TG)),
        DMAverage({0, 0}, shared_allocator<ComplexType>{TG.TG_local()})
  {
    using std::copy_n;
    using std::fill_n;

    app_log(1,"  --  Adding 2RDM (TwoRDM) estimator. -- ");

    std::string rot_file, h5_path;
    rot_file = pt.get<std::string>("rotation", "");
    h5_path = pt.get<std::string>("path", "/");

    if (rot_file != "")
    {
      if (not file_exists(rot_file))
      {
        app_error(" Error: File with rotation matrix does not exist: {}",rot_file);
        APP_ABORT("");
      }
      apply_rotation = true;
      int dim[2];

      hdf_archive dump;
      if (TG.Node().root())
      {
        if (!dump.open(rot_file, H5F_ACC_RDONLY))
          APP_ABORT("Error opening orbitals file for full2rdm estimator.");
        if (dump.push(h5_path, false)<0)
          APP_ABORT("Error in full2rdm: path not found.");
        stdCMatrix R;
        if (!dump.readEntry(R, "RotationMatrix"))
          APP_ABORT("Error reading RotationMatrix.");
        if (R.size(1) != NMO)
          APP_ABORT("Error Wrong dimensions in RotationMatrix.");
        dim[0] = R.size(0);
        dim[1] = 0;
        // conjugate rotation matrix
        std::transform(R.origin(), R.origin() + R.num_elements(), R.origin(),
                       [](const auto& c) { return std::conj(c); });
        TG.Node().broadcast_n(dim, 2, 0);
        XRot = sharedCMatrix({dim[0], NMO}, make_node_allocator<ComplexType>(TG));
        copy_n(R.origin(), R.num_elements(), make_device_ptr(XRot.origin()));
        if (TG.Node().root())
          TG.Cores().broadcast_n(raw_pointer_cast(XRot.origin()), XRot.num_elements(), 0);

        dump.pop();
        dump.close();
      }
      else
      {
        TG.Node().broadcast_n(dim, 2, 0);
        XRot = sharedCMatrix({dim[0], NMO}, make_node_allocator<ComplexType>(TG));
        if (TG.Node().root())
          TG.Cores().broadcast_n(raw_pointer_cast(XRot.origin()), XRot.num_elements(), 0);
      }
      TG.Node().barrier();

      dm_size = XRot.size(0) * XRot.size(0) * XRot.size(0) * XRot.size(0);
    }
    else
    {
      dm_size = NMO * NMO * NMO * NMO;
    }

    // (a,a,a,a), (a,a,b,b)
    nspinblocks = 2;
    if (walker_type == COLLINEAR)
    {
      nspinblocks = 3; // (a,a,a,a), (a,a,b,b), (b,b,b,b)
    }
    else if (walker_type == NONCOLLINEAR)
      APP_ABORT(" Error: NONCOLLINEAR not yet implemented. \n\n");

    dm_size *= nspinblocks;

    using std::fill_n;
    writer = (TG.Global().rank() == 0);

    DMAverage = mpi3CMatrix({nave, dm_size}, shared_allocator<ComplexType>{TG.TG_local()});
    fill_n(DMAverage.origin(), DMAverage.num_elements(), ComplexType(0.0, 0.0));
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  template<class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, MatG&& G, [[maybe_unused]] MatG_host&& G_host, 
                  HostCVec1&& Xw, [[maybe_unused]] bool impsamp)
  {
    using std::copy_n;
    static_assert(std::decay<MatG>::type::dimensionality == 4, "Wrong dimensionality");
    static_assert(std::decay<MatG_host>::type::dimensionality == 4, "Wrong dimensionality");
    // assumes G[nwalk][spin][M][M]
    RUNTIME_CHECK(G.size(0) == Xw.size(0), "");

    if (apply_rotation)
      acc_with_rotation(iav, G, Xw);
    else
      acc_no_rotation(iav, G, Xw);
  }

/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/
  template<class... Args>
  void accumulate_reference_configuration([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_reference_configuration ");
  }

  template<class... Args>
  void accumulate_excited_configuration_first([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_first ");
  }

  template<class... Args>
  void accumulate_excited_configuration_second([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_second ");
  }

  template<class HostCVec>
  void print(int iblock, hdf_archive& dump, HostCVec&& Wsum)
  {
    using std::fill_n;
    const int n_zero = 9;

    if (TG.TG_local().root())
    {
      ma::scal(ComplexType(1.0 / block_size), DMAverage);
      TG.TG_heads().reduce_in_place_n(raw_pointer_cast(DMAverage.origin()), DMAverage.num_elements(), std::plus<>(), 0);
      if (writer)
      {
        dump.push(std::string("FullTwoRDM"));
        for (int i = 0; i < nave; ++i)
        {
          dump.push(std::string("Average_") + std::to_string(i));
          std::string padded_iblock =
              std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
          stdCVector_ref DMAverage_(raw_pointer_cast(DMAverage[i].origin()), {dm_size});
          dump.write(DMAverage_, "two_rdm_" + padded_iblock);
          dump.write(Wsum[i], "denominator_" + padded_iblock);
          dump.pop();
        }
        dump.pop();
      }
    }
    TG.TG_local().barrier();
    fill_n(DMAverage.origin(), DMAverage.num_elements(), ComplexType(0.0, 0.0));
  }

private:
  TaskGroup_& TG;

  WALKER_TYPES walker_type;

  bool writer;

  int block_size;

  int nave;

  int nspinblocks;

  int dm_size;

  bool apply_rotation;

  sharedCMatrix XRot;
  stdCVector Grot;

  // DMAverage (nave, nspinblocks, x*NMO*x*NMO), x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  mpi3CMatrix DMAverage;

  template<class MatG, class CVec>
  void acc_no_rotation(int iav, MatG&& G, CVec&& Xw)
  {
    // doing this 1 walker at a time and not worrying about speed
    int nw(G.size(0));

    int i0, iN;
    std::tie(i0, iN) = FairDivideBoundary(TG.TG_local().rank(), NMO * NMO, TG.TG_local().size());
    int dN           = iN - i0;

    size_t M2(NMO * NMO);
    size_t M4(M2 * M2);
    DeviceBufferManager buffer_manager;
    StaticMatrix R({dN, NMO * NMO}, buffer_manager.get_generator().template get_allocator<ComplexType>());
    CMatrix_ref Q(R.origin(), {NMO * NMO, NMO});

    // put this in shared memory!!!
    StaticMatrix Gt({NMO, NMO}, buffer_manager.get_generator().template get_allocator<ComplexType>());
    CMatrix_ref GtC(Gt.origin(), {NMO * NMO, 1});
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    if (Grot.size() < R.num_elements())
      Grot = stdCVector(iextensions<1u>(R.num_elements()));
#endif

    if (TG.TG_local().size() > 1)
      APP_ABORT(" ncores > 1 not yet allowed in full2rdm.\n\n");

    // (ikjl) = Gik * Gjl - (same spin) Gil Gjk
    if (walker_type == COLLINEAR)
    {
      for (int iw = 0; iw < nw; iw++)
      {
        CMatrix_ref Gup(make_device_ptr(G[iw][0].origin()), {NMO * NMO, 1});
        CMatrix_ref Gdn(make_device_ptr(G[iw][1].origin()), {NMO * NMO, 1});
        // use ger !!!!

        //  (a,a,a,a)
        ma::product(Gup.sliced(i0, iN), ma::T(Gup), R);
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
        using std::copy_n;
        copy_n(R.origin(), R.num_elements(), Grot.origin());
        ma::axpy(Xw[iw], Grot, DMAverage[iav].sliced(size_t(i0) * M2, size_t(iN) * M2));
#else
        ma::axpy(Xw[iw], R.flatted(), DMAverage[iav].sliced(size_t(i0) * M2, size_t(iN) * M2));
#endif
        TG.TG_local().barrier();
        ma::transpose(G[iw][0], Gt);
        // EXX: (ikjl) -= Gil Gjk = Gt_kj  x G[i]l  for each i
        // parallelize this!!!
        for (int i = 0; i < NMO; ++i)
        {
          ma::product(ComplexType(-1.0), GtC, G[iw][0].sliced(i, i + 1), ComplexType(0.0), Q);
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
          using std::copy_n;
          copy_n(Q.origin(), Q.num_elements(), Grot.origin());
          ma::axpy(Xw[iw], Grot.sliced(0, Q.num_elements()),
                   DMAverage[iav].sliced(size_t(i * NMO) * M2, size_t((i + 1) * NMO) * M2));
#else
          ma::axpy(Xw[iw], Q.flatted(), 
                   DMAverage[iav].sliced(size_t(i * NMO) * M2, size_t((i + 1) * NMO) * M2));
#endif
        }

        //  (a,a,b,b)
        ma::product(Gup.sliced(i0, iN), ma::T(Gdn), R);
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
        using std::copy_n;
        copy_n(R.origin(), R.num_elements(), Grot.origin());
        ma::axpy(Xw[iw], Grot, DMAverage[iav].sliced(M4 + size_t(i0) * M2, M4 + size_t(iN) * M2));
#else
        ma::axpy(Xw[iw], R.flatted(), DMAverage[iav].sliced(M4 + size_t(i0) * M2, M4 + size_t(iN) * M2));
#endif

        //  (b,b,b,b)
        ma::product(Gdn.sliced(i0, iN), ma::T(Gdn), R);
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
        using std::copy_n;
        copy_n(R.origin(), R.num_elements(), Grot.origin());
        ma::axpy(Xw[iw], Grot, 
                 DMAverage[iav].sliced(2 * M4 + size_t(i0) * M2, 2 * M4 + size_t(iN) * M2));
#else
        ma::axpy(Xw[iw], R.flatted(), 
                 DMAverage[iav].sliced(2 * M4 + size_t(i0) * M2, 2 * M4 + size_t(iN) * M2));
#endif
        TG.TG_local().barrier();
        ma::transpose(G[iw][1], Gt);
        // EXX: (ikjl) -= Gil Gjk = Gt_kj  x G[i]l  for each i
        // parallelize this!!!
        for (int i = 0; i < NMO; ++i)
        {
          ma::product(ComplexType(-1.0), GtC, G[iw][1].sliced(i, i + 1), ComplexType(0.0), Q);
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
          using std::copy_n;
          copy_n(Q.origin(), Q.num_elements(), Grot.origin());
          ma::axpy(Xw[iw], Grot.sliced(0, Q.num_elements()),
                   DMAverage[iav].sliced(2 * M4 + size_t(i * NMO) * M2, 2 * M4 + size_t((i + 1) * NMO) * M2));
#else
          ma::axpy(Xw[iw], Q.flatted(), 
                   DMAverage[iav].sliced(2 * M4 + size_t(i * NMO) * M2, 2 * M4 + size_t((i + 1) * NMO) * M2));
#endif
        }
      }
    }
    else
    {
      APP_ABORT(" Error: Complete full2rdm. ");  
    }
    TG.TG_local().barrier();
  }

  template<class MatG, class CVec>
  void acc_with_rotation([[maybe_unused]] int iav, [[maybe_unused]] MatG&& G, [[maybe_unused]] CVec&& Xw)
  {
    APP_ABORT(" Error: Complete full2rdm. ");  
  }
};

} // namespace afqmc
} // namespace sfqmc

#endif
