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

#ifndef SFQMC_AFQMC_FULL1RDM_HPP
#define SFQMC_AFQMC_FULL1RDM_HPP

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
 * Observable class that calculates the walker averaged 1 RDM.
 * The resulting RDM will be [spin][x*NMO][x*NMO],
 * where x:2 for NONCOLLINEAR and 1 for everything else.
 */
class full1rdm : public AFQMCInfo
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
  using StaticMatrix     = boost::multi::static_array<ComplexType, 2, stack_alloc_type>;

public:
  full1rdm(afqmc::TaskGroup_& tg_, AFQMCInfo& info, ptree pt, WALKER_TYPES wlk, int nave_ = 1, int bsize = 1)
      : AFQMCInfo(info),
        TG(tg_),
        walker_type(wlk),
        writer(false),
        block_size(bsize),
        nave(nave_),
        hdf_walker_output(""),
        nskip_walker_output(0),
        apply_rotation(false),
        XRot({0, 0}, make_node_allocator<ComplexType>(TG)),
        print_from_list(false),
        index_list({0, 0}, shared_allocator<int>{TG.Node()}),
        DMAverage({0, 0}, shared_allocator<ComplexType>{TG.TG_local()})
  {
    using std::copy_n;
    using std::fill_n;

    app_log(1,"  --  Adding Full 1RDM (OneRDM) estimator. -- ");
    std::string rot_file, h5_path; 
    hdf_walker_output = pt.get<std::string>("walker_output", "");
    nskip_walker_output = pt.get<int>("nskip_output", 0);
    rot_file = pt.get<std::string>("rotation", "");
    h5_path = pt.get<std::string>("path", "/");
    print_from_list = pt.get<bool>("with_index_list", false);

    if (rot_file != "")
    {
      if (not file_exists(rot_file))
      {
        app_error(" Error: File with rotation matrix does not exist: {}",rot_file);
        APP_ABORT("");
      }
      apply_rotation  = true;
      int dim[2];

      hdf_archive dump;
      if (TG.Node().root())
      {
        if (!dump.open(rot_file, H5F_ACC_RDONLY))
          APP_ABORT("Error opening orbitals file for n2r estimator.");
        if (dump.push(h5_path, false)<0)
          APP_ABORT("Error in full1rdm: path not found.");
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
        stdIMatrix I;
        if (print_from_list)
        {
          if (!dump.readEntry(I, "Indices"))
            APP_ABORT("Error reading Indices.");
          if (I.size(1) != 2)
            APP_ABORT("Error Wrong dimensions in Indices.");
          dim[1] = I.size(0);
        }
        TG.Node().broadcast_n(dim, 2, 0);
        XRot = sharedCMatrix({dim[0], NMO}, make_node_allocator<ComplexType>(TG));
        copy_n(R.origin(), R.num_elements(), make_device_ptr(XRot.origin()));
        if (TG.Node().root())
          TG.Cores().broadcast_n(raw_pointer_cast(XRot.origin()), XRot.num_elements(), 0);
        if (print_from_list)
        {
          index_list = mpi3IMatrix({dim[1], 2}, shared_allocator<int>{TG.Node()});
          copy_n(I.origin(), I.num_elements(), make_device_ptr(index_list.origin()));
          if (TG.Node().root())
            TG.Cores().broadcast_n(raw_pointer_cast(index_list.origin()), index_list.num_elements(), 0);
        }

        dump.pop();
        dump.close();
      }
      else
      {
        TG.Node().broadcast_n(dim, 2, 0);
        XRot = sharedCMatrix({dim[0], NMO}, make_node_allocator<ComplexType>(TG));
        if (TG.Node().root())
          TG.Cores().broadcast_n(raw_pointer_cast(XRot.origin()), XRot.num_elements(), 0);
        if (print_from_list)
        {
          index_list = mpi3IMatrix({dim[1], 2}, shared_allocator<int>{TG.Node()});
          if (TG.Node().root())
            TG.Cores().broadcast_n(raw_pointer_cast(index_list.origin()), index_list.num_elements(), 0);
        }
      }
      TG.Node().barrier();

      if (print_from_list)
        dm_size = index_list.size(0);
      else
        dm_size = XRot.size(0) * XRot.size(0);
    }
    else
    {
      // can also add print_from_list option without rotation later on
      dm_size = NMO * NMO;
    }

    if (walker_type == COLLINEAR)
      dm_size *= 2;
    else if (walker_type == NONCOLLINEAR)
      dm_size *= 4;

    if (hdf_walker_output != std::string(""))
    {
      hdf_walker_output = "G" + std::to_string(TG.TG_heads().rank()) + "_" + hdf_walker_output;
      hdf_archive dump;
      if (not dump.create(hdf_walker_output))
      {
        app_log(1,"Problems creating walker output hdf5 file: {}",hdf_walker_output); 
        APP_ABORT("Problems creating walker output hdf5 file.");
      }
      dump.push("FullOneRDM");
      dump.push("Metadata");
      dump.write(NMO, "NMO");
      dump.write(NAEA, "NUP");
      dump.write(NAEB, "NDOWN");
      int wlk_t_copy = walker_type; // the actual data type of enum is implementation-defined. convert to int for file
      dump.write(wlk_t_copy, "WalkerType");
      dump.pop();
      dump.pop();
      dump.close();
    }

    using std::fill_n;
    writer = (TG.Global().rank() == 0);

    DMAverage = mpi3CMatrix({nave, dm_size}, shared_allocator<ComplexType>{TG.TG_local()});
    fill_n(DMAverage.origin(), DMAverage.num_elements(), ComplexType(0.0, 0.0));
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  template<class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, MatG&& G, MatG_host&& G_host, HostCVec1&& Xw, [[maybe_unused]] bool impsamp)
  {
    static_assert(std::decay<MatG>::type::dimensionality == 4, "Wrong dimensionality");
    static_assert(std::decay<MatG_host>::type::dimensionality == 4, "Wrong dimensionality");
    // assumes G[nwalk][spin][M][M]
    RUNTIME_CHECK(G_host.size(0) == Xw.size(0), "");
    RUNTIME_CHECK(G_host[0].num_elements() == DMAverage.size(1), "");
    
    if (apply_rotation) {
      // Grot = Xc * G * H(Xc)

      if (walker_type == NONCOLLINEAR)
        APP_ABORT("Error: Not yet implemented: acc_with_rotation && noncollinear.");

      int nw(G.size(0));  
      int i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(TG.TG_local().rank(), 
                                            int(XRot.size(0)), TG.TG_local().size());
      int nX   = XRot.size(0);
      int npts = (iN - i0) * nX;
      DeviceBufferManager buffer_manager;
      StaticMatrix T1({(iN - i0), NMO}, buffer_manager.get_generator().template get_allocator<ComplexType>());
      StaticMatrix T2({(iN - i0), nX}, buffer_manager.get_generator().template get_allocator<ComplexType>());
      if (Grot.size() != npts)
        Grot = stdCVector(iextensions<1u>(npts));      

      // round-robin for now
      // can batch over walkers if too slow on GPU!
      for (int iw = 0; iw < nw; iw++)
      {
        if (i0 == iN || i0 == XRot.size(0))
          break;
        ma::product(XRot.sliced(i0, iN), G[iw][0], T1);
        ma::product(T1, ma::H(XRot), T2);
        copy_n(T2.origin(), T2.num_elements(), Grot.origin());
        if (print_from_list)
        {
          for (int i = 0; i < index_list.size(0); i++)
          {
            if (index_list[i][0] >= i0 && index_list[i][0] < iN)
            {
              int ij = (index_list[i][0] - i0) * nX + index_list[i][1];
              DMAverage[iav][i] += Xw[iw] * Grot[ij];
            }
          }
        }
        else
          ma::axpy(Xw[iw], Grot, DMAverage[iav].sliced(i0 * nX, i0 * nX + npts));
        if (walker_type == COLLINEAR)
        {
          ma::product(XRot.sliced(i0, iN), G[iw][1], T1);
          ma::product(T1, ma::H(XRot), T2);
          copy_n(T2.origin(), T2.num_elements(), Grot.origin());
          if (print_from_list)
          {
            for (int i = 0, ie = index_list.size(0); i < ie; i++)
            {
              if (index_list[i][0] >= i0 && index_list[i][0] < iN)
              {
                int ij = (index_list[i][0] - i0) * nX + index_list[i][1];
                DMAverage[iav][i + ie] += Xw[iw] * Grot[ij];
              }
            }
          }
          else
            ma::axpy(Xw[iw], Grot, DMAverage[iav].sliced((nX + i0) * nX, (nX + i0) * nX + npts));
        }
      }

    } else {

      int i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(TG.TG_local().rank(), int(DMAverage.size(1)), 
                                          TG.TG_local().size());

      // DMAverage[iav][ij] += sum_iw G_host[iw][ij] Xw[iw] = T( G_host ) * Xw 
      auto G2D = G_host.rotated().flatted().flatted().sliced(i0, iN).unrotated();
      ma::product(ComplexType(1.0, 0.0), ma::T(G2D), Xw, ComplexType(1.0, 0.0),
                DMAverage[iav].sliced(i0, iN));

    }
    TG.TG_local().barrier();
  }

  // Second interface, including factorized G in addition to full G and G_host 
  template<class Mat1, class Mat2, class Mat3, class Mat4,
           class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, [[maybe_unused]] Mat1&& Sa, [[maybe_unused]] Mat2&& Ga,
                  [[maybe_unused]] Mat3&& Sb, [[maybe_unused]] Mat4&& Gb,
                  MatG&& G, MatG_host&& G_host, HostCVec1&& Xw, bool impsamp)
  {
    accumulate(iav,std::forward<MatG>(G),std::forward<MatG_host>(G_host),
                   std::forward<HostCVec1>(Xw),impsamp);
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
        dump.push(std::string("FullOneRDM"));
        for (int i = 0; i < nave; ++i)
        {
          dump.push(std::string("Average_") + std::to_string(i));
          std::string padded_iblock =
              std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
          stdCVector_ref DMAverage_(raw_pointer_cast(DMAverage[i].origin()), {dm_size});
          dump.write(DMAverage_, "one_rdm_" + padded_iblock);
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

  int dm_size;

  std::string hdf_walker_output;

  int nskip_walker_output;

  bool apply_rotation;

  sharedCMatrix XRot;
  stdCVector Grot;

  bool print_from_list;

  mpi3IMatrix index_list;

  // DMAverage (nave, spin*x*NMO*x*NMO), x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  mpi3CMatrix DMAverage;

};

} // namespace afqmc
} // namespace sfqmc

#endif
