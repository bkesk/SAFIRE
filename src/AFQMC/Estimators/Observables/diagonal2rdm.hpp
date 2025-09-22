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

#ifndef SFQMC_AFQMC_DIAGONAL2RDM_HPP
#define SFQMC_AFQMC_DIAGONAL2RDM_HPP

#include "AFQMC/config.h"
#include <vector>
#include <string>
#include <iostream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Numerics/ma_operations.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged diagonal of the 2 RDM 
 */
class diagonal2rdm : public AFQMCInfo
{
  // allocators
  using Allocator = device_allocator<ComplexType>;

  // type defs
  using pointer       = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;

  using CVector_ref    = boost::multi::array_ref<ComplexType, 1, pointer>;
  using CMatrix_ref    = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CVector        = boost::multi::array<ComplexType, 1, Allocator>;
  using CMatrix        = boost::multi::array<ComplexType, 2, Allocator>;
  using stdCVector_ref = boost::multi::array_ref<ComplexType, 1>;
  using stdCMatrix_ref = boost::multi::array_ref<ComplexType, 2>;
  using mpi3CVector    = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3CMatrix    = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using mpi3CTensor    = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;
  using mpi3C4Tensor   = boost::multi::array<ComplexType, 4, shared_allocator<ComplexType>>;

public:
  diagonal2rdm(afqmc::TaskGroup_& tg_, AFQMCInfo& info, ptree pt, WALKER_TYPES wlk, int nave_ = 1, int bsize = 1)
      : AFQMCInfo(info),
        block_size(bsize),
        nave(nave_),
        TG(tg_),
        walker_type(wlk),
        writer(false),
        hdf_walker_output(""),
        DMAverage({0, 0}, shared_allocator<ComplexType>{TG.TG_local()})
  {
    app_log(1,"  --  Adding Diagonal 2RDM (Diag2RDM) estimator. -- ");
    hdf_walker_output = pt.get<std::string>("walker_output", "");

    if (hdf_walker_output != std::string(""))
    {
      hdf_walker_output = "G" + std::to_string(TG.TG_heads().rank()) + "_" + hdf_walker_output;
      hdf_archive dump;
      if (not dump.create(hdf_walker_output))
      {
        app_error("Problems creating walker output hdf5 file: {}", hdf_walker_output);
        APP_ABORT("Problems creating walker output hdf5 file.");
      }
      dump.push("DiagTwoRDM");
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
    writer  = (TG.Global().rank() == 0);
    dm_size = NMO * (2 * NMO - 1);
    if (walker_type == CLOSED)
      dm_size -= NMO * (NMO - 1) / 2;

    DMAverage = mpi3CMatrix({nave, dm_size}, shared_allocator<ComplexType>{TG.TG_local()});
    fill_n(DMAverage.origin(), DMAverage.num_elements(), ComplexType(0.0, 0.0));
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  template<class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, MatG&& G, MatG_host&& G_host, HostCVec1&& Xw, [[maybe_unused]] bool impsamp)
  {
    static_assert(std::decay<MatG>::type::dimensionality == 4, "Wrong dimensionality");
    static_assert(std::decay<MatG_host>::type::dimensionality == 4, "Wrong dimensionality");
    using std::copy_n;
    using std::fill_n;

    // assumes G[nwalk][spin][M][M]
    int nw(G.size(0));
    RUNTIME_CHECK(G.size(0) == Xw.size(0), "");

    int i0, iN;
    std::tie(i0, iN) = FairDivideBoundary(TG.TG_local().rank(), dm_size, TG.TG_local().size());

    // no parallelization over ncores for now, fix if needed 
    if(TG.TG_local().root())
    {
      for (int iw = 0; iw < nw; iw++)
      {
        ComplexType* ptr(DMAverage[iav].origin());
        if (walker_type == CLOSED)
        {
          auto&& Gu_ = G_host[iw][0];
          for (int i = 0; i < NMO; i++)
          {
            for (int j = i + 1; j < NMO; j++, ptr++)
              *ptr += Xw[iw] * (Gu_[i][i] * Gu_[j][j] - Gu_[i][j] * Gu_[j][i]);
            for (int j = NMO, j0 = 0; j < 2 * NMO; j++, j0++, ptr++)
              *ptr += Xw[iw] * (Gu_[i][i] * Gu_[j0][j0]);
          }
        }
        else if (walker_type == COLLINEAR)
        {
          auto&& Gu_ = G_host[iw][0];
          auto&& Gd_ = G_host[iw][1];
          for (int i = 0; i < 2 * NMO; i++)
          {
            if (i < NMO)
            {
              for (int j = i + 1; j < NMO; j++, ptr++)
                *ptr += Xw[iw] * (Gu_[i][i] * Gu_[j][j] - Gu_[i][j] * Gu_[j][i]);
              for (int j = NMO, j0 = 0; j < 2 * NMO; j++, j0++, ptr++)
                *ptr += Xw[iw] * (Gu_[i][i] * Gd_[j0][j0]);
            }
            else
            {
              int i_0 = i - NMO;
              for (int j = i + 1, j_0 = i + 1 - NMO; j < 2 * NMO; j++, j_0++, ptr++)
                *ptr += Xw[iw] * (Gd_[i_0][i_0] * Gd_[j_0][j_0] - Gd_[i_0][j_0] * Gd_[j_0][i_0]);
              ;
            }
          }
        }
        else
        {
          auto&& G_ = G_host[iw][0];
          for (int i = 0; i < 2 * NMO; i++)
            for (int j = i + 1; j < 2 * NMO; j++, ptr++)
              *ptr += Xw[iw] * (G_[i][i] * G_[j][j] - G_[i][j] * G_[j][i]);
        }
      }
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
        dump.push(std::string("DiagTwoRDM"));
        for (int i = 0; i < nave; ++i)
        {
          dump.push(std::string("Average_") + std::to_string(i));
          std::string padded_iblock =
              std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
          stdCVector_ref DMAverage_(raw_pointer_cast(DMAverage[i].origin()), {dm_size});
          dump.write(DMAverage_, "diag_two_rdm_" + padded_iblock);
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
  int block_size;

  int nave;

  TaskGroup_& TG;

  WALKER_TYPES walker_type;

  int dm_size;

  bool writer;

  std::string hdf_walker_output;

  // DMAverage (nave, spin*spin*x*NMO*(x*NMO-1)/2 ), x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  mpi3CMatrix DMAverage;
};

} // namespace afqmc
} // namespace sfqmc

#endif
