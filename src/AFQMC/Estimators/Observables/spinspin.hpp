/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

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
 * Observable class that calculates the walker averaged spin*spin correlation
 */
class spinspinobs : public AFQMCInfo
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
  spinspinobs(afqmc::TaskGroup_& tg_, AFQMCInfo& info, ptree pt, WALKER_TYPES wlk, int nave_ = 1, int bsize = 1)
      : AFQMCInfo(info),
        block_size(bsize),
        nave(nave_),
        TG(tg_),
        walker_type(wlk),
        writer(false),
        hdf_walker_output(""),
        DMAverage({0, 0}, shared_allocator<ComplexType>{TG.TG_local()})
  {
    app_log(1,"  --  Adding Spin*Spin (spinspinobs) estimator. -- ");
    hdf_walker_output = pt.get<std::string>("walker_output", "");

    //ref = pt.get<int>("ref",0);

    if (hdf_walker_output != std::string(""))
    {
      hdf_walker_output = "G" + std::to_string(TG.TG_heads().rank()) + "_" + hdf_walker_output;
      hdf_archive dump;
      if (not dump.create(hdf_walker_output))
      {
        app_error("Problems creating walker output hdf5 file: {}", hdf_walker_output);
        APP_ABORT("Problems creating walker output hdf5 file.");
      }
      dump.push("SpinSpin");
      dump.push("Metadata");
      dump.write(NMO, "NMO");
      dump.write(NAEA, "NUP");
      dump.write(NAEB, "NDOWN");
      //dump.write(ref, "REF");
      int wlk_t_copy = walker_type; // the actual data type of enum is implementation-defined. convert to int for file
      dump.write(wlk_t_copy, "WalkerType");
      dump.pop();
      dump.pop();
      dump.close();
    }

    using std::fill_n;
    writer  = (TG.Global().rank() == 0);
    // currently 2* upper triangular since <SiSj> = <SjSi>
    dm_size = NMO*(NMO+1); 
    //if (walker_type == CLOSED)
    //  dm_size -= NMO * (NMO - 1) / 2;

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
        ComplexType* ptrXY(DMAverage[iav].origin());
        ComplexType* ptrZ(DMAverage[iav].origin()+ (int) NMO*(NMO+1)/2);
        if (walker_type == CLOSED)
        {
          // TODO: optimize this
          auto&& Gu_ = G_host[iw][0];
          for (int ref = 0; ref < NMO; ref++ ){
            for (int j = ref; j < NMO; j++, ptrZ++, ptrXY++)
            {
              if(j==ref){
                *ptrZ += Xw[iw]/4.0 * (Gu_[ref][ref]*(1.0-Gu_[j][j])
                                      +Gu_[ref][ref]*(1.0-Gu_[j][j]));
                *ptrXY += Xw[iw]/2.0 * (Gu_[ref][j]*(1.0-Gu_[j][ref])
                                      +Gu_[ref][j]*(1.0-Gu_[j][ref]));
              }
              else{
                *ptrZ += Xw[iw]/4.0 * (Gu_[ref][ref]*(Gu_[j][j]-Gu_[j][j])
                                  +Gu_[ref][ref]*(Gu_[j][j]-Gu_[j][j])
                                  -Gu_[ref][j]*Gu_[j][ref] // only U2 term
                                  -Gu_[ref][j]*Gu_[j][ref]);

                *ptrXY += Xw[iw]/2.0 * (Gu_[ref][j]*(-Gu_[j][ref])
                                       +Gu_[ref][j]*(-Gu_[j][ref]));
              }
            }
          }
        }
        else if (walker_type == COLLINEAR)
        {
          auto&& Gu_ = G_host[iw][0];
          auto&& Gd_ = G_host[iw][1];

          for (int ref = 0; ref < NMO; ref++ ){
            for (int j = ref; j < NMO; j++, ptrZ++,ptrXY++)
            {
              if(j==ref){
                *ptrZ += Xw[iw]/4.0 * (Gu_[ref][ref]*(1.0-Gd_[j][j])
                                      +Gd_[ref][ref]*(1.0-Gu_[j][j]));
                *ptrXY += Xw[iw]/2.0 * (Gu_[ref][j]*(1.0-Gd_[j][ref])
                                      +Gd_[ref][j]*(1.0-Gu_[j][ref]));
              }
              else{
                *ptrZ += Xw[iw]/4.0 * (Gu_[ref][ref]*(Gu_[j][j]-Gd_[j][j])
                                  +Gd_[ref][ref]*(Gd_[j][j]-Gu_[j][j])
                                  -Gu_[ref][j]*Gu_[j][ref] // only U2 term
                                  -Gd_[ref][j]*Gd_[j][ref]);

                *ptrXY += Xw[iw]/2.0 * (Gu_[ref][j]*(-Gd_[j][ref])
                                       +Gd_[ref][j]*(-Gu_[j][ref]));
              }
            }
          }
        }
        else
        {
          auto&& G_ = G_host[iw][0];
          for (int ref = 0; ref < NMO; ref++){
            for (int j = ref; j < NMO; j++, ptrZ++,ptrXY++)
            {
              if(j==ref){
                *ptrZ += Xw[iw]/4.0 * (G_[ref][ref]*(1.0-G_[j+NMO][j+NMO])
                                      +G_[ref+NMO][ref+NMO]*(1.0-G_[j][j])
                                      -G_[ref][j+NMO]*(-G_[j+NMO][ref])
                                      -G_[ref+NMO][j]*(-G_[j][ref+NMO]));

                *ptrXY += Xw[iw]/2.0 * (G_[ref][j]*(1.0-G_[j+NMO][ref+NMO])
                                      +G_[ref+NMO][j+NMO]*(1.0-G_[j][ref])
                                      +G_[j+NMO][j]*G_[ref][ref+NMO]
                                      +G_[j][j+NMO]*G_[ref+NMO][ref]
                                      );
              }
              else{
                *ptrZ += Xw[iw]/4.0 * (G_[ref][ref]*(G_[j][j]-G_[j+NMO][j+NMO])
                                  +G_[ref+NMO][ref+NMO]*(G_[j+NMO][j+NMO]-G_[j][j])
                                  -G_[ref][j]*G_[j][ref] // only U2 term
                                  -G_[ref+NMO][j+NMO]*G_[j+NMO][ref+NMO]
                                  -G_[ref][j+NMO]*(-G_[j+NMO][ref])
                                  -G_[ref+NMO][j]*(-G_[j][ref+NMO]));

                *ptrXY += Xw[iw]/2.0 * (G_[ref][j]*(-G_[j+NMO][ref+NMO])
                                      +G_[ref+NMO][j+NMO]*-G_[j][ref]
                                      +G_[j+NMO][j]*G_[ref][ref+NMO]
                                      +G_[j][j+NMO]*G_[ref+NMO][ref]
                                      );
              }
            }
          }
        }
      }
    }
    TG.TG_local().barrier();
  }

/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/
  template<class... Args>
  void accumulate_reference_configuration([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_reference_configuration ");
  }

  template<class... Args>
  void accumulate_excited_configuration_first([[maybe_unused]]Args&&... args)
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
        dump.push(std::string("SpinSpin"));
        for (int i = 0; i < nave; ++i)
        {
          dump.push(std::string("Average_") + std::to_string(i));
          std::string padded_iblock =
              std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
          stdCVector_ref DMAverage_(raw_pointer_cast(DMAverage[i].origin()), {dm_size});
          dump.write(DMAverage_, "spinspin_" + padded_iblock);
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
  // reference site to compute <S_ref \cdot S_j>
  //int ref;

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

