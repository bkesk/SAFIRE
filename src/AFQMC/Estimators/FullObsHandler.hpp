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

#ifndef SFQMC_AFQMC_FULLOBSHANDLER_HPP
#define SFQMC_AFQMC_FULLOBSHANDLER_HPP

#include <vector>
#include <string>
#include <iostream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/Estimators/Observables/Observable.hpp"
#include "AFQMC/config.h"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "Numerics/ma_operations.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Memory/buffer_managers.h"

namespace sfqmc
{
namespace afqmc
{
/*
 * This class manages a list of "full" observables.
 * Full observables are those that have a walker dependent left-hand side, 
 * which result from back propagation. 
 * This implementation of the class assumes a multi-determinant trial wavefunction,  
 * resulting in the loop over "references" (each determinant in the trial wavefunciton
 * being back-propagated). 
 * Given a walker set and an array of (back propagated) slater matrices, 
 * this routine will calculate and accumulate all requested observables.
 * To make the implementation of the BackPropagated class cleaner, 
 * this class also handles all the hdf5 I/O (given a hdf archive).
 */
class FullObsHandler : public AFQMCInfo
{
  // allocators
  using sharedAllocator = localTG_allocator<ComplexType>;

  using shared_pointer       = typename std::allocator_traits<sharedAllocator>::pointer;
  using const_shared_pointer = typename std::allocator_traits<sharedAllocator>::const_pointer;

  using devCMatrix_ptr = boost::multi::array_ptr<ComplexType, 2, device_ptr<ComplexType>>;

  using sharedCVector      = boost::multi::array<ComplexType, 1, sharedAllocator>;
  using sharedCVector_ref  = boost::multi::array_ref<ComplexType, 1, shared_pointer>;
  using sharedCMatrix_ref  = boost::multi::array_ref<ComplexType, 2, shared_pointer>;
  using sharedC4Tensor_ref = boost::multi::array_ref<ComplexType, 4, shared_pointer>;

  using mpi3C4Tensor = boost::multi::array<ComplexType, 4, shared_allocator<ComplexType>>;

  using stdCVector     = boost::multi::array<ComplexType, 1>;
  using stdCMatrix     = boost::multi::array<ComplexType, 2>;
  using stdCVector_ref = boost::multi::array_ref<ComplexType, 1>;

  using shm_stack_alloc_type = LocalTGBufferManager::template allocator_t<ComplexType>;
  using StaticSHMVector      = boost::multi::static_array<ComplexType, 1, shm_stack_alloc_type>;
  using StaticSHM3Tensor     = boost::multi::static_array<ComplexType, 3, shm_stack_alloc_type>;
  using StaticSHM4Tensor     = boost::multi::static_array<ComplexType, 4, shm_stack_alloc_type>;

public:
  FullObsHandler(afqmc::TaskGroup_& tg_,
                 AFQMCInfo& info,
                 std::string name_,
                 ptree pt,
                 WALKER_TYPES wlk,
                 Wavefunction& wfn)
      : AFQMCInfo(info),
        TG(tg_),
        walker_type(wlk),
        wfn0(wfn),
        writer(false),
        block_size(1),
        nave(1),
        name(name_),
        nspins((walker_type == COLLINEAR) ? 2 : 1),
        G4D_host({0, 0, 0, 0}, shared_allocator<ComplexType>{TG.TG_local()})
  {
    using std::fill_n;
    std::vector<int> nback_prop_interval_multipliers = io::get_value_or_vector<int>(pt, "measure_interval_multiplier", {DEFAULT_MEASURE_INTERVAL_MULTIPLIER});
    nave = nback_prop_interval_multipliers.size();

    block_size = pt.get<int>("block_size", 1);
    if (nave <= 0)
      APP_ABORT("naverages <= 0 is not allowed.");

    for(const ptree::value_type &it : pt)
    {
      std::string cname = it.first;
      io::tolower(cname);
      if (cname == "onerdm")
      {
        properties.emplace_back(Observable(full1rdm(TG, info, it.second, walker_type, nave, block_size)));
      }
      else if (cname == "gfock" || cname == "genfock" || cname == "ekt")
      {
        properties.emplace_back(Observable(
            generalizedFockMatrix(TG, info, it.second, walker_type, wfn0, nave, block_size)));
      }
      else if (cname == "diag2rdm")
      {
        properties.emplace_back(Observable(diagonal2rdm(TG, info, it.second, walker_type, nave, block_size)));
      }
      else if (cname == "twordm")
      {
        properties.emplace_back(Observable(full2rdm(TG, info, it.second, walker_type, nave, block_size)));
      }
      else if (cname == "n2r" || cname == "ontop2rdm")
      {
#if defined(ENABLE_DEVICE)
        ptree pt1 = it.second;
        bool use_host_memory = pt1.get<bool>("use_host_memory", false);
        if (use_host_memory)
        {
          properties.emplace_back(Observable(
              n2r<device_allocator<ComplexType>>(TG, info, it.second, walker_type, false, device_allocator<ComplexType>{},
                                                 device_allocator<ComplexType>{}, nave, block_size)));
        }
        else
#endif
        {
          properties.emplace_back(Observable(
              n2r<shared_allocator<ComplexType>>(TG, info, it.second, walker_type, true,
                                                 shared_allocator<ComplexType>{TG.TG_local()},
                                                 shared_allocator<ComplexType>{TG.Node()}, nave, block_size)));
        }
      }
      else if (cname == "realspace_correlators")
      {
        properties.emplace_back(Observable(realspace_correlators(TG, info, it.second, walker_type, nave, block_size)));
      }
      else if (cname == "correlators")
      {
        properties.emplace_back(Observable(atomcentered_correlators(TG, info, it.second, walker_type, nave, block_size)));
      }
      else if (cname == "pair_correlators")
      {
        properties.emplace_back(Observable(pair_correlator(TG, info, it.second, walker_type, nave, block_size)));
      }
      else if (cname == "spinspin")
      {
        properties.emplace_back(Observable(spinspinobs(TG, info, it.second, walker_type, nave, block_size)));
      }
    }

    if (properties.size() == 0)
      APP_ABORT("empty observables list is not allowed.");

    Gdims = std::make_tuple(NMO, NMO);
    if (walker_type == NONCOLLINEAR)
      Gdims = std::make_tuple(2 * NMO, 2 * NMO);
    dm_size = nspins * std::get<0>(Gdims) * std::get<1>(Gdims);

    writer = (TG.Global().rank() == 0);

    denominator = stdCVector(iextensions<1u>{nave});
    fill_n(denominator.begin(), denominator.num_elements(), ComplexType(0.0, 0.0));
  }

  void print(int iblock, hdf_archive& dump)
  {
    using std::fill_n;

    if (TG.TG_local().root())
    {
      ma::scal(ComplexType(1.0 / block_size), denominator);
      TG.TG_heads().reduce_in_place_n(raw_pointer_cast(denominator.origin()), denominator.num_elements(), std::plus<>(), 0);
    }

    for (auto& v : properties)
      v.print(iblock, dump, denominator);
    fill_n(denominator.origin(), denominator.num_elements(), ComplexType(0.0, 0.0));
  }

  template<class WlkSet, class MatR, class HostCVec, class MatD>
  void accumulate(int iav, WlkSet& wset, MatR&& Refs, HostCVec&& wgt, MatD&& DevlogdetR, bool impsamp)
  {
    if (iav < 0 || iav >= nave)
      APP_ABORT("Runtime Error: iav out of range in full1rdm::accumulate. \n\n");

    int nw(wset.size());
    int nrefs(Refs.size(1));
    double LogOverlapFactor(wset.getLogOverlapFactor());

    auto SMA = wset.SlaterMatricesN(Alpha);
    auto SMB = wset.SlaterMatricesN( (walker_type == COLLINEAR) ? Beta : Alpha );
    auto RefsA = wset.SlaterMatricesAux(Alpha);
    auto RefsB = wset.SlaterMatricesAux( (walker_type == COLLINEAR) ? Beta : Alpha );
    auto SDetOp = wfn0.getSlaterDetOperations();

    LocalTGBufferManager shm_buffer_manager;
    StaticSHM4Tensor G4D({nw, nspins, std::get<0>(Gdims), std::get<1>(Gdims)},
                         shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    StaticSHM3Tensor GA({nw, SMA.size(2), SMA.size(1)},
                         shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    StaticSHM3Tensor GB({nw, (nspins-1)*SMB.size(2), (nspins-1)*SMB.size(1)},
                         shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    StaticSHMVector DevOv(iextensions<1u>{2 * nw},
                         shm_buffer_manager.get_generator().template get_allocator<ComplexType>());

    if (G4D_host.num_elements() != G4D.num_elements())
    {
      G4D_host = mpi3C4Tensor(G4D.extensions(), shared_allocator<ComplexType>{TG.TG_local()});
      TG.TG_local().barrier();
    }
    RUNTIME_CHECK(DevlogdetR.size(0) == nw, "");

    stdCVector Xw(iextensions<1u>{nw});
    stdCVector Ov(iextensions<1u>{2 * nw});
    stdCMatrix logdetR(DevlogdetR);
    stdCVector scl_wgt(wgt);

    if (impsamp)
      denominator[iav] += std::accumulate(wgt.begin(), wgt.end(), ComplexType(0.0));
    else
    {
      APP_ABORT(" Finish implementation of free projection. \n\n");
    }

    // calculate all overlaps and accumulate denominator 
    if( nrefs > 1 ) {

      // calculate logdetR_shift and apply shift to logdetR 
      // logdetR_shift[w] = (1/Nd) * sum_d logdetR[w][d]
      stdCVector logdetR_shift(iextensions<1u>{nw}, ComplexType(0.0));
      ma::accumulate(1, ComplexType(1.0/double(logdetR.size(1))), logdetR, logdetR_shift);
      // apply shift: logdetR[w][d] = logdetR[w][d] - logdetR_shift[w]
      ma::elementwise(ma::TOp_MINUS, 0, logdetR_shift, logdetR);    
        
      std::fill_n(Xw.origin(), Xw.num_elements(), ComplexType(0.0, 0.0));
      for (int iref = 0, is = 0; iref < nrefs; iref++, is += nspins)
      {
        // conjugated here!
        ComplexType CIcoeff(std::conj(wfn0.getReferenceWeight(iref)));

        //1. Calculate Green functions
        for (int iw = 0; iw < nw; iw++)
          copy_n(Refs[iw][iref].origin(), RefsA[iw].num_elements(), RefsA[iw].origin());
        SDetOp->BatchedOverlap(RefsA, SMA, LogOverlapFactor, DevOv.sliced(0, nw));

        if (walker_type == COLLINEAR)
        {
          // batched copy_n ?
          for (int iw = 0; iw < nw; iw++)
            copy_n(Refs[iw][iref].origin() + RefsA[iw].num_elements(), RefsB[iw].num_elements(),
                   RefsB[iw].origin());
          SDetOp->BatchedOverlap(RefsB, SMB, LogOverlapFactor, DevOv.sliced(nw, 2 * nw));
        }

        //2.accumulate CI[n] * Ov[n] * R[n]
        copy_n(DevOv.origin(), 2 * nw, Ov.origin());
        if (walker_type == CLOSED)
        {
          for (int iw = 0; iw < nw; iw++)
            Xw[iw] += CIcoeff * Ov[iw] * Ov[iw] * 
                        std::conj(std::exp(logdetR[iw][iref] + logdetR[iw][iref]) );
        }
        else if (walker_type == COLLINEAR)
        {
          for (int iw = 0; iw < nw; iw++)
            Xw[iw] += CIcoeff * Ov[iw] * Ov[iw + nw] * 
                        std::conj( std::exp(logdetR[iw][2 * iref] + logdetR[iw][2 * iref + 1]) );
        }
        else if (walker_type == NONCOLLINEAR)
        {
          for (int iw = 0; iw < nw; iw++)
            Xw[iw] += CIcoeff * Ov[iw] * std::conj( std::exp(logdetR[iw][iref]) );
        }
      }
      
      // scale walker weights
      for(int i=0; i<nw; i++)
        scl_wgt[i] /= Xw[i];

    }

    // calculate GF and accumulate
    std::fill_n(Xw.origin(), Xw.num_elements(), ComplexType(1.0, 0.0));
    for (int iref = 0, is = 0; iref < nrefs; iref++, is += nspins)
    {
      // conjugated here!
      ComplexType CIcoeff(std::conj(wfn0.getReferenceWeight(iref)));

      //1. Calculate Green functions
      for (int iw = 0; iw < nw; iw++)
        copy_n(Refs[iw][iref].origin(), RefsA[iw].num_elements(), RefsA[iw].origin());
      // compact GF  
      SDetOp->BatchedDensityMatrix(RefsA, SMA, GA, LogOverlapFactor, DevOv.sliced(0, nw), true);
      // Full GF
      ma::complex_conjugate(RefsA);
      ma::productStridedBatched(RefsA, GA, G4D.rotated()[0].unrotated());

      if (walker_type == COLLINEAR)
      {
        // batched copy_n ?
        for (int iw = 0; iw < nw; iw++)
          copy_n(Refs[iw][iref].origin() + RefsA[iw].num_elements(), RefsB[iw].num_elements(),
                 RefsB[iw].origin());
        // compact GF  
        SDetOp->BatchedDensityMatrix(RefsB, SMB, GB, LogOverlapFactor, DevOv.sliced(nw, 2 * nw), true);
        // Full GF
        ma::complex_conjugate(RefsB);
        ma::productStridedBatched(RefsB, GB, G4D.rotated()[1].unrotated());
      }

      //2. calculate and accumulate appropriate weights
      copy_n(scl_wgt.origin(), nw, Xw.origin());
      if (nrefs > 1)
      {
        copy_n(DevOv.origin(), 2 * nw, Ov.origin());
        if (walker_type == CLOSED)
        {
          for (int iw = 0; iw < nw; iw++)
            Xw[iw] *= CIcoeff * Ov[iw] * Ov[iw] * 
                    std::conj( std::exp(logdetR[iw][iref] + logdetR[iw][iref]) );
        }
        else if (walker_type == COLLINEAR)
        {
          for (int iw = 0; iw < nw; iw++)
            Xw[iw] *= CIcoeff * Ov[iw] * Ov[iw + nw] * 
                    std::conj( std::exp(logdetR[iw][2 * iref] + logdetR[iw][2 * iref + 1]) );
        }
        else if (walker_type == NONCOLLINEAR)
        {
          for (int iw = 0; iw < nw; iw++)
            Xw[iw] *= CIcoeff * Ov[iw] * std::conj( std::exp(logdetR[iw][iref]) );
        }
      }

      // MAM: Since most of the simpler estimators need G4D in host memory,
      //      I'm providing a copy of the structure there already
      TG.TG_local().barrier();
      int i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(TG.TG_local().rank(), 
                                            int(G4D_host.num_elements()), TG.TG_local().size());
      copy_n(make_device_ptr(G4D.origin()) + i0, iN - i0, raw_pointer_cast(G4D_host.origin()) + i0);
      TG.TG_local().barrier();

      //3. accumulate references
      for (auto& v : properties)
        v.accumulate(iav, RefsA, GA, RefsB, GB, G4D, G4D_host, Xw, impsamp);
    }
  }

private:
  TaskGroup_& TG;

  WALKER_TYPES walker_type;

  Wavefunction& wfn0;

  bool writer;

  int block_size;

  int nave;

  std::string name;

  int nspins;
  int dm_size;
  std::tuple<int, int> Gdims;

  std::vector<Observable> properties;

  // denominator (nave, ...)
  stdCVector denominator;

  // space for G in host space
  mpi3C4Tensor G4D_host;
};

} // namespace afqmc

} // namespace sfqmc

#endif
