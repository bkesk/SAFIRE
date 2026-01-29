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

#include <vector>
#include <string>
#include <iostream>

#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Estimators/Observables/Observable.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * This class manages a list of observables evaluated at the mixed distribution.
 * Mixed distribution observables are those that have the trial wave function as the left
 * hand side. 
 * Given a walker set and an array of slater matrices (references), 
 * this routine will calculate and accumulate all requested observables.
 * This class also handles all the hdf5 I/O (given a hdf archive).
 * Since n-body observables (n>1) require an explicit sum over references, 
 * they are treated separately if the number of references is > 1.
 * A customized implementation for PHMSD will be ubilt in the future if needed. 
 */
template<MEMORY_SPACE MEM>
class MixedObsHandler : public AFQMCInfo
{
public:
  MixedObsHandler(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                 AFQMCInfo& info,
                 std::string name_,
                 ptree pt,
                 WALKER_TYPES wlk,
                 Wavefunction<MEM>& wfn_)
      : AFQMCInfo(info),
        mpi(_mpi),
        walker_type(wlk),
        wfn(std::addressof(wfn_)),
        name(name_),
        denominator(1)
  {
    for(const ptree::value_type &it : pt)
    {
      std::string cname = it.first;
      io::tolower(cname);
      if (cname == "onerdm")
      {
        properties_1body.emplace_back(Observable(full1rdm(mpi, info, it.second, walker_type, 1)));
      }
      else if (cname == "gfock" || cname == "genfock" || cname == "ekt")
      {
//        properties.emplace_back(Observable(
//            generalizedFockMatrix(mpi, info, it.second, walker_type, *wfn, 1, block_size)));
      }
      else if (cname == "diag2rdm")
      {
//        properties.emplace_back(Observable(diagonal2rdm(mpi, info, it.second, walker_type, 1, block_size)));
      }
      else if (cname == "twordm")
      {
//        properties.emplace_back(Observable(full2rdm(mpi, info, it.second, walker_type, 1, block_size)));
      }
      else if (cname == "n2r" || cname == "ontop2rdm")
      {
#if defined(ENABLE_DEVICE)
        ptree pt1 = it.second;
        bool use_host_memory = pt1.get<bool>("use_host_memory", false);
        if (use_host_memory)
        {
//          properties.emplace_back(Observable(
//              n2r<device_allocator<ComplexType>>(mpi, info, it.second, walker_type, false, device_allocator<ComplexType>{},
//                                                 device_allocator<ComplexType>{}, 1, block_size)));
        }
        else
#endif
        {
//          properties.emplace_back(Observable(
//              n2r<shared_allocator<ComplexType>>(mpi, info, it.second, walker_type, true,
//                                                 shared_allocator<ComplexType>{TG.TG_local()},
//                                                 shared_allocator<ComplexType>{TG.Node()}, 1, block_size)));
        }
      }
      else if (cname == "realspace_correlators")
      {
//        properties.emplace_back(Observable(realspace_correlators(mpi, info, it.second, walker_type, 1, block_size)));
      }
      else if (cname == "correlators")
      {
//        properties.emplace_back(Observable(atomcentered_correlators(mpi, info, it.second, walker_type, 1, block_size)));
      }
      else if (cname == "pair_correlators")
      {
//        properties.emplace_back(Observable(pair_correlator(mpi, info, it.second, walker_type, 1, block_size)));
      }
      else if (cname == "spinspin")
      {
//        properties.emplace_back(Observable(spinspinobs(mpi, info, it.second, walker_type, 1, block_size)));
      }
    }

    ncalls = 0;
    denominator() = ComplexType(0.0, 0.0);
    utils::check((properties.size()+properties_1body.size()) != 0, "empty observables list is not allowed.");
  }

  void print(int iblock, h5::group *g)
  {
    denominator(0) *= ComplexType(1.0 / double(ncalls));
    mpi->comm.all_reduce_in_place_n(&denominator(0),1,std::plus<>());

    for (auto& v : properties_1body)
      v.print(iblock, g, denominator);
    for (auto& v : properties)
      v.print(iblock, g, denominator);

    ncalls=0;
    denominator(0) = ComplexType(0.0, 0.0);
  }

  template<class WlkSet>
  void accumulate(WlkSet& wset)
  {
    int nwalk = wset.size();
    int nspin = ( walker_type == COLLINEAR ? 2 : 1 );
    int npol = ( walker_type == NONCOLLINEAR ? 2 : 1 );    
    int nrefs = wfn->total_number_of_references();

    nda::array<ComplexType, 1> wgt(nwalk);
    wset.getProperty(WEIGHT, wgt);
    ncalls++;
    denominator(0) += std::accumulate(wgt.begin(), wgt.end(), ComplexType(0.0));

// MAM: Implement this in the wavefuntion with a lambda function that is applied
// e.g. [] (auto && G) { 
//        for (auto& v : properties)
//          v.accumulate(0, G, G, wgt, true);
// }
// This way each wfn can optimize their evaluation of green functions and 
// there is no need reimplement algorithms here

    memory::buffered_array<MEM,ComplexType,1> Ov(nwalk);
    memory::buffered_array<MEM,ComplexType,4> G(nwalk,nspin,npol*NMO,npol*NMO); 
    auto G2D = nda::reshape(G,std::array<long,2>{nwalk,nspin*npol*NMO*npol*NMO});

    if( nrefs==1 || properties.size()==0 )  {

      // 1. Calculate mixed density matrix
      wfn->MixedDensityMatrix(wset,G2D,Ov,false);

      //2. accumulate 
      if constexpr (MEM == HOST_MEMORY) {
        for (auto& v : properties_1body)
          v.accumulate(0, G, G, wgt, true);
        for (auto& v : properties)
          v.accumulate(0, G, G, wgt, true);
      } else {
        auto Gh = nda::to_host(G);
        for (auto& v : properties_1body)
          v.accumulate(0, G, Gh, wgt, true);
        for (auto& v : properties)
          v.accumulate(0, G, Gh, wgt, true);
      }

    }
    else 
    {
      utils::check(false, "finish");
/*
      // use the fact that Observables accumulate between calls to print to sum over configurations
      // MAM: UNTESTED!!!
      auto Orbs = wfn->getReferences();  // (ndet,nspin)(nel,npol*NMO)
      int nrefs = Orbs.extent(0); 

      memory::buffered_array<M,ComplexType,2> Gtmp(nwalk,nspin*npol*NMO*npol*NMO); 
      nda::array<ComplexType, 1> Xw(nwalk, ComplexType(1.0, 0.0));
      nda::array<ComplexType, 1> Oh(nwalk, ComplexType(1.0, 0.0));
      nda::array<ComplexType, 1> detR(nwalk, ComplexType(1.0, 0.0));
      nda::array<ComplexType, 1> log_m(nwalk);
      wset.getProperty(OVLP, log_m);  // use as reference

      if (impsamp)
        denominator[0] += std::accumulate(wgt.begin(), wgt.end(), ComplexType(0.0));
      else
      {
        utils::check(false, " Finish implementation of free projection. \n\n");
      }

      for (int iref = 0; iref < nrefs; iref++)
      {
        // conjugated here!
        ComplexType CIcoeff(std::conj(wfn->getReferenceWeight(iref)));

        wfn->DensityMatrix(wset, OrbMats(iref,all), Gtmp, Ov, false);

        Oh() = Ov(); 
        if (nrefs > 1)
        {
          if (walker_type == CLOSED)
          {
            for (int iw = 0; iw < nw; iw++)
              Xw[iw] = CIcoeff * Ov[iw] * Ov[iw] * std::conj(detR[iw][iref] * detR[iw][iref]);
          }
          else if (walker_type == COLLINEAR)
          {
            for (int iw = 0; iw < nw; iw++)
              Xw[iw] = CIcoeff * Ov[iw] * Ov[iw + nw] * std::conj(detR[iw][2 * iref] * detR[iw][2 * iref + 1]);
          }
          else if (walker_type == NONCOLLINEAR)
          {
            for (int iw = 0; iw < nw; iw++)
            Xw[iw] = CIcoeff * Ov[iw] * std::conj(detR[iw][iref]);
          }
        }

        //3. accumulate references
        if constexpr (MEM == HOST_MEMORY) {
          for (auto& v : properties)
            v.accumulate_reference(0, iref, Gtmp, Gtmp, wgt, Xw, Ov, impsamp);
        } else {
          auto Gh = nda::to_host(G);
          for (auto& v : properties)
            v.accumulate_reference(0, iref, G, Gh, wgt, Xw, Ov, impsamp);
        }
      }
      //4. accumulate block (normalize and accumulate sum over references)
      if constexpr (MEM == HOST_MEMORY) {
        for (auto& v : properties)
          v.accumulate_reference(0, iref, G, G, wgt, Xw, Ov, impsamp);
      } else {
        auto Gh = nda::to_host(G);
        for (auto& v : properties)
          v.accumulate_reference(0, iref, G, Gh, wgt, Xw, Ov, impsamp);
    for (auto& v : properties)
        v.accumulate_block(0, wgt, impsamp);
      }
    }
*/
    }
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  Wavefunction<MEM>* wfn;

  int ncalls = 0;

  std::string name;

  std::vector<Observable> properties_1body;
  std::vector<Observable> properties;

  nda::array<ComplexType,1> denominator;

};

} // namespace afqmc

} // namespace sfqmc

