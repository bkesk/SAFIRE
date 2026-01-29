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
#include <fstream>

#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Estimators/TimeEvolvedObsHandler.hpp"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{

namespace detail
{

inline int get_number_of_averages(ptree pt)
{
  std::string measure_at = pt.get<std::string>("measure_at");
  std::vector<int> measure_at_blocks = io::str2vec<int>(measure_at);
  // sort the requested blocks and remove repeated
  std::sort(measure_at_blocks.begin(), measure_at_blocks.end());
  {
    auto last = std::unique(measure_at_blocks.begin(), measure_at_blocks.end());
    measure_at_blocks.erase(last, measure_at_blocks.end());
    for(auto v: measure_at_blocks)
      if(v <= 0)
        APP_ABORT(" Error: measure_at must be > 0.");
  }
  return measure_at_blocks.size();
}

}

/*
 * Implements back propagation by evolving operators forward in time.
 * Based on 10.1103/PhysRevA.100.023621.
 */
template<MEMORY_SPACE MEM>
class BPWithTimeEvolvedOperators : public EstimatorBase<MEM>
{

public:
  BPWithTimeEvolvedOperators(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                          AFQMCInfo& info,
                          std::string name,
                          ptree exec_pt,
                          ptree est_pt,
                          WALKER_TYPES wlk,
                          WalkerSet<MEM>& wset,
                          Wavefunction<MEM>& wfn,
                          Propagator<MEM>& prop,
                          bool impsamp_ = true)
      : EstimatorBase<MEM>(info),
        mpi(_mpi),
        walker_type(wlk),
        nspin( (walker_type==COLLINEAR) ? 2 : 1 ),
        npol( (walker_type==NONCOLLINEAR) ? 2 : 1 ),
//        observ0(TG, info, name, est_pt, wlk, detail::get_number_of_averages(est_pt), wfn),
        prop0(std::addressof(prop)),
        iblock(0),
        block_size(1),
        path_restoration(false),
        importanceSampling(impsamp_),
        extra_path_restoration(false),
        first(true),
        wgt_factors(iextensions<1u>{wset.size()}, ComplexType(1.0)),
        // hard-wired for the full MO space. Add choices later
        X({wset.size(), nspin, npol*NMO, npol*NMO}),
        Y({wset.size(), nspin, npol*NMO, npol*NMO}),
        M({wset.size(), nspin, npol*NMO, npol*NMO})
  {
    // KE: Current implementation is not consistent with the new driver conventions. Adding a block until we refactor here.
    utils::check(false,"BPWithTimeEvolvedOperators is not yet implemented. Use BackPropagatedEstimator instead.\n");

    app_log(1, "\n  --   Back Propagation with Time Evolved Operators -- \n"); 
    int driver_nstep, driver_nsubstep, driver_nstabilize;
    driver_nstep = exec_pt.get<int>("steps", 1);
    driver_nsubstep = exec_pt.get<int>("substeps", 1);
    driver_nstabilize = exec_pt.get<int>("bp_walker_ortho_interval", 1);
    if( driver_nstabilize>driver_nstep or driver_nstep%driver_nstabilize != 0)
      APP_ABORT(" Error: Back Propagation with time evolved operators requires steps >= ortho and steps%ortho==0. ");

    measure_interval = est_pt.get<int>("measure_interval", 1);

    // to trigger default value
    path_restoration  = est_pt.get<bool>("path_restoration", false);
    extra_path_restoration = est_pt.get<bool>("extra_path_restoration", false);
    block_size = est_pt.get<int>("block_size", 1);
    nblocks_equil = est_pt.get<int>("equil", 0);
    nblock_between_bp_starts = est_pt.get<int>("period", -1);
    if (extra_path_restoration)
    {
      utils::check(false,"  Error: extra_path_restoration not yet working.");
      path_restoration       = true;
      extra_path_restoration = true;
    }
    std::string measure_at = est_pt.get<std::string>("measure_at");
    measure_at_blocks = io::str2vec<int>(measure_at);
    // sort the requested blocks and remove repeated
    std::sort(measure_at_blocks.begin(), measure_at_blocks.end());
    {
      auto last = std::unique(measure_at_blocks.begin(), measure_at_blocks.end());
      measure_at_blocks.erase(last, measure_at_blocks.end());
      for(auto v: measure_at_blocks) 
        if(v <= 0)
          APP_ABORT(" Error: measure_at must be > 0.");
    }

// MAM: later on, enable measure_at to be zero, which will lead to the Mixed Distribution!

    if (measure_at_blocks.size() == 0)
      APP_ABORT("Error:  Empty measure_at.");

    nblocks_equil = std::max(0,nblocks_equil);
    if(nblock_between_bp_starts < 0)
      nblock_between_bp_starts = measure_at_blocks.back();

    // make sure nblocks_between is larger or equal than measure_at_blocks.back()
    if(nblock_between_bp_starts < measure_at_blocks.back())
    {
      app_warning(" BPWithTimeEvolvedOperators: period should be larger or equal than");
      app_warning("          the largest number of blocks in measure_at.");
      app_warning("          Re-setting period to: {}", measure_at_blocks.back());
      nblock_between_bp_starts = measure_at_blocks.back();
    } 

    if(extra_path_restoration)
      app_log(1," Using path restoration with modification to include extra time segment. "); 
    else if(path_restoration)
      app_log(1," Using path restoration. "); 
    else
      app_log(1," Path restoration is not used "); 
    app_log(1," Number of equilibration blocks: {}", nblocks_equil);
    app_log(1," Number of blocks between the start of BP: {}", nblock_between_bp_starts);
    app_log(1," Number of blocks in local averaging: {}", block_size);
    app_log(1," Number of time measurements: {}", measure_at_blocks.size());
    app_log(1," Measuring at blocks (relative to each BP start): ");
    for(auto v: measure_at_blocks) 
      app_log(1, "{} ", v);

    // number of time propagations per block in current driver
    steps_per_block = driver_nstep*driver_nsubstep; 

    int ncv(prop0->global_number_of_cholesky_vectors());
    wset.resize_bp(steps_per_block, ncv, 1);
    wset.setBPPos(0);
    if (nblocks_equil == 0) 
      reset(wset);
  }

  ~BPWithTimeEvolvedOperators() {}

  void accumulate_step([[maybe_unused]] double time,
                       [[maybe_unused]] WalkerSet& wset,
                       [[maybe_unused]] std::vector<ComplexType>& curData) {}

  void accumulate_block([[maybe_unused]] double time, WalkerSet& wset)
  {
    // always set to false
    accumulated_in_last_block = false;

    // return if within equilibration time
    if(iblock < nblocks_equil) {
      iblock++;
      return;  
    }   

    // check if this is the start of BP
    if( iblock == nblocks_equil ) {
      // start of BP
      // start bp in WalkerSet and store SM
      iblock++;
      reset(wset);
      return;
    }

    // counter within BP block  
    int bp_blk = (iblock - nblocks_equil)%nblock_between_bp_starts;
    if( bp_blk == 0 ) bp_blk = nblock_between_bp_starts;

    // check if we are in "dead" period between last BP measurement and BP restart
    if( bp_blk > measure_at_blocks.back() ) {
      iblock++;
      if( bp_blk == nblock_between_bp_starts ) 
        reset(wset);
      return;
    } 

    // We are within the BP measurement phase
    // 1. Propagate X, Y matrices forward and accumulate M 
//    prop0->PropagateOperators(steps_per_block, wset, X, Y, M);
/*
    // 2. accumulate weights if using path restoration
    utils::check(wgt_factors.extent(0) == wset.extent(), "Size mismatch");
    if (path_restoration)
    {
      auto factors = nda::to_host(wset.getWeightFactors());
      int hpos(wset.getHistoryPos()); // position where next step goes... go bach in history...
      int maxpos(wset.HistoryBufferLength());
      int nbp(steps_per_block);
      for (int k = 0; k < nbp; k++)
      {
        // start going back since position is advanced for next step already
        hpos = ((hpos == 0) ? maxpos - 1 : hpos - 1); 
        for (int i = 0; i < wgt_factors.size(); i++)
          wgt_factors[i] *= factors[hpos][i];
      }
    }
    else if (!importanceSampling)
    {
      stdCVector phase(iextensions<1u>{wset.size()});
      wset.getProperty(PHASE, phase);
      for (int i = 0; i < wgt_factors.size(); i++)
        wgt_factors[i] *= phase[i];
    }

    // 3. Calculate observables if needed
    for(int iav=0; iav<measure_at_blocks.size(); iav++) {
      if(bp_blk == measure_at_blocks[iav]) { 
        // 3.a add walker weights at current time
        stdCVector wgt(iextensions<1u>{wset.size()});
        wset.getProperty(WEIGHT, wgt);
        for (int i = 0; i < wgt.size(); i++)
          wgt[i] *= wgt_factors[i];
        // accumulate expects Yc = conj(Y)
        ma::complex_conjugate(Y.flatted());
        observ0.accumulate(iav, wset, wgt, X, Y, M, importanceSampling);
        // conjugate Y back!
        ma::complex_conjugate(Y.flatted());
      }
    }

    if( bp_blk == nblock_between_bp_starts ) {
      // last measurement on this block, full reset 
      reset(wset);
      accumulated_in_last_block = true;
    } else {
      // reset wset BP pos if we need to collect fields
      if(bp_blk < measure_at_blocks.back())
        wset.setBPPos(0);
    }

    // increase counter
    iblock++;
*/
  }

  void tags([[maybe_unused]] std::ofstream& out)
  {
  }

  int get_measurement_interval()
  {
    app_log(1, "Warning: BPWithTimeEvolvedOperators is hard_coded to use measurement_interval == 1.\n");
    return measure_interval;
  }

  void print([[maybe_unused]] std::ofstream& out, h5::file& file, [[maybe_unused]] WalkerSet& wset)
  {
    if(number_of_references==0) return;
    if (accumulated_in_last_block)
    {
      if (mpi->comm.root())
      {
        h5::group grp(file);
        h5::group g1 = ( grp.has_key("Observables") ? grp.open_group("Observables") :
                                                      grp.create_group("Observables") );
        h5::group g2 = ( g1.has_key("BackPropagated") ? g1.open_group("BackPropagated") :
                                                g1.create_group("BackPropagated") );
        if(first) {
          first = false;
          if (write_metadata)
          {
            h5::group g3 = g2.create_group("Metadata"); // can this already exist??? 
// sync with new framework
            //h5::h5_write(g3,"BackPropSteps",nback_prop_steps);
//            std::vector<double> times;
//            for(auto v:measure_at_blocks)
//              times.push_back(v*steps_per_block);
//            h5::h5_write(g3,"BackPropTimes", measure_at_blocks);
            int sz = measure_at_blocks.size();
            h5::h5_write(g3, "NumAverages", sz);
            int one(1);
            h5::h5_write(g3, "NumReferences", one);
            write_metadata = false;
          }
        }
//        observ0.print(iblock, &g2);
      } else {
        h5::group *g = nullptr;
//        observ0.print(iblock, g);
      }
    }
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE;

  int nspin;
  int npol;
  int accumulated_in_last_block = 0;

//  TimeEvolvedObsHandler<MEM> observ0;

  Propagator<MEM>* prop0;

  std::vector<int> measure_at_blocks;

  int steps_per_block = 1;
  int iblock        = 0;
  int nblocks_equil = 0;
  int nblock_between_bp_starts = 0;

  // Block size over which RDM will be averaged.
  int block_size = 1;
  // Whether to restore cosine projection and real local energy apprximation for weights
  // along back propagation path.
  bool path_restoration = true;
  bool importanceSampling = true;
  bool extra_path_restoration = true;

  int first = true;

  bool write_metadata = true;

  int measure_interval = 1;

  nda::vector<ComplexType> wgt_factors;

// if memory is a problem, you can keep these in host memory
// and use buffer space for calculations 
  // State matrices for the evolved operators. X->c^+, Y->c
  C4Tensor X;
  C4Tensor Y;
  // Accumulates the scalar terms coming from the stabilization procedure
  memory::array<MEM, M;  

/*
  template<class WlkSet>
  void reset(WlkSet& wset)
  {
    wset.setBPPos(0);
// this BP scheme does not need SMs at earlier times
//    for (auto it = wset.begin(); it < wset.end(); ++it)
//      it->setSlaterMatrixN();

    wgt_factors() = ComplexType(1.0);
    //if(extra_path_restoration) {
     //accumulate phase and cosine factors from previous history 
    //}   

    // initialize X, Y, M
    // hard-wired for the native basis set. Add choices later...
    M() = ComplexType(0.0);
    ma::set_identity(X.flatted());
    ma::set_identity(Y.flatted());
  }
*/

};
} // namespace afqmc
} // namespace sfqmc

