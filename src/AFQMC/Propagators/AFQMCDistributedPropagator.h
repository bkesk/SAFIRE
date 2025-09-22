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

#ifndef SFQMC_AFQMC_DISTRIBUTEDPROPAGATOR_H
#define SFQMC_AFQMC_DISTRIBUTEDPROPAGATOR_H

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <tuple>

#include "hdf/hdf_archive.h"
#include "Utilities/Random.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "mpi3/shm/mutex.hpp"
#include "Memory/buffer_managers.h"

#include "AFQMC/Wavefunctions/Wavefunction.hpp"

#include "AFQMC/Propagators/AFQMCBasePropagator.h"

namespace sfqmc
{
namespace afqmc
{
/*
 * AFQMC propagator using distributed Cholesky matrices over nodes in the TG. 
 * Shared memory parallelization is used for on-node concurrency.
 *  - General case. Both vbias and vHS are assumed to need reduction over all nodes
 *    in TG.
 */
template<bool SP>
class AFQMCDistributedPropagator : public AFQMCBasePropagator<SP>
{
  using Base = AFQMCBasePropagator<SP>;

public:
  using Base::hybrid_propagation;
  using Base::free_propagation;
  using Base::global_number_of_cholesky_vectors;
  using Base::Orthogonalize;
  using Base::generateP1;

protected:
  using Base::assemble_X;
  using Base::reset_nextra;
  using Base::parse;
  using Base::apply_propagators;
  using Base::apply_propagators_batched;
  using Base::apply_bound_vbias;
  using Base::Orthogonalize_impl;

  using Base::NMO;
  using Base::NAEA;
  using Base::NAEB;
  using Base::transposed_vHS_;
  using Base::transposed_G_;
  using Base::new_overlaps;
  using Base::new_energies;
  using Base::old_dt;
  using Base::importance_sampling;
  using Base::apply_constrain;
  using Base::upper_cutoff_scale;
  using Base::lower_cutoff_scale;
  using Base::work;
  using Base::nbatched_propagation;
  using Base::TG;
  using Base::buffer_manager;
  using Base::device_buffer_manager;
  using Base::wfn;
  using Base::H1ext;
  using Base::denseP1;
  using Base::P1s;
  using Base::P1d;
  using Base::P1s_inv;
  using Base::P1d_inv;
  using Base::vMF;
  using Base::rng;
  using Base::SDetOp;
  using Base::vbias_bound;
  using Base::substractMF;
  using Base::free_projection;
  using Base::debug_verbosity;
  using Base::hybrid;
  using Base::rng_block_size;
  using Base::nspins_in_vHS;

  using SPRealType = typename Base::SPRealType;
  using SPComplexType = typename Base::SPComplexType;
  using CVector = typename Base::CVector;
  using CMatrix = typename Base::CMatrix;
  using C3Tensor = typename Base::C3Tensor;
  using SPCMatrix_ref = typename Base::SPCMatrix_ref;
  using StaticMatrix = typename Base::StaticMatrix;
  using StaticSPVector = typename Base::StaticSPVector;
  using StaticSPMatrix = typename Base::StaticSPMatrix;
  using Static3Tensor = typename Base::Static3Tensor;
  using stdSPCMatrix_ref = typename Base::stdSPCMatrix_ref;
  using C3Tensor_ref = typename Base::C3Tensor_ref;
  using C4Tensor_ref = typename Base::C4Tensor_ref;

public:
  AFQMCDistributedPropagator(AFQMCInfo& info,
                             ptree pt_in,
                             afqmc::TaskGroup_& tg_,
                             Wavefunction& wfn_,
                             utils::DeviceRandomGenerator_t* r)
      : Base(info, pt_in, tg_, wfn_, r),
        core_comm(tg_.TG().split(tg_.getLocalTGRank(), tg_.TG().rank()))
  //            ,core_comm()
  {
    //      core_comm = std::move(tg_.TG().split(tg_.getLocalTGRank()));
    RUNTIME_CHECK(TG.getNGroupsPerTG() > 1, "");
  }

  ~AFQMCDistributedPropagator() {}

  AFQMCDistributedPropagator(AFQMCDistributedPropagator const& other) = delete;
  AFQMCDistributedPropagator& operator=(AFQMCDistributedPropagator const& other) = delete;
  //AFQMCDistributedPropagator(AFQMCDistributedPropagator&& other) = default;
  AFQMCDistributedPropagator(AFQMCDistributedPropagator&& other) : Base(std::move(other)), core_comm()
  {
    // move constructor for communicator seems broken
    core_comm = TG.TG().split(TG.getLocalTGRank(), TG.TG().rank());
  }
  AFQMCDistributedPropagator& operator=(AFQMCDistributedPropagator&& other) = delete;

  template<class WlkSet>
  void Propagate(int steps, WlkSet& wset, RealType E1, RealType dt, int fix_bias = 1)
  {
    int nblk   = steps / fix_bias;
    int nextra = steps % fix_bias;
    for (int i = 0; i < nblk; i++)
    {
      step(fix_bias, wset, E1, dt);
      update_memory_managers();
    }
    if (nextra > 0)
      step(nextra, wset, E1, dt);
    TG.TG_local().barrier();
  }

  template<class WlkSet, class CTens, class CMat>
  void BackPropagate([[maybe_unused]] int steps, [[maybe_unused]] int nStabalize, 
                     [[maybe_unused]]  WlkSet& wset, [[maybe_unused]]  CTens&& Refs, 
                     [[maybe_unused]]  CMat&& logdetR)
  {
    APP_ABORT(" Error: Finish BackPropagate.");
  }

  template<class WlkSet, class Mat1, class Mat2, class Mat3>
  void PropagateOperators(int steps, WlkSet& wset,  Mat1&& X, Mat2&& Y, Mat3&& M);

protected:
  // new communicator over similar cores in a TG
  // every core communicates a segment to increase effective bandwidth
  boost::mpi3::communicator core_comm;

  // additional dimension for temporary computation
  C3Tensor MFfactor;
  C3Tensor hybrid_weight;

  template<class WlkSet>
  void step(int steps, WlkSet& wset, RealType E1, RealType dt);
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Propagators/AFQMCDistributedPropagator.icc"

#endif
