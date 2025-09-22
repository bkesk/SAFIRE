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

#ifndef SFQMC_AFQMC_DISTRIBUTEDPROPAGATORDISTCV_H
#define SFQMC_AFQMC_DISTRIBUTEDPROPAGATORDISTCV_H

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
 *   - Specialized algorithm for case when vbias doesn't need to be reduced over 
 *     nodes in a TG.
 */
template<bool SP>
class AFQMCDistributedPropagatorDistCV : public AFQMCBasePropagator<SP>
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
  using Base::MFfactor;
  using Base::hybrid_weight;
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
  using Base::hybrid;
  using Base::nspins_in_vHS;
  using Base::debug_verbosity;

  using SPRealType = typename Base::SPRealType;
  using SPComplexType = typename Base::SPComplexType;
  using CVector = typename Base::CVector; 
  using CMatrix = typename Base::CMatrix; 
  using C3Tensor = typename Base::C3Tensor;
  using SPCMatrix_ref = typename Base::SPCMatrix_ref; 
  using StaticMatrix = typename Base::StaticMatrix; 
  using StaticSPMatrix = typename Base::StaticSPMatrix; 
  using Static3Tensor = typename Base::Static3Tensor; 
  using stdSPCMatrix_ref = typename Base::stdSPCMatrix_ref; 
  using C3Tensor_ref = typename Base::C3Tensor_ref; 
  using C4Tensor_ref = typename Base::C4Tensor_ref; 
//  using  = typename Base::; 
  using mpi3SPCVector = typename Base::mpi3SPCVector; 

public:
  AFQMCDistributedPropagatorDistCV(AFQMCInfo& info,
                                   ptree pt_in,
                                   afqmc::TaskGroup_& tg_,
                                   Wavefunction& wfn_,
                                   utils::DeviceRandomGenerator_t* r)
      : Base(info, pt_in, tg_, wfn_, r),
        bpX(iextensions<1u>{1}, shared_allocator<ComplexType>{TG.TG_local()}),
        req_Gsend(MPI_REQUEST_NULL),
        req_Grecv(MPI_REQUEST_NULL),
        req_vsend(MPI_REQUEST_NULL),
        req_vrecv(MPI_REQUEST_NULL),
        req_Xsend(MPI_REQUEST_NULL),
        req_Xrecv(MPI_REQUEST_NULL),
        req_X2send(MPI_REQUEST_NULL),
        req_X2recv(MPI_REQUEST_NULL),
        req_bpvsend(MPI_REQUEST_NULL),
        req_bpvrecv(MPI_REQUEST_NULL)
  {
    RUNTIME_CHECK(TG.getNGroupsPerTG() > 1, "");

    low_memory_step = pt_in.get<bool>("low_memory", true);
    if (low_memory_step)
      app_log(1," Using low memory distributed propagation. ");
  }

  ~AFQMCDistributedPropagatorDistCV()
  {
    if (req_Grecv != MPI_REQUEST_NULL)
      MPI_Request_free(&req_Grecv);
    if (req_Gsend != MPI_REQUEST_NULL)
      MPI_Request_free(&req_Gsend);
    if (req_vrecv != MPI_REQUEST_NULL)
      MPI_Request_free(&req_vrecv);
    if (req_vsend != MPI_REQUEST_NULL)
      MPI_Request_free(&req_vsend);
    if (req_X2recv != MPI_REQUEST_NULL)
      MPI_Request_free(&req_X2recv);
    if (req_X2send != MPI_REQUEST_NULL)
      MPI_Request_free(&req_X2send);
    if (req_Xrecv != MPI_REQUEST_NULL)
      MPI_Request_free(&req_Xrecv);
    if (req_Xsend != MPI_REQUEST_NULL)
      MPI_Request_free(&req_Xsend);
    if (req_bpvrecv != MPI_REQUEST_NULL)
      MPI_Request_free(&req_bpvrecv);
    if (req_bpvsend != MPI_REQUEST_NULL)
      MPI_Request_free(&req_bpvsend);
  }

  AFQMCDistributedPropagatorDistCV(AFQMCDistributedPropagatorDistCV const& other) = delete;
  AFQMCDistributedPropagatorDistCV& operator=(AFQMCDistributedPropagatorDistCV const& other) = delete;
  AFQMCDistributedPropagatorDistCV(AFQMCDistributedPropagatorDistCV&& other)                 = default;
  AFQMCDistributedPropagatorDistCV& operator=(AFQMCDistributedPropagatorDistCV&& other) = delete;

  template<class WlkSet>
  void Propagate(int steps, WlkSet& wset, RealType E1, RealType dt, int fix_bias = 1)
  {
    int nblk   = steps / fix_bias;
    int nextra = steps % fix_bias;
    if (low_memory_step)
    {
      for (int i = 0; i < nblk; i++)
      {
        step_collective(fix_bias, wset, E1, dt);
        update_memory_managers();
      }
      if (nextra > 0)
        step_collective(nextra, wset, E1, dt);
    }
    else
    {
      for (int i = 0; i < nblk; i++)
      {
        step(fix_bias, wset, E1, dt);
        update_memory_managers();
      }
      if (nextra > 0)
        step(nextra, wset, E1, dt);
    }
    TG.TG_local().barrier();
  }

  template<class WlkSet, class CTens, class CMat>
  void BackPropagate(int steps, int nStabalize, WlkSet& wset, CTens&& Refs, CMat&& logdetR);

  template<class WlkSet, class Mat1, class Mat2, class Mat3>
  void PropagateOperators(int steps, WlkSet& wset,  Mat1&& X, Mat2&& Y, Mat3&& M);

protected:
  mpi3SPCVector bpX;
  std::vector<int> bpx_counts, bpx_displ;

  bool buffer_reallocated    = false;
  bool buffer_reallocated_bp = false;
  bool low_memory_step       = true;

  MPI_Request req_Gsend, req_Grecv;
  MPI_Request req_vsend, req_vrecv;

  MPI_Request req_Xsend, req_Xrecv;
  MPI_Request req_X2send, req_X2recv;
  MPI_Request req_bpvsend, req_bpvrecv;

  template<class WlkSet>
  void step(int steps, WlkSet& wset, RealType E1, RealType dt);

  template<class WlkSet>
  void step_collective(int steps, WlkSet& wset, RealType E1, RealType dt);
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Propagators/AFQMCDistributedPropagatorDistCV.icc"

#endif
