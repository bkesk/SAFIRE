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


#include <vector>

#include "nda/tensor.hpp"

#include "config.h"
#include "AFQMC/config.h"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "AFQMC/Propagators/AFQMCBasePropagator.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/operations/exp.hpp"
#include "numerics/operations/tensor.hpp"

namespace sfqmc
{
namespace afqmc
{

/*
 * Constructs the various 1-body propagators for the given timestep
 * If Pinv is true, only the backward-direction propagators are constructed, P1x_inv.
 * If Pinv is false, forward-direction propagators are constructed.
 * In addition, if Pinv is false and the backward-direction propagators have previously been
 * constructed, they are updated for the current timestep.  
 */
template<MEMORY_SPACE MEM>
void AFQMCBasePropagator<MEM>::generateP1(double dt, WALKER_TYPES walker_type, bool Pinv)
{
  using nda::range;
  auto all = range::all;
  bool build_inv = Pinv or ( P1s_inv.size() > 0 ? (P1s_inv(0).capacity() > 0) : false );

  old_dt = dt;

  app_log(1, "\n  - Generating a new 1-body propagator with timestep: {}",dt);

  // update hamiltonian factorization parameters if needed (e.g. in discrete factorization)
  bool discrete_propg = false;
  for ( int i=0; i<FieldTypes.size(); i++ ) {
    int v(FieldTypes[i]);
    if( (PropagatorTypes(v) == DiscreteChargePropagator) or
        (PropagatorTypes(v) == DiscreteSpinPropagator) ) {
      discrete_propg = true;
      break;
    }
  }
  // discrete propagators setup their own vMF
  memory::buffered_array<MEM,ComplexType,1> vMF_discrete(vMF.extent(0)); 
  if(discrete_propg) {
    int npol         = (walker_type == NONCOLLINEAR or walker_type == NONCOLLINEAR_FT) ? 2 : 1;
    int nspin        = (walker_type == COLLINEAR or walker_type == COLLINEAR_FT) ? 2 : 1;
    nda::array<ComplexType,1> nMF(2*NMO, ComplexType(0.0));
    // setup sparse vector to generate <nI>
    auto Gmf_shm = wfn->G_MF();
    if(mpi->comm.root()) {
      auto Gmf = nda::to_host(Gmf_shm()); 
      for(int i=0; i<npol*NMO; i++)
        nMF(i) = Gmf(0,i,i);
      if(walker_type == COLLINEAR)
        for(int i=0; i<NMO; i++)
          nMF(i+NMO) = Gmf(1,i,i);
    }
    mpi->broadcast(nMF);
    wfn->update_potentials(dt,nMF,vMF_discrete,natural_shift);
  }

  bool head_shared = ( MEM==HOST_MEMORY ? mpi->node_comm.root() : true ); 
  if(head_shared) vMF() = ComplexType(0.0);
  mpi->comm.barrier();

  // calculate vMF for the current time step
  if (substractMF)
  { 
    {
     auto hamtype(wfn->getHamType());
      memory::buffered_array<MEM,ComplexType,1> vt(vMF.shape());
      // collective call
      wfn->vMF(vt, dt);
      if(hamtype == ModelHamiltonian) { 
        // depending on charge/spin, you should also set imag/real parts to zero
        // overwrite vMF if needed
        if(discrete_propg) {
          for ( int i=0; i<FieldTypes.size(); i++ ) {
            int v(FieldTypes(i));
            if( (PropagatorTypes(v) == DiscreteChargePropagator) or
                (PropagatorTypes(v) == DiscreteSpinPropagator) ) {
              vt(nda::range(i,i+1)) = vMF_discrete(nda::range(i,i+1));
            }
          }
        }
      } else {
        // continuous propagator, charge decomposition. vt should be real
        math::zero_imag(vt);
      }
      if(head_shared) vMF() = vt(); 
    }
    if constexpr (MEM==HOST_MEMORY) { 
      if(mpi->node_comm.root()) mpi->internode_comm.broadcast_n(vMF.data(),vMF.size(),0);
    } else {
      mpi->broadcast(vMF());
    }
  }
  mpi->comm.barrier();

  if(mpi->comm.root()) {
    auto v_h = nda::to_host(vMF());
    RealType vmax = 0;
    for (int i = 0; i < v_h.size(); i++)
      vmax = std::max(vmax, std::abs(v_h(i)));
    app_log(1," Largest component of Mean-field subtraction potential: {}",vmax);
    if (vmax > vbias_bound) {
      app_warning(" WARNING: Mean-field subtraction potential has components outside vbias_bound.");
      app_warning("          Consider increasing vbias_bound. max(vMF[n])={}, vbias_bound={} ",
  		    vmax,vbias_bound);
    }
  }
  mpi->comm.barrier();

  // assemble H1(i,j) = dt * (h(i,j) + vn0(i,j) + sum_n vMF[n]*vn(i,j,n))
  // H1 should have the same spin structure as walker_type
  // everypne computes until I write a csr_matrix in shared memory
  int nspin = (walker_type == COLLINEAR or walker_type == COLLINEAR_FT ? 2 : 1);
  int npol  = (walker_type == NONCOLLINEAR or walker_type == NONCOLLINEAR_FT ? 2 : 1);

  // resize if needed
  if(P1d.shape() != std::array<long,3>{nspin,npol*NMO,npol*NMO}) {
    mpi->comm.barrier();
    P1d = memory::make_shared_array<MEM,ComplexType,3>(mpi,std::array<long,3>{nspin,npol*NMO,npol*NMO});
  }
  if(build_inv and (P1d_inv.shape() != std::array<long,3>{nspin,npol*NMO,npol*NMO})) {
    mpi->comm.barrier();
    P1d_inv = memory::make_shared_array<MEM,ComplexType,3>(mpi,std::array<long,3>{nspin,npol*NMO,npol*NMO});
  } 
  mpi->comm.barrier();

  // H1 is in host
  if(head_shared) {
    auto vMF_h = nda::to_host(vMF());
    auto H1 = wfn->getOneBodyPropagatorMatrix(dt, vMF_h);
    utils::check(H1.shape() == std::array<long,3>{nspin,npol*NMO,npol*NMO}, "Shape mismatch.");
    if(external_H1) nda::tensor::add(ComplexType(1.0),H1ext(),"sij",ComplexType(1.0),H1(),"sij");

    nda::tensor::scale(ComplexType(-0.5),H1);
    for(int i=0; i<nspin; ++i) 
      P1d()(i,all,all) = math::exp_hermitian(H1(i,all,all), printP1eV);
    if(build_inv) {
      nda::tensor::scale(ComplexType(-1.0),H1);
      for(int i=0; i<nspin; ++i) 
        P1d_inv()(i,all,all) = math::exp_hermitian(H1(i,all,all), printP1eV);
    }
  }
  mpi->comm.barrier();

  if(P1s.size() != nspin) P1s.resize(nspin); 
  for(int i=0; i<nspin; i++) 
    P1s(i) = math::sparse::to_csr<MEM>(P1d()(i,all,all),1e-8); 
  if(build_inv) { 
    if(P1s_inv.size() != nspin) P1s_inv.resize(nspin); 
    for(int i=0; i<nspin; i++) 
      P1s_inv(i) = math::sparse::to_csr<MEM>(P1d_inv()(i,all,all),1e-8); 
  }
  mpi->comm.barrier();

}

template void AFQMCBasePropagator<HOST_MEMORY>::generateP1(double,WALKER_TYPES,bool);

#if defined(ENABLE_DEVICE)
template void AFQMCBasePropagator<DEVICE_MEMORY>::generateP1(double,WALKER_TYPES,bool);
#endif

} // namespace afqmc


} // namespace sfqmc
