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

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "Utilities/app_loggers.h"
#include "Utilities/FairDivide.hpp"
#include "Memory/utilities.hpp"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/config.h"
#include "AFQMCBasePropagator.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"

namespace sfqmc
{
namespace afqmc
{

template<bool SP>
void AFQMCBasePropagator<SP>::reset_nextra(int nextra)
{
  if (nextra <= 0)
    return;
  if (last_nextra != nextra)
  {
    last_nextra = nextra;
    for (int n = 0; n < nextra; n++)
    {
      int n0, n1;
      std::tie(n0, n1) = FairDivideBoundary(n, TG.getNCoresPerTG(), nextra);
      if (TG.getLocalTGRank() >= n0 && TG.getLocalTGRank() < n1)
      {
        last_task_index = n;
        break;
      }
    }
    local_group_comm = shared_communicator(TG.TG_local().split(last_task_index, TG.TG_local().rank()));
  }
  if (last_task_index < 0 || last_task_index >= nextra)
    APP_ABORT("Error: Problems in AFQMCBasePropagator::reset_nextra()");
}

/*
 * Constructs the various 1-body propagators for the given timestep
 * If Pinv is true, only the backward-direction propagators are constructed, P1x_inv.
 * If Pinv is false, forward-direction propagators are constructed.
 * In addition, if Pinv is false and the backward-direction propagators have previously been
 * constructed, they are upated for the current timestep.  
 */
template<bool SP>
void AFQMCBasePropagator<SP>::generateP1(double dt, WALKER_TYPES walker_type, bool Pinv)
{
  using std::fill_n;
  using std::copy_n;
  using P1shm = ma::sparse::csr_matrix<ComplexType, int, int, shared_allocator<ComplexType>,
                                         ma::sparse::is_root>;
  bool build_inv = Pinv or (P1s_inv[0].capacity() > 0);
  boost::multi::array<ComplexType, 1> vMF_(vMF.extensions());

  old_dt = dt;

  app_log(1, "\n  - Generating a new 1-body propagator with timestep: {}",dt);


  // calculate vMF for the current time step
  fill_n(vMF.origin(), vMF.num_elements(), ComplexType(0));
  if (substractMF)
  { 
    wfn.vMF(vMF, dt);
    // MAM: For chemistry hamiltonians, v is hermitian and hence vMF real.
    //      Truncated sparse hamiltonian might mess this up, so make real "by-hand"
    ma::zero_complex_part(vMF);

    if (not wfn.distribution_over_cholesky_vectors() and (TG.TG().size() > TG.TG_local().size()) )
    { 
      if (not TG.TG_local().root())
        fill_n(vMF.origin(), vMF.num_elements(), ComplexType(0));
      TG.TG().all_reduce_in_place_n(raw_pointer_cast(vMF.origin()), vMF.num_elements(), std::plus<>());
    }
  }

  // getOneBodyPropagatorMatrix expects vMF on the host, so make temporary copy
  copy_n(vMF.origin(), vMF.num_elements(), vMF_.origin());
  RealType vmax = 0, v_ = 0;
  for (int i = 0; i < vMF_.size(); i++)
    v_ = std::max(v_, std::abs(vMF_[i]));
  TG.Global().all_reduce_n(&v_, 1, &vmax, boost::mpi3::max<>());
  app_log(1," Largest component of Mean-field subtraction potential: {}",vmax);
  if (vmax > vbias_bound) {
    app_warning(" WARNING: Mean-field subtraction potential has components outside vbias_bound.");
    app_warning("          Consider increasing vbias_bound. max(vMF[n])={}, vbias_bound={} ",
		    vmax,vbias_bound);
  }

  // assemble H1(i,j) = dt * (h(i,j) + vn0(i,j) + sum_n vMF[n]*vn(i,j,n))
  // H1 should have the same spin structure as walker_type
  auto H1(wfn.getOneBodyPropagatorMatrix(TG, dt, vMF_));

  if (walker_type == CLOSED) {

    if(external_H1)
      APP_ABORT(" Error: External propagator (P1) not yet working with CLOSED walker.");
    RUNTIME_CHECK(H1.size(0) == NMO && H1.size(1) == NMO, "");

    if(not Pinv)
      P1s[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1, printP1eV);
    if(build_inv) {
      ma::scal(ComplexType(-1.0), H1.flatted());
      P1s_inv[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1, printP1eV);
    }

  } else if (walker_type == COLLINEAR) {

    RUNTIME_CHECK(H1.size(0) == 2*NMO && H1.size(1) == NMO, "");
    auto&& H1a(H1.sliced(0,NMO));
    auto&& H1b(H1.sliced(NMO,2*NMO));
    if (external_H1) {
      // scale H1ext by dt
      ma::add(ComplexType(1.0), H1a, ComplexType(dt), H1ext[0], H1a );
      ma::add(ComplexType(1.0), H1b, ComplexType(dt), H1ext[1], H1b );
    }
    if(not Pinv) {
      P1s[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1a, printP1eV);
      P1s[1] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1b, printP1eV);
    }
    if(build_inv) {
      ma::scal(ComplexType(-1.0), H1.flatted());
      P1s_inv[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1a, printP1eV);
      P1s_inv[1] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1b, printP1eV);
    }

  } else if (walker_type == NONCOLLINEAR) {

    RUNTIME_CHECK(H1.size(0) == 2*NMO && H1.size(1) == 2*NMO, "");
    if (external_H1) {
      // scale H1ext by dt
      auto&& H1a(H1({0,NMO},{0,NMO}));
      auto&& H1b(H1({NMO,2*NMO},{NMO,2*NMO}));
      ma::add(ComplexType(1.0), H1a, ComplexType(dt), H1ext[0], H1a );
      ma::add(ComplexType(1.0), H1b, ComplexType(dt), H1ext[1], H1b );
    }
    if(not Pinv) 
      P1s[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1, printP1eV);
    if(build_inv) {
      ma::scal(ComplexType(-1.0), H1.flatted());
      P1s_inv[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1, printP1eV);
    }
  } else if (walker_type == FULLYPOLARIZED) {
    
    if(external_H1)
      APP_ABORT(" Error: External propagator (P1) not yet working with FULLYPOLARIZED walker.");
    RUNTIME_CHECK(H1.size(0) == NMO && H1.size(1) == NMO, "");

    if(not Pinv)
      P1s[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1, printP1eV);
    if(build_inv) {
      ma::scal(ComplexType(-1.0), H1.flatted());
      P1s_inv[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1, printP1eV);
    }
  }


  if(denseP1) {
    if(not Pinv) { 
      P1d = node3CTensor( {P1s.size(), P1s[0].size(0), P1s[0].size(1)},
                        make_node_allocator<ComplexType>(TG) );
      int nspin = (walker_type == COLLINEAR ? 2 : 1);
#if defined(ENABLE_DEVICE)
      for(int n=0; n<nspin; n++)
        ma::Matrix2MAREF('N',P1s[n],P1d[n]);
#else
      if(TG.Node().root()) {
        for(int n=0; n<nspin; n++)
          ma::Matrix2MAREF('N',P1s[n],P1d[n]);
      }
#endif
    }
    if(build_inv) {
      P1d_inv = node3CTensor( {P1s_inv.size(), P1s_inv[0].size(0), P1s_inv[0].size(1)},
                        make_node_allocator<ComplexType>(TG) );
      int nspin = (walker_type == COLLINEAR ? 2 : 1);
#if defined(ENABLE_DEVICE)
      for(int n=0; n<nspin; n++)
        ma::Matrix2MAREF('N',P1s_inv[n],P1d_inv[n]);
#else
      if(TG.Node().root()) {
        for(int n=0; n<nspin; n++)
          ma::Matrix2MAREF('N',P1s_inv[n],P1d_inv[n]);
      }
#endif
    }
    TG.Node().barrier();
  }
}

template void AFQMCBasePropagator<true>::reset_nextra(int);
template void AFQMCBasePropagator<false>::reset_nextra(int);

template void AFQMCBasePropagator<true>::generateP1(double,WALKER_TYPES,bool);
template void AFQMCBasePropagator<false>::generateP1(double,WALKER_TYPES,bool);

} // namespace afqmc


} // namespace sfqmc
