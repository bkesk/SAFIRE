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

#include <vector>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "Utilities/app_loggers.h"
#include "Memory/utilities.hpp"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/config.h"
#include "AFQMCModelPropagator.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"

namespace sfqmc
{
namespace afqmc
{

template<bool SP>
void AFQMCModelPropagator<SP>::generateP1(double dt, WALKER_TYPES walker_type)
{
  using std::fill_n;
  using std::copy_n;
  using P1shm = ma::sparse::csr_matrix<ComplexType, int, int, shared_allocator<ComplexType>,
                                         ma::sparse::is_root>;
  old_dt = dt;

  app_log(1, "\n  - Generating a new 1-body propagator with timestep: {}", dt);

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
  boost::multi::array<ComplexType, 1> vMF_discrete(vMF.extensions());
  if(discrete_propg) {
    int npol         = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin        = (walker_type == COLLINEAR) ? 2 : 1;
    Vector<ComplexType> nMF(iextensions<1u>{2*NMO}); 
    CMatrix Gmf({nspin * npol * NMO, npol * NMO},ComplexType(0.0,0.0)); 
    // setup sparse vector to generate <nI>
    wfn.G_MF(Gmf);
    for(int i=0; i<npol*NMO; i++)
      nMF[i] = ComplexType(Gmf[i][i]);	
    if(walker_type == COLLINEAR)
      for(int i=0; i<NMO; i++)
        nMF[i+NMO] = ComplexType(Gmf[i+NMO][i]);	
    wfn.update_potentials(dt,nMF,vMF_discrete,natural_shift);
  }

  // calculate vMF for the current time step
  fill_n(vMF.origin(), vMF.num_elements(), ComplexType(0));
  boost::multi::array<ComplexType, 1> vMF_(vMF.extensions(), ComplexType(0));
  if (substractMF)
  {
    wfn.vMF(vMF, dt);
    copy_n(vMF.origin(), vMF.num_elements(), vMF_.origin());
    if(discrete_propg) {
      for ( int i=0; i<FieldTypes.size(); i++ ) {
        int v(FieldTypes[i]);
        if( (PropagatorTypes(v) == DiscreteChargePropagator) or 
            (PropagatorTypes(v) == DiscreteSpinPropagator) ) {
	  vMF_[i] = vMF_discrete[i]; 
        }
      }
      // copy potential changes back to device array      
      copy_n(vMF_.origin(), vMF_.num_elements(), vMF.origin());
    }
  }

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
      APP_ABORT(" Error: Spin dependent propagator (P1) with CLOSED walker not yet working.");
    RUNTIME_CHECK(H1.size(0) == NMO && H1.size(1) == NMO, "");
    P1s[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1, printP1eV);

  } else if (walker_type == COLLINEAR) {

    RUNTIME_CHECK(H1.size(0) == 2*NMO && H1.size(1) == NMO, "");
    auto&& H1a(H1.sliced(0,NMO));
    auto&& H1b(H1.sliced(NMO,2*NMO));
    if (external_H1) {
      // scale H1ext by dt
      ma::add(ComplexType(1.0), H1a, ComplexType(dt), H1ext[0], H1a );
      ma::add(ComplexType(1.0), H1b, ComplexType(dt), H1ext[1], H1b );
    }
    P1s[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1a, printP1eV);
    P1s[1] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1b, printP1eV);

  } else if (walker_type == NONCOLLINEAR) {

    RUNTIME_CHECK(H1.size(0) == 2*NMO && H1.size(1) == 2*NMO, "");
    if (external_H1) {
      // scale H1ext by dt
      auto&& H1a(H1({0,NMO},{0,NMO}));
      auto&& H1b(H1({NMO,2*NMO},{NMO,2*NMO}));
      ma::add(ComplexType(1.0), H1a, ComplexType(dt), H1ext[0], H1a );
      ma::add(ComplexType(1.0), H1b, ComplexType(dt), H1ext[1], H1b );
    }
    P1s[0] = generate1BodyPropagator<P1shm>(TG, 1e-8, H1, printP1eV);

  }
  if(denseP1) {
    P1d = node3CTensor( {P1s.size(), P1s[0].size(0), P1s[0].size(1)}, 
                        make_node_allocator<ComplexType>(TG) );
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    for(int n=0; n<nspin; n++)
      ma::Matrix2MAREF('N',P1s[n],P1d[n]);
#else
    if(TG.Node().root()) {
      for(int n=0; n<nspin; n++)
        ma::Matrix2MAREF('N',P1s[n],P1d[n]);
    }
#endif
    TG.Node().barrier();
  }  
  
}

// instantiate templates to keep cpp
template void AFQMCModelPropagator<true>::generateP1(double,WALKER_TYPES);
template void AFQMCModelPropagator<false>::generateP1(double,WALKER_TYPES);

} // namespace afqmc


} // namespace sfqmc
