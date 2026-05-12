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

#include <cstdlib>
#include <algorithm>
#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <utility>
#include <vector>
#include <numeric>
#include <functional>

#include "config.h"
#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "utilities/h5_utils.hpp"
#include "numerics/nda_functions.hpp"

#include "nda/h5.hpp"
#include "nda/tensor.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/shared_array.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "RealDenseHamiltonian.h"

namespace sfqmc
{
namespace afqmc
{

// NOTE: remove AFQMCInfo object from Hamiltonian generators, NMO/nup/ndown should be provided by calling routine

template<MEMORY_SPACE MEM> HamiltonianOperations<MEM> 
RealDenseHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  using nda::range;
  std::string base_error(" Error in RealDenseHamiltonian::getHamiltonianOperations():\n   ");
  auto all  = range::all;
  int nspin = ((type != COLLINEAR) ? 1 : 2);
  int npol  = ((type == NONCOLLINEAR) ? 2 : 1);
  int ndet = PsiT.extent(0);
  int nact_up = PsiT(0,0).extent(0);
  int NMO = PsiT(0,0).extent(1)/npol;
  int nspin_in_PsiT = PsiT.extent(1);
  int nspin_in_H1 = 1, npol_in_H1 = 1; // read/broadcast below
  int nspin_in_H2 = 1, npol_in_H2 = 1; // read/broadcast below
  utils::check(PsiT(0,0).extent(1)%npol==0, base_error + "Psi.size(1)%npol != 0");
  utils::check(nspin_in_PsiT == 1 or nspin_in_PsiT == nspin, base_error + "Size mismatch PsiT");
  utils::check(nspin==1 or npol==1, base_error + "Both nspin and npol can not be >1 simultaneously."); 

  // MAM: should this be zero with CLOSED shell???
  int nact_dn = ( type == FULLYPOLARIZED or type == NONCOLLINEAR ? 0l :
              (type == CLOSED ? nact_up : PsiT(0,nspin_in_PsiT-1).extent(0) ) );
  bool head_shared = (MEM==HOST_MEMORY ? mpi->node_comm.root() : true );

  std::vector<long> Idata(8);
  ComplexType E0;
  h5::file file;
  if (mpi->comm.root()) 
  {
    file = h5::file(fileName,'r');
    h5::group g = h5::group(file).open_group("Hamiltonian"); 

    h5::h5_read(g,"dims",Idata);

    std::vector<RealType> E_(2);
    h5::h5_read(g,"Energies",E_);
    E0 = E_[0] + E_[1];

    {
      // Too many choices, consider forcing standard structure (e.g. [ns_f][np_f*NMO][np_f*NMO]
      auto l = h5::array_interface::get_dataset_info(g,"hcore");
      if(l.rank() == 2) {
        // [2*NMO][NMO] 
        if( l.lengths[0]==2*NMO and l.lengths[1]==NMO ) {
          nspin_in_H1=2;
          npol_in_H1=1; 
        } else {
          nspin_in_H1=1;
          utils::check(l.lengths[0] == l.lengths[1], base_error + "Size mismatch");
          utils::check(l.lengths[0]==NMO or l.lengths[0]==2*NMO, base_error +  "Size mismatch");
          npol_in_H1 = l.lengths[0]/NMO;
        }
      } else if(l.rank() == 3) {
        if (l.lengths[0]==2*NMO and l.lengths[1]==2*NMO)
        {
          nspin_in_H1=1;
          npol_in_H1=2;
          // checking l.lengths[0] == l.lengths[1] is unnecessary, see condition above
          utils::check(l.lengths[2]==2, base_error + "Incorrect tailing dimension of hcore in h5 file. Expected 2, got {}",l.lengths[2]);
        } else {
          nspin_in_H1=l.lengths[0];
          utils::check(l.lengths[1] == l.lengths[2], base_error + "Size mismatch");
          utils::check(l.lengths[1]==NMO or l.lengths[1]==2*NMO, base_error + "Size mismatch");
          npol_in_H1 = l.lengths[1]/NMO;
        }
      } else {
        utils::check(false, base_error + "Invalid hcore rank:{}",l.rank());
      }
      utils::check(nspin_in_H1 == 1 or nspin_in_H1 == nspin, 
                   base_error +  "Invalid nspin_in_H1:{}",nspin_in_H1);
      utils::check(npol_in_H1 == 1 or npol_in_H1 == npol, 
                   base_error +  "Invalid npol_in_H1:{}",npol_in_H1);
    }
    {
      // cholesky tensor
      h5::group vgrp = g.open_group("DenseFactorized");
      auto l = h5::array_interface::get_dataset_info(vgrp,"L");
      if(l.rank()==2) {
        //[nspin_in_H2*npol_in_H2*NMO*NMO]][ncv]
        if( nspin > 1 ) nspin_in_H2 = l.lengths[0] / (NMO*NMO); 
        if( npol > 1 ) npol_in_H2 = l.lengths[0] / (NMO*NMO); 
        utils::check( l.lengths[0] == nspin_in_H2*npol_in_H2*NMO*NMO, 
                      base_error + "Inconsistent size of DenseFactorized/L:({}, {}). Incompatible with nspin_in_H2:{}, npol_in_H2:{}, NMO:{} found in hcore",l.lengths[0],l.lengths[1],nspin_in_H2,npol_in_H2,NMO);
      } else if(l.rank()==3 or l.rank()==4) {
        //rank:3 [nspin_in_H2*npol_in_H2][NMO*NMO]][ncv]
        //rank:4 [nspin_in_H2*npol_in_H2][NMO][NMO]][ncv]
        if( nspin > 1 ) nspin_in_H2 = l.lengths[0]; 
        if( npol > 1 ) npol_in_H2 = l.lengths[0];  
        utils::check( l.lengths[0] == nspin_in_H2*npol_in_H2, 
                      base_error +  "Inconsistent size of DenseFactorized/L:({}, ...). Incompatible with nspin_in_H2:{}, npol_in_H2:{} found in hcore",l.lengths[0],nspin_in_H2,npol_in_H2);
      } else {
        utils::check(false, "Invalid Cholesky vector rank:{} ",l.rank());
      }
    }
  }
  mpi->comm.broadcast_n(Idata.begin(), 8, 0);
  mpi->comm.broadcast_n(&E0, 1, 0);
  mpi->comm.broadcast_n(&nspin_in_H1, 1, 0);
  mpi->comm.broadcast_n(&npol_in_H1, 1, 0);
  mpi->comm.broadcast_n(&nspin_in_H2, 1, 0);
  mpi->comm.broadcast_n(&npol_in_H2, 1, 0);

  // number of cholesky vectors
  int ncv = Idata[7];
  
  // allocate shared arrays
  auto H1 = memory::make_shared_array<HOST_MEMORY,ComplexType,3>(mpi,
                      {nspin_in_H1,npol_in_H1*NMO,npol_in_H1*NMO});
  auto Likn = memory::make_shared_array<MEM,RealType,4>(mpi,
                      {nspin_in_H2*npol_in_H2,NMO,NMO,ncv});

  if(mpi->comm.root()) {
    h5::group g = h5::group(file).open_group("Hamiltonian"); 
    {
      // hcore
      auto l = h5::array_interface::get_dataset_info(g,"hcore");
      auto h_ = nda::reshape(H1(),nspin_in_H1*npol_in_H1*NMO,npol_in_H1*NMO);
      sfqmc::utils::h5_read(g,"hcore",h_);
    }
    {
      // cholesky tensor
      h5::group vgrp = g.open_group("DenseFactorized"); 
      auto l = h5::array_interface::get_dataset_info(vgrp,"L");
      if(l.rank()==2) {
        //[nspin_in_H2*npol_in_H2*NMO*NMO]][ncv]
        utils::check( l.lengths[0] == nspin_in_H2*npol_in_H2*NMO*NMO  and
                      l.lengths[1] == ncv, base_error +  "Inconsistent size of DenseFactorized/L:({}, {}). Incompatible with nspin_in_H2:{}, npol_in_H2:{}, NMO:{} found in hcore",l.lengths[0],l.lengths[1],nspin_in_H2,npol_in_H2,NMO);
        auto L_ = nda::reshape(Likn(),std::array<long,2>{nspin_in_H2*npol_in_H2*NMO*NMO,ncv});
        utils::h5_read(vgrp,"L",L_);
      } else if(l.rank()==3) {
        //[nspin_in_H2*npol_in_H2][NMO*NMO]][ncv]
        utils::check( l.lengths[0] == nspin_in_H2*npol_in_H2  and
                      l.lengths[1] == NMO*NMO  and
                      l.lengths[2] == ncv, base_error +  "Inconsistent size of DenseFactorized/L:({}, {}, {}). Incompatible with nspin_in_H2:{}, npol_in_H2:{}, NMO:{} found in hcore",l.lengths[0],l.lengths[1],l.lengths[2],nspin_in_H2,npol_in_H2,NMO);
        auto L_ = nda::reshape(Likn(),std::array<long,3>{nspin_in_H2*npol_in_H2,NMO*NMO,ncv});
        utils::h5_read(vgrp,"L",L_);
      } else if(l.rank()==4) {
        //[nspin_in_H2*npol_in_H2][NMO][NMO]][ncv]
        utils::check( l.lengths[0] == nspin_in_H2*npol_in_H2  and
                      l.lengths[1] == NMO  and
                      l.lengths[2] == NMO  and
                      l.lengths[3] == ncv, base_error +  "Inconsistent size of DenseFactorized/L:({}, {}, {}, {}). Incompatible with nspin_in_H2:{}, npol_in_H2:{}, NMO:{} found in hcore",l.lengths[0],l.lengths[1],l.lengths[2],l.lengths[3],nspin_in_H2,npol_in_H2,NMO);
        auto L_ = nda::reshape(Likn(),std::array<long,4>{nspin_in_H2*npol_in_H2,NMO,NMO,ncv});
        utils::h5_read(vgrp,"L",L_);
      } else {
        utils::check(false, "Invalid Cholesky vector rank:{} ",l.rank());
      }
    }
  }
  if(mpi->node_comm.root()) mpi->internode_comm.broadcast_n(H1.data(),H1.size(),0); 
  if constexpr (MEM==HOST_MEMORY) {
    if(mpi->node_comm.root()) mpi->internode_comm.broadcast_n(Likn.data(),Likn.size(),0); 
  } else {
    mpi->broadcast(Likn());
  }
  mpi->comm.barrier();

  long nel[] = {nact_up, (type == COLLINEAR ? nact_dn : 0l) };
  auto haj = memory::make_shared_array<MEM,ComplexType,3>(mpi,std::array<long,3>{ndet, nel[0]+nel[1], npol*NMO});
// use nspin_in_PsiT and propagate into HamOps
  nda::array<memory::shared_array<MEM,ComplexType,5>,1> Lnak(nspin);
  for (int is = 0; is < nspin; is++)
    Lnak(is) = std::move(memory::make_shared_array<MEM,ComplexType,5>(mpi,
             {ndet,npol,ncv,(is==0?nact_up:nact_dn),NMO}));

  // for simplicity
  for (int id = 0, itot=0; id<ndet; id++) {
    for(long is=0; is<nspin; ++is, ++itot) {
      if( itot%mpi->comm.size() != mpi->comm.rank() ) continue;
      auto Aai = math::sparse::to_array<'N'>(PsiT(id,is%nspin_in_PsiT));

      // H1
      {
        int is_f = is%nspin_in_H1;
        auto h_ = haj()(id,range(is*nact_up,nact_up+is*nact_dn),all);
        nda::array<ComplexType,2> hc(npol*NMO,npol*NMO);
        hc() = ComplexType(0.0);
        if(npol_in_H1==1) {
          for(int p=0; p<npol; p++)
            for(int a=0; a<NMO; a++)
              for(int b=0; b<NMO; b++)
                hc(p*NMO+a,p*NMO+b) = ComplexType(H1()(is_f,a,b));
        } else {
          for(int a=0; a<npol*NMO; a++)
            for(int b=0; b<npol*NMO; b++)
              hc(a,b) = ComplexType(H1()(is_f,a,b));
        }  
        if constexpr (MEM==HOST_MEMORY) {
          nda::blas::gemm(Aai,hc,h_);
        } else {
          memory::array<MEM,ComplexType,2> hc_d(hc);
          nda::blas::gemm(Aai,hc_d,h_);
        }
      } 

      // Lnak
      {
        int is_f = is%nspin_in_H2;
        auto Aai_r = memory::to_real_view(Aai);
        auto L_r = memory::to_real_view(Lnak(is)()(id,nda::ellipsis{}));
        for(int p=0; p<npol; ++p) {
          int ip_f = p%npol_in_H2;
          nda::range rng(ip_f*NMO,(ip_f+1)*NMO);
          auto Aai_is = Aai_r(all,nda::range(p*NMO,(p+1)*NMO),all);
          nda::tensor::contract(RealType(1.0),Aai_is,"aic",Likn()(is_f,rng,rng,all),"ijn",
                                RealType(0.0),L_r(p,all,range(nel[is]),all,all),"najc");
        }
      }
    } // is
  }
  mpi->comm.barrier();
  if constexpr (MEM==HOST_MEMORY) {
    if(mpi->node_comm.root()) {
      mpi->internode_comm.all_reduce_in_place_n(haj.data(),haj.size(),std::plus<>{}); 
      for(int is=0; is<nspin; ++is)
        mpi->internode_comm.all_reduce_in_place_n(Lnak(is).data(),Lnak(is).size(),std::plus<>{}); 
    }
  } else {
    mpi->all_reduce(haj(),std::plus<>{});
    for(int is=0; is<nspin; ++is)
      mpi->all_reduce(Lnak(is)(),std::plus<>{});
  }
  mpi->comm.barrier();

  // exchange potential, parallelize over i:{0,NMO} to avoid temporary memory
  auto v0 = memory::make_shared_array<HOST_MEMORY,RealType,3>(mpi,std::array<long,3>{nspin_in_H2*npol_in_H2, NMO, NMO});
  auto [n0, n1] = itertools::chunk_range(0, NMO, mpi->comm.size(), mpi->comm.rank());
  // calculate v0(i,l) = -0.5 sum_j sum_n L[i][j][n] L[j][l][n] = -0.5 sum_j sum_n L[i][j][n] L[l][j][n]
  if(n1>n0)
  {
    if constexpr (MEM==HOST_MEMORY) {
      for(int is=0, isp=0; is<nspin_in_H2; ++is)
        for(int ip=0; ip<npol_in_H2; ++ip, ++isp)
          nda::tensor::contract(RealType(-0.5),Likn()(isp,range(n0,n1),all,all),"ijn",
                                              Likn()(isp,all,all,all),"ljn",
                                RealType(0.0),v0()(isp,range(n0,n1),all),"il");
    } else {
      memory::array<MEM,RealType,2> vt(n1-n0, NMO);
      for(int is=0, isp=0; is<nspin_in_H2; ++is)
        for(int ip=0; ip<npol_in_H2; ++ip, ++isp) {
          nda::tensor::contract(RealType(-0.5),Likn()(isp,range(n0,n1),all,all),"ijn",
                                              Likn()(isp,all,all,all),"ljn",
                                RealType(0.0),vt,"il");
          v0()(isp,range(n0,n1),all) = vt();
        }
    }
  }
  mpi->comm.barrier();
  if constexpr (MEM==HOST_MEMORY) {
    if(mpi->node_comm.root()) mpi->internode_comm.all_reduce_in_place_n(v0.data(),v0.size(),std::plus<>{});
  } else {
    mpi->all_reduce(v0(),std::plus<>{});
  }
  mpi->comm.barrier();

  return HamiltonianOperations<MEM>(Real3IndexFactorization<MEM>(mpi,type,NMO,nact_up,nact_dn,
       std::move(H1), std::move(haj), std::move(Likn), std::move(Lnak), std::move(v0), E0,
       max_memory_MB));
}

// instantiate templates
template HamiltonianOperations<HOST_MEMORY> 
  RealDenseHamiltonian::getHamiltonianOperations<HOST_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<HOST_MEMORY>,2>const&);
#if defined(ENABLE_DEVICE)
template HamiltonianOperations<DEVICE_MEMORY>
  RealDenseHamiltonian::getHamiltonianOperations<DEVICE_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<DEVICE_MEMORY>,2>const&);
#endif

} // namespace afqmc
} // namespace sfqmc
