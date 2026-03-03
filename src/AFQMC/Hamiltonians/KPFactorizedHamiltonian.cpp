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
#include <iomanip>
#include <vector>
#include <utility>
#include <numeric>

#include "config.h"
#include "utilities/check.hpp"
#include "utilities/h5_utils.hpp"
#include "numerics/nda_functions.hpp"
#include "AFQMC/config.h"

#include "nda/h5.hpp"
#include "nda/tensor.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

#include "KPFactorizedHamiltonian.h"
#include "AFQMC/Utilities/wfn_utils.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"

#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/shared_array.hpp"

namespace sfqmc
{
namespace afqmc
{

// PsiT[idet][ispin][ik] -> csr_mat( nel_up/nel_dn, NMO )
template<MEMORY_SPACE MEM> HamiltonianOperations<MEM>
KPFactorizedHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
               std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
               nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  using nda::range;
  std::string base_error(" Error in KPFactorizedHamiltonian::getHamiltonianOperations_shared: \n    ");
  auto all = range::all;
  std::string format;  // only meaningful at root
  double GBx = sizeof(ComplexType)/double(1024*1024*1024);
  ComplexType zero(0.0),one(1.0);
  long nspin = (type == COLLINEAR?2:1);
  long npol = (type == NONCOLLINEAR?2:1);
  int nspin_in_PsiT = PsiT.extent(1);
  long ndet = PsiT.extent(0);
  long nel_up = PsiT(0,0).extent(0);
  long NMO = PsiT(0,0).extent(1)/npol;
  utils::check(PsiT(0,0).extent(1)%npol==0, base_error + "Psi.size(1)%npol != 0");
  utils::check(nspin_in_PsiT==1 or nspin_in_PsiT==nspin, "Size mismatch");
  utils::check(ndet==1, "Error: ndet > 1 not yet implemented in KPFactorizedHamiltonian::getHamiltonianOperations.");
  long nel_dn = ( type == FULLYPOLARIZED or type == NONCOLLINEAR ? 0l :
              (type == CLOSED ? nel_up : PsiT(0,nspin_in_PsiT-1).extent(0) ) );
  for(int i=0; i<ndet; ++i)
    for(int ip=0; ip<npol; ++ip) {
      utils::check(PsiT(i,0).shape() == std::array<long,2>{nel_up,NMO},"PsiT shape mismatch.");
      if(type == COLLINEAR)
        utils::check(PsiT(i,nspin_in_PsiT-1).shape() == std::array<long,2>{nel_dn,NMO},"PsiT shape mismatch.");
    }
  utils::check(nel_up >= nel_dn, base_error + "nel_up:{} < nel_dn:{} not allowed.",nel_up,nel_dn);

  // Hamiltonian variables
  int nspin_in_file = nspin;
  int npol_in_file = npol;
  // BZ variables
  int nkpts = 1;
  int nkpts_ibz = 1;
  int nqpts_ibz = 1;
  int nbnd = 1;
  int Q0_index = -1;
  nda::array<int,1> minusq;
  nda::array<int,2> qk_to_k2;
  nda::array<int,1> kp_to_ibz;
  nda::array<int,1> qp_to_ibz;
  nda::array<bool,1> kp_trev;
  nda::array<bool,1> qp_trev;
  nda::array<int,1> kp_trev_pair;
  nda::array<int,1> nchol;
  nda::array<double,2> qpoints;
  int number_of_trev_kpoint_pairs=0;

  h5::file file;
  // Read nbnd, BZ info, etc from h5. Only root reads
  if (mpi->comm.root())
  {
    file = h5::file(fileName,'r');
    h5::group grp = h5::group(file);
    format = get_hamiltonian_format(grp);
    if(format.substr(0,6) == "coqui") {
      // open subgroup
      h5::group hgrp = grp.open_group("System");
      {
        int n;
        h5::group bz = hgrp.open_group("BZ");
        // read nkpts
        h5::h5_read_attribute(bz,"number_of_kpoints",n);
        nkpts = long(n);
        h5::h5_read_attribute(bz,"number_of_kpoints_ibz",n);
        nkpts_ibz = long(n);
        h5::h5_read_attribute(bz,"number_of_qpoints",n);
        utils::check(n==nkpts, base_error + "nqpts != nkpts, nqpts:{}, nkpts:{}",n,nkpts);
        h5::h5_read_attribute(bz,"number_of_qpoints_ibz",n);
        nqpts_ibz = long(n);
        // nbnd
        h5::h5_read_attribute(hgrp,"number_of_bands",n);  // per kpoint
        // check nbnd 
        utils::check(NMO%nkpts==0, base_error + "NMO%nkpts != 0, NMO:{}, nkpts:{}",NMO,nkpts);
        utils::check((NMO/nkpts)==n, base_error + "nbnd:{} differs from file n:{}",nbnd,n);
        nbnd = long(NMO/nkpts);
        // read nspin_in_file
        h5::h5_read_attribute(hgrp,"number_of_spins",n);
        nspin_in_file = long(n);
        // read npol_in_file
//        h5::h5_read_attribute(bz,"number_of_polarizations",n);
//        npol_in_file=long(n);
        utils::check((nspin_in_file==1) or (nspin_in_file==nspin),
                     base_error + " Incompatible nspin:{} in h5 file.",nspin_in_file);
//        utils::check((npol_in_file==1) or (npol_in_file==npol), 
//                     base_error + " Incompatible npol:{} in h5 file.",npol_in_file);
        minusq.resize(nkpts);
        nda::h5_read(bz,"qminus",minusq);
        qk_to_k2.resize(nkpts,nkpts);
        nda::h5_read(bz,"qk_to_k2",qk_to_k2);
        qp_to_ibz.resize(nkpts);
        nda::h5_read(bz,"qp_to_ibz",qp_to_ibz);
        kp_trev.resize(nkpts);
        nda::h5_read(bz,"kp_trev",kp_trev);
        kp_trev_pair.resize(nkpts);
        nda::h5_read(bz,"kp_trev_pair",kp_trev_pair);
        qp_trev.resize(nkpts);
        nda::h5_read(bz,"qp_trev",qp_trev);
        qpoints.resize(nkpts,3);
        nda::h5_read(bz,"qpoints",qpoints);
        Q0_index=-1;
        for(int i=0; i<nkpts; i++)
          if(nda::sum(qpoints(i,all)*qpoints(i,all)) < 1e-8) {
            utils::check(Q0_index<0, "Error: Multiple points with Q=0: {}, {}",Q0_index,i);
            Q0_index = i;
          }
        utils::check(Q0_index>=0, "Error: Problems finding Q=0");
      }
      {
        // interaction
        h5::group igrp = grp.open_group("Interaction");
        nchol.resize(nqpts_ibz); 
        {
          auto l = h5::array_interface::get_dataset_info(igrp,"Vq0");
          utils::check(l.rank() == 6, "Rank mismatch");
          utils::check(l.lengths[1] == nspin_in_file*npol_in_file and
                       l.lengths[2] == nkpts and
                       l.lengths[3] == nbnd and
                       l.lengths[4] == nbnd, "Size mismatch");
          nchol(0) = l.lengths[0];
        }
        for(int Q=0; Q<nqpts_ibz; ++Q) {
          auto l = h5::array_interface::get_dataset_info(igrp,"Vq"+std::to_string(Q));
          utils::check(l.rank() == 6, "Rank mismatch");
          utils::check(l.lengths[1] == nspin_in_file*npol_in_file and
                       l.lengths[2] == nkpts and
                       l.lengths[3] == nbnd and
                       l.lengths[4] == nbnd, "Size mismatch");
          if(Q <= minusq(Q))
            nchol(Q) = l.lengths[0];
          else
            nchol(Q) = nchol(minusq(Q)); 
        }
      }
    } else if(format.substr(0,4) == "std") {
      // MAM: The "std" format, written for pyscf and the old fortran QE converter,
      //      was/is limited to spin independent basis sets. Generalize this if needed...
      // Current implementation is limited to cases with a consistent number of bands 
      // per kpoint, unlikely we will go back to the more general case.
      nspin_in_file = 1;
      npol_in_file  = 1;
      h5::group hgrp = grp.open_group("Hamiltonian");
      std::vector<int> Idata(8);
      h5::h5_read(hgrp,"dims",Idata);
      nkpts = Idata[2];
      nkpts_ibz = Idata[2];
      nqpts_ibz = Idata[2];
      utils::check(Idata[3] == NMO, " Error: NMO differs from value in integral file. ");

      Idata.resize(nkpts);
      std::vector<double> Ddata(nkpts);  
      h5::h5_read(hgrp,"NMOPerKP",Idata);
      utils::check(Idata.size() == nkpts, "size(NMOPerKP):{} != nkpts",Idata.size());
      nbnd = Idata[0];
      for(int i=1; i<nkpts; ++i)
        utils::check(Idata[i] == nbnd, "Inconsistent number of bands per kpoint. We now require all kpoints to have a consistent number of bands (NMOPerKP)."); 
      minusq.resize(nkpts);
      nda::h5_read(hgrp,"MinusK",minusq);
      qk_to_k2.resize(nkpts,nkpts);
      nda::h5_read(hgrp,"QKTok2",qk_to_k2);
      nchol.resize(nqpts_ibz);
      nda::h5_read(hgrp,"NCholPerKP",nchol);
      utils::check(NMO == nbnd*nkpts, " Error: NMO:{}, nkpts:{}, nbnd:{}",NMO,nkpts,nbnd); 
      Q0_index=-1;
      for (int Q = 0; Q < nkpts; Q++)
      {
        if (minusq(Q) == Q)
        {
          bool found = true;
          for (int KI = 0; KI < nkpts; KI++)
            if (KI != qk_to_k2(Q,KI))
            { 
              found = false;
              break;
            }
          if (found)
          {
            Q0_index = Q;
            break;
          }
        }
      }  
      utils::check(Q0_index>=0, "Error: Problems finding Q=0");
    } else {
      utils::check(false,"Unknown file format:",format);
    }
  } // root

  mpi->comm.broadcast_value(nspin_in_file);
  mpi->comm.broadcast_value(npol_in_file);
  mpi->comm.broadcast_value(nbnd);
  mpi->comm.broadcast_value(nkpts);
  mpi->comm.broadcast_value(nkpts_ibz);
  mpi->comm.broadcast_value(nqpts_ibz);
  mpi->comm.broadcast_value(Q0_index);
  if(not mpi->comm.root()) {
    minusq.resize(nkpts);
    qk_to_k2.resize(nkpts,nkpts);
    nchol.resize(nqpts_ibz);
  }
  mpi->broadcast(nchol);
  mpi->broadcast(minusq);
  mpi->broadcast(qk_to_k2);

  utils::check(nqpts_ibz==nkpts, "Error: Symmetry not yet implemented.");

  int nchol_av = nda::sum(nchol)/double(nchol.extent(0));
  app_log(1," # k-points: {}", nkpts);
  app_log(1," # Q-points (IBZ): {}", nqpts_ibz);
  app_log(1," Average # of cholesky vectors {}", nchol_av);

  // If q==minusq(q), qmap(q) has the position of q in Lbnk.
  nda::array<int,1> Qmap(nkpts,-1);  
  int number_of_symmetric_Q = 0;  // number of q==minusq(q)
  int number_of_allocated_Q = 0;
  for (int Q = 0; Q < nkpts; Q++) {
    if(minusq(Q) == Q) Qmap(Q) = (number_of_symmetric_Q++);
    if(Q <= minusq(Q)) number_of_allocated_Q++;
  }

  // Read hamiltonian components from h5. Only root reads
  // H0: /System/H0:   [nspin][nkpts][npol*nbnd][npol*nbnd]
  auto H1 = memory::make_shared_array<HOST_MEMORY,ComplexType,4>(mpi,
                      {nspin_in_file,nkpts,npol_in_file*nbnd,npol_in_file*nbnd});

  app_log(2, "KPFactorizedHamiltonian: Allocating Lijn: {} GB",
    number_of_allocated_Q*nspin_in_file*npol_in_file*nkpts*nbnd*nbnd*nchol_av*GBx);
  // L(Q)(ispin*ip,ik,i,j,n): Since each qpoint has its own nchol
  // dummy allocation if Q > minusq(Q)
  nda::array<memory::shared_array<MEM,ComplexType,6>,1> LQ(nkpts); 
  for (int Q = 0; Q < nkpts; Q++) 
  { 
    if(Q <= minusq(Q))
      LQ(Q) = std::move(memory::make_shared_array<MEM,ComplexType,6>(mpi,
             {nspin_in_file,npol_in_file,nkpts,nbnd,nbnd,nchol(Q)}));
    else
      LQ(Q) = std::move(memory::make_shared_array<MEM,ComplexType,6>(mpi,{1,1,1,1,1,1}));
  }

  if (mpi->comm.root()) 
  {
    h5::group grp = h5::group(file);

    if(format == "std") {

      utils::check(npol_in_file==1 and nspin_in_file==1, "KPFactorized: std format requires nspin_in_file==1 and npol_in_file==1.");

      h5::group hgrp = grp.open_group("Hamiltonian");

      // only spin independent hamiltonians right now!
      // now read H1_kpK
      for (int K = 0; K < nkpts; K++)
      {
        auto h_ = H1()(0,K,all,all);
        utils::h5_read(hgrp,std::string("H1_kp") + std::to_string(K), h_);
      }

      // now read KPFactorized/L
      h5::group lgrp = hgrp.open_group("KPFactorized");
      for(int Q=0; Q<nkpts; ++Q) {
        if(Q <= minusq(Q)) {
          auto L2d = nda::reshape(LQ(Q)()(0,0,nda::ellipsis{}),
                                  std::array<long,2>{nkpts,nbnd*nbnd*nchol(Q)});
          utils::h5_read(lgrp,std::string("L") + std::to_string(Q),L2d);
        }
      }
      // normalization (1/sqrt(nkpts)) assummed to be included

    } else if(format == "coqui") {

      h5::group sgrp = grp.open_group("System");
      auto h_ = H1();
      utils::h5_read(sgrp,"H0",h_);

      h5::group igrp = grp.open_group("Interaction");
      for(int Q=0; Q<nkpts; ++Q) {
        if(Q <= minusq(Q)) {
          // too much memory???
          nda::array<ComplexType,5> L(nchol(Q),nspin_in_file*npol_in_file,nkpts,nbnd,nbnd);
          utils::h5_read(igrp,"Vq"+std::to_string(Q),L);
          utils::check( L.shape() == std::array<long,5>{nchol(Q),nspin_in_file*npol_in_file,nkpts,nbnd,nbnd}, "Size mismatch from h5 dataset.");
          auto L2d = nda::reshape(L,std::array<long,2>{nchol(Q),nspin_in_file*npol_in_file*nkpts*nbnd*nbnd});
          if constexpr (MEM==HOST_MEMORY) {
            auto LQ2d = nda::reshape(LQ(Q)(),std::array<long,2>{nspin_in_file*npol_in_file*nkpts*nbnd*nbnd,nchol(Q)});
            LQ2d() = nda::transpose(L2d()); 
          } else {
            nda::array<ComplexType,2> L2(nspin_in_file*npol_in_file*nkpts*nbnd*nbnd,nchol(Q));
            L2() = nda::transpose(L2d());
            auto LQ_1d = nda::flatten(LQ(Q)()); 
            LQ_1d() = nda::flatten(L2());
          }
          // normalize
          nda::tensor::scale(ComplexType(1.0/std::sqrt(RealType(nkpts))), LQ(Q)());
        }
      } // Q
    } // format
  } // root
  mpi->comm.barrier();
  if(mpi->node_comm.root()) 
    mpi->internode_comm.all_reduce_in_place_n(H1.data(),H1.size(),std::plus<>{});
  if constexpr (MEM==HOST_MEMORY) {
    if(mpi->node_comm.root()) {
      for(int Q=0; Q<nkpts; ++Q)
        if(Q <= minusq(Q))
          mpi->internode_comm.all_reduce_in_place_n(LQ(Q).data(),LQ(Q).size(),std::plus<>{});
    }
  } else {
    for(int Q=0; Q<nkpts; ++Q)
      if(Q <= minusq(Q))
        mpi->all_reduce(LQ(Q)(),std::plus<>{});
  }
  mpi->comm.barrier();

 // kpoint dependent occupations
  nda::array<int,2> nocc(nspin,nkpts);
  if(mpi->comm.root())
    nocc = nocc_per_kpoint(type,nkpts,PsiT);
  mpi->broadcast(nocc);
  auto nocc_max = nda::max_element(nocc);
  utils::check(nel_up == nda::sum(nocc(0,all)), "Error: Mismatch in number of electrons: nel_up:{} sum(nel_up(k)):{}",nel_up,nda::sum(nocc(0,all)));
  if(type == COLLINEAR)
    utils::check(nel_dn == nda::sum(nocc(1,all)), "Error: Mismatch in number of electrons: ndown:{} sum(ndown(k)):{}",nel_dn,nda::sum(nocc(1,all)));

  /* half-rotate LQ and H1:
   * Given that PsiT = H(SM),
   * Lank(Q)(idet,ispin,a,j,n) = sum_i PsiT(idet,ispin)(a,ip,i) * LQ(Q)(ispin*ip,ik,i,j,n)
   *  where a includes the kpoint index implicitly, e.g. a:{0,nup/ndown}
   */
  app_log(2, "KPFactorizedHamiltonian: Allocating Lank: {} GB",
        (number_of_symmetric_Q+nkpts)*ndet*nspin_in_file*npol_in_file*nel_up*nbnd*nchol_av*GBx);
  nda::array<memory::shared_array<MEM,ComplexType,6>,1> Lank(nkpts); 
  nda::array<memory::shared_array<MEM,ComplexType,6>,1> Lbnk(number_of_symmetric_Q); 
// should be nspin_in_PsiT instead of nspin!!!
  for(int Q=0; Q<nkpts; ++Q) {
    Lank(Q) = std::move(memory::make_shared_array<MEM,ComplexType,6>(mpi,
           {ndet,nspin,nkpts,nocc_max,nchol(Q),npol*nbnd}));
    if(Q==minusq(Q)) 
      Lbnk(Qmap(Q)) = std::move(memory::make_shared_array<MEM,ComplexType,6>(mpi,
             {ndet,nspin,nkpts,nocc_max,nchol(Q),npol*nbnd}));
  }
  mpi->node_comm.barrier();
  if constexpr (MEM == HOST_MEMORY) {
    if(mpi->node_comm.root()) { 
      for(int Q=0; Q<nkpts; ++Q) Lank(Q)() = zero;
      for(int Q=0; Q<number_of_symmetric_Q; ++Q) Lbnk(Q)() = zero;
    }
  } else {
    for(int Q=0; Q<nkpts; ++Q) Lank(Q)() = zero;
    for(int Q=0; Q<number_of_symmetric_Q; ++Q) Lbnk(Q)() = zero;
  }
  mpi->node_comm.barrier();
  for(long id=0, itot=0; id<ndet; ++id) {
    for(long Q=0; Q<nkpts; ++Q) {
      int Qm = minusq(Q);
      for(long is=0; is<nspin; ++is) {
        int is_ = is%nspin_in_file;
        for(long ik=0; ik<nkpts; ++ik, ++itot) {
          if( itot%mpi->comm.size() != mpi->comm.rank() ) continue;
          int n0 = ( ik==0 ? 0 : nda::sum(nocc(is,range(ik))) );
          int nk = nocc(is,ik);
          for(long ip=0; ip<npol; ++ip) {
            int ip_ = ip%npol_in_file;
            if(Q <= Qm) 
            {
              // L[Q,k,k2=k-Q]
              auto Aai = math::sparse::to_array<'N'>(PsiT(id,is%nspin_in_PsiT),range(n0,n0+nk),
                                                     range(ip*NMO+ik*nbnd,ip*NMO+(ik+1)*nbnd));
              auto Lijn = LQ(Q)()(is_,ip_,ik,all,all,all);
              auto L_ = Lank(Q)()(id,is,ik,range(nk),all,range(ip*nbnd,(ip+1)*nbnd));
              utils::check(Lijn.extent(2)==L_.extent(1), "Size mismatch.");
              nda::tensor::contract(one,Aai,"ai",Lijn,"ijn",zero,L_,"anj");
            } else {
              // L[Q,k,k2=k-Q]
              int k2 = qk_to_k2(Q,ik);
              auto Abj = math::sparse::to_array<'N'>(PsiT(id,is%nspin_in_PsiT),range(n0,n0+nk),
                                                     range(ip*NMO+ik*nbnd,ip*NMO+(ik+1)*nbnd));
              auto Lljn = LQ(Qm)()(is_,ip_,k2,all,all,all);
              auto L_ = Lank(Q)()(id,is,ik,range(nk),all,range(ip*nbnd,(ip+1)*nbnd));
              utils::check(Lljn.extent(2)==L_.extent(1), "Size mismatch.");
              nda::tensor::contract(one,Abj,"bj",nda::conj(Lljn),"ljn",zero,L_,"bnl");
            }
            if(Q==Qm) {
              // conj(L[Q,k,k2](lj,n) * A[k2]bj, careful: Lbnk is indexed by k2! 
              int k2 = qk_to_k2(Q,ik);
              int n0b = ( k2==0 ? 0 : nda::sum(nocc(is,range(k2))) );
              int nb = nocc(is,k2);
              auto Abj = math::sparse::to_array<'N'>(PsiT(id,is%nspin_in_PsiT),range(n0b,n0b+nb),
                                                     range(ip*NMO+k2*nbnd,ip*NMO+(k2+1)*nbnd));
              auto Lljn = LQ(Q)()(is_,ip_,ik,all,all,all);
              auto L_ = Lbnk(Qmap(Q))()(id,is,k2,range(nb),all,range(ip*nbnd,(ip+1)*nbnd));
              utils::check(Lljn.extent(2)==L_.extent(1), "Size mismatch.");
              nda::tensor::contract(one,Abj,"bj",nda::conj(Lljn),"ljn",zero,L_,"bnl");
            }
          } // ip
        } // ik
      } // is
    }  // Q
  }  // id
  mpi->comm.barrier();
  if constexpr (MEM==HOST_MEMORY) {
    if(mpi->node_comm.root()) {
      for(int Q=0; Q<nkpts; ++Q) 
        if(Q <= minusq(Q)) 
          mpi->internode_comm.all_reduce_in_place_n(Lank(Q).data(),Lank(Q).size(),std::plus<>{});
      for(int Q=0; Q<number_of_symmetric_Q; ++Q) 
        mpi->internode_comm.all_reduce_in_place_n(Lbnk(Q).data(),Lbnk(Q).size(),std::plus<>{});
    }
  } else {
    for(int Q=0; Q<nkpts; ++Q) 
      if(Q <= minusq(Q)) 
        mpi->all_reduce(Lank(Q)(),std::plus<>{});
    for(int Q=0; Q<number_of_symmetric_Q; ++Q) 
      mpi->all_reduce(Lbnk(Q)(),std::plus<>{});
  }
  mpi->comm.barrier();

  // haj(idet,a_is_ik,j_ip_ik) = sum_j PsiT(idet,is)(a_is,i) H1(is,ik,i,j_ip)
  long nel[] = {nel_up, (type == COLLINEAR ? nel_dn : 0l) };
  auto haj = memory::make_shared_array<MEM,ComplexType,3>(mpi,std::array<long,3>{ndet, nel[0]+nel[1], npol*NMO}); 
  if constexpr (MEM == HOST_MEMORY) {
    if(mpi->node_comm.root()) haj() = zero; 
  } else {
    haj() = zero; 
  }
  mpi->node_comm.barrier();

  for(long id=0, itot=0; id<ndet; ++id) {
    for(long is=0; is<nspin; ++is) {
      for(long ik=0; ik<nkpts; ++ik, ++itot) {
        if( itot%mpi->comm.size() != mpi->comm.rank() ) continue;
        int n0 = ( ik==0 ? 0 : nda::sum(nocc(is,range(ik))) );
        int nk = nocc(is,ik);
        for(long ip1=0; ip1<npol; ++ip1) {
          int ip1_ = ip1%npol_in_file;
          auto Aai = math::sparse::to_array<'N'>(PsiT(id,is%nspin_in_PsiT),range(n0,n0+nk),range(ip1*NMO+ik*nbnd,ip1*NMO+(ik+1)*nbnd));
          // haj = PsiT * H1
          for(long ip2=0; ip2<npol; ++ip2) {
            int ip2_ = ip2%npol_in_file;
            auto h_ = haj()(id,range(is*nel_up+n0,is*nel_up+n0+nk),nda::range(ip2*NMO+ik*nbnd,ip2*NMO+(ik+1)*nbnd));
            if constexpr (MEM==HOST_MEMORY) {
              auto hij = H1()(is%nspin_in_file,ik,range(ip1_*nbnd,(ip1_+1)*nbnd),range(ip2_*nbnd,(ip2_+1)*nbnd));
              nda::blas::gemm(one,Aai,hij,one,h_);
            } else {
              memory::array<MEM,ComplexType,2> hij(H1()(is%nspin_in_file,ik,range(ip1_*nbnd,(ip1_+1)*nbnd),range(ip2_*nbnd,(ip2_+1)*nbnd)));
              nda::blas::gemm(one,Aai,hij,one,h_);
            }
          } // ip2
        } // ip1
      } // ik
    }  // is
  }  // id
  mpi->comm.barrier();
  if constexpr (MEM==HOST_MEMORY) {
    if(mpi->node_comm.root()) mpi->internode_comm.all_reduce_in_place_n(haj.data(),haj.size(),std::plus<>{});
  } else {
    mpi->all_reduce(haj(),std::plus<>{});
  }
  mpi->comm.barrier();

  // calculate vn0
  auto v0 = memory::make_shared_array<HOST_MEMORY,ComplexType,4>(mpi,std::array<long,4>{nspin_in_file*npol_in_file, nkpts, nbnd, nbnd});
  if(mpi->node_comm.root()) v0() = ComplexType(0.0);
  // v0(s,k,i,l) = -0.5*sum_k,q sum_j <i_k,j_k-q|j_k-q,l_k>
  //         = -0.5 sum_kq sum_j,n L[Q](s,k,i,j,n) conj( L[Q](s,k,l,j) ) 
  {
    memory::buffered_array<MEM,ComplexType,2> vt(nbnd,nbnd);
    for(long is=0, isp=0, itot=0; is<nspin_in_file; is++) {
      int is_ = is%nspin_in_file;
      for(long ip=0; ip<npol_in_file; ip++, isp++) {
        int ip_ = ip%npol_in_file;
        for(long ik=0; ik<nkpts; ik++, ++itot) {
          if( itot%mpi->comm.size() != mpi->comm.rank() ) continue;
          vt() = zero;
          for(long Q=0; Q<nkpts; Q++) {
            if(Q<=minusq(Q)) {
              auto Lijn = LQ(Q)()(is_,ip_,ik,nda::ellipsis{});
              nda::tensor::contract(one,Lijn,"ijn",nda::conj(Lijn),"ljn",one,vt,"il");
            } else {
              // L[Q,k,k2](i,j,n) = conj( L[-Q,k2,k](j,i,n) )
              int Qm = minusq(Q);
              int k2 = qk_to_k2(Q,ik); 
              auto Lijn = LQ(Qm)()(is_,ip_,k2,nda::ellipsis{});
              nda::tensor::contract(one,nda::conj(Lijn),"jin",Lijn,"jln",one,vt,"il");
            }
          }
          v0()(isp,ik,all,all) = vt();        
        } // ik
      } // ip
    } // is
  }
  if(mpi->node_comm.root()) mpi->internode_comm.all_reduce_in_place_n(v0.data(),v0.size(),std::plus<>{});
  mpi->comm.barrier();

  ComplexType E0 = NuclearCoulombEnergy + FrozenCoreEnergy;

  return HamiltonianOperations<MEM>(
      KP3IndexFactorization<MEM>(mpi,type,nbnd,Q0_index,std::move(nocc),std::move(minusq),
        std::move(qk_to_k2),std::move(Qmap),std::move(H1),std::move(haj),std::move(LQ),
        std::move(Lank),std::move(Lbnk),std::move(v0),E0,buffer_size));
}

template HamiltonianOperations<HOST_MEMORY>
  KPFactorizedHamiltonian::getHamiltonianOperations<HOST_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<HOST_MEMORY>,2>const&);
#if defined(ENABLE_DEVICE)
template HamiltonianOperations<DEVICE_MEMORY> 
  KPFactorizedHamiltonian::getHamiltonianOperations<DEVICE_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<DEVICE_MEMORY>,2>const&);
#endif 

} // namespace afqmc
} // namespace sfqmc
