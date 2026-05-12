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
////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <algorithm>
#include <complex>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <vector>
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

#include "KPTHCHamiltonian.h"
#include "AFQMC/Utilities/wfn_utils.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"

#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/shared_array.hpp"

namespace sfqmc
{
namespace afqmc
{
// Coqui h5 structure
// H1: /System/H0:   [nspin,nkpts_ibz,nbnd,nbnd]
// X: /Interaction/collocation_matrix:  [nspins,nkpts,nbnd,Nu]
// L: /Interaction/factorized_coulomb_matrix:  [Nq][Nu][Nv]  
// Z: /Interaction/coulomb_matrix:  [Nq][Nu][Nv]  
// with half rotation:
// Xrot: /Interaction/collocation_matrix_half_rotated:  [nspins,nkpts,nbnd,Nu]
// Zrot: /Interaction/half_rotated_coulomb_matrix:      [Nq][Nu][Nv]

// PsiT[idet][ispin][ik] -> csr_mat( nel_up/nel_dn, NMO )
template<MEMORY_SPACE MEM> HamiltonianOperations<MEM> 
KPTHCHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
               std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
               nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  using nda::range;

  std::string base_error(" Error in KPTHCHamiltonian::getHamiltonianOperations():\n   ");

  auto all = range::all;
  std::string format;  // only meaningful at root
  long nspin = (type == COLLINEAR?2:1);
  long npol = (type == NONCOLLINEAR?2:1);
  int nspin_in_PsiT = PsiT.extent(1);
  long ndet = PsiT.extent(0); 
  long nel_up = PsiT(0,0).extent(0);
  long NMO = PsiT(0,0).extent(1)/npol;
  utils::check(PsiT(0,0).extent(1)%npol==0, base_error + "Psi.size(1)%npol != 0");
  utils::check(nspin_in_PsiT==1 or nspin_in_PsiT==nspin, "Size mismatch");
  utils::check(ndet==1, "Error: ndet > 1 not yet implemented in KPTHCHamiltonian::getHamiltonianOperations.");
  long nel_dn = ( type == FULLYPOLARIZED or type == NONCOLLINEAR ? 0l :
              (type == CLOSED ? nel_up : PsiT(0,nspin_in_PsiT-1).extent(0) ) );
  for(int i=0; i<ndet; ++i) 
    for(int ip=0; ip<npol; ++ip) {
      utils::check(PsiT(i,0).shape() == std::array<long,2>{nel_up,NMO},"PsiT shape mismatch.");
      if(type == COLLINEAR)
        utils::check(PsiT(i,nspin_in_PsiT-1).shape() == std::array<long,2>{nel_dn,NMO},"PsiT shape mismatch.");
    }
  utils::check(nel_up >= nel_dn, base_error + "nel_up:{} < nel_dn:{} not allowed.",nel_up,nel_dn);

  // THC variables
  long nspin_in_file = nspin;
  long npol_in_file = npol;
  // ISDF variables
  long nu = 1; // number of interpolating points/vectors
  long nv = 1; // number of columns of Vuv matrix
  long nu_rot = 1; // number of interpolating points/vectors
  long nv_rot = 1; // number of columns of Vuv matrix
  long have_rot_coul = 0;  
  // BZ variables
  long nkpts = 1;
  long nkpts_ibz = 1;
  long nqpts_ibz = 1;
  long nbnd = 1; 
  long Q0_index = 0;
  nda::array<int,1> minusq;
  nda::array<int,2> qk_to_k2;
  nda::array<int,1> kp_to_ibz; 
  nda::array<int,1> qp_to_ibz; 
  nda::array<bool,1> kp_trev;  
  nda::array<bool,1> qp_trev;  
  nda::array<int,1> kp_trev_pair;
  nda::array<double,2> qpoints; 
  long number_of_trev_kpoint_pairs=0;
  
  // only root reads
  h5::file file;
  if (mpi->comm.root())
  {
    file = h5::file(fileName,'r');
    h5::group grp = h5::group(file);
    format = get_hamiltonian_format(grp);
    // open subgroup
    utils::check(format == "coqui", base_error + " Only coqui format is allowed. Format found:{}",format);
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
//      h5::h5_read_attribute(bz,"number_of_polarizations",n);
//      npol_in_file=long(n);
      utils::check((nspin_in_file==1) or (nspin_in_file==nspin), 
                   base_error + " Incompatible nspin:{} in h5 file.",nspin_in_file);
//      utils::check((npol_in_file==1) or (npol_in_file==npol), 
//                   base_error + " Incompatible npol:{} in h5 file.",npol_in_file);
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
    // read from /Interaction 
    {
      h5::group igrp = grp.open_group("Interaction");
      // check consistency
// MAM: fix this, integer types in h5 file are not consistent!!!
      long n;
      // read nspin_in_file
      h5::h5_read_attribute(igrp,"number_of_spins",n);
      utils::check(nspin_in_file==n,
                   base_error + " Incompatible nspin:{} in h5::/Interaction.",n);
      // read npol_in_file
//      h5::h5_read_attribute(igrp,"number_of_polarizations",n);
//      utils::check(npol_in_file==n,
//                   base_error + " Incompatible npol:{} in h5::/Interaction.",n);
      // nbnd 
      h5::h5_read_attribute(igrp,"number_of_bands",n);
      utils::check(nbnd==n,base_error + " Incompatible nbnd:{} in h5::/Interaction.",n);
      // nkpts
      h5::h5_read_attribute(igrp,"number_of_kpoints",n);
      utils::check(nkpts==n,base_error + " Incompatible nkpts:{} in h5::/Interaction.",n);
      // nqpts
      h5::h5_read_attribute(igrp,"number_of_qpoints",n);
      utils::check(nkpts==n,base_error + " Incompatible nqpts:{} in h5::/Interaction.",n);
      // now read dimensions
      auto lX = h5::array_interface::get_dataset_info(igrp,"collocation_matrix");
      nu = lX.lengths[3];
      utils::check((lX.lengths[0]==nspin_in_file) and (lX.lengths[1]==nkpts) and 
                   (lX.lengths[2]==nbnd),
                   base_error + "Incompatible dimensions of collocation_matrix.");
      auto lL = h5::array_interface::get_dataset_info(igrp,"factorized_coulomb_matrix");
      nv = lL.lengths[2];
      utils::check((lL.lengths[0]==nqpts_ibz) and (lL.lengths[1]==nu), 
                   base_error + "Incompatible dimensions of factorized_coulomb_matrix.");
      if(igrp.has_key("collocation_matrix_half_rotated")) {
        have_rot_coul = 1;
        auto lXr = h5::array_interface::get_dataset_info(igrp,"collocation_matrix_half_rotated");
        utils::check((lXr.lengths[0]==nspin_in_file) and (lXr.lengths[1]==nkpts) and 
                     (lXr.lengths[2]==nbnd),
                      base_error + "Incompatible dimensions of collocation_matrix_half_rotated.");
        nu_rot = lXr.lengths[3];
        auto lZr = h5::array_interface::get_dataset_info(igrp,"half_rotated_coulomb_matrix");
        nv_rot = lZr.lengths[2];
        utils::check((lZr.lengths[0]==nqpts_ibz) and (lZr.lengths[1]==nu_rot),
                     base_error + "Incompatible dimensions of half_rotated_coulomb_matrix.");
      } else {
        have_rot_coul = 0;
        nu_rot = nu;
        nv_rot = nv;
        auto lZ = h5::array_interface::get_dataset_info(igrp,"coulomb_matrix");
        utils::check((lZ.lengths[0]==nqpts_ibz) and (lZ.lengths[1]==nu) and (lZ.lengths[2]==nu), 
                     base_error + "Incompatible dimensions of coulomb_matrix.");
        
      }
    }
  }
  mpi->comm.broadcast_value(nspin_in_file);
  mpi->comm.broadcast_value(npol_in_file);
  mpi->comm.broadcast_value(nu);
  mpi->comm.broadcast_value(nv);
  mpi->comm.broadcast_value(nu_rot);
  mpi->comm.broadcast_value(nv_rot);
  mpi->comm.broadcast_value(have_rot_coul);
  mpi->comm.broadcast_value(nbnd);
  mpi->comm.broadcast_value(nkpts);
  mpi->comm.broadcast_value(nkpts_ibz);
  mpi->comm.broadcast_value(nqpts_ibz);
  mpi->comm.broadcast_value(Q0_index);
  if(not mpi->comm.root()) {
    minusq.resize(nkpts);
    qk_to_k2.resize(nkpts,nkpts);
  }
  mpi->broadcast(minusq);
  mpi->broadcast(qk_to_k2);

  utils::check(nqpts_ibz==nkpts, "Error: Symmetry not yet implemented.");

  // allocate and read H1. 
  // H0: /System/H0:   [nspin][nkpts][npol*nbnd][npol*nbnd]
  auto H1 = memory::make_shared_array<HOST_MEMORY,ComplexType,4>(mpi,
                      {nspin_in_file,nkpts,npol_in_file*nbnd,npol_in_file*nbnd});
  // X: /Interaction/collocation_matrix:  [nspins,nkpts,npol*nbnd,Nu]
  // L: /Interaction/factorized_coulomb_matrix:  [nqpts][Nu][Nv] 
  // Z: /Interaction/coulomb_matrix:  [Nq][Nu][Nv]  
  // with half rotation:
  // Xrot: /Interaction/collocation_matrix_half_rotated:  [nspins,nkpts,nbnd,Nu]
  // Zrot: /Interaction/half_rotated_coulomb_matrix:      [Nq][Nu][Nv]
  auto Xsiu = memory::make_shared_array<MEM,ComplexType,4>(mpi,
                      {nspin_in_file,nkpts,npol_in_file*nbnd,nu});
  auto Luv = memory::make_shared_array<MEM,ComplexType,3>(mpi,{nqpts_ibz,nu,nv});
  std::optional<decltype(Luv)> Zuv = std::nullopt;
  // half-rotated 
  std::optional<decltype(Xsiu)> Xsiu_rot = std::nullopt;
  std::optional<decltype(Luv)> Zuv_rot = std::nullopt;
  if(have_rot_coul) {
    Xsiu_rot = std::make_optional(memory::make_shared_array<MEM,ComplexType,4>(mpi,
                      {nspin_in_file,nkpts,npol_in_file*nbnd,nu_rot}));
    Zuv_rot = std::make_optional(memory::make_shared_array<MEM,ComplexType,3>(mpi,{nqpts_ibz,nu_rot,nu_rot}));
  } else {
    Zuv = std::make_optional(memory::make_shared_array<MEM,ComplexType,3>(mpi,{nqpts_ibz,nu,nu}));
  }

  if (mpi->comm.root())
  {
    h5::group grp = h5::group(file);
    h5::group hgrp = grp.open_group("System");
    utils::h5_read(hgrp,"H0",H1());
    {  // Interaction 
      h5::group igrp = grp.open_group("Interaction");
      utils::h5_read(igrp,"collocation_matrix",Xsiu());
      utils::h5_read(igrp,"factorized_coulomb_matrix",Luv());
      if(have_rot_coul) {
        utils::h5_read(igrp,"collocation_matrix_half_rotated",(*Xsiu_rot)());
        utils::h5_read(igrp,"half_rotated_coulomb_matrix",(*Zuv_rot)());
      } else {
        utils::h5_read(igrp,"coulomb_matrix",(*Zuv)());
      }
    }  // Interaction 
  }

  // broadcast, careful with shared memory
  if(mpi->node_comm.root()) 
    mpi->internode_comm.broadcast_n(H1.data(),H1.size(),0);
  if constexpr (MEM==HOST_MEMORY) {
    if(mpi->node_comm.root()) {
      mpi->internode_comm.broadcast_n(Xsiu.data(),Xsiu.size(),0);
      mpi->internode_comm.broadcast_n(Luv.data(),Luv.size(),0);
      if(have_rot_coul) {
        mpi->internode_comm.broadcast_n(Zuv_rot->data(),Zuv_rot->size(),0);
        mpi->internode_comm.broadcast_n(Xsiu_rot->data(),Xsiu_rot->size(),0);
      } else {
        mpi->internode_comm.broadcast_n(Zuv->data(),Zuv->size(),0);
      }
    }
  } else {
    mpi->broadcast(Xsiu);
    mpi->broadcast(Luv);
    if(have_rot_coul) {
      mpi->broadcast(*Zuv_rot);
      mpi->broadcast(*Xsiu_rot);
    } else {
      mpi->broadcast(*Zuv);
    }
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

  // Y = PsiT*conj(X): (since PsiT is already conjugated/transposed) 
  auto Ydsau = memory::make_shared_array<MEM,ComplexType,6>(mpi,
                      {ndet,nspin,npol,nkpts,nocc_max,nu});
  if constexpr (MEM == HOST_MEMORY) {
    if(mpi->node_comm.root()) Ydsau() = ComplexType(0.0);
  } else {
    Ydsau() = ComplexType(0.0);
  }
  std::optional<decltype(Ydsau)> Ydsau_rot = std::nullopt;
  mpi->comm.barrier();
  
  for(long id=0, itot=0; id<ndet; ++id) 
    for(long is=0; is<nspin; ++is) 
      for(long ik=0; ik<nkpts; ++ik, ++itot) 
      {
        if( itot%mpi->comm.size() != mpi->comm.rank() ) continue; 
        using matrix_t = memory::buffered_array<MEM,ComplexType,2>;
        long is_ = is%nspin_in_file;
        int n0 = ( ik==0 ? 0 : nda::sum(nocc(is,range(ik))) );
        int nel = nocc(is,ik); 
        for(int ip=0; ip<npol; ++ip) { 
          auto Aai = math::sparse::to_array<'N'>(PsiT(id,is%nspin_in_PsiT),range(n0,n0+nel),range(ip*NMO+ik*nbnd,ip*NMO+(ik+1)*nbnd));
          auto Yau = Ydsau()(id,is,ip,ik,range(nel),all);
          long ip_ = ip%npol_in_file;
          auto Xiu = Xsiu()(is_,ik,range(ip_*nbnd,(ip_+1)*nbnd),all);
          nda::tensor::contract(Aai,"ai", nda::conj(Xiu),"iu",Yau,"au");
        }
      }
  mpi->comm.barrier();
  if constexpr (MEM==HOST_MEMORY) {
    if(mpi->node_comm.root()) 
      mpi->internode_comm.all_reduce_in_place_n(Ydsau.data(),Ydsau.size(),std::plus<>{});
  } else {
    mpi->comm.all_reduce_in_place_n(Ydsau.data(),Ydsau.size(),std::plus<>{});
  }
  mpi->comm.barrier();
  // what to do with Ydsau_rot???

  // now calculate v0
  // Partition work over spin, pol, kpts. 
  // v0(s,k,i,l) = -0.5*sum_k,q sum_j <i_k,j_k-q|j_k-q,l_k>
  //         = -0.5 sum_kq sum_j,u,v ma::conj(Piu(s,k,i,u)) ma::conj(Piu(s,k-q,j,v)) Mquv Piu(s,k-q,j,u) Piu(s,k,l,v)
  auto v0 = memory::make_shared_array<HOST_MEMORY,ComplexType,4>(mpi,std::array<long,4>{nspin_in_file*npol_in_file, nkpts, nbnd, nbnd});
  if(mpi->node_comm.root()) v0() = ComplexType(0.0);
  mpi->comm.barrier();
  if(have_rot_coul) {
    utils::check(false, "Finish");
  } else {
    {
      memory::buffered_array<MEM,ComplexType,2> Tuv(nu,nu);
      memory::buffered_array<MEM,ComplexType,2> Wuv(nu,nu);
      memory::buffered_array<MEM,ComplexType,2> vt(nbnd,nbnd);
      for(long is=0, isp=0, itot=0; is<nspin_in_file; is++) {
        for(long ip=0; ip<npol_in_file; ip++, isp++) {
          for(long ik=0; ik<nkpts; ik++, ++itot) {
            if( itot%mpi->comm.size() != mpi->comm.rank() ) continue; 
            // sum over q
            Tuv() = ComplexType(0.0);
            Wuv() = ComplexType(0.0);
            for(long iq=0; iq<nkpts; iq++) {
              int k2 = 0; // qp_to_k2(ik,iq);  // no symmetry yet!!!
              auto Xiu_k2 = Xsiu()(is,k2,range(ip*nbnd,(ip+1)*nbnd),all);
              auto Zu = (*Zuv)()(iq,all,all);
              nda::tensor::contract(Xiu_k2, "iu", nda::conj(Xiu_k2), "iv", Tuv, "uv");
              if constexpr (MEM==HOST_MEMORY) 
                Tuv() = Zu() * Tuv();
              else
                nda::tensor::elementwise(ComplexType(1.0), Zu, "uv", 
                                         ComplexType(1.0), Tuv, "uv", nda::tensor::op::MUL);
              nda::tensor::add(ComplexType(1.0),Tuv,"uv",ComplexType(1.0),Wuv,"uv"); 
            } 
            auto Xiu = Xsiu()(is,ik,range(ip*nbnd,(ip+1)*nbnd),all);
            auto Tiv = Tuv(range(nbnd),all);
            nda::tensor::contract(nda::conj(Xiu), "iu", Wuv, "uv", Tiv, "iv");
            nda::tensor::contract(ComplexType(-0.5),Tiv, "iv", Xiu, "jv", 
                                  ComplexType(0.0),vt, "ij");
            v0()(isp,ik,all,all) = vt(); 
          }
        }
      }
    } 
    mpi->comm.barrier();
    if(mpi->node_comm.root()) 
      mpi->internode_comm.all_reduce_in_place_n(v0.data(),v0.size(),std::plus<>{}); 
  }
  mpi->comm.barrier();

  // keep in full basis, better for batched
  // MAM: this uses a lot more memory/compute, but can be batched into a single call (energy eval)
  // time the two versions and change to k-dependent haj if the timing in gpu is not that different
  // You can also write a kernel that dispatches all the gemms from device side using the new library
  long nel[] = {nel_up, (type == COLLINEAR ? nel_dn : 0l) }; 
  auto haj = memory::make_shared_array<MEM,ComplexType,3>(mpi,std::array<long,3>{ndet, nel[0]+nel[1], npol*NMO});
  if constexpr (MEM == HOST_MEMORY) {
    if(mpi->node_comm.root()) haj() = ComplexType(0.0);
  } else {
    haj() = ComplexType(0.0);
  }
  mpi->node_comm.barrier();
  {
    for(long id=0, itot=0; id<ndet; ++id) {
      for(long is=0; is<nspin; ++is) {
        for(long ik=0; ik<nkpts; ++ik, ++itot) {
          if( itot%mpi->comm.size() != mpi->comm.rank() ) continue; 
          int n0 = ( ik==0 ? 0 : nda::sum(nocc(is,range(ik))) );
          int nk = nocc(is,ik); 
          for(long ip1=0; ip1<npol; ++ip1) {
            int ip1_ = ip1%npol_in_file;
            auto Aai = math::sparse::to_array<'N'>(PsiT(id,is%nspin_in_PsiT),range(n0,n0+nk),range(ip1*NMO+ik*nbnd,ip1*NMO+(ik+1)*nbnd));
            for(long ip2=0; ip2<npol; ++ip2) {
              int ip2_ = ip2%npol_in_file;
              auto h_ = haj()(id,range(is*nel_up+n0,is*nel_up+n0+nk),nda::range(ip2*NMO+ik*nbnd,ip2*NMO+(ik+1)*nbnd));
              if constexpr (MEM==HOST_MEMORY) {
                auto hij = H1()(is%nspin_in_file,ik,range(ip1_*nbnd,(ip1_+1)*nbnd),range(ip2_*nbnd,(ip2_+1)*nbnd));
                nda::blas::gemm(ComplexType(1.0),Aai,hij,ComplexType(1.0),h_);
              } else {
                memory::array<MEM,ComplexType,2> hij(H1()(is%nspin_in_file,ik,range(ip1_*nbnd,(ip1_+1)*nbnd),range(ip2_*nbnd,(ip2_+1)*nbnd)));
                nda::blas::gemm(ComplexType(1.0),Aai,hij,ComplexType(1.0),h_);
              }
            }
          }
        }
      }  // is
    }  // id
  }
  mpi->comm.barrier();
  if constexpr (MEM==HOST_MEMORY) { 
    if(mpi->node_comm.root()) mpi->internode_comm.all_reduce_in_place_n(haj.data(),haj.size(),std::plus<>{}); 
  } else {
    mpi->all_reduce(haj(),std::plus<>{}); 
  }
  mpi->comm.barrier();

  ComplexType E0 = NuclearCoulombEnergy + FrozenCoreEnergy;
  return HamiltonianOperations<MEM>(KPTHCOps<MEM>(mpi,type,NMO,nel_up,nel_dn,nkpts,Q0_index,std::move(nocc),
     std::move(minusq),std::move(qk_to_k2),std::move(H1),std::move(haj),std::move(Xsiu),
     std::move(Ydsau),std::move(Luv),std::move(Zuv),std::move(Xsiu_rot),std::move(Ydsau_rot),
     std::move(Zuv_rot),std::move(v0),E0)); 
}

template HamiltonianOperations<HOST_MEMORY> 
  KPTHCHamiltonian::getHamiltonianOperations<HOST_MEMORY>(WALKER_TYPES, 
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>, 
     nda::array<PsiT_Matrix<HOST_MEMORY>,2>const&);
#if defined(ENABLE_DEVICE)
template HamiltonianOperations<DEVICE_MEMORY> 
  KPTHCHamiltonian::getHamiltonianOperations<DEVICE_MEMORY>(WALKER_TYPES, 
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>, 
     nda::array<PsiT_Matrix<DEVICE_MEMORY>,2>const&);
#endif

} // namespace afqmc
} // namespace sfqmc
