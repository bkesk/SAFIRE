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

#pragma once

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check_strides.hpp"
#include "numerics/shared_array/shared_array.hpp"
#include "numerics/nda_functions.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class KP3IndexFactorization
{
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:
  static const HamiltonianTypes HamOpType = KPFactorized;
  HamiltonianTypes getHamType() const { return HamOpType; }

  // since arrays can be in host, can't assume that types are consistent
  KP3IndexFactorization(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
          WALKER_TYPES type,
          int nbnd_,
          int q0,
          nda::array<int,2>&& nocc_,
          nda::array<int,1>&& minusq_,
          nda::array<int,2>&& qk_to_k2_,
          nda::array<int,1>&& qmap_,
          memory::shared_array<HOST_MEMORY,ComplexType,4>&& hij_,
          memory::shared_array<MEM,ComplexType,3>&& haj_,
          nda::array<memory::shared_array<MEM,ComplexType,6>,1>&& lq_,
          nda::array<memory::shared_array<MEM,ComplexType,6>,1>&& la_,
          nda::array<memory::shared_array<MEM,ComplexType,6>,1>&& lb_,
          memory::shared_array<HOST_MEMORY,ComplexType,4>&& vexx_,
          ComplexType e0_,
          int bf_size = 4096)
      : mpi(_mpi),
        walker_type(type),
        nkpts(nocc_.extent(1)),
        nbnd(nbnd_),
        nup(nda::sum(nocc_(0,nda::range::all))),
        ndown(walker_type==COLLINEAR ? nda::sum(nocc_(1,nda::range::all)) : 0),
        nocc_max(nda::max_element(nocc_)),
        nchol_max(0),
        nocc(std::move(nocc_)),
        minusq(std::move(minusq_)),
        qk_to_k2(std::move(qk_to_k2_)),
        Qmap(std::move(qmap_)),
        ncv0(nkpts),
        Q0_index(q0),
        hij(std::move(hij_)),
        haj(std::move(haj_)),
        LQ(std::move(lq_)),
        Lank(std::move(la_)),
        Lbnk(std::move(lb_)),
        vexx(std::move(vexx_)),
        default_buffer_size_in_MB(bf_size),
        E0(e0_)
  {
    auto all = nda::range::all;
    // setup
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO = nkpts*nbnd;  
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;
    int ndet = haj.extent(0);
    
    // check sizes
    utils::check( hij.shape()==std::array<long,4>{nstot,nkpts,nptot*nbnd,nptot*nbnd},
      "KP3IndexFactorization: Size mismatch.");
    utils::check( haj.shape()==std::array<long,3>{ndet,nup+ndown,npol*NMO},
      "KP3IndexFactorization: Size mismatch.");
    utils::check( vexx.shape()==std::array<long,4>{nstot*nptot,nkpts,nbnd,nbnd},
      "KP3IndexFactorization: Size mismatch.");
    utils::check( LQ.extent(0)==nkpts, "KP3IndexFactorization: Size mismatch.");
    utils::check( Lank.extent(0)==nkpts, "KP3IndexFactorization: Size mismatch.");
    for(int iq=0; iq<nkpts; ++iq) {
      if(iq<=minusq(iq)) {
        int nc = LQ(iq).extent(5); 
        utils::check( LQ(iq).shape()==std::array<long,6>{nstot,nptot,nkpts,nbnd,nbnd,nc},
          "KP3IndexFactorization: Size mismatch.");
      } 
      {
        int nc = Lank(iq).extent(4); 
        utils::check(Lank(iq).shape()==std::array<long,6>{ndet,nspin,nkpts,nocc_max,nc,npol*nbnd},
            "KP3IndexFactorization: Size mismatch.");
      }
      if(iq==minusq(iq)) {
        int nc = Lbnk(Qmap(iq)).extent(4); 
        utils::check( Lbnk(Qmap(iq)).shape()==std::array<long,6>{ndet,nspin,nkpts,nocc_max,nc,npol*nbnd},
            "KP3IndexFactorization: Size mismatch.");
      } 
    } // iq

    ncvecs = 0;
    for(int iq=0; iq<nkpts; ++iq)        
    { 
      ncv0(iq) = ncvecs;
      ncvecs += LQ(std::min(iq,minusq(iq))).extent(5);
      nchol_max = std::max(nchol_max,int(LQ(std::min(iq,minusq(iq))).extent(5)));
    }
  }

  ~KP3IndexFactorization() = default;

  KP3IndexFactorization(const KP3IndexFactorization& other) = default;
  KP3IndexFactorization& operator=(const KP3IndexFactorization& other) = default;
  KP3IndexFactorization(KP3IndexFactorization&& other)                 = default;
  KP3IndexFactorization& operator=(KP3IndexFactorization&& other) = default;

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    using nda::range;
    auto all = range::all;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO = nkpts*nbnd;
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;

    nda::array<ComplexType, 3> H1(nspin, npol*NMO, npol*NMO);
    H1() = ComplexType(0.0);

    // v[nstot][nwalk=1][nptot*NMO][NMO]
    nda::array<ComplexType, 4> v;
    {
      memory::buffered_array<MEM,ComplexType,2> vMF_2d(1,vMF.size());
      vMF_2d(0,all) = vMF();
      v = std::move(nda::to_host(vHS(vMF_2d, dt)));
    }

    //
    for (int is = 0; is < nspin; is++) {
      int is_ = is%nstot;
      for (int p1 = 0; p1 < npol; p1++) {
        int p1_ = p1%nptot;
        // vHS finite 'q' contributions (full NMO*NMO) 
//        for (int I = 0; I < NMO; I++)
//          for (int J = 0 ; J < NMO; J++)
//              H1(is,p1*NMO+I,p1*NMO+J) += v(0,is_,p1_*NMO+I,J);

        // hij and vexx only have q=0 contributions  
        for (int p2 = 0; p2 < npol; p2++) {
          int p2_ = p2%nptot;
          for(int ik=0, i0=0; ik<nkpts; ik++, i0+=nbnd) {
            for (int i = 0; i < nbnd; i++) {
              for (int j = 0 ; j < nbnd; j++) {
                if(p1==p2) {
                  H1(is,p1*NMO+i0+i,p2*NMO+i0+j) +=
                     dt * (hij()(is_,ik,p1_*NMO+i,p2_*NMO+j) + vexx()(is_*nptot+p1_,ik,i,j));
                } else {
                  // only spin-orbit terms here coming from hij
                  H1(is,p1*NMO+i0+i,p2*NMO+i0+j) += dt * hij()(is_,ik,p1_*NMO+i,p2_*NMO+j);
                }
              }
            }
          }
        }
      }
    }

    // now hermitize and check
    long cnt = 0;
    for (int is = 0; is < nspin; is++) {
      for (int i = 0; i < npol*NMO; i++) {
        for (int j = i+1 ; j < npol*NMO; j++)
        {
          if(cnt <= 10 and (std::abs(H1(is,i,j) - std::conj(H1(is,j,i))) > 1e-5 )) {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 is not hermitian: ispin:{},i:{},j:{},H1(is,i,j):{},H1(is,j,i):{} ",is,i,j,H1(is,i,j),H1(is,j,i));
            if(cnt==10) app_warning("Suppressing further warnings!");
          }
          H1(is,i,j) = 0.5 * (H1(is,i,j) + std::conj(H1(is,j,i)));
          H1(is,j,i) = std::conj(H1(is,i,j));
        }
      }
    }
    return H1;
  }

  void runtime_optimization(nda::MemoryArrayOfRank<2> auto const& G)
  {  /*nothing to do right now*/ }

  nda::array<int,1> getFieldTypes() const {
    int nvc = number_of_cholesky_vectors();
    nda::array<int,1> v(nvc, int(ContinuousChargePropagator));
    return v;
  }

  // nothing to update 
  template<class... Args> void update_potentials([[maybe_unused]] Args&&... args) {}

  void energy(nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              int idet,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    memory::check_memory_space<MEM>(E,G);
    using nda::range;
    auto all = range::all;
    ComplexType one(1.0), zero(0.0);
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nup = nda::sum(nocc(0,all));
    int ndown = (walker_type==COLLINEAR ? nda::sum(nocc(1,all)) : 0);
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    int nqpts = LQ.extent(0);
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "Size mismatch.");
    utils::check(G.extent(1) == nel*npol*nkpts*nbnd, "Size mismatch.");
    utils::check(idet>=0 and idet<haj.extent(0), "Invalid idet:{}",idet);
    utils::check_strides(E,G);
    ComplexType scl = (walker_type == CLOSED ? ComplexType(2.0) : ComplexType(1.0));

    // addH1
    E() = ComplexType(0.0);
    if (addH1)
    {
      E(all,0) = E0;
      auto haj_2d = nda::reshape(haj(),std::array<long,2>{haj.extent(0),haj.extent(1)*haj.extent(2)});
      nda::tensor::contract(scl, G, "wi", haj_2d(idet,all), "i", one, E(all,0), "w");
    }
    if (not(addEJ || addEXX))
      return;

    // Does the CPU benefit from batch_size>1???
    int batch_size = nkpts*nkpts;
    if (addEXX)
    {
      long Bytes = long(default_buffer_size_in_MB) * 1024L * 1024L;
      Bytes /= size_t(nwalk * nocc_max * nocc_max * nchol_max * sizeof(ComplexType));
      int bz0 = std::max(1, int(Bytes));
      // batch_size includes the factor of 2 from Q/Qm pair
      batch_size = std::min(bz0, (nkpts * nkpts));
    }
batch_size=10;

    memory::buffered_array<MEM,ComplexType,2> Kleft((addEJ?nwalk:0),ncvecs);
    memory::buffered_array<MEM,ComplexType,2> Kright((addEJ?nwalk:0),ncvecs);
    if(addEJ) Kright() = zero;
    if(addEJ) Kleft() = zero;

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());

    // G in new layout
    memory::buffered_array<MEM,ComplexType,4> GKK(nkpts,nkpts,nwalk*nocc_max,npol*nbnd);

    // work array 
    memory::buffered_array<MEM,ComplexType,1> T1buff(batch_size*nwalk*nocc_max*nocc_max*nchol_max);
    memory::buffered_array<MEM,ComplexType,1> T2buff(batch_size*nwalk*nocc_max*nocc_max*nchol_max);
    T1buff() = zero;
    T2buff() = zero;

    if (addEXX)
    {
      int batch_cnt(0);
      // Lank(Q)(idet,ispin,nkpts, nocc_max, nchol, npol*nbnd)

      using A_t = decltype(nda::transpose(Lank(0)()(0,0,0,0,all,all))); 
      using B_t = decltype(GKK(0,0,all,all));
      std::vector<A_t> Av;
      std::vector<B_t> Bv;
      std::vector<B_t> Cv;
      Av.reserve(2*batch_size);
      Bv.reserve(2*batch_size);
      Cv.reserve(2*batch_size);

      // keeps track of diagonal terms, e.g. Ka==Kb
      std::vector<int> kdiag;
      kdiag.reserve(batch_size);

      for (int is  = 0; is  < nspin; ++is) {

        // reorder G
        Gc_to_GKKwaj(is,is*nup,G5d,GKK);
      
        for (int Q = 0; Q < nqpts; ++Q)
        {
          int Qm    = minusq(Q);
          int nchol = Lank(Q).extent(4);
          // Lank(Q)(ndet, nspin, nkpts, nocc_max, nchol, npol*nbnd)
          auto La = nda::reshape(Lank(Q)()(idet,is,nda::ellipsis{}),std::array<long,3>{nkpts,nocc_max*nchol,npol*nbnd});
          auto LB = ( Q==Qm ? Lbnk(Qmap(Q))() : Lank(Qm)() ); 
          auto Lb = nda::reshape(LB(idet,is,nda::ellipsis{}),std::array<long,3>{nkpts,nocc_max*nchol,npol*nbnd}); 

          auto Tl3d = nda::reshape(T1buff,std::array<long,3>{batch_size,nwalk*nocc_max,nocc_max*nchol});   
          auto Tl5d = nda::reshape(T1buff,std::array<long,5>{batch_size,nwalk,nocc_max,nocc_max,nchol});   
          auto Tr3d = nda::reshape(T2buff,std::array<long,3>{batch_size,nwalk*nocc_max,nocc_max*nchol});   
          auto Tr5d = nda::reshape(T2buff,std::array<long,5>{batch_size,nwalk,nocc_max,nocc_max,nchol});   

          // simple implementation for now
          Av.clear();
          Bv.clear();
          Cv.clear();
          kdiag.clear();
          batch_cnt = 0;

          for (int Ka = 0; Ka < nkpts; ++Ka)
          {
            int K0 = (Q==Qm) ? 0 : Ka;
            for (int Kb = K0; Kb < nkpts; ++Kb)
            {
              int Kl = qk_to_k2(Qm,Kb);
              int Kk  = qk_to_k2(Q,Ka);
              if( Ka == Kb ) kdiag.emplace_back(batch_cnt);

              // L[Kb][Kl] * G [Ka][Kl]
              // L(nkpts, nocc_max*nchol, nbnd)
              Av.emplace_back(nda::transpose(Lb(Kb,all,all)));
              // GKK(nkpts,nkpts,nwalk*nocc_max,npol*nbnd)
              Bv.emplace_back(GKK(Ka,Kl,all,all));
              Cv.emplace_back(Tr3d(batch_cnt,all,all));

              Av.emplace_back(nda::transpose(La(Ka,all,all)));
              Bv.emplace_back(GKK(Kb,Kk,all,all));
              Cv.emplace_back(Tl3d(batch_cnt++,all,all));

              if (batch_cnt >= batch_size)
              {
                nda::blas::gemm_batch<false>(ComplexType(1.0),Bv,Av,ComplexType(0.0),Cv);
                
                // add factor of 0.5 and accumulate on Kleft/Kright if needed
                if constexpr (MEM==HOST_MEMORY) { 
                  for(auto i: kdiag) {
                    if(addEJ) {
                      // T5d(batch_size,nwalk,nocc_max,nocc_max,nchol)
                      for(int a=0; a<nocc_max; ++a) {
                        Kleft(all,range(ncv0(Q),ncv0(Q)+nchol)) += Tl5d(i,all,a,a,all);
                        Kright(all,range(ncv0(Q),ncv0(Q)+nchol)) += Tr5d(i,all,a,a,all);
                      }
                    }
                    // only scale if Q!=Qm, otherwise full tensor is scaled below 
                    if(Q!=Qm) nda::tensor::scale(ComplexType(0.5),Tl3d(i,all,all));
                  }
                } else {
                  // custom kernel will be faster 
                  utils::check(false,"finish"); // add kernel
                } 
                if(Q==Qm) nda::tensor::scale(ComplexType(0.5),Tl3d);

                nda::tensor::contract(-scl,Tl5d(range(batch_cnt),nda::ellipsis{}),"lwabn",
                           Tr5d(range(batch_cnt),nda::ellipsis{}),"lwban",one,E(all,1),"w");

                // reset
                Av.clear();
                Bv.clear();
                Cv.clear();
                kdiag.clear();
                batch_cnt = 0;
              }
            }
          }

          if (batch_cnt > 0)
          {
            nda::blas::gemm_batch<false>(ComplexType(1.0),Bv,Av,ComplexType(0.0),Cv);

            // add factor of 0.5 and accumulate on Kleft/Kright if needed
            if constexpr (MEM==HOST_MEMORY) {
              for(auto i: kdiag) {
                if(addEJ) {
                  // T5d(batch_size,nwalk,nocc_max,nocc_max,nchol)
                  for(int a=0; a<nocc_max; ++a) {
                    Kleft(all,range(ncv0(Q),ncv0(Q)+nchol)) += Tl5d(i,all,a,a,all);
                    Kright(all,range(ncv0(Q),ncv0(Q)+nchol)) += Tr5d(i,all,a,a,all);
                  }
                }
                // scale diagonal terms to get correct EXX below
                if(Q!=Qm) nda::tensor::scale(ComplexType(0.5),Tl3d(i,all,all));
              }
            } else {
              // custom kernel will be faster 
              utils::check(false,"finish"); // add kernel
            }
            if(Q==Qm) nda::tensor::scale(ComplexType(0.5),Tl3d);

            nda::tensor::contract(-scl,Tl5d(range(batch_cnt),nda::ellipsis{}),"lwabn",
                          Tr5d(range(batch_cnt),nda::ellipsis{}),"lwban",one,E(all,1),"w");

            // reset
            Av.clear();
            Bv.clear();
            Cv.clear();
            kdiag.clear();
            batch_cnt = 0;
          }
        } // Q
      } // is
    }
    if (addEJ)
    {
      utils::check(addEXX, "Error: addEJ without addEXX not yet implemented.");
      nda::tensor::contract(ComplexType(0.5*scl*scl),Kleft,"wn",Kright,"wn",zero,E(all,2),"w");
    }
  }

  void energy(SpinTypes spin_component,
              nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              int idet,
              nda::MemoryArrayOfRank<2> auto && EJn,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    memory::check_memory_space<MEM>(E,G);
    using nda::range;
    auto all = range::all;
    ComplexType one(1.0), zero(0.0);
    int is = (spin_component == Alpha ? 0 : 1);
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nup = nda::sum(nocc(0,all));
    int ndown = (walker_type==COLLINEAR ? nda::sum(nocc(1,all)) : 0);
    int nel  = (is==0 ? nup : ndown);
    int nqpts = LQ.extent(0);
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "Size mismatch.");
    utils::check(G.extent(1) == nel*npol*nkpts*nbnd, "Size mismatch.");
    utils::check(idet>=0 and idet<haj.extent(0), "Invalid idet:{}",idet);
    utils::check_strides(E,G);
    ComplexType scl = (walker_type == CLOSED ? ComplexType(2.0) : ComplexType(1.0));

    // addH1
    E() = ComplexType(0.0);
    if (addH1)
    {
      if(spin_component==Alpha) 
        E(all,0) = E0;
      auto haj_2d = nda::reshape(haj(),std::array<long,2>{haj.extent(0),haj.extent(1)*haj.extent(2)});
      long n0 = (spin_component==Alpha ? 0l : long(nup));
      long nx = npol*nkpts*nbnd;
      nda::tensor::contract(scl, G, "wi", haj_2d(idet,range(n0*nx,(n0+nel)*nx)), "i", 
                            one, E(all,0), "w");
    }
    if (not(addEJ || addEXX))
      return;

    // Does the CPU benefit from batch_size>1???
    int batch_size = nkpts*nkpts;
    if (addEXX)
    {
      long Bytes = long(default_buffer_size_in_MB) * 1024L * 1024L;
      Bytes /= size_t(nwalk * nocc_max * nocc_max * nchol_max * sizeof(ComplexType));
      int bz0 = std::max(1, int(Bytes));
      // batch_size includes the factor of 2 from Q/Qm pair
      batch_size = std::min(bz0, (nkpts * nkpts));
    }

    if(addEJ) { 
      utils::check(EJn.extent(0)==nwalk and EJn.extent(1)==2*ncvecs, "Size mismatch");
      EJn() = zero;
    }

    memory::buffered_array<MEM,ComplexType,2> Kleft((addEJ?nwalk:0),ncvecs);
    memory::buffered_array<MEM,ComplexType,2> Kright((addEJ?nwalk:0),ncvecs);
    if(addEJ) Kright() = zero;
    if(addEJ) Kleft() = zero;

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());

    // G in new layout
    memory::buffered_array<MEM,ComplexType,4> GKK(nkpts,nkpts,nwalk*nocc_max,npol*nbnd);

    // work array 
    memory::buffered_array<MEM,ComplexType,1> T1buff(batch_size*nwalk*nocc_max*nocc_max*nchol_max);
    memory::buffered_array<MEM,ComplexType,1> T2buff(batch_size*nwalk*nocc_max*nocc_max*nchol_max);
    T1buff() = zero;
    T2buff() = zero;

    if (addEXX)
    {
      int batch_cnt(0);
      // Lank(Q)(idet,ispin,nkpts, nocc_max, nchol, npol*nbnd)

      using A_t = decltype(nda::transpose(Lank(0)()(0,0,0,0,all,all))); 
      using B_t = decltype(GKK(0,0,all,all));
      std::vector<A_t> Av;
      std::vector<B_t> Bv;
      std::vector<B_t> Cv;
      Av.reserve(2*batch_size);
      Bv.reserve(2*batch_size);
      Cv.reserve(2*batch_size);

      // keeps track of diagonal terms, e.g. Ka==Kb
      std::vector<int> kdiag;
      kdiag.reserve(batch_size);

      // reorder G
      Gc_to_GKKwaj(is,0,G5d,GKK);
    
      for (int Q = 0; Q < nqpts; ++Q)
      {
        int Qm    = minusq(Q);
        int nchol = Lank(Q).extent(4);
        // Lank(Q)(ndet, nspin, nkpts, nocc_max, nchol, npol*nbnd)
        auto La = nda::reshape(Lank(Q)()(idet,is,nda::ellipsis{}),std::array<long,3>{nkpts,nocc_max*nchol,npol*nbnd});
        auto LB = ( Q==Qm ? Lbnk(Qmap(Q))() : Lank(Qm)() ); 
        auto Lb = nda::reshape(LB(idet,is,nda::ellipsis{}),std::array<long,3>{nkpts,nocc_max*nchol,npol*nbnd}); 

        auto Tl3d = nda::reshape(T1buff,std::array<long,3>{batch_size,nwalk*nocc_max,nocc_max*nchol});   
        auto Tl5d = nda::reshape(T1buff,std::array<long,5>{batch_size,nwalk,nocc_max,nocc_max,nchol});   
        auto Tr3d = nda::reshape(T2buff,std::array<long,3>{batch_size,nwalk*nocc_max,nocc_max*nchol});   
        auto Tr5d = nda::reshape(T2buff,std::array<long,5>{batch_size,nwalk,nocc_max,nocc_max,nchol});   

        // simple implementation for now
        Av.clear();
        Bv.clear();
        Cv.clear();
        kdiag.clear();
        batch_cnt = 0;

        for (int Ka = 0; Ka < nkpts; ++Ka)
        {
          int K0 = (Q==Qm) ? 0 : Ka;
          for (int Kb = K0; Kb < nkpts; ++Kb)
          {
            int Kl = qk_to_k2(Qm,Kb);
            int Kk  = qk_to_k2(Q,Ka);
            if( Ka == Kb ) kdiag.emplace_back(batch_cnt);

            // L[Kb][Kl] * G [Ka][Kl]
            // L(nkpts, nocc_max*nchol, nbnd)
            Av.emplace_back(nda::transpose(Lb(Kb,all,all)));
            // GKK(nkpts,nkpts,nwalk*nocc_max,npol*nbnd)
            Bv.emplace_back(GKK(Ka,Kl,all,all));
            Cv.emplace_back(Tr3d(batch_cnt,all,all));

            Av.emplace_back(nda::transpose(La(Ka,all,all)));
            Bv.emplace_back(GKK(Kb,Kk,all,all));
            Cv.emplace_back(Tl3d(batch_cnt++,all,all));

            if (batch_cnt >= batch_size)
            {
              nda::blas::gemm_batch<false>(ComplexType(1.0),Bv,Av,ComplexType(0.0),Cv);
              
              // add factor of 0.5 and accumulate on Kleft/Kright if needed
              if constexpr (MEM==HOST_MEMORY) { 
                for(auto i: kdiag) {
                  if(addEJ) {
                    // T5d(batch_size,nwalk,nocc_max,nocc_max,nchol)
                    for(int a=0; a<nocc_max; ++a) {
                      Kleft(all,range(ncv0(Q),ncv0(Q)+nchol)) += Tl5d(i,all,a,a,all);
                      Kright(all,range(ncv0(Q),ncv0(Q)+nchol)) += Tr5d(i,all,a,a,all);
                    }
                  }
                  // only scale if Q!=Qm, otherwise full tensor is scaled below 
                  if(Q!=Qm) nda::tensor::scale(ComplexType(0.5),Tl3d(i,all,all));
                }
              } else {
                // custom kernel will be faster 
                utils::check(false,"finish"); // add kernel
              } 
              if(Q==Qm) nda::tensor::scale(ComplexType(0.5),Tl3d);

              nda::tensor::contract(-scl,Tl5d(range(batch_cnt),nda::ellipsis{}),"lwabn",
                         Tr5d(range(batch_cnt),nda::ellipsis{}),"lwban",one,E(all,1),"w");

              // reset
              Av.clear();
              Bv.clear();
              Cv.clear();
              kdiag.clear();
              batch_cnt = 0;
            }
          }
        }

        if (batch_cnt >= batch_size)
        {
          nda::blas::gemm_batch<false>(ComplexType(1.0),Bv,Av,ComplexType(0.0),Cv);

          // add factor of 0.5 and accumulate on Kleft/Kright if needed
          if constexpr (MEM==HOST_MEMORY) {
            for(auto i: kdiag) {
              if(addEJ) {
                // T5d(batch_size,nwalk,nocc_max,nocc_max,nchol)
                for(int a=0; a<nocc_max; ++a) {
                  Kleft(all,range(ncv0(Q),ncv0(Q)+nchol)) += Tl5d(i,all,a,a,all);
                  Kright(all,range(ncv0(Q),ncv0(Q)+nchol)) += Tr5d(i,all,a,a,all);
                }
              }
              // scale diagonal terms to get correct EXX below
              if(Q!=Qm) nda::tensor::scale(ComplexType(0.5),Tl3d(i,all,all));
            }
          } else {
            // custom kernel will be faster 
            utils::check(false,"finish"); // add kernel
          }
          if(Q==Qm) nda::tensor::scale(ComplexType(0.5),Tl3d);

          nda::tensor::contract(-scl,Tl5d(range(batch_cnt),nda::ellipsis{}),"lwabn",
                        Tr5d(range(batch_cnt),nda::ellipsis{}),"lwban",one,E(all,1),"w");

          // reset
          Av.clear();
          Bv.clear();
          Cv.clear();
          kdiag.clear();
          batch_cnt = 0;
        }
      } // Q
    }

    if (addEJ)
    {
      utils::check(addEXX, "Error: addEJ without addEXX not yet implemented.");
      nda::tensor::contract(ComplexType(0.5),Kleft,"wn",Kright,"wn",zero,E(all,2),"w");
      if(spin_component == Alpha) {
        // store as Kl,Kr
        EJn(all,range(0,ncvecs)) = Kleft();
        EJn(all,range(ncvecs,2*ncvecs)) = Kright();
      } else {
        // store as Kr,Kl
        EJn(all,range(0,ncvecs)) = Kright();
        EJn(all,range(ncvecs,2*ncvecs)) = Kleft();
      }
      // gets multiplied by 2 in PHMSD, so undo it here!	
      nda::tensor::scale(ComplexType(1.0/std::sqrt(2.0)),EJn);
    }
  }

  template<class... Args>
  void ph_reference_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_reference_energy not implemented yet. ");
  }

  template<class... Args>
  void ph_excited_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_excited_energy not implemented yet. ");
  }

  auto vHS_sparse(nda::MemoryArrayOfRank<2> auto && X, double dt)
  {
    utils::check(false, "vHS_sparse not implemented in THCOps.");
    nda::array<csrMat<ComplexType>,1> spvHS;
    return spvHS();
  }

  auto vHS(nda::MemoryArrayOfRank<2> auto && X, double dt)
  {
    memory::check_memory_space<MEM>(X);
    using nda::range;
    auto all  = range::all;
    ComplexType one(1.0), zero(0.0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nwalk = X.extent(0);
    int NMO   = nkpts*nbnd;
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;

    // Note: Allocate first, to make better use of memory pool
    // vHS[nspin_in_vHS][nwalk][npol_in_vHS*NMO][NMO]
    memory::buffered_array<MEM,ComplexType,4> v(nwalk,nstot,nptot*NMO,NMO);
    auto v7d = nda::reshape(v,std::array<long,7>{nwalk,nstot,nptot,nkpts,nbnd,nkpts,nbnd});
    v() = ComplexType(0.0);

    auto X3d = nda::reshape(X,std::array<long,3>{nwalk,2,ncvecs});
    {
      // "rotate" X
      //  XIJ = 0.5*a*(Xn+ -i*Xn-), XJI = 0.5*a*(Xn+ +i*Xn-)
      RealType a = std::sqrt(dt)*0.5;
      memory::buffered_array<MEM,ComplexType,2> T(nwalk,ncvecs);
      T() = X3d(all,0,all);
      nda::tensor::add(ComplexType(0.0,-a),X3d(all,1,all),"wn",ComplexType(a),T,"wn");
      nda::tensor::add(ComplexType(a),X3d(all,0,all),"wn",ComplexType(0.0,a),X3d(all,1,all),"wn");
      X3d(all,0,all) = T();
      //  then combine Q/(-Q) pieces if Q != -Q
      //  X(Q)np = (X(Q)np + X(-Q)nm)
      for(int iq=0; iq<nkpts; iq++)  {
        int nchol = Lank(iq).extent(4);
        if( iq != minusq(iq) ) {
          nda::tensor::add(ComplexType(1.0),X3d(all,1,range(ncv0(iq),ncv0(iq)+nchol)),"wn",ComplexType(1.0),X3d(all,0,range(ncv0(iq),ncv0(iq)+nchol)),"wn");
        }
      }
    }

    memory::buffered_array<MEM,ComplexType,5> vKK(nwalk,npol,nkpts,nbnd,nbnd);
    for(int is=0; is<nstot; ++is) {

      for (int Q = 0; Q < nkpts; ++Q)
      { 
        int nchol = Lank(Q).extent(4);
        // v[nw][i(in K)][k(in Q(K))] += sum_n LQK[i][k][n] X[Q][0][n][nw]
        if (Q <= minusq(Q))
        {
          // ci^dagger cj term
          // LQ(Q)(ispin, ipol, ik, i, j, nchol) 
          nda::tensor::contract(one,LQ(Q)()(is,nda::ellipsis{}),"pkijn",
               X3d(all,0,range(ncv0(Q),ncv0(Q)+nchol)),"wn",zero,vKK,"wpkij");

          // accumulate on v7d(nwalk,nstot,nptot,nkpts,nbnd,nkpts,nbnd)
          // write kernel if this is too slow, or fix streams in cutensor!!!
          for(int ik=0; ik<nkpts; ++ik) {
            int k2 = qk_to_k2(Q,ik);
            nda::tensor::add(one,vKK(all,all,ik,all,all),"wpij",
                             one,v7d(all,is,all,ik,all,k2,all),"wpij");
          }
        } else {
          // cj^dagger ci term
          // the kpoint index of LQ(Qm) refers to k2 
          nda::tensor::contract(one,nda::conj(LQ(minusq(Q))()(is,nda::ellipsis{})),"pkijn",
               X3d(all,0,range(ncv0(Q),ncv0(Q)+nchol)),"wn",zero,vKK,"wpkij");

          // accumulate on v7d(nwalk,nstot,nptot,nkpts,nbnd,nkpts,nbnd)
          // write kernel if this is too slow, or fix streams in cutensor!!!
          for(int ik=0; ik<nkpts; ++ik) {
            int k2 = qk_to_k2(Q,ik);
            nda::tensor::add(one,vKK(all,all,k2,all,all),"wpji",
                             one,v7d(all,is,all,ik,all,k2,all),"wpij");
          }
        }
        // v[nw][k(in Q(K))][i(in K)] += sum_n conj(LQK[i][k][n]) X[Q][n-][nw]
        if(Q == minusq(Q)) {
          // cj^dagger ci term for Q==minusq(Q)
          // the kpoint index refers to k2 
          nda::tensor::contract(one,nda::conj(LQ(Q)()(is,nda::ellipsis{})),"pkijn",
               X3d(all,1,range(ncv0(Q),ncv0(Q)+nchol)),"wn",zero,vKK,"wpkij");

          // accumulate on v7d(nwalk,nstot,nptot,nkpts,nbnd,nkpts,nbnd)
          // write kernel if this is too slow, or fix streams in cutensor!!!
          for(int ik=0; ik<nkpts; ++ik) {
            int k2 = qk_to_k2(Q,ik);
// this needs checking !!!
            nda::tensor::add(one,vKK(all,all,ik,all,all),"wpij",
                             one,v7d(all,is,all,k2,all,ik,all),"wpji");
          }
        }

      } // Q
    } // is

    return v;
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    memory::check_memory_space<MEM>(G,v);
    using nda::range;
    auto all = range::all;
    ComplexType one(1.0), zero(0.0);
    int nwalk = G.extent(0);
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check_strides(G,v);
    utils::check(v.shape() == std::array<long,2>{nwalk,2*ncvecs}, "vbias: Size mismatch.");
    if(haj.extent(0) == 1) // ndet==1, G half rotated
      utils::check(G.extent(1) == nel*npol*nkpts*nbnd, "vbias: Size mismatch.");
    else // ndet>1, full G 
      utils::check(G.extent(1) == nspin*npol*nkpts*nbnd*npol*nkpts*nbnd, "vbias: Size mismatch.");
    // needed to build G3d easily, can build from strides if needed
    utils::check(G.is_contiguous(), "Layout mismatch");
    utils::check(v.is_contiguous(), "Layout mismatch");

    // scale a by sqrt(dt)
    RealType scl = (walker_type == CLOSED ? 2.0 : 1.0);
    ComplexType a = ComplexType(std::sqrt(dt)*0.5*scl);
    ComplexType ia = ComplexType(0.0,std::sqrt(dt)*0.5*scl);

    // v4d
    auto v3d = nda::reshape(v, std::array<long,3>{nwalk,2,ncvecs});
    v() = ComplexType(0.0);

    utils::check(G.is_contiguous(), "Layout mismatch");

    // G in new layout
    memory::buffered_array<MEM,ComplexType,5> GQK(nkpts,nwalk,nkpts,nocc_max,npol*nbnd);

    if (haj.extent(0) == 1)
    {
      memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());
      for(int is=0; is<nspin; ++is)  {

        // do I gain anything by doing all Q at the same time, uses more memory!
        Gc_to_GQKwaj(is,is*nup,G5d,GQK);
        for(int iq=0; iq<nkpts; ++iq) {

          int nchol = Lank(iq).extent(4);
          int iqm = minusq(iq);
          memory::buffered_array<MEM,ComplexType,2> vn(nwalk,nchol);
     
          nda::tensor::contract(one,Lank(iq)()(0,is,nda::ellipsis{}),"kanj",
                                    GQK(iq,nda::ellipsis{}),"wkaj",zero,vn,"wn");

          // accumulate on v3d
          // v+ = a*(v[Q]+v[-Q])
          nda::tensor::add(a,vn,"wn",one,v3d(all,0,range(ncv0(iq),ncv0(iq)+nchol)),"wn");
          // v- = -a*i*(v[Q]-v[-Q])
          nda::tensor::add(-ia,vn,"wn",one,v3d(all,1,range(ncv0(iq),ncv0(iq)+nchol)),"wn");

          if(iq == iqm) {
            nda::tensor::contract(one,Lbnk(Qmap(iq))()(0,is,nda::ellipsis{}),"kanj",
                                      GQK(iq,nda::ellipsis{}),"wkaj",zero,vn,"wn");
            // accumulate on v3d
            nda::tensor::add(a,vn,"wn",one,v3d(all,0,range(ncv0(iqm),ncv0(iqm)+nchol)),"wn");
            nda::tensor::add(ia,vn,"wn",one,v3d(all,1,range(ncv0(iqm),ncv0(iqm)+nchol)),"wn");
          } else {
            nda::tensor::add(a,vn,"wn",one,v3d(all,0,range(ncv0(iqm),ncv0(iqm)+nchol)),"wn");
            nda::tensor::add(ia,vn,"wn",one,v3d(all,1,range(ncv0(iqm),ncv0(iqm)+nchol)),"wn");
          }

        } // iq
      } // is  
    }
    else
    {
      utils::check(false," Error: KP3Index not yet implemented for multiple references.");
    }

    //vbias_from_v1(halfa, v1, v);
  }

  template<class... Args> void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

  /// Returns the number of spins and polarizations in the VHS potential.
  std::tuple<int,int> vHS_dims() const {
    return std::make_tuple(1,1);
  }
  int number_of_ke_vectors() const { return 2 * ncvecs; }
  int number_of_cholesky_vectors() const { return 2 * ncvecs; }

  nda::array<ComplexType, 2> getHSPotentials()
  { return nda::array<ComplexType, 2>{}; }

protected:
  // keeping communicators here seems unnecesary
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int nkpts=0;
  int nbnd=0;
  int nup=0;
  int ndown=0;
  int nocc_max=0;
  int nchol_max=0;
  int ncvecs=0;     // total number of cholesky vectors

  nda::array<int,2> nocc;
         
  // BZ information
  nda::array<int,1> minusq;
  nda::array<int,2> qk_to_k2;
  nda::array<int,1> Qmap;  // location of Q (Q==minusq(Q)) in Lbnk
  nda::array<int,1> ncv0;  // location of first cholesky vector for each Q
  int Q0_index = 0;

  // H1[nspin][nk][npol*nbnd][npol*nbnd]
  memory::shared_array<HOST_MEMORY,ComplexType,4> hij;

  // half rotated one body hamiltonian: [ndet][nup+ndn][npol*NMO]. Kept in full basis
  memory::shared_array<MEM,ComplexType,3> haj;

  // LQ(Q)(ispin, ipol, ik, i, j, nchol) 
  nda::array<memory::shared_array<MEM,ComplexType,6>,1> LQ; 

  // Lank(Q)(ndet, nspin, nkpts, nocc_max, nchol, npol*nbnd) 
  nda::array<memory::shared_array<MEM,ComplexType,6>,1> Lank; 

  // Lbnk(Qmap(Q))(ndet, nspin, nkpts, nocc_max, nchol, npol*nbnd), only for q==minusq(q) 
  nda::array<memory::shared_array<MEM,ComplexType,6>,1> Lbnk; 

  // vexx(i,l) = -0.5 * sum_j <ij|jl> : [nspin][nk][npol*nbnd][npol*nbnd]
  memory::shared_array<HOST_MEMORY,ComplexType,4> vexx;

  int default_buffer_size_in_MB=2000;

  ComplexType E0 = ComplexType(0.0);

  // maps Q (only for those with Qmap >=0) to the corresponding sector in vbias
//  stdIVector Q2vbias;

  // Changes the layout of G:
  //    From: G(nwalk,nel_tot,npol,nkpts,nbnd)
  //    To:   GKK(nkpts,nkpts,nwalk*nocc_max,npol*nbnd)
  void Gc_to_GKKwaj(int is, int n0, nda::MemoryArrayOfRank<5> auto const& G,
                        nda::MemoryArrayOfRank<4> auto && GKK)
  {
    using nda::range;
    auto all  = range::all;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int nwalk = G.extent(0); 
    int nup   = nda::sum(nocc(0,all));
    int ndown = (walker_type==COLLINEAR ? nda::sum(nocc(1,all)) : 0);
 
    auto G6d = nda::reshape(GKK,std::array<long,6>{nkpts,nkpts,nwalk,nocc_max,npol,nbnd});
    GKK() = ComplexType(0.0);
    if constexpr (MEM==HOST_MEMORY) {
      for(int ik1=0; ik1<nkpts; ++ik1) 
      {
        int nk1 = nocc(is,ik1);
        for(int ik2=0; ik2<nkpts; ++ik2) 
        { 
          for(int ip=0; ip<npol; ++ip) 
            for(int iw=0; iw<nwalk; ++iw) 
              G6d(ik1,ik2,iw,range(nk1),ip,all) = G(iw,range(n0,n0+nk1),ip,ik2,all);
        } 
        n0 += nk1;
      }
    } else {
      // figure out how to run cutensor calls concurrently, streams don't seem to do it
      for(int ik1=0; ik1<nkpts; ++ik1) 
      {
        int nk1 = nocc(is,ik1);
        nda::tensor::add(ComplexType(1.0),G(all,range(n0,n0+nk1),all,all,all),"wapkj",
                         ComplexType(0.0),G6d(ik1,all,all,range(nk1),all,all),"kwapj");
        n0 += nk1;
      }
    }
  }

  // Changes the layout of G:
  //    From: G(nwalk,nel_tot,npol,nkpts,nbnd)
  //    To:   GQK(nkpts,nwalk,nkpts,nocc_max,npol*nbnd)
  void Gc_to_GQKwaj(int is, int n0, nda::MemoryArrayOfRank<5> auto const& G,
                    nda::MemoryArrayOfRank<5> auto && GQK)
  {
    using nda::range;
    auto all  = range::all;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int nwalk = G.extent(0);
    int nup   = nda::sum(nocc(0,all));
    int ndown = (walker_type==COLLINEAR ? nda::sum(nocc(1,all)) : 0);

    auto G6d = nda::reshape(GQK,std::array<long,6>{nkpts,nkpts,nwalk,nocc_max,npol,nbnd});
    GQK() = ComplexType(0.0);
    if constexpr (MEM==HOST_MEMORY) {
      for(int iq=0; iq<nkpts; ++iq)
      {
        int nk0 = n0; 
        for(int ik=0; ik<nkpts; ++ik)
        {
          int k2 = qk_to_k2(iq,ik);
          int nk = nocc(is,ik);
          for(int ip=0; ip<npol; ++ip)
            for(int iw=0; iw<nwalk; ++iw)
              G6d(iq,iw,ik,range(nk),ip,all) = G(iw,range(nk0,nk0+nk),ip,k2,all);
          nk0 += nk;
        }
      }
    } else {
      // figure out how to run cutensor calls concurrently, streams don't seem to do it
      for(int iq=0; iq<nkpts; ++iq)
      {
        int nk0 = n0;
        for(int ik=0; ik<nkpts; ++ik)
        {
          int k2 = qk_to_k2(iq,ik);
          int nk = nocc(is,ik);
          nda::tensor::add(ComplexType(1.0),G(all,range(nk0,nk0+nk),all,k2,all),"wapj",
                           ComplexType(0.0),G6d(iq,all,ik,range(nk),all,all),"wapj");
          nk0 += nk;
        }
      }
    }
  }

};

} // namespace afqmc

} // namespace sfqmc

