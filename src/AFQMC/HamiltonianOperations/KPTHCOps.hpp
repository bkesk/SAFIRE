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

/*
 * v+(q,n) = 0.5 * ( L(q,n) + dagger(L(q,n)) )
 * v-(q,n) = i*0.5 * ( L(q,n) - dagger(L(q,n)) )
 * L(q,n) = sum_k sum_ab L^{k,q}_{ab,n} dagger( c^{k}_a ) c^{k-q}_b
 */


template<MEMORY_SPACE _MEM>
class KPTHCOps
{
  static constexpr MEMORY_SPACE MEM = _MEM;
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:
  static constexpr HamiltonianTypes HamOpType = KPTHC;
  constexpr HamiltonianTypes getHamType() const { return KPTHC; }

  /*
   * nup/ndown stands for number of alpha/beta electrons
   */
  KPTHCOps(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> ctxt,
         WALKER_TYPES type,
         int nmo_,
         int nup_, 
         int ndn_, 
         int nkpts_,
         int q0_,
         nda::array<int,2>&& nocc_per_kp_,
         nda::array<int,1>&& minusq_,
         nda::array<int,2>&& qk_to_k2_,
         memory::shared_array<HOST_MEMORY,ComplexType,4>&& hij_,
         memory::shared_array<MEM,ComplexType,3>&& haj_,
         memory::shared_array<MEM,ComplexType,4>&& x_,
         memory::shared_array<MEM,ComplexType,6>&& y_,
         memory::shared_array<MEM,ComplexType,3>&& l_,
         std::optional<memory::shared_array<MEM,ComplexType,3>>&& z_,
         std::optional<memory::shared_array<MEM,ComplexType,4>>&& x_rot_,
         std::optional<memory::shared_array<MEM,ComplexType,6>>&& y_rot_,
         std::optional<memory::shared_array<MEM,ComplexType,3>>&& z_rot_,
         memory::shared_array<HOST_MEMORY,ComplexType,4>&& v0_,
         ComplexType e0_)
      : mpi(ctxt), 
        walker_type(type),
        NMO(nmo_),
        nup(nup_),
        ndown(ndn_),
        nelec{nup, ndown},
        nkpts(nkpts_),
        nbnd(NMO/nkpts),
        nocc_per_kp(nocc_per_kp_),
        minusq(std::move(minusq_)),
        qk_to_k2(std::move(qk_to_k2_)),
        Q0_index(q0_),
        hij(std::move(hij_)),
        haj(std::move(haj_)),
        _Xsiu_(std::move(x_)),
        _Ydsau_(std::move(y_)),
        _Luv_(std::move(l_)),
        _Zuv_(std::move(z_)),
        _Xsiu_rot_(std::move(x_rot_)),
        _Ydsau_rot_(std::move(y_rot_)),
        _Zuv_rot_(std::move(z_rot_)),
        vexx(std::move(v0_)),
        E0(e0_)
  {
    // make sure dimensions are compatible
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nu = _Luv_.extent(1);
    int nv = _Luv_.extent(2);
    int nu_rot = (_Xsiu_rot_.has_value() ? _Xsiu_rot_->shape()[3] : nu);
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;
    int ndet = haj.extent(0);
    int nqpts_ibz = _Luv_.extent(0);
    auto nocc_max = nda::max_element(nocc_per_kp);
    utils::check(vexx.shape() == std::array<long,4>{nstot*nptot,nkpts,nbnd,nbnd},"KPTHCOps: Size mismatch"); 
    utils::check(hij.shape() == std::array<long,4>{nstot,nkpts,nptot*nbnd,nptot*nbnd},"KPTHCOps: Size mismatch"); 
    utils::check(haj.shape() == std::array<long,3>{ndet,nel,npol*NMO},"KPTHCOps: Size mismatch"); 
    utils::check(_Xsiu_.shape() == std::array<long,4>{nstot,nkpts,npol*nbnd,nu},"KPTHCOps: Size mismatch"); 
    utils::check(_Ydsau_.shape() == std::array<long,6>{ndet,nspin,npol,nkpts,nocc_max,nu},"KPTHCOps: Size mismatch"); 
    utils::check(_Luv_.shape() == std::array<long,3>{nqpts_ibz,nu,nv},"KPTHCOps: Size mismatch"); 
    if(_Zuv_.has_value())
      utils::check(_Zuv_->shape() == std::array<long,3>{nqpts_ibz,nu,nu},"KPTHCOps: Size mismatch"); 
    if(_Xsiu_rot_.has_value())
      utils::check(_Xsiu_rot_->shape() == std::array<long,4>{nstot,nkpts,npol*nbnd,nu_rot},"KPTHCOps: Size mismatch"); 
    if(_Ydsau_rot_.has_value())
      utils::check(_Ydsau_rot_->shape() == std::array<long,6>{ndet,nspin,npol,nkpts,nocc_max,nu_rot},"KPTHCOps: Size mismatch"); 
    if(_Zuv_rot_.has_value())
      utils::check(_Zuv_rot_->shape() == std::array<long,3>{nqpts_ibz,nu_rot,nu_rot},"KPTHCOps: Size mismatch"); 
  }

  ~KPTHCOps() = default; 

  KPTHCOps(KPTHCOps const& other) = default;
  KPTHCOps& operator=(KPTHCOps const& other) = default;

  KPTHCOps(KPTHCOps&& other) = default;
  KPTHCOps& operator=(KPTHCOps&& other) = default;

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    using nda::range;
    auto all = range::all;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
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
        for (int I = 0; I < NMO; I++) 
          for (int J = 0 ; J < NMO; J++) 
              H1(is,p1*NMO+I,p1*NMO+J) += v(0,is_,p1_*NMO+I,J); 

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
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    int nocc_max = _Ydsau_.extent(4);
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "THC::energy: Size mismatch.");
    utils::check(G.extent(1) == nel*npol*NMO, "THC::energy: Size mismatch.");
    utils::check_strides(E,G);
    RealType scl = (walker_type == CLOSED ? 2.0 : 1.0);

    // addH1
    E() = ComplexType(0.0);
    if (addH1)
    {
      E(all,0) = E0; 
      auto haj_2d = nda::reshape(haj(),std::array<long,2>{haj.extent(0),haj.extent(1)*haj.extent(2)});
      nda::tensor::contract(ComplexType(scl), G, "wi", haj_2d(idet,all), "i", 
                            ComplexType(1.0), E(all,0), "w");
    }
    if (not(addEJ || addEXX))
      return;

    // get array_views to the correct data and correct determinant
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_() );
    const auto Ysau = ( has_rot ? (*_Ydsau_rot_)()(idet,nda::ellipsis{}) : _Ydsau_()(idet,nda::ellipsis{}) );
    const auto Zuv = ( has_rot ? (*_Zuv_rot_)() : (*_Zuv_)() );

    int nu = Zuv.extent(1);
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;

    // calculate how many walkers can be done concurrently
//    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
//    Bytes /= long((nu * nu + nu + nu * nup) * sizeof(ComplexType));
//    int nwmax = std::min(nwalk, std::max(1, int(Bytes)));
    int nwmax = nwalk;

    utils::check(G.is_contiguous(), "Layout mismatch"); 
    memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());

    // separate GPU implementation if needed!
    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      // Guv[nspin][nu][nv]
      memory::buffered_array<MEM,ComplexType,3> Guv(nw,nu,nu);
      memory::buffered_array<MEM,ComplexType,3> Wuv(nw,nu,nu);
      // Guu: summed over spin, polarization and k
      memory::buffered_array<MEM,ComplexType,4> Guu(2,nw,nkpts,nu);
      memory::buffered_array<MEM,ComplexType,1> Tbuff(nw*nocc_max*nu);
      Guu() = ComplexType(0.0);
      for (int ispin = 0; ispin < nspin; ++ispin)
      {
        long is_ = long(ispin)%nstot; 
        for (int p1 = 0; p1 < npol; ++p1)
        {
          long ip1_ = long(p1)%nptot; 
          for (int p2 = 0; p2 < npol; ++p2)
          {
 
            for (int k1 = 0; k1 < nkpts; ++k1) {
            for (int k2 = 0; k2 < nkpts; ++k2) {

              // Buffer space
              get_Guv(ispin, p1, p2, k1, k2, G5d(range(iw, iw + nw), nda::ellipsis{}), 
                      Guv, Tbuff, idet);

              if(p1==p2) {
                //accumulate Guu
                // find q such that qk_to_k2(q,k1) = k2
                int iq=-1;
                for(int i=0; i<nkpts; i++)
                  if(qk_to_k2(i,k1)==k2) {
                    iq=i;
                    break;
                  }
                utils::check(iq>=0, "Error: Problems mapping {k1,k2} to q.");
                if constexpr (MEM==HOST_MEMORY) {
                  for(int i=0; i<nw; i++)
                    Guu(0,i,iq,all) += nda::diagonal(Guv(i,all,all));
                } else {
                  std::array<long,2> str = {Guv.strides()[0],Guv.strides()[1]+1};
                  nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm({nw,nu},str);
                  memory::array_view<MEM,ComplexType,2> Guv_diag(idxm, Guv.data());
                  nda::tensor::add(ComplexType(1.0),Guv_diag,ComplexType(1.0),Guu(0,all,iq,all));
                }
                // now right hand side, where need to find q such that qk_to_k2(q,k2) = k1
                iq=-1;
                for(int i=0; i<nkpts; i++)
                  if(qk_to_k2(i,k2)==k1) {
                    iq=i;
                    break;
                  }
                utils::check(iq>=0, "Error: Problems mapping {k2,k1} to q.");
                if constexpr (MEM==HOST_MEMORY) {
                  for(int i=0; i<nw; i++)
                    Guu(1,i,iq,all) += nda::diagonal(Guv(i,all,all));
                } else {
                  std::array<long,2> str = {Guv.strides()[0],Guv.strides()[1]+1};
                  nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm({nw,nu},str);
                  memory::array_view<MEM,ComplexType,2> Guv_diag(idxm, Guv.data());
                  nda::tensor::add(ComplexType(1.0),Guv_diag,ComplexType(1.0),Guu(1,all,iq,all));
                }
              }

              for (int iq = 0; iq < nkpts; ++iq) {
                int qk1 = qk_to_k2(iq,k1);
                int qk2 = qk_to_k2(iq,k2);
                int n0 = ( qk2==0 ? 0 : nda::sum(nocc_per_kp(ispin,range(qk2))) );
                int nel_qk2 = nocc_per_kp(ispin,qk2);
                memory::array_view<MEM,ComplexType,3> Tau(std::array<long,3>{nw,nel_qk2,nu},Tbuff.data());

                Wuv() = Guv();
                if constexpr (MEM==HOST_MEMORY) {
                  for(int i=0; i<nw; ++i)
                    Wuv(i,all,all) *= Zuv(iq,all,all);
                } else {
                  nda::tensor::elementwise(ComplexType(1.0),Zuv(iq,all,all),"uv",
                                           ComplexType(1.0),Wuv,"wuv",nda::tensor::op::MUL);
                }

                // R[w,u][b] = sum_v Guv[w,u][v] * Yau[b][v]
                auto Yau = Ysau(ispin,p2,qk2,range(nel_qk2),all);
                nda::tensor::contract(Yau,"av",Wuv,"wuv",Tau,"wau"); 

                // reuse Guv memory
                memory::array_view<MEM,ComplexType,3> Twbi(std::array<long,3>{nw,nel_qk2,nbnd},Wuv.data());
                //T[w][b][k] = sum_u R[w][u][b] * Xiu[k][u]
                auto Xiu = Xsiu(is_,qk1,range(ip1_*nbnd,(ip1_+1)*nbnd),all);
                nda::tensor::contract(Tau,"wau",Xiu,"iu",Twbi,"wai"); 

                // E[w] = sum_ai T[w][a][i] * G[w][a][i] 
                auto Gwai = G5d(range(iw, iw + nw),range(ispin*nup+n0,ispin*nup+n0+nel_qk2),p1,qk1,all); 
                memory::buffered_array<MEM,ComplexType,1> Ew(nw);
                nda::tensor::contract(ComplexType(-0.5*scl/double(nkpts)),Twbi,"wai",Gwai,"wai",ComplexType(0.0),Ew,"w"); 
                nda::tensor::add(ComplexType(1.0),Ew,ComplexType(1.0),E(range(iw, iw + nw), 1));

              } // iq
            } // k2
            } //k1
          } //p2
        } // p1
      } //is
      if (addEJ)
      {
        memory::buffered_array<MEM,ComplexType,3> Twu(nw,nkpts,nu);
        memory::buffered_array<MEM,ComplexType,1> Ew(nw);
        nda::tensor::contract(Guu(0,all,all,all),"wqu",Zuv,"quv",Twu,"wqv");
        nda::tensor::contract(ComplexType(RealType(0.5*scl*scl/nkpts)),Guu(1,all,all,all),"wqv",
                              Twu,"wqv", ComplexType(0.0),Ew,"w"); 
        nda::tensor::add(ComplexType(1.0),Ew,ComplexType(1.0),E(range(iw, iw + nw), 2));
      }
      iw += nw;
    }
  }

  void energy([[maybe_unused]] SpinTypes spin_component,
              [[maybe_unused]] nda::MemoryArrayOfRank<2> auto && E,
              [[maybe_unused]] nda::MemoryArrayOfRank<2> auto const& Gc,
              [[maybe_unused]] int nd,
              [[maybe_unused]] nda::MemoryArrayOfRank<2> auto && EJn,
              [[maybe_unused]] bool addH1  = true,
              [[maybe_unused]] bool addEJ  = true,
              [[maybe_unused]] bool addEXX = true)
  {
    utils::check(false," Error: spin-dependent energy not implemented ");
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

  // returns v[nwalk, nspin_in_basis*npol_in_basis, NMO, NMO]
  // no spin-orbit vHS yet
  auto vHS(nda::MemoryArrayOfRank<2> auto && X, double dt)
  {
    memory::check_memory_space<MEM>(X);
    using nda::range;
    auto all = range::all;
    int nchol = 2 * nkpts * _Luv_().extent(2);
    int nwalk = X.extent(0);
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;

    utils::check(X.is_contiguous(), "Layout mismatch");
    utils::check_strides(X);
    utils::check(X.shape() == std::array<long,2>{nwalk,nchol}, "THC::vbias: Size mismatch.");

    // Note: Allocate first, to make better use of memory pool
    // vHS[nspin_in_vHS][nwalk][npol_in_vHS*NMO][NMO]
    memory::buffered_array<MEM,ComplexType,4> v(nwalk,nstot,nptot*NMO,NMO);
    auto v7d = nda::reshape(v,std::array<long,7>{nwalk,nstot,nptot,nkpts,nbnd,nkpts,nbnd});
    v() = ComplexType(0.0);

    // scale by sqrt(dt)
    RealType a(std::sqrt(dt/double(nkpts))*0.5);

    // get array_views to the correct data and correct determinant
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_() );
    const auto Luv = _Luv_(); 
    int nu = Luv.extent(1);
    int nv = Luv.extent(2);

    auto X4d = nda::reshape(X,std::array<long,4>{nwalk,2,nkpts,nu});

    {
      // "rotate" X
      //  XIJ = 0.5*a*(Xn+ -i*Xn-), XJI = 0.5*a*(Xn+ +i*Xn-)
      memory::buffered_array<MEM,ComplexType,3> T(nwalk,nkpts,nu);
      T() = X4d(all,0,all,all);
      nda::tensor::add(ComplexType(0.0,-a),X4d(all,1,all,all),"wqn",ComplexType(a),T,"wqn");
      nda::tensor::add(ComplexType(a),X4d(all,0,all,all),"wqn",ComplexType(0.0,a),X4d(all,1,all,all),"wqn");
      X4d(all,0,all,all) = T();
      //  then combine Q/(-Q) pieces if Q != -Q
      //  X(Q)np = (X(Q)np + X(-Q)nm)
      for(int iq=0; iq<nkpts; iq++)  {
        if( iq != minusq(iq) ) {
          nda::tensor::add(ComplexType(1.0),X4d(all,1,minusq(iq),all),"wqn",ComplexType(1.0),X4d(all,0,iq,all),"wqn");
        }
      } 
    }

    // work array
    memory::buffered_array<MEM,ComplexType,3> Twqu(nwalk,nkpts,nu);
    // T(w,q,u) = sum_v L(q,u,n) * X(w,0,q,n) 
    nda::tensor::contract(ComplexType(1.0),X4d(all,0,all,all),"wqn",Luv,"qun",
                          ComplexType(0.0),Twqu,"wqu");

    // v[w][is*npol+ip][i][j] = sum_u conj(X[is][ip*NMO+i][u]) * X[is][ip*NMO+j][u] * T[u][w] 
    if constexpr (MEM==HOST_MEMORY) {

      memory::buffered_array<MEM,ComplexType,3> Qwiu(nwalk,nbnd,nu);
      for(int ik=0; ik<nkpts; ++ik) {
        for(int iq=0; iq<nkpts; ++iq) { 
          int k2 = qk_to_k2(iq,ik);
          for( int is=0; is<nstot; ++is) {
            for( int ip=0; ip<nptot; ++ip) {
        
              auto Xiu = Xsiu(is,ik,range(ip*nbnd,(ip+1)*nbnd),all); 
              auto Xju = Xsiu(is,k2,range(ip*nbnd,(ip+1)*nbnd),all); 
 
              // Qwiu[w][i][u] = T[w][u] * conj(Piu[i][u])
              for(int w=0; w<nwalk; ++w)
                for(int i=0; i<nbnd; ++i)
                  Qwiu(w,i,all) = Twqu(w,iq,all) * nda::conj(Xiu(i,all));

              // v(nstot,nwalk,nptot,nkpts,nbnd,nkpts,nbnd)
              auto vij = v7d(all,is,ip,ik,all,k2,all);
              nda::tensor::contract(ComplexType(1.0),Qwiu,"wiu",Xju,"ju",
                                    ComplexType(1.0),vij,"wij");

            }  // ip
          } // is
        } // iq
      } // ik

      memory::array_view<MEM,ComplexType,2> Twu(std::array<long,2>{nwalk,nu},Twqu.data());
      for(int iq=0; iq<nkpts; ++iq) {
        if(iq != minusq(iq)) continue;

        // T(w,q,u) = sum_v L(q,u,n) * X(w,0,q,n) 
        // can submit all q's with group batched gemm
        nda::tensor::contract(ComplexType(1.0),X4d(all,1,iq,all),"wn",
                                               nda::conj(Luv(iq,all,all)),"un",
                              ComplexType(0.0),Twu,"wu");

        for(int ik=0; ik<nkpts; ++ik) {
          int k2 = qk_to_k2(iq,ik);
          for( int is=0; is<nstot; ++is) {
            for( int ip=0; ip<nptot; ++ip) {
        
              auto Xiu = Xsiu(is,k2,range(ip*nbnd,(ip+1)*nbnd),all);
              auto Xju = Xsiu(is,ik,range(ip*nbnd,(ip+1)*nbnd),all);
 
              // Qwiu[w][i][u] = T[w][u] * Piu[i][u]
              for(int w=0; w<nwalk; ++w)
                for(int i=0; i<nbnd; ++i)
                  Qwiu(w,i,all) = Twu(w,all) * Xiu(i,all);

              auto vij = v7d(all,is,ip,k2,all,ik,all);
              nda::tensor::contract(ComplexType(1.0),Qwiu,"wiu",nda::conj(Xju),"ju",
                                    ComplexType(1.0),vij,"wij");

            }  // ip
          } // is
        } // ik
      } // iq

      // now missing contributions from q==-q terms
    } else {
      // use group batched gemm for this, since k-q matrix is not properly strided
      utils::check(false,"finish");
    } // MEM
    return v;
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    memory::check_memory_space<MEM>(G,v);
    using nda::range;
    auto all = range::all;
    int nchol = 2 * nkpts * _Luv_().extent(2);
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check_strides(G,v);
    utils::check(v.shape() == std::array<long,2>{nwalk,nchol}, "THC::vbias: Size mismatch.");
    if(haj.extent(0) == 1) // ndet==1, G half rotated
      utils::check(G.extent(1) == nel*npol*NMO, "THC::vbias: Size mismatch.");
    else // ndet>1, full G 
      utils::check(G.extent(1) == nspin*npol*NMO*npol*NMO, "THC::vbias: Size mismatch.");
    // needed to build G3d easily, can build from strides if needed
    utils::check(G.is_contiguous(), "Layout mismatch");
    utils::check(v.is_contiguous(), "Layout mismatch");

    // finish!!!
    utils::check(npol==1, "Error: vbias with npol>1 not yet implemented");

    // scale a by sqrt(dt)
    RealType a(std::sqrt(dt/double(nkpts))*0.5);

    // get array_views to the correct data and correct determinant
    const auto Luv = _Luv_(); 
    int nu = Luv.extent(1);
    int nv = Luv.extent(2);

    // v4d
    auto v4d = nda::reshape(v, std::array<long,4>{nwalk,2,nkpts,nv});

    // zero
    v() = ComplexType(0.0);

    if (haj.extent(0) == 1)
    {
      // can loop over walker blocks when memory is limited
      memory::array_view<MEM,const ComplexType,5> G5d(std::array<long,5>{nwalk,nel,npol,nkpts,nbnd},G.data());
      memory::buffered_array<MEM,ComplexType,3> Awu(nwalk,nkpts,nu); 
      memory::array_view<MEM,ComplexType,3> Awn(std::array<long,3>{nwalk,nkpts,nv},Awu.data()); 
      memory::buffered_array<MEM,ComplexType,3> Bwu(nwalk,nkpts,nu); 
      for(int ik=0; ik<nkpts; ik++) {

        // A(w,k2,u) = sum_a,j Y(d,s,k,a,u) * G(w,k,a,k2,j) * X(s,k2,j)
        Gu_from_compact(G5d, Awu, ik, range(nkpts), 0);

        // B(q) = A(k-q), now reorder according to k-q
        for(int iq=0; iq<nkpts; iq++) {
          int k2 = qk_to_k2(iq,ik);
          Bwu(all,iq,all) = Awu(all,k2,all);
        }

        //A(w,q,n) = sum_u B(w,q,u)*L(q,u,n)
        nda::tensor::contract(Bwu,"wqu",Luv,"qun",Awn,"wqn"); 

        //v+(w,q,n) += 0.5*T(w,q,n) 
        nda::tensor::add(ComplexType(a), Awn, "wqn", 
                         ComplexType(1.0), v4d(all,0,all,all), "wqn");

        //v-(w,q,n) += -i*0.5*T(w,q,n) 
        nda::tensor::add(ComplexType(0.0,-a), Awn, "wqn", 
                         ComplexType(1.0), v4d(all,1,all,all), "wqn");

        // how to batch these???
        for(int iq=0; iq<nkpts; iq++) {
          //if(q != -q) v(w,-q,n) -= i*0.5*T(w,q,n)
          if(iq != minusq(iq)) {
            nda::tensor::add(ComplexType(a), Awn(all,iq,all), "wn", 
                             ComplexType(1.0), v4d(all,0,minusq(iq),all), "wn");
            nda::tensor::add(ComplexType(0.0,a), Awn(all,iq,all), "wn", 
                             ComplexType(1.0), v4d(all,1,minusq(iq),all), "wn");
          }
        }

        // how to batch these???
        // now v- contributions for q == -q
        for(int iq=0; iq<nkpts; iq++) {
          if(iq != minusq(iq)) continue;
          // reuse memory
          memory::array_view<MEM,ComplexType,2> A2dwu(std::array<long,2>{nwalk,nu},Awu.data());
          // 3d view with a single kpoint, needed to call Gu_from_compact
          memory::array_view<MEM,ComplexType,3> A3dwu(std::array<long,3>{nwalk,1,nu},Awu.data());
          memory::array_view<MEM,ComplexType,2> B2dwn(std::array<long,2>{nwalk,nv},Bwu.data()); 
          int k2 = qk_to_k2(iq,ik);

          // G(w,u) = sum_a,j Y(d,s,k-q,a,u) * G(w,k-q,a,k,j) * X(s,k,j)
          Gu_from_compact(G5d, A3dwu, k2, range(ik,ik+1), 0);
              
          //T(w,n) = sum_u G(w,u)* conj(L(q,u,n))
          nda::tensor::contract(A2dwu,"wu",nda::conj(Luv(iq,all,all)),"un",B2dwn,"wn");
             
          //v+(w,q,n) += 0.5*T(w,q,n) 
          nda::tensor::add(ComplexType(a), B2dwn, "wn", 
                           ComplexType(1.0), v4d(all,0,iq,all), "wn");

          //v-(w,q,n) -= i*0.5*T(w,q,n) 
          nda::tensor::add(ComplexType(0.0,a), B2dwn, "wn", 
                           ComplexType(1.0), v4d(all,1,iq,all), "wn");
        } // iq* 

      } // ik 
    }
    else
    {
      utils::check(false," Error: THC not yet implemented for multiple references.");
      // multideterminant is not half-rotated, so use Likn
      // which spin???
//      memory::array_view<MEM,const ComplexType,4> G3d(std::array<long,4>{nwalk,nspin,npol*NMO,npol*NMO},G.data());
    }
  }

  template<class... Args> void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }
  
  /// Returns the number of spins and polarizations in the VHS potential.
  std::tuple<int,int> vHS_dims() const {
    return std::make_tuple(_Xsiu_().shape()[0],_Xsiu_().shape()[2]/nbnd);
  }
  int number_of_ke_vectors() const { 
    utils::check(_Zuv_.has_value() or _Zuv_rot_.has_value(), "Missing Zuv/Zuv_rot.");
    if(_Zuv_.has_value()) return nkpts*_Zuv_->extent(1);
    else return nkpts*_Zuv_rot_->extent(1); 
  }
  int number_of_cholesky_vectors() const { return 2 * nkpts * _Luv_().extent(2); }

  nda::array<ComplexType, 2> getHSPotentials() 
  { return nda::array<ComplexType, 2>{}; }

protected:
  // for a given ispin, k
  // G(w,k2,u) = sum_a,j Y(d,s,k,a,u) * G(w,k,a,k2,j) * X(s,k2,p,j,u)
  void Gu_from_compact(nda::MemoryArrayOfRank<5> auto const& G,
               nda::MemoryArrayOfRank<3> auto && Gwku, int ik, nda::range k2_rng, int idet)
  {
    using nda::range;
    auto all = range::all;
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    bool has_rot = _Xsiu_rot_.has_value();
    int num_k2 = k2_rng.size();
    // [nstot][nkpts][nptot*nbnd][nu]
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_());
    // [nstot][nptot][nkpts][nocc_max][nu]
    const auto Ysau = ( has_rot ? (*_Ydsau_rot_)()(idet,nda::ellipsis{}) :
                                 _Ydsau_()(idet,nda::ellipsis{}) );
    long nw   = G.extent(0);
    long nu   = Xsiu.extent(3);

    utils::check(G.shape() == std::array<long,5>{nw,nel,npol,nkpts,nbnd}, 
                 "THC::Gu_from_compact: Shape mismatch");
    utils::check(Gwku.shape() == std::array<long,3>{nw,num_k2,nu}, "THC::Gu_from_compact: Shape mismatch");
    Gwku() = ComplexType(0.0);
    ComplexType a = (walker_type == CLOSED) ? ComplexType(2.0) : ComplexType(1.0);
    for( int is=0; is<nspin; is++ ) {
      int n0 = ( ik==0 ? 0 : nda::sum(nocc_per_kp(is,range(ik))) );
      int nel_k = nocc_per_kp(is,ik);
      for( int ip=0; ip<npol; ip++ ) {
        
        memory::buffered_array<MEM,ComplexType,4> Tw(nw,num_k2,nel_k,nu);    
        auto Xju = Xsiu(is%nstot,k2_rng,range((ip%nptot)*nbnd,(ip%nptot+1)*nbnd),all);
        auto Yau = Ysau(is%nstot,ip%nptot,ik,range(nel_k),all);

        auto Gwakj = G(all,range(is*nup+n0,is*nup+n0+nel_k),ip,k2_rng,all);
        //  T(w,k2,a,u) = G(w,k,a,p,k2,j) * X(s,k2,p,j,u)
        nda::tensor::contract(Gwakj,"wakj",Xju,"kju",Tw,"wkau");
        // Gwku(w,k2,u) = scl * sum_a Y(s,p,k,a,u) * T(w,k2,a,u) 
        nda::tensor::contract(ComplexType(a),Tw,"wkau",Yau,"au",ComplexType(1.0),Gwku,"wku");
      
      } // npol 
    } // nspin 
  }

  // Computes Guv and Guu for a set of walkers
  // rotMuv is partitioned along 'u'
  // G[w][nel*nmo]
  // Guv[w][nu][nv]
  // Twav[w][nel][nv]
  void get_Guv(int ispin, int p1, int p2, int k1, int k2, 
         nda::MemoryArrayOfRank<5> auto const& G, 
         nda::MemoryArrayOfRank<3> auto && Guv, 
         nda::MemoryArrayOfRank<1> auto && Tbuff, int idet)
  {
// MAM: in GPU implementation, do multiple k1 or k2 simultaneously
    using nda::range;
    auto all = range::all;
    int nstot = hij.extent(0);
    int nptot = hij.extent(2)/nbnd;
    long p2_ = long(p2)%nptot;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    int n0 = ( k1==0 ? 0 : nda::sum(nocc_per_kp(ispin,range(k1))) );
    int nel_k1 = nocc_per_kp(ispin,k1);
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xiu = ( has_rot ? (*_Xsiu_rot_)()(ispin%nstot,k2,range(p2_*nbnd,(p2_+1)*nbnd),all) : 
                                  _Xsiu_()(ispin%nstot,k2,range(p2_*nbnd,(p2_+1)*nbnd),all) );
    const auto Yau = ( has_rot ? (*_Ydsau_rot_)()(idet,ispin,p1,k1,range(nel_k1),all) : 
                                 _Ydsau_()(idet,ispin,p1,k1,range(nel_k1),all) );
    int nw   = int(G.extent(0));
    int nu = Xiu.extent(1);

    // G5d[w][a][j]
    utils::check(G.shape() == std::array<long,5>{nw,nel,npol,nkpts,nbnd}, "THC::get_Guv: Shape mismatch");
    utils::check(Tbuff.size() >= nw*nel_k1*nu, "THC::get_Guv: Twav size mismatch.");
    memory::array_view<MEM,ComplexType,3> Twav(std::array<long,3>{nw,nel_k1,nu},Tbuff.data());

    auto Gwai = G(all,range(ispin*nup+n0,ispin*nup+n0+nel_k1),p2,k2,all);
    // Twav[w][a][v] = sum_j G[w][a][j] X[j][v]
    nda::tensor::contract(Gwai,"wai",Xiu,"iv",Twav,"wav");
    // G[w][u][v] = sum_a X[a][u] Twav[w][a][v]
    nda::tensor::contract(Yau,"au",Twav,"wav",Guv,"wuv");

//    over a range of k2
//    auto Gwai = G(all,range(ispin*nup+n0,ispin*nup+n0+nel_k1),p2,all,all);
//    // Twav[w][a][k2][v] = sum_j G[w][a][k2][j] X[j][v]
//    nda::tensor::contract(Gwai,"waki",Xiu,"kiv",Twav,"wakv");
//    // G[w][u][v] = sum_a X[a][u] Twav[w][a][v]
//    nda::tensor::contract(Yau,"au",Twav,"wakv",Guv,"wkuv");
  }

protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int NMO, nup, ndown;
  int nelec[2];
  int nkpts, nbnd;

  nda::array<int,2> nocc_per_kp;

  // BZ information
  nda::array<int,1> minusq;
  nda::array<int,2> qk_to_k2;
  int Q0_index = 0;

  // H1[nspin][nk][npol*nbnd][npol*nbnd]
  memory::shared_array<HOST_MEMORY,ComplexType,4> hij;

  // half rotated one body hamiltonian: [ndet][nup+ndn][npol*NMO]. Kept in full basis
  memory::shared_array<MEM,ComplexType,3> haj;

  // Xsiu[nspin][nk][npol*nbnd][nu]
  memory::shared_array<MEM,ComplexType,4> _Xsiu_;

  // Ydsau[ndet][nspin][ipol][nk][nocc_max][nu]
  memory::shared_array<MEM,ComplexType,6> _Ydsau_;

  // Luv[nu][nv]
  memory::shared_array<MEM,ComplexType,3> _Luv_;

  // Zuv[nu][nv]
  std::optional<decltype(_Luv_)> _Zuv_;

  // Xsiu_rot[nspin][npol*nbnd][nu_rot]
  std::optional<decltype(_Xsiu_)> _Xsiu_rot_;

  // Ydsau_rot[ndet][nspin][ipol][nocc_max][nu_rot]
  std::optional<decltype(_Ydsau_)> _Ydsau_rot_; 

  // Zuv_rot[nu_rot][nu_rot]
  std::optional<decltype(_Luv_)> _Zuv_rot_; 

  // vexx(i,l) = -0.5 * sum_j <ij|jl> : [nspin][nk][npol*nbnd][npol*nbnd]
  memory::shared_array<HOST_MEMORY,ComplexType,4> vexx;

  ComplexType E0;

};

} // namespace afqmc

} // namespace sfqmc

