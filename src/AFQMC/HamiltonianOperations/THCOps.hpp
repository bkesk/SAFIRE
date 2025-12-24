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

template<MEMORY_SPACE _MEM, bool REAL>
class THCOps
{
  static constexpr MEMORY_SPACE MEM = _MEM;

  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:
  static constexpr HamiltonianTypes HamOpType = THC;
  constexpr HamiltonianTypes getHamType() const { return THC; }

  THCOps()
  {
    utils::check(false,"Default constructor for THCOps disabled.");
  }

  /*
   * nup/ndown stands for number of alpha/beta electrons
   */
  THCOps(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> ctxt,
         WALKER_TYPES type,
         long nmo_,
         long nup_, 
         long ndn_, 
         memory::shared_array<HOST_MEMORY,ComplexType,3>&& hij_,
         memory::shared_array<MEM,ComplexType,3>&& haj_,
         memory::shared_array<MEM,ValueType,3>&& x_,
         memory::shared_array<MEM,ComplexType,5>&& y_,
         memory::shared_array<MEM,ValueType,2>&& l_,
         std::optional<memory::shared_array<MEM,ValueType,2>>&& z_,
         std::optional<memory::shared_array<MEM,ValueType,3>>&& x_rot_,
         std::optional<memory::shared_array<MEM,ComplexType,5>>&& y_rot_,
         std::optional<memory::shared_array<MEM,ValueType,2>>&& z_rot_,
         memory::shared_array<HOST_MEMORY,ComplexType,3>&& v0_,
         ComplexType e0_)
      : mpi(ctxt), 
        walker_type(type),
        NMO(nmo_),
        nup(nup_),
        ndown(ndn_),
        nelec{nup, ndown},
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
    int nu = _Luv_.extent(0);
    int nv = _Luv_.extent(1);
    int nu_rot = (_Xsiu_rot_.has_value() ? _Xsiu_rot_->shape()[2] : nu);
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    int nstot = hij.extent(0);
    int nptot = hij.extent(1)/NMO;
    int ndet = haj.extent(0);
    utils::check(vexx.shape() == std::array<long,3>{nstot*nptot,NMO,NMO},"THCOps: Size mismatch"); 
    utils::check(hij.shape() == std::array<long,3>{nstot,nptot*NMO,nptot*NMO},"THCOps: Size mismatch"); 
    utils::check(haj.shape() == std::array<long,3>{ndet,nel,npol*NMO},"THCOps: Size mismatch"); 
    utils::check(_Xsiu_.shape() == std::array<long,3>{nstot,npol*NMO,nu},"THCOps: Size mismatch"); 
    utils::check(_Ydsau_.shape() == std::array<long,5>{ndet,nspin,npol,nup,nu},"THCOps: Size mismatch"); 
    utils::check(_Luv_.shape() == std::array<long,2>{nu,nv},"THCOps: Size mismatch"); 
    if(_Zuv_.has_value())
      utils::check(_Zuv_->shape() == std::array<long,2>{nu,nu},"THCOps: Size mismatch"); 
    if(_Xsiu_rot_.has_value())
      utils::check(_Xsiu_rot_->shape() == std::array<long,3>{nstot,npol*NMO,nu_rot},"THCOps: Size mismatch"); 
    if(_Ydsau_rot_.has_value())
      utils::check(_Ydsau_rot_->shape() == std::array<long,5>{ndet,nspin,npol,nup,nu_rot},"THCOps: Size mismatch"); 
    if(_Zuv_rot_.has_value())
      utils::check(_Zuv_rot_->shape() == std::array<long,2>{nu_rot,nu_rot},"THCOps: Size mismatch"); 
  }

  ~THCOps() = default; 

  THCOps(THCOps const& other) = default;
  THCOps& operator=(THCOps const& other) = default;

  THCOps(THCOps&& other) = default;
  THCOps& operator=(THCOps&& other) = default;

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    using nda::range;
    auto all = range::all;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nstot = hij.extent(0);
    int nptot = hij.extent(1)/NMO;
    utils::check(vMF.size() == number_of_cholesky_vectors(), "Size mismatch");

    // v[nstot][nwalk=1][nptot*NMO][NMO]
    nda::array<ComplexType, 4> v;
    {
      memory::buffered_array<MEM,ComplexType,2> vMF_2d(1,vMF.size());
      vMF_2d(0,all) = vMF();
      v = std::move(vHS(vMF_2d, dt));
      utils::check(v.shape() == std::array<long,4>{1,nspin,npol*NMO,NMO}, "Size mismatch");
    }

    nda::array<ComplexType, 3> H1(nspin, npol*NMO, npol*NMO);
    H1() = ComplexType(0.0);
    
    // add hij(nstot,nptot*NMO,nptot*NMO) + vexx(nstot*nptot,NMO,NMO) and symmetrize
    //
    for (int is = 0; is < nspin; is++) {
      int is_ = is%nstot;
      for (int p1 = 0; p1 < npol; p1++) {
        int p1_ = p1%nptot;
        for (int p2 = 0; p2 < npol; p2++) {
          int p2_ = p2%nptot;
          for (int i = 0; i < NMO; i++) {
            for (int j = 0 ; j < NMO; j++)
            {
              if(p1==p2) {
                H1(is,p1*NMO+i,p2*NMO+j) = v(0,is_,p1_*NMO+i,j) + 
                                           dt * (hij()(is_,p1_*NMO+i,p2_*NMO+j) + vexx()(is_*nptot+p1_,i,j));
              } else {
                // only spin-orbit terms here coming from hij
                H1(is,p1*NMO+i,p2*NMO+j) = dt * hij()(is_,p1_*NMO+i,p2_*NMO+j); 
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
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "THC::energy: Size mismatch.");
    utils::check(G.extent(1) == nel*npol*NMO, "THC::energy: Size mismatch.");
    utils::check_strides(E,G);
    // limiting G to contiguous arrays for simplicity now, reconsider if necessary
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

    int nu = Zuv.extent(0);
    long nstot = _Xsiu_().shape()[0];
    long nptot = _Xsiu_().shape()[1]/NMO; 

    // calculate how many walkers can be done concurrently
    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
    Bytes /= long((nu * nu + nu + nu * nup) * sizeof(ComplexType));
    int nwmax = std::min(nwalk, std::max(1, int(Bytes)));
    
    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel,npol*NMO},G.data());

    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      // Guv[nspin][nu][nv]
      memory::buffered_array<MEM,ComplexType,3> Guv(nw,nu,nu);
      // Guu[u]: summed over spin
      memory::buffered_array<MEM,ComplexType,2> Guu(nw,nu);
      Guu() = ComplexType(0.0);
      for (int ispin = 0; ispin < nspin; ++ispin)
      {
        long is_ = long(ispin)%nstot; 
        for (int p1 = 0; p1 < npol; ++p1)
        {
          long ip1_ = long(p1)%nptot; 
          for (int p2 = 0; p2 < npol; ++p2)
          {
            // Buffer space
            memory::buffered_array<MEM,ComplexType,3> Tav(nw,nelec[ispin],nu);
            Guv_Guu(ispin, p1, p2, G3d(range(iw, iw + nw), all, all), Guv, Guu, Tav, idet);

            if constexpr (MEM==HOST_MEMORY) {
              for(int i=0; i<nw; ++i)
                Guv(i,nda::ellipsis{}) *= Zuv();
            } else {
	      if constexpr(REAL) {
                auto Guv4d = memory::to_real_view(Guv);
                nda::tensor::elementwise(RealType(1.0),Zuv,"uv",
                                         RealType(1.0),Guv4d,"wuvc",nda::tensor::op::MUL);
              } else {
                nda::tensor::elementwise(ComplexType(1.0),Zuv,"uv",
                                         ComplexType(1.0),Guv,"wuv",nda::tensor::op::MUL);
              }
            }

            // R[w,u][b] = sum_v Guv[w,u][v] * rotcXau[b][v]
            auto Yau = Ysau(ispin,p2,range(nelec[ispin]),all);
            nda::tensor::contract(Yau,"av",Guv,"wuv",Tav,"wau"); 

            // reuse Guv memory
            memory::array_view<MEM,ComplexType,3> Twbi(std::array<long,3>{nw,nelec[ispin],NMO},Guv.data());
            //T[w][b][k] = sum_u R[w][u][b] * Piu[k][u]
	    if constexpr(REAL) {
              auto Xiu = Xsiu(is_,range(ip1_*NMO,(ip1_+1)*NMO),all);
              auto Ta4d = memory::to_real_view(Tav);
              auto Tb4d = memory::to_real_view(Twbi);
              nda::tensor::contract(Ta4d,"wauc",Xiu,"iu",Tb4d,"waic"); 
	    } else {
              auto Xiu = Xsiu(is_,range(ip1_*NMO,(ip1_+1)*NMO),all);
              nda::tensor::contract(Tav,"wau",Xiu,"iu",Twbi,"wai"); 
            }

            // E[w] = sum_ai T[w][a][i] * G[w][a][i] 
            auto Gwai = G3d(range(iw, iw + nw),range(ispin*nup,nup+ispin*ndown),range(p1*NMO,(p1+1)*NMO)); 
            memory::buffered_array<MEM,ComplexType,1> Ew(nw);
            nda::tensor::contract(ComplexType(-0.5*scl),Twbi,"wai",Gwai,"wai",ComplexType(0.0),Ew,"w"); 

            nda::tensor::add(ComplexType(1.0),Ew,ComplexType(1.0),E(range(iw, iw + nw), 1));
          }
        }
      }
      if (addEJ)
      {
        memory::buffered_array<MEM,ComplexType,2> Twu(nw,nu);
        memory::buffered_array<MEM,ComplexType,1> Ew(nw);
	if constexpr (REAL) {
          // use strategy in Guv_Guu
          auto Guu3d = memory::to_real_view(Guu);
          auto Twu3d = memory::to_real_view(Twu);
          nda::tensor::contract(Guu3d,"wuc",Zuv,"uv",Twu3d,"wvc");
	} else {
          nda::blas::gemm(Guu,Zuv,Twu);
	}
        nda::tensor::contract(ComplexType(RealType(0.5 * scl * scl)),nda::conj(Guu),"wu",Twu,"wu",
                              ComplexType(0.0),Ew,"w"); 
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
    int nchol = ( REAL ? _Luv_().extent(1) : 2 * _Luv_().extent(1) );
    int nwalk = X.extent(0);
    long nstot = _Xsiu_().shape()[0];
    long nptot = _Xsiu_().shape()[1]/NMO;
    utils::check_strides(X);
    // limiting X/v to contiguous arrays for simplicity now, reconsider if necessary
    utils::check(X.shape() == std::array<long,2>{nwalk,nchol}, "THC::vbias: Size mismatch.");

    // Note: Allocate first, to make better use of memory pool
    // vHS[nspin_in_vHS][nwalk][npol_in_vHS*NMO][NMO]
    memory::buffered_array<MEM,ComplexType,4> v(nwalk,nstot,nptot*NMO,NMO);
    v() = ComplexType(0.0);

    // scale by sqrt(dt)
    RealType a(std::sqrt(dt));

    // get array_views to the correct data and correct determinant
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_() );
    const auto Luv = _Luv_(); 
    int nu = Luv.extent(0);

    // calculate how many walkers can be done concurrently
    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
    Bytes /= size_t(NMO * nu * sizeof(ComplexType));
    int nwmax = std::min(nwalk, std::max(1, int(Bytes)));

    // work array
    memory::buffered_array<MEM,ComplexType,2> Twu(nwmax,nu);
    auto X_r = memory::to_real_view(X);
    auto Twu_r = memory::to_real_view(Twu);
    memory::array_view<MEM,const RealType,2> Luv2(std::array<long,2>{nu,nchol},reinterpret_cast<RealType const*>(Luv.data()));

    // T[u][w] = sum_v L[u][v] * X[v][w] 
    nda::tensor::contract(X_r,"wvc",Luv2,"uv",Twu_r,"wuc");

    // v[w][is*npol+ip][i][j] = sum_u conj(X[is][ip*NMO+i][u]) * X[is][ip*NMO+j][u] * T[u][w] 
    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      memory::buffered_array<MEM,ComplexType,3> Qwiu(nw,NMO,nu);
      for( int is=0; is<nstot; ++is) {
        for( int ip=0; ip<nptot; ++ip) {
       
          auto Xiu = Xsiu(is,range(ip*NMO,(ip+1)*NMO),all); 
          if constexpr (REAL) {

            auto Qwiu_r = memory::to_real_view(Qwiu);
            // Qwiu[w][i][u] = T[w][u] * conj(Piu[i][u])
            if constexpr (MEM==HOST_MEMORY) {
              for(int w=0; w<nw; ++w)
                for(int i=0; i<NMO; ++i)
                  Qwiu(w,i,all) = Twu(w,all) * Xiu(i,all);
            } else {
              nda::tensor::elementwise_trinary(1.0,Twu_r,"wuc",1.0,Xiu,"iu",0.0,Qwiu_r,"wiuc",nda::tensor::op::MUL,nda::tensor::op::SUM);
            }
            
            auto vij = v(range(iw,iw+nw),is,range(ip*NMO,(ip+1)*NMO),all);
            auto vij_r = memory::to_real_view(vij);
            nda::tensor::contract(a,Qwiu_r,"wiuc",Xiu,"ju",
                                  RealType(0.0),vij_r,"wijc");

          } else {
 
            // Qwiu[w][i][u] = T[w][u] * conj(Piu[i][u])
            if constexpr (MEM==HOST_MEMORY) {
              for(int w=0; w<nw; ++w)
                for(int i=0; i<NMO; ++i)
                  Qwiu(w,i,all) = Twu(w,all) * nda::conj(Xiu(i,all));
            } else {
              nda::tensor::elementwise_trinary(ComplexType(1.0),Twu,"wu",ComplexType(1.0),nda::conj(Xiu),"iu",ComplexType(0.0),Qwiu,"wiu",nda::tensor::op::MUL,nda::tensor::op::SUM); 
            }

            auto vij = v(range(iw,iw+nw),is,range(ip*NMO,(ip+1)*NMO),all);
            nda::tensor::contract(ComplexType(a),Qwiu,"wiu",Xiu,"ju",
                                  ComplexType(0.0),vij,"wij");

          }
        }
      }
      iw += nw;
    }
    return v;
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    memory::check_memory_space<MEM>(G,v);
    using nda::range;
    auto all = range::all;
    int nchol = ( REAL ? _Luv_().extent(1) : 2 * _Luv_().extent(1) );
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check_strides(G,v);
    // limiting G to contiguous arrays for simplicity now, reconsider if necessary
    utils::check(v.shape() == std::array<long,2>{nwalk,nchol}, "THC::vbias: Size mismatch.");
    if(haj.extent(0) == 1) // ndet==1, G half rotated
      utils::check(G.extent(1) == nel*npol*NMO, "THC::vbias: Size mismatch.");
    else // ndet>1, full G 
      utils::check(G.extent(1) == nspin*npol*NMO*npol*NMO, "THC::vbias: Size mismatch.");
    utils::check(G.is_contiguous(), "Layout mismatch");

    // scale a by sqrt(dt)
    RealType a(std::sqrt(dt));

    // get array_views to the correct data and correct determinant
    const auto Luv = _Luv_(); 
    int nu = Luv.extent(0);

    if (haj.extent(0) == 1)
    {
      memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel,npol*NMO},G.data());
      memory::buffered_array<MEM,ComplexType,2> Guu(nwalk,nu);
      Guu_from_compact(G3d, Guu, 0);
      auto Guu_3d= memory::to_real_view(Guu);
      auto v_3d = memory::to_real_view(v);
      memory::array_view<MEM,const RealType,2> Luv2(std::array<long,2>{nu,nchol},reinterpret_cast<RealType const*>(Luv.data()));
      nda::tensor::contract(a,Guu_3d,"wuc",Luv2,"uv",RealType(0.0),v_3d,"wvc");
    }
    else
    {
      utils::check(false," Error: THC not yet implemented for multiple references.");
      // multideterminant is not half-rotated, so use Likn
      // which spin???
      memory::array_view<MEM,const ComplexType,4> G3d(std::array<long,4>{nwalk,nspin,npol*NMO,npol*NMO},G.data());
    }
  }

  template<class... Args> void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }
  
  /// Returns the number of spins and polarizations in the VHS potential.
  std::tuple<int,int> vHS_dims() const {
    return std::make_tuple(_Xsiu_().shape()[0],_Xsiu_().shape()[1]/NMO);
  }
  int number_of_ke_vectors() const { 
    utils::check(_Zuv_.has_value() or _Zuv_rot_.has_value(), "Missing Zuv/Zuv_rot.");
    if(_Zuv_.has_value()) return _Zuv_->extent(0);
    else return _Zuv_rot_->extent(0); 
  }
  int number_of_cholesky_vectors() const { return ( REAL ? _Luv_().extent(1) : 2 * _Luv_().extent(1) ); }

  // add nspin_in_basis to allow for a spin independent basis too
  nda::array<ComplexType, 2> getHSPotentials() 
  { return nda::array<ComplexType, 2>{}; }

protected:
  // Guu[nu][nwalk]
  void Guu_from_compact(nda::MemoryArrayOfRank<3> auto const& G,
               nda::MemoryArrayOfRank<2> auto && Guu,
               int idet)
  {
    using nda::range;
    auto all = range::all;
    long nstot = _Xsiu_().shape()[0];
    long nptot = _Xsiu_().shape()[1]/NMO;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_());
    const auto Ysau = ( has_rot ? (*_Ydsau_rot_)()(idet,nda::ellipsis{}) :
                                 _Ydsau_()(idet,nda::ellipsis{}) );
    long nw   = G.extent(0);
    long nu   = Xsiu.extent(2);

    // G3d[w][a][j]
    utils::check(G.shape() == std::array<long,3>{nw,nel,npol*NMO}, "THC::Guv_Guu: Shape mismatch");
    utils::check(Guu.shape() == std::array<long,2>{nw,nu}, "THC::Guv_Guu: Shape mismatch");
    Guu() = ComplexType(0.0);
    ComplexType a = (walker_type == CLOSED) ? ComplexType(2.0) : ComplexType(1.0);
    for( int is=0; is<nspin; is++ ) {
      for( int ip=0; ip<npol; ip++ ) {
        
        memory::buffered_array<MEM,ComplexType,3> Twau(nw,nelec[is],nu);    
        auto Xiu = Xsiu(is%nstot,range(ip%nptot*NMO,(ip%nptot+1)*NMO),all);
        auto Yau = Ysau(is%nstot,ip%nptot,range(nelec[is]),all);

        if constexpr (REAL) {
          auto G4d = memory::to_real_view(G);
          auto Gwaic = G4d(all,range(is*nup,nup+is*ndown),range(ip*NMO,(ip+1)*NMO),all); 
          auto T4d = memory::to_real_view(Twau);
          nda::tensor::contract(Gwaic,"waic",Xiu,"iu",T4d,"wauc");
        } else {
          auto Gwai = G(all,range(is*nup,nup+is*ndown),range(ip*NMO,(ip+1)*NMO)); 
          nda::tensor::contract(Gwai,"wai",Xiu,"iu",Twau,"wau");
        }
        // Gwu[w][u] = a * sum_a T1[w][a][u] * cXau[a][u]
        nda::tensor::contract(ComplexType(a),Twau,"wau",Yau,"au",ComplexType(1.0),Guu,"wu");
      
      } // npol 
    } // nspin 
  }

/*
  // Guu[nu][nwalk]
  template<class MatA, class MatB>
  void Guu_from_full(MatA const& G, MatB&& Guu)
  {
    int nmo_ = int(Piu.size(0));
    int nu   = int(Piu.size(1));
    int u0, uN;
    std::tie(u0, uN) = FairDivideBoundary(comm->rank(), nu, comm->size());
    int nwalk        = G.size(0);

    RUNTIME_CHECK(G.size(0) == Guu.size(1), "");
    RUNTIME_CHECK(Guu.size(0) == nu, "");
    RUNTIME_CHECK(G.size(1) == nmo_ * nmo_, "");

    // calculate how many walkers can be done concurrently
    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
    Bytes /= size_t(nmo_ * nu * sizeof(SPComplexType));
    int nwmax = std::min(nwalk, std::max(1, int(Bytes)));

    ComplexType a = (walker_type == CLOSED) ? ComplexType(2.0) : ComplexType(1.0);
    Array<SPComplexType, 2> T1({nwmax * nmo_, (uN - u0)},
                               device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    comm->barrier();
    ma::fill(Guu.sliced(u0,uN), SPComplexType(0.0));

    APP_ABORT(" Error: Finish Guu_from_full \n\n");
    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      Array_cref<SPComplexType, 2> Giw(make_device_ptr(G[iw].origin()), {nw * nmo_, nmo_});
      //        ma::product(Giw,Piu({0,nmo_},{u0,uN}),T1);
      // Guu[u+u0][w] = alpha * sum_i T[w][i][u] * P[i][u]
      // this looks wrong!!!
      //Awiu_Biu_Cuw(uN - u0, nw, nmo_, SPComplexType(a), T1.origin(), make_device_ptr(Piu.origin()) + u0, nu,
      //             make_device_ptr(Guu[u0].origin()) + iw, nwalk);
      //ma::Awiu_Biu_Cuw(SPComplexType(a), T1.partitioned(nwmax).sliced(0, nw),
      //		       Piu.range({u0,uN},1), Guu.sliced(u0,uN).range({iw,iw+nw},1));
      iw += nw;
    }
    comm->barrier();
  }
*/

  // Computes Guv and Guu for a set of walkers
  // rotMuv is partitioned along 'u'
  // G[w][nel*nmo]
  // Guv[w][nu][nv]
  // Guu[w][v], accumulated on this routine, sum over spin is outside
  // Twav[w][nel][nv]
  void Guv_Guu(int ispin, int p1, int p2, nda::MemoryArrayOfRank<3> auto const& G, 
         nda::MemoryArrayOfRank<3> auto && Guv, nda::MemoryArrayOfRank<2> auto && Guu, 
         nda::MemoryArrayOfRank<3> auto && Twav, int idet)
  {
    using nda::range;
    auto all = range::all;
    long nstot = _Xsiu_().shape()[0]; 
    long nptot = _Xsiu_().shape()[1]/NMO; 
    long ip_ = long(p2)%nptot;
    range M_rng(ip_*NMO,(ip_+1)*NMO);
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xiu = ( has_rot ? (*_Xsiu_rot_)()(ispin%nstot,M_rng,all) : 
                                  _Xsiu_()(ispin%nstot,M_rng,all) );
    const auto Yau = ( has_rot ? (*_Ydsau_rot_)()(idet,ispin,p1,range(nelec[ispin]),all) : 
                                 _Ydsau_()(idet,ispin,p1,range(nelec[ispin]),all) );
    int nw   = int(G.extent(0));

    // G3d[w][a][j]
    utils::check(G.shape() == std::array<long,3>{nw,nel,npol*NMO}, "THC::Guv_Guu: Shape mismatch");
    utils::check(Twav.extent(1) == nelec[ispin], "THC::Guv_Guu: Twav size mismatch.");

    if constexpr (REAL) {
      static_assert(std::decay_t<decltype(G)>::is_stride_order_C(), "Stride mismatch");
      static_assert(std::decay_t<decltype(Twav)>::is_stride_order_C(), "Stride mismatch");
      auto G4d = memory::to_real_view(G);
      auto T4d = memory::to_real_view(Twav);
      // choose electron range compatible with ispin
      auto Gwaic = G4d(all,range(ispin*nup,nup+ispin*ndown),range(p2*NMO,(p2+1)*NMO),all);
      // Twav[w][a][v] = sum_j G[w][a][j] X[j][v]
      nda::tensor::contract(Gwaic,"wajc",Xiu,"jv",T4d,"wavc");
    } else {
      auto Gwai = G(all,range(ispin*nup,nup+ispin*ndown),range(p2*NMO,(p2+1)*NMO));
      // Twav[w][a][v] = sum_j G[w][a][j] X[j][v]
      nda::tensor::contract(Gwai,"wai",Xiu,"iv",Twav,"wav");
    }
    // G[w][u][v] = sum_a X[a][u] Twav[w][a][v]
    nda::tensor::contract(Yau,"au",Twav,"wav",Guv,"wuv");

    // Gwv = Gwvv, 
    if(p1==p2) {
      if constexpr (MEM==HOST_MEMORY) {
        for(int i=0; i<nw; i++)
          Guu(i,all) += nda::diagonal(Guv(i,all,all));
      } else {
        std::array<long,2> str = {Guv.strides()[0],Guv.strides()[1]+1};
        nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm(Guu.shape(),str);
        memory::array_view<MEM,ComplexType,2> Guv_diag(idxm, Guv.data());
        nda::tensor::add(ComplexType(1.0),Guv_diag,ComplexType(1.0),Guu);   
      }
    }
  }
/*
  / *
    // since this is for energy, only compact is accepted
    // Computes Guv and Guu for a single walker
    // As opposed to the other Guu routines,
    //  this routine expects G for the walker in matrix form
    // rotMuv is partitioned along 'u'
    // G[nel][nmo]
    // Guv[nu][nu]
    // Guu[u]: summed over spin
    // T1[nel_][nu]
    template<class MatA, class MatB, class MatC, class MatD>
    void Guv_Guu2(MatA const& G, MatB&& Guv, MatC&& Guu, MatD&& T1, int k) {

      static_assert(std::decay<MatA>::type::dimensionality == 2, "Wrong dimensionality");
      static_assert(std::decay<MatB>::type::dimensionality == 2, "Wrong dimensionality");
      static_assert(std::decay<MatC>::type::dimensionality == 1, "Wrong dimensionality");
      static_assert(std::decay<MatD>::type::dimensionality == 2, "Wrong dimensionality");
      int nmo_ = int(rotPiu.size(0));
      int nu = int(rotMuv.size(0));  // potentially distributed over nodes
      int nv = int(rotMuv.size(1));  // not distributed over nodes
      RUNTIME_CHECK(rotPiu.size(1) == nv, "");
      int v0,vN;
      std::tie(v0,vN) = FairDivideBoundary(comm->rank(),nv,comm->size());
      int nu0 = rotnmu0; 
      ComplexType zero(0.0,0.0);

      RUNTIME_CHECK(Guu.size(0) == nv, "");
      RUNTIME_CHECK(Guv.size(0) == nu, "");
      RUNTIME_CHECK(Guv.size(1) == nv, "");

      // sync first
      comm->barrier();
      int nel_ = (walker_type==CLOSED)?nup:(nup+ndown);
      RUNTIME_CHECK(G.size(0) == size_t(nel_), "");
      RUNTIME_CHECK(G.size(1) == size_t(nmo_), "");
      RUNTIME_CHECK(T1.size(0) == size_t(nel_), "");
      RUNTIME_CHECK(T1.size(1) == size_t(nv), "");

      ma::product(G,rotPiu({0,nmo_},{v0,vN}),
                  T1(T1.extension(0),{v0,vN}));
      // This operation might benefit from a 2-D work distribution
      ma::product(rotcXau[k].sliced(nu0,nu0+nu),
                  T1(T1.extension(0),{v0,vN}),
                  Guv(Guv.extension(0),{v0,vN}));
      for(int v=v0; v<vN; ++v)
        if( v < nu0 || v >= nu0+nu ) {
          Guu[v] = ma::dot(rotcXau[k][v],T1(T1.extension(0),v)); 
        } else
         Guu[v] = Guv[v-nu0][v];
      comm->barrier();
    }
*/
protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int NMO, nup, ndown;
  int nelec[2];

  // H1[nspin][npol*NMO][npol*NMO]
  memory::shared_array<HOST_MEMORY,ComplexType,3> hij;

  // half rotated one body hamiltonian: [ndet][nup+ndn][npol*NMO]
  memory::shared_array<MEM,ComplexType,3> haj;

  // Xsiu[nspin][npol*NMO][nu]
  memory::shared_array<MEM,ValueType,3> _Xsiu_;

  // Ydsau[ndet][nspin][ipol][nup][nu]
  memory::shared_array<MEM,ComplexType,5> _Ydsau_;

  // Luv[nu][nv]
  memory::shared_array<MEM,ValueType,2> _Luv_;

  // Zuv[nu][nv]
  std::optional<decltype(_Luv_)> _Zuv_;

  // Xsiu_rot[nspin][npol*NMO][nu_rot]
  std::optional<decltype(_Xsiu_)> _Xsiu_rot_;

  // Ydsau_rot[ndet][nspin][ipol][nup][nu_rot]
  std::optional<decltype(_Ydsau_)> _Ydsau_rot_; 

  // Zuv_rot[nu_rot][nu_rot]
  std::optional<decltype(_Luv_)> _Zuv_rot_; 

  // vexx(i,l) = -0.5 * sum_j <ij|jl>
  memory::shared_array<HOST_MEMORY,ComplexType,3> vexx;

  ComplexType E0;

  long default_buffer_size_in_MB = 4L * 1024L;
};

} // namespace afqmc

} // namespace sfqmc

