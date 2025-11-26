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
 *
 */


template<MEMORY_SPACE _MEM>
class KPTHCOps
{
  static constexpr MEMORY_SPACE MEM = _MEM;

public:
  static constexpr HamiltonianTypes HamOpType = KPTHC;
  constexpr HamiltonianTypes getHamType() const { return KPTHC; }

  /*
   * nup/ndown stands for number of alpha/beta electrons
   */
  KPTHCOps(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> ctxt,
         WALKER_TYPES type,
         long nmo_,
         long nup_, 
         long ndn_, 
         long nkpts_,
         memory::shared_array<MEM,ComplexType,4>&& hij_,
         memory::shared_array<MEM,ComplexType,3>&& haj_,
         memory::shared_array<MEM,ComplexType,4>&& x_,
         memory::shared_array<MEM,ComplexType,6>&& y_,
         memory::shared_array<MEM,ComplexType,3>&& l_,
         std::optional<memory::shared_array<MEM,ComplexType,3>>&& z_,
         std::optional<memory::shared_array<MEM,ComplexType,4>>&& x_rot_,
         std::optional<memory::shared_array<MEM,ComplexType,6>>&& y_rot_,
         std::optional<memory::shared_array<MEM,ComplexType,3>>&& z_rot_,
         memory::shared_array<MEM,ComplexType,4>&& v0_,
         ComplexType e0_)
      : mpi(ctxt), 
        walker_type(type),
        NMO(nmo_),
        nup(nup_),
        ndown(ndn_),
        nelec{nup, ndown},
        nkpts(nkpts_),
        nbnd(NMO/nkpts),
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
/*
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
    utils::check(vexx.shape() == std::array<long,3>{nstot*nptot,NMO,NMO},"KPTHCOps: Size mismatch"); 
    utils::check(hij.shape() == std::array<long,3>{nstot,nptot*NMO,nptot*NMO},"KPTHCOps: Size mismatch"); 
    utils::check(haj.shape() == std::array<long,3>{ndet,nel,npol*NMO},"KPTHCOps: Size mismatch"); 
    utils::check(_Xsiu_.shape() == std::array<long,3>{nstot,npol*NMO,nu},"KPTHCOps: Size mismatch"); 
    utils::check(_Ydsau_.shape() == std::array<long,5>{ndet,nspin,npol,nup,nu},"KPTHCOps: Size mismatch"); 
    utils::check(_Luv_.shape() == std::array<long,2>{nu,nv},"KPTHCOps: Size mismatch"); 
    if(_Zuv_.has_value())
      utils::check(_Zuv_->shape() == std::array<long,2>{nu,nu},"KPTHCOps: Size mismatch"); 
    if(_Xsiu_rot_.has_value())
      utils::check(_Xsiu_rot_->shape() == std::array<long,3>{nstot,npol*NMO,nu_rot},"KPTHCOps: Size mismatch"); 
    if(_Ydsau_rot_.has_value())
      utils::check(_Ydsau_rot_->shape() == std::array<long,5>{ndet,nspin,npol,nup,nu_rot},"KPTHCOps: Size mismatch"); 
    if(_Zuv_rot_.has_value())
      utils::check(_Zuv_rot_->shape() == std::array<long,2>{nu_rot,nu_rot},"KPTHCOps: Size mismatch"); 
*/
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
    int nptot = hij.extent(1)/NMO;

    nda::array<ComplexType, 3> H1(nspin, npol*NMO, npol*NMO);
    H1() = ComplexType(0.0);

/*
    // v[nstot][nwalk=1][nptot*NMO][NMO]
    nda::array<ComplexType, 4> v;
    {
      memory::buffered_array<MEM,ComplexType,2> vMF_2d(vMF.size(),1);
      vMF_2d(all,0) = vMF();
      v = std::move(nda::to_host(vHS(vMF_2d, dt)));
    }
    
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
                H1(is,p1*NMO+i,p2*NMO+j) = v(is_,0,p1_*NMO+i,j) + 
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
 */     
    return H1;
  }

  nda::array<int,1> getFieldTypes() const {
    int nvc = number_of_cholesky_vectors();
    nda::array<int,1> v(nvc, int(ContinuousChargePropagator));
    return v;
  }

  void energy(nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              int idet,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
/*
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
              nda::tensor::elementwise(ComplexType(1.0),Guv,"wuv",
                                       ComplexType(1.0),Zuv,"uv",nda::tensor::op::MUL);
            }

            // R[w,u][b] = sum_v Guv[w,u][v] * rotcXau[b][v]
            auto Yau = Ysau(ispin,p2,range(nelec[ispin]),all);
            nda::tensor::contract(Yau,"av",Guv,"wuv",Tav,"wau"); 

            // reuse Guv memory
            memory::array_view<MEM,ComplexType,3> Twbi(std::array<long,3>{nw,nelec[ispin],NMO},Guv.data());
            //T[w][b][k] = sum_u R[w][u][b] * Piu[k][u]
            auto Xiu = Xsiu(is_,range(ip1_*NMO,(ip1_+1)*NMO),all);
            nda::tensor::contract(Tav,"wau",Xiu,"iu",Twbi,"wai"); 

            // E[w] = sum_ai T[w][a][i] * G[w][a][i] 
            auto Gwai = G3d(range(iw, iw + nw),range(ispin*nup,nup+ispin*ndown),range(p1*NMO,(p1+1)*NMO)); 
            memory::buffered_array<MEM,ComplexType,1> Ew(nw);
            nda::tensor::contract(ComplexType(-0.5*scl),Twbi,"wai",Gwai,"wai",ComplexType(0.0),Ew,"w"); 

            if constexpr (MEM==HOST_MEMORY) 
              E(range(iw, iw + nw), 1) += Ew();
            else
              utils::check(false,"finish");
          }
        }
      }
      if (addEJ)
      {
        memory::buffered_array<MEM,ComplexType,2> Twu(nw,nu);
        memory::buffered_array<MEM,ComplexType,1> Ew(nw);
        nda::blas::gemm(Guu,Zuv,Twu);
        nda::tensor::contract(ComplexType(RealType(0.5 * scl * scl)),Guu,"wu",Twu,"wu",
                              ComplexType(0.0),Ew,"w"); 
// NEED ACCUMULATE WITH CASTING
        if constexpr (MEM==HOST_MEMORY) 
          E(range(iw, iw + nw), 2) += Ew();
        else
          utils::check(false,"finish");
      }
      iw += nw;
    }
*/
  }
/*
  template<class Mat, class MatB, class MatC>
  void energy([[maybe_unused]] SpinTypes spin_component,
              [[maybe_unused]] Mat&& E,
              [[maybe_unused]] MatB const& Gc,
              [[maybe_unused]] int nd,
              [[maybe_unused]] MatC&& EJn,
              [[maybe_unused]] bool addH1  = true,
              [[maybe_unused]] bool addEJ  = true,
              [[maybe_unused]] bool addEXX = true)
  {
    APP_ABORT(" Error: spin-dependent energy not implemented ");
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
*/

  // returns v[nwalk, nspin_in_basis*npol_in_basis, NMO, NMO]
  // no spin-orbit vHS yet
  auto vHS(nda::MemoryArrayOfRank<2> auto const& X, double dt)
  {
    memory::check_memory_space<MEM>(X);
    using nda::range;
    auto all = range::all;
    int nchol = 2 * _Luv_().extent(1);
    int nwalk = X.extent(1);
    long nstot = _Xsiu_().shape()[0];
    long nptot = _Xsiu_().shape()[1]/NMO;

/*
    utils::check_strides(X);
    utils::check(X.shape() == std::array<long,2>{nchol,nwalk}, "THC::vbias: Size mismatch.");
*/

    // Note: Allocate first, to make better use of memory pool
    // vHS[nspin_in_vHS][nwalk][npol_in_vHS*NMO][NMO]
    memory::buffered_array<MEM,ComplexType,4> v(nstot,nwalk,nptot*NMO,NMO);
    v() = ComplexType(0.0);

/*
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
    nda::tensor::contract(X_r,"vwc",Luv2,"uv",Twu_r,"wuc");

    // v[w][is*npol+ip][i][j] = sum_u conj(X[is][ip*NMO+i][u]) * X[is][ip*NMO+j][u] * T[u][w] 
    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      memory::buffered_array<MEM,ComplexType,3> Qwiu(nw,NMO,nu);
      for( int is=0; is<nstot; ++is) {
        for( int ip=0; ip<nptot; ++ip) {
       
          auto Xiu = Xsiu(is,range(ip*NMO,(ip+1)*NMO),all); 
 
          // Qwiu[w][i][u] = T[w][u] * conj(Piu[i][u])
          if constexpr (MEM==HOST_MEMORY) {
            for(int w=0; w<nw; ++w)
              for(int i=0; i<NMO; ++i)
                Qwiu(w,i,all) = Twu(w,all) * nda::conj(Xiu(i,all));
          } else {
            nda::tensor::elementwise(Twu,"wu",nda::conj(Xiu),"iu",Qwiu,"wiu",nda::tensor::op::MUL); 
          }

          auto vij = v(is,range(iw,iw+nw),range(ip*NMO,(ip+1)*NMO),all);
          nda::tensor::contract(ComplexType(a),Qwiu,"wiu",Xiu,"ju",
                                ComplexType(0.0),vij,"wij");

        }
      }
      iw += nw;
    }
*/
    return v;
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {

/*
 
// In gpu, compute all q's simultaneously since layout is compatible with strided batch
for k

for q
G(w,u) = sum_a,j Y(s,k,a,u) * G(w,k,a,k-q,j) * X(s,k-q,j) 
T(w,q,n) = sum_u G(w,u)*L(q,u,n)
v(w,q,n) += 0.5*T(w,q,n) * (q==0 ? 2.0 : 1.0)
if(q != -q) v(w,-q,n) -= i*0.5*T(w,q,n)
end q

for *q (limited to cases where q == -q)
G(w,u) = sum_a,j Y(s,k-q,a,u) * G(w,k-q,a,k,j) * X(s,k,j) 
T(w,q,n) = sum_u G(w,u)*conj(L(q,u,n))
// only negative component!
v(w,-q,n) -= i*0.5*T(w,q,n)
end q

end k
 
 */
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
    RealType a(std::sqrt(dt));

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
      for(int is=0; is<nkpts; is++) {
        for(int ik=0; ik<nkpts; ik++) {

          // A(w,k2,u) = sum_a,j Y(d,s,k,a,u) * G(w,k,a,k2,j) * X(s,k2,j)
          Gu_from_compact(G5d, Awu, is, ik, 0);

          // B(q) = A(k-q), now reorder according to k-q
          for(int iq=0; iq<nkpts; iq++) {
            int k2 = 0; // qk_to_k2(iq,ik);
            Bwu(all,iq,all) = Awu(all,k2,all);
          }

          //A(w,q,n) = sum_u B(w,q,u)*L(q,u,n)
          nda::tensor::contract(Luv,"qun",Bwu,"wqu",Awn,"wqn"); 

          //v+(w,q,n) += 0.5*T(w,q,n) * (q==0 ? 2.0 : 1.0)
          nda::tensor::add(ComplexType(0.5), Awn, "wqn", 
                           ComplexType(1.0), v4d(all,0,all,all), "wqn");

          // how to batch these???
          for(int iq=0; iq<nkpts; iq++) {
            //if(q != -q) v(w,-q,n) -= i*0.5*T(w,q,n)
            if(iq != minusq(iq))
              nda::tensor::add(ComplexType(0.0,-0.5), Awn(all,iq,all), "wn", 
                               ComplexType(1.0), v4d(all,1,minusq(iq),all), "wn");
          }
/*
          // now v- contributions for q == -q
          for(int iq=0; iq<nkpts; iq++) {
            if(iq != minusq(iq)) continue;
            int k2 = 0; // qk_to_k2(iq,ik);
            // G(w,u) = sum_a,j Y(d,s,k-q,a,u) * G(w,k-q,a,k,j) * X(s,k,j)
            Gu_from_compact(G5d, Gwu, is, k2, ik, 0);
              
            //T(w,q,n) = sum_u G(w,u)* conj(L(q,u,n))
            nda::tensor::contract(nda::conj(Luv(iq,all,all)),"un",Gwu,"wu",Twn,"nw");
             
            //v(w,-q,n) -= i*0.5*T(w,q,n) 
            v(range(iq*nv,(iq+1)*nv),all) -= ComplexType(0.0,-0.5) * Twn();
          } // iq* 
*/
        } // ik 
      }  // is
    }
    else
    {
      utils::check(false," Error: THC not yet implemented for multiple references.");
      // multideterminant is not half-rotated, so use Likn
      // which spin???
//      memory::array_view<MEM,const ComplexType,4> G3d(std::array<long,4>{nwalk,nspin,npol*NMO,npol*NMO},G.data());
    }
  }
/*
  template<class Mat, class MatB>
  void generalizedFockMatrix([[maybe_unused]] Mat&& G, [[maybe_unused]] MatB&& Fp, [[maybe_unused]] MatB&& Fm)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

*/
  
  /// Returns the number of spins and polarizations in the VHS potential.
  auto vHS_dims() const {
    return std::make_tuple(_Xsiu_().shape()[0],_Xsiu_().shape()[2]/NMO);
  }
  bool distribution_over_cholesky_vectors() const { return false; }
  int number_of_ke_vectors() const { 
    utils::check(_Zuv_.has_value() or _Zuv_rot_.has_value(), "Missing Zuv/Zuv_rot.");
    if(_Zuv_.has_value()) return nkpts*_Zuv_->extent(1);
    else return nkpts*_Zuv_rot_->extent(1); 
  }
  int number_of_cholesky_vectors() const { return 2 * nkpts * _Luv_().extent(2); }

  bool fast_ph_energy() const { return false; }
  nda::array<ComplexType, 2> getHSPotentials() 
  { return nda::array<ComplexType, 2>{}; }

protected:
  // for a given ispin, k1, k2
  // G(w,u) = sum_a,j Y(d,s,k1,a,u) * G(w,k,a,k2,j) * X(s,k2,j)
  void Gu_from_compact(nda::MemoryArrayOfRank<5> auto const& G,
               nda::MemoryArrayOfRank<2> auto && Gwu,
               int ispin, int k1, int k2, int idet)
  {
  }

  // for a given ispin, k
  // G(w,k2,u) = sum_a,j Y(d,s,k1,a,u) * G(w,k,a,k2,j) * X(s,k2,j)
  void Gu_from_compact(nda::MemoryArrayOfRank<5> auto const& G,
               nda::MemoryArrayOfRank<3> auto && Gwku,
               int ispin, int k, int idet)
  {
/*
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

        auto Gwai = G(all,range(is*nup,nup+is*ndown),range(ip*NMO,(ip+1)*NMO)); 
        nda::tensor::contract(Gwai,"wai",Xiu,"iu",Twau,"wau");
        // Gwu[w][u] = a * sum_a T1[w][a][u] * cXau[a][u]
        nda::tensor::contract(ComplexType(a),Twau,"wau",Yau,"au",ComplexType(1.0),Guu,"wu");
      
      } // npol 
    } // nspin 
*/
  }

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
/*
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

    auto Gwai = G(all,range(ispin*nup,nup+ispin*ndown),range(p2*NMO,(p2+1)*NMO));
    // Twav[w][a][v] = sum_j G[w][a][j] X[j][v]
    nda::tensor::contract(Gwai,"wai",Xiu,"iv",Twav,"wav");
    // G[w][u][v] = sum_a X[a][u] Twav[w][a][v]
    nda::tensor::contract(Yau,"au",Twav,"wav",Guv,"wuv");

    // Gwv = Gwvv, 
    if(p1==p2) {
      if constexpr (MEM==HOST_MEMORY) {
        for(int i=0; i<nw; i++)
          Guu(i,all) += nda::diagonal(Guv(i,all,all));
      } else {
        utils::check(false, "finish");
      }
    }
*/
  }
protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int NMO, nup, ndown;
  int nelec[2];
  int nkpts, nbnd;

  // BZ information
  nda::array<int,1> minusq;
  nda::array<int,2> qk_to_k2;
  int Q0_index = 0;

  // H1[nspin][nk][npol*nbnd][npol*nbnd]
  memory::shared_array<MEM,ComplexType,4> hij;

  // half rotated one body hamiltonian: [ndet][nup+ndn][npol*NMO]. Kept in full basis
  memory::shared_array<MEM,ComplexType,3> haj;

  // Xsiu[nspin][nk][npol*nbnd][nu]
  memory::shared_array<MEM,ComplexType,4> _Xsiu_;

  // Ydsau[ndet][nspin][ipol][nk][nup][nu]
  memory::shared_array<MEM,ComplexType,6> _Ydsau_;

  // Luv[nu][nv]
  memory::shared_array<MEM,ComplexType,3> _Luv_;

  // Zuv[nu][nv]
  std::optional<decltype(_Luv_)> _Zuv_;

  // Xsiu_rot[nspin][npol*nbnd][nu_rot]
  std::optional<decltype(_Xsiu_)> _Xsiu_rot_;

  // Ydsau_rot[ndet][nspin][ipol][nup][nu_rot]
  std::optional<decltype(_Ydsau_)> _Ydsau_rot_; 

  // Zuv_rot[nu_rot][nu_rot]
  std::optional<decltype(_Luv_)> _Zuv_rot_; 

  // vexx(i,l) = -0.5 * sum_j <ij|jl> : [nspin][nk][npol*nbnd][npol*nbnd]
  memory::shared_array<MEM,ComplexType,4> vexx;

  ComplexType E0;

};

} // namespace afqmc

} // namespace sfqmc

