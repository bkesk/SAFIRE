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

template<MEMORY_SPACE _MEM, bool MP, bool REAL>
class THCOps
{
  static constexpr MEMORY_SPACE MEM = _MEM;

  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type; 

  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  using SPValueType   = typename to_working_precision<MP,ValueType>::type; 

public:
  static const HamiltonianTypes HamOpType = THC;
  HamiltonianTypes getHamType() const { return HamOpType; }

  /*
   * nup/ndown stands for number of alpha/beta electrons
   */
  THCOps(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> ctxt,
         WALKER_TYPES type,
         long nmo_,
         long nup_, 
         long ndn_, 
         memory::shared_array<MEM,ComplexType,3>&& hij_,
         memory::shared_array<MEM,ComplexType,3>&& haj_,
         memory::shared_array<MEM,SPValueType,3>&& x_,
         memory::shared_array<MEM,SPComplexType,5>&& y_,
         memory::shared_array<MEM,SPValueType,2>&& l_,
         std::optional<memory::shared_array<MEM,SPValueType,2>>&& z_,
         std::optional<memory::shared_array<MEM,SPValueType,3>>&& x_rot_,
         std::optional<memory::shared_array<MEM,SPComplexType,5>>&& y_rot_,
         std::optional<memory::shared_array<MEM,SPValueType,2>>&& z_rot_,
         memory::shared_array<MEM,ComplexType,3>&& v0_,
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
        v0(std::move(v0_)),
        E0(e0_)
  {
/*
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    gnmu    = Luv.size(1);
    grotnmu = rotMuv.size(1);
    if (npol > 1)
      APP_ABORT(" Error: THC not yet implemented for non-collinear calculations.");
    if ((walker_type == NONCOLLINEAR) and ndown > 0)
      APP_ABORT(" Error in THC: Noncollinear calculation with ndown>0. "); 
    if (haj.size() > 1)
      APP_ABORT(" Error: THC not yet implemented for multiple references.");
    utils::check(comm, "THC: nullptr communicator.");
    // current partition over 'u' for L/Piu
    utils::check(Luv.size(0) == Piu.size(1), " THC: Shape mismatch");
    for (int i = 0; i < rotcXau.size(); i++)
    {
      // rot Ps are not yet distributed
      utils::check(rotcXau[i].size(1) == rotPiu.size(1), " THC: Shape mismatch");
      if (walker_type == CLOSED)
        utils::check(rotcXau[i].size(0) == nup, " THC: Shape mismatch");
      else if (walker_type == COLLINEAR)
        utils::check(rotcXau[i].size(0) == nup + ndown, " THC: Shape mismatch");
      else if (walker_type == NONCOLLINEAR)
        utils::check(rotcXau[i].size(0) == npol * (nup + ndown), " THC: Shape mismatch" );
    }
    for (int i = 0; i < cXau.size(); i++)
    {
      utils::check(cXau[i].size(1) == Luv.size(0), " THC: Shape mismatch");
      if (walker_type == CLOSED)
        utils::check(cXau[i].size(0) == nup, " THC: Shape mismatch");
      else if (walker_type == COLLINEAR)
        utils::check(cXau[i].size(0) == nup + ndown, " THC: Shape mismatch");
      else if (walker_type == NONCOLLINEAR)
        utils::check(cXau[i].size(0) == npol * (nup + ndown), " THC: Shape mismatch");
    }
    utils::check(Piu.size(0) == nspin * npol * NMO, " THC: Shape mismatch");
    utils::check(rotPiu.size(0) == nspin * npol * NMO, " THC: Shape mismatch");
    utils::check(vn0.size(0) == nspin * NMO, " THC: Shape mismatch");
    utils::check(vn0.size(1) == NMO, " THC: Shape mismatch");
    utils::check(hij.size(0) == nspin * npol * NMO, " THC: Shape mismatch");
    utils::check(hij.size(1) == npol * NMO, " THC: Shape mismatch");
*/
  }

  ~THCOps() {}

  THCOps(THCOps const& other) = default;
  THCOps& operator=(THCOps const& other) = default;

  THCOps(THCOps&& other) = default;
  THCOps& operator=(THCOps&& other) = default;

/*
  boost::multi::array<ComplexType, 2> getOneBodyPropagatorMatrix(TaskGroup_& TG, double dt,
                                                                 boost::multi::array<ComplexType, 1> const& vMF)
  {
    using std::copy_n;
    if(walker_type == NONCOLLINEAR)
      APP_ABORT("Error: Noncollinear not yet implemented in THCOps.\n ");
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;

    ShmArray<ComplexType, 1> vMF_(vMF, shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    ShmArray<ComplexType, 1> P1D(iextensions<1u>{nspin * NMO * NMO}, ComplexType(0),
                                 shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    auto P0 = P1D.partitioned(nspin*NMO);

    vHS(vMF_, P1D, dt);
    if (TG.TG_Cores().size() > 1 && TG.TG_local().root())
      TG.TG_Cores().all_reduce_in_place_n(raw_pointer_cast(P1D.origin()), P1D.num_elements(), std::plus<>());
    TG.TG().barrier();

    boost::multi::array<ComplexType, 2> H1({nspin * npol * NMO, npol * NMO});
    
    // copy hij since it must have full spinor structure
    std::fill_n(H1.origin(), H1.num_elements(), ComplexType(0));

    // add hij + vn0 and symmetrize
    using ma::conj;
    for (int i = 0; i < NMO; i++)
    {
      H1[i][i] = P0[i][i] + dt * (hij[i][i] + vn0[i][i]);
      if(walker_type == COLLINEAR)
        H1[NMO+i][i] = P0[NMO+i][i] + dt * (hij[NMO+i][i] + vn0[NMO+i][i]);
      else if(walker_type == NONCOLLINEAR)
        H1[NMO+i][NMO+i] = P0[NMO+i][i] + dt * (hij[NMO+i][NMO+i] + vn0[NMO+i][i]);

      for (int j = i + 1; j < NMO; j++)
      {
        H1[i][j] = P0[i][j] + dt * (hij[i][j] + vn0[i][j]);
        H1[j][i] = P0[j][i] + dt * (hij[j][i] + vn0[j][i]);
        if (std::abs(H1[i][j] - ma::conj(H1[j][i])) > 1e-5)
        {
          app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 is not hermitian. ");
          app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[i][j],H1[j][i]);
          app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[i][j],hij[j][i]);
          app_warning("             vn0[I,J]:{}, vn0[J,I]:{} ",vn0[i][j],vn0[j][i]);
        }
        H1[i][j] = 0.5 * (H1[i][j] + ma::conj(H1[j][i]));
        H1[j][i] = ma::conj(H1[i][j]);
        if(walker_type == COLLINEAR) {
          H1[NMO+i][j] = P0[NMO+i][j] + dt * (hij[NMO+i][j] + vn0[NMO+i][j]);
          H1[NMO+j][i] = P0[NMO+j][i] + dt * (hij[NMO+j][i] + vn0[NMO+j][i]);
          // This is really cutoff dependent!!!
          if (std::abs(H1[NMO+i][j] - ma::conj(H1[NMO+j][i])) > 1e-6)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (beta) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[NMO+i][j],H1[NMO+j][i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[NMO+i][j],hij[NMO+j][i]);
            app_warning("             vn0[I,J]:{}, vn0[J,I]:{} ",vn0[NMO+i][j],vn0[NMO+j][i]);
          }
          H1[NMO+i][j] = 0.5 * (H1[NMO+i][j] + ma::conj(H1[NMO+j][i]));
          H1[NMO+j][i] = ma::conj(H1[NMO+i][j]);
        } else if(walker_type == NONCOLLINEAR) {
          // dn/dn 
          H1[NMO+i][NMO+j] = P0[NMO+i][j] + dt * (hij[NMO+i][NMO+j] + vn0[NMO+i][j]);
          H1[NMO+j][NMO+i] = P0[NMO+j][i] + dt * (hij[NMO+j][NMO+i] + vn0[NMO+j][i]);

          // spin-orbit terms, a-b and b-a are hij only!
          H1[NMO+i][j] = dt * hij[NMO+i][j];
          H1[NMO+j][i] = dt * hij[NMO+j][i];

          H1[i][NMO+j] = dt * hij[i][NMO+j];
          H1[j][NMO+i] = dt * hij[j][NMO+i];

          // This is really cutoff dependent!!!
          if (std::abs(H1[NMO+i][NMO+j] - ma::conj(H1[NMO+j][NMO+i])) > 1e-6) // b-b
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (beta-beta) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[NMO+i][NMO+j],H1[NMO+j][NMO+i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[NMO+i][NMO+j],hij[NMO+j][NMO+i]);
            app_warning("             vn0[I,J]:{}, vn0[J,I]:{} ",vn0[NMO+i][NMO+j],vn0[NMO+j][NMO+i]);
          }

          if (std::abs(H1[i][NMO+j] - ma::conj(H1[NMO+j][i])) > 1e-6)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (spin-flip) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[i][NMO+j],H1[NMO+j][i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[i][NMO+j],hij[NMO+j][i]);
          }

          H1[NMO+i][NMO+j] = 0.5 * (H1[NMO+i][NMO+j] + ma::conj(H1[NMO+j][NMO+i]));
          H1[NMO+j][NMO+i] = ma::conj(H1[NMO+i][NMO+j]);

          H1[i][NMO+j] = 0.5 * (H1[i][NMO+j] + ma::conj(H1[NMO+j][i]));
          H1[NMO+j][i] = ma::conj(H1[i][NMO+j]);

          H1[NMO+i][j] = 0.5 * (H1[NMO+i][j] + ma::conj(H1[j][NMO+i]));
          H1[j][NMO+i] = ma::conj(H1[NMO+i][j]);
        }
      }
    }

    return H1;
  }

  template<class TVec>
  void getFieldTypes(TVec&& v) {
    int localnvc = local_number_of_cholesky_vectors();
    RUNTIME_CHECK(v.size() == localnvc, "");
    using std::fill_n;
    fill_n( v.origin(), v.size(), ContinuousChargePropagator );
  }
*/
  void energy(nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              int idet,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    using nda::range;
    using GType = nda::get_value_t<decltype(G)>;
    auto all = range::all;
    int nwalk = G.extent(0);
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "THC::energy: Size mismatch.");
    utils::check(G.extent(1) == nel*npol*NMO, "THC::energy: Size mismatch.");
    utils::check_strides(E,G);
    // limiting G to contiguous arrays for simplicity now, reconsider if necessary
    utils::check(G.is_contiguous(), "Layout mismatch");

    // addH1
    E() = ComplexType(0.0);
    if (addH1)
    {
      E(all,0) = E0; 
      auto haj_2d = nda::reshape(haj(),std::array<long,2>{haj.extent(0),haj.extent(1)*haj.extent(2)});
      nda::blas::gemv(ComplexType(1.0), G, haj_2d(idet,all), ComplexType(1.0), E(all,0));
    }
    if (not(addEJ || addEXX))
      return;

    if (idet > 0)
      APP_ABORT(" Error: THC not yet implemented for multiple references.");

    // get array_views to the correct data and correct determinant
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xsiu = ( has_rot ? (*_Xsiu_rot_)() : _Xsiu_() );
    const auto Ysau = ( has_rot ? (*_Ydsau_rot_)()(idet,nda::ellipsis{}) : _Ydsau_()(idet,nda::ellipsis{}) );
    const auto Zuv = ( has_rot ? (*_Zuv_rot_)() : (*_Zuv_)() );

    int nu = Zuv.extent(0);
    long nstot = _Xsiu_().shape()[0];
    long nptot = _Xsiu_().shape()[1]/NMO; 

    // calculate how many walkers can be done concurrently
    long mem_needs(0);
    if (not std::is_same_v<GType, SPComplexType>)
      mem_needs += G.size();
    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
    Bytes -= mem_needs * long(sizeof(SPComplexType));
    Bytes /= long((nu * nu + nu + nu * nup) * sizeof(SPComplexType));
    int nwmax = std::min(nwalk, std::max(1, int(Bytes)));
    
    memory::buffered_array<MEM,SPComplexType,1> Gbuff(mem_needs);
    SPComplexType const* Gptr = nullptr;
    // setup origin of Gsp and copy_n_cast if necessary
    if constexpr (std::is_same_v<GType, SPComplexType>)
    {
      Gptr = reinterpret_cast<SPComplexType const*>(G.data()); 
    }
    else
    {
      auto Gb = nda::reshape(Gbuff,G.shape());
      nda::copy_cast(G,Gb);
      Gptr = Gbuff.data(); 
    }
    // fine because G is assumed contiguous, otherwise build nda::idx_map with custom strides
    memory::array_view<MEM,const SPComplexType,3> G3d(std::array<long,3>{nwalk,nel,npol*NMO},Gptr);

    SPRealType scl = (walker_type == CLOSED ? 2.0 : 1.0);
    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      // Guv[nspin][nu][nv]
      memory::buffered_array<MEM,SPComplexType,3> Guv(nw,nu,nu);
      // Guu[u]: summed over spin
      memory::buffered_array<MEM,SPComplexType,2> Guu(nw,nu);
      Guu() = SPComplexType(0.0);
      for (int ispin = 0; ispin < nspin; ++ispin)
      {
        long is_ = long(ispin)%nstot; 
        for (int p1 = 0; p1 < npol; ++p1)
        {
          long ip1_ = long(p1)%nptot; 
          for (int p2 = 0; p2 < npol; ++p2)
          {
            // Buffer space
            memory::buffered_array<MEM,SPComplexType,3> Tav(nw,nelec[ispin],nu);
            Guv_Guu(ispin, p1, p2, G3d(range(iw, iw + nw), all, all), Guv, Guu, Tav, idet);

            if constexpr (MEM==HOST_MEMORY) {
              for(int i=0; i<nw; ++i)
                Guv(i,nda::ellipsis{}) *= Zuv();
            } else {
              nda::tensor::elementwise(SPComplexType(1.0),Guv,"wuv",
                                       SPComplexType(1.0),Zuv,"uv",nda::tensor::op::MUL);
            }

            // R[w,u][b] = sum_v Guv[w,u][v] * rotcXau[b][v]
            auto Yau = Ysau(ispin,p2,range(nelec[ispin]),all);
            nda::tensor::contract(Yau,"av",Guv,"wuv",Tav,"wau"); 

            // reuse Guv memory
            memory::array_view<MEM,SPComplexType,3> Twbi(std::array<long,3>{nw,nelec[ispin],NMO},Guv.data());
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
            memory::buffered_array<MEM,SPComplexType,1> Ew(nw);
            nda::tensor::contract(SPComplexType(-0.5*scl),Twbi,"wai",Gwai,"wai",SPComplexType(0.0),Ew,"w"); 

            if constexpr (MEM==HOST_MEMORY) 
              E(range(iw, iw + nw), 1) += Ew();
            else
              utils::check(false,"finish");
          }
        }
      }
      if (addEJ)
      {
        memory::buffered_array<MEM,SPComplexType,2> Twu(nw,nu);
        memory::buffered_array<MEM,SPComplexType,1> Ew(nw);
	if constexpr (REAL) {
          // use strategy in Guv_Guu
          utils::check(false,"finish");
	} else {
          nda::blas::gemm(Guu,Zuv,Twu);
	}
        nda::tensor::contract(SPComplexType(SPRealType(0.5 * scl * scl)),Guu,"wu",Twu,"wu",
                              SPComplexType(0.0),Ew,"w"); 
// NEED ACCUMULATE WITH CASTING
        if constexpr (MEM==HOST_MEMORY) 
          E(range(iw, iw + nw), 2) += Ew();
        else
          utils::check(false,"finish");
      }
      mpi->comm.barrier();
      iw += nw;
    }
    mpi->comm.barrier();
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

  template<class MatE, class MatO, class MatG, class MatQ, class MatB, class index_aos>
  void fast_energy([[maybe_unused]] MatE&& E,
                   [[maybe_unused]] MatO&& Ov,
                   [[maybe_unused]] MatG const& GrefA,
                   [[maybe_unused]] MatG const& GrefB,
                   [[maybe_unused]] MatQ const& QQ0A,
                   [[maybe_unused]] MatQ const& QQ0B,
                   [[maybe_unused]] MatB&& Qwork,
                   [[maybe_unused]] ph_excitations<int, ComplexType> const& abij,
                   [[maybe_unused]] std::array<index_aos, 2> const& det_couplings)
  {
    APP_ABORT(" Error: fast_energy not yet working");
    if (haj.size() != 1)
      APP_ABORT(" Error: Single reference implementation currently in THCOps::fast_energy.");
    if (walker_type != CLOSED)
      APP_ABORT(" Error: THCOps::fast_energy requires walker_type==CLOSED.");
    /*
       * E[nspins][maxn_unique_confg][nwalk][3]
       * Ov[nspins][maxn_unique_confg][nwalk]
       * GrefA[nwalk][nup][NMO]
       * GrefB[nwalk][ndown][NMO]
       * QQ0A[nwalk][nup][NAEA]
       * QQ0B[nwalk][nup][NAEA]
       */
    /*
      static_assert(std::decay<MatE>::type::dimensionality==4, "Wrong dimensionality");
      static_assert(std::decay<MatO>::type::dimensionality==3, "Wrong dimensionality");
      static_assert(std::decay<MatG>::type::dimensionality==3, "Wrong dimensionality");
      static_assert(std::decay<MatQ>::type::dimensionality==3, "Wrong dimensionality");
      //static_assert(std::decay<MatB>::type::dimensionality==3, "Wrong dimensionality");
      int nspin = E.size(0);
      int nrefs = haj.size();
      int nwalk = GrefA.size(0);
      int naoa_ = QQ0A.size(1);
      int naob_ = QQ0B.size(1);
      int nmo_ = rotPiu.size(0);
      int nu = rotMuv.size(0);
      int nu0 = rotnmu0; 
      int nv = rotMuv.size(1);
      int nel_ = rotcXau[0].size(1);
      // checking
      RUNTIME_CHECK(E.size(2) == nwalk, "");
      RUNTIME_CHECK(E.size(3) == 3, "");
      RUNTIME_CHECK(Ov.size(0) == nspin, "");
      RUNTIME_CHECK(Ov.size(1) == E.size(1), "");
      RUNTIME_CHECK(Ov.size(2) == nwalk, "");
      RUNTIME_CHECK(GrefA.size(1) == naoa_, "");
      RUNTIME_CHECK(GrefA.size(2) == nmo_, "");
      RUNTIME_CHECK(GrefB.size(0) == nwalk, "");
      RUNTIME_CHECK(GrefB.size(1) == naob_, "");
      RUNTIME_CHECK(GrefB.size(2) == nmo_, "");
      // limited to single reference now
      RUNTIME_CHECK(rotcXau.size() == nrefs, "");
      RUNTIME_CHECK(nel_ == naoa_, "");
      RUNTIME_CHECK(nel_ == naob_, "");

      using ma::T;
      int u0,uN;
      std::tie(u0,uN) = FairDivideBoundary(comm->rank(),nu,comm->size());
      int v0,vN;
      std::tie(v0,vN) = FairDivideBoundary(comm->rank(),nv,comm->size());
      int k0,kN;
      std::tie(k0,kN) = FairDivideBoundary(comm->rank(),nel_,comm->size());
      // right now the algorithm uses 2 copies of matrices of size nuxnv in COLLINEAR case,
      // consider moving loop over spin to avoid storing the second copy which is not used
      // simultaneously
      size_t memory_needs = nu*nv + nv + nu  + nel_*(nv+2*nu+2*nel_);
      set_shmbuffer(memory_needs);
      size_t cnt=0;
      // if Alpha/Beta have different references, allocate the largest and
      // have distinct references for each
      // Guv[nu][nv]
      boost::multi::array_ref<ComplexType,2> Guv(raw_pointer_cast(SM_TMats.origin()),{nu,nv});
      cnt+=Guv.num_elements();
      // Gvv[v]: summed over spin
      boost::multi::array_ref<ComplexType,1> Gvv(raw_pointer_cast(SM_TMats.origin())+cnt,iextensions<1u>{nv});
      cnt+=Gvv.num_elements();
      // S[nel_][nv]
      boost::multi::array_ref<ComplexType,2> Scu(raw_pointer_cast(SM_TMats.origin())+cnt,{nel_,nv});
      cnt+=Scu.num_elements();
      // Qub[nu][nel_]:
      boost::multi::array_ref<ComplexType,2> Qub(raw_pointer_cast(SM_TMats.origin())+cnt,{nu,nel_});
      cnt+=Qub.num_elements();
      boost::multi::array_ref<ComplexType,1> Tuu(raw_pointer_cast(SM_TMats.origin())+cnt,iextensions<1u>{nu});
      cnt+=Tuu.num_elements();
      boost::multi::array_ref<ComplexType,2> Jcb(raw_pointer_cast(SM_TMats.origin())+cnt,{nel_,nel_});
      cnt+=Jcb.num_elements();
      boost::multi::array_ref<ComplexType,2> Xcb(raw_pointer_cast(SM_TMats.origin())+cnt,{nel_,nel_});
      cnt+=Xcb.num_elements();
      boost::multi::array_ref<ComplexType,2> Tub(raw_pointer_cast(SM_TMats.origin())+cnt,{nu,nel_});
      cnt+=Tub.num_elements();
      RUNTIME_CHECK(cnt <= memory_needs, "");
      boost::multi::static_array<ComplexType,3,dev_buffer_type> eloc({2,nwalk,3}
                        device_buffer_manager.get_generator().template get_allocator<ComplexType>());
      std::fill_n(eloc.origin(),eloc.num_elements(),ComplexType(0.0));

      RealType scl = (walker_type==CLOSED?2.0:1.0);
      if(comm->root()) {
        std::fill_n(raw_pointer_cast(E.origin()),E.num_elements(),ComplexType(0.0));
        std::fill_n(raw_pointer_cast(Ov[0][1].origin()),nwalk*(Ov.size(1)-1),ComplexType(0.0));
        std::fill_n(raw_pointer_cast(Ov[1][1].origin()),nwalk*(Ov.size(1)-1),ComplexType(0.0));
        auto Ea = E[0][0];
        auto Eb = E[1][0];
        boost::multi::array_cref<ComplexType,2> G2DA(raw_pointer_cast(GrefA.origin()),
                                          {nwalk,GrefA[0].num_elements()});
        ma::product(ComplexType(1.0),G2DA,haj[0],ComplexType(0.0),Ea(Ea.extension(0),0));
        boost::multi::array_cref<ComplexType,2> G2DB(raw_pointer_cast(GrefA.origin()),
                                          {nwalk,GrefA[0].num_elements()});
        ma::product(ComplexType(1.0),G2DB,haj[0],ComplexType(0.0),Eb(Eb.extension(0),0));
        for(int i=0; i<nwalk; i++) {
            Ea[i][0] += E0;
            Eb[i][0] += E0;
        }
      }

      for(int wi=0; wi<nwalk; wi++) {

        { // Alpha
          auto Gw = GrefA[wi];
          boost::multi::array_cref<ComplexType,1> G1D(raw_pointer_cast(Gw.origin()),
                                                        iextensions<1u>{Gw.num_elements()});
          Guv_Guu2(Gw,Guv,Gvv,Scu,0);
          if(u0!=uN)
            ma::product(rotMuv.sliced(u0,uN),Gvv,
                      Tuu.sliced(u0,uN));
          auto Mptr = rotMuv[u0].origin();
          auto Gptr = raw_pointer_cast(Guv[u0].origin());
          for(size_t k=0, kend=(uN-u0)*nv; k<kend; ++k, ++Gptr, ++Mptr)
            (*Gptr) *= (*Mptr);
          if(u0!=uN)
            ma::product(Guv.sliced(u0,uN),rotcXau[0],
                      Qub.sliced(u0,uN));
          comm->barrier();
          if(k0!=kN)
            ma::product(Scu.sliced(k0,kN),Qub,
                      Xcb.sliced(k0,kN));
          // Tub = rotcXau.*Tu
          auto rPptr = rotcXau[0][nu0+u0].origin();
          auto Tuuptr = Tuu.origin()+u0;
          auto Tubptr = Tub[u0].origin();
          for(size_t u_=u0; u_<uN; ++u_, ++Tuuptr)
            for(size_t k=0; k<nel_; ++k, ++rPptr, ++Tubptr)
              (*Tubptr) = (*Tuuptr)*(*rPptr);
          comm->barrier();
          // Jcb = Scu*Tub
          if(k0!=kN)
            ma::product(Scu.sliced(k0,kN),Tub,
                      Jcb.sliced(k0,kN));
          for(int c=k0; c<kN; ++c)
            eloc[0][wi][1] += -0.5*scl*Xcb[c][c];
          for(int c=k0; c<kN; ++c)
            eloc[0][wi][2] += 0.5*scl*scl*Jcb[c][c];
          calculate_ph_energies(0,comm->rank(),comm->size(),
                                E[0],Ov[0],QQ0A,Qwork,
                                rotMuv,
                                abij,det_couplings);
        }

        { // Beta: Unnecessary in CLOSED walker type (on Walker)
          auto Gw = GrefB[wi];
          boost::multi::array_cref<ComplexType,1> G1D(raw_pointer_cast(Gw.origin()),
                                                        iextensions<1u>{Gw.num_elements()});
          Guv_Guu2(Gw,Guv,Gvv,Scu,0);
          if(u0!=uN)
            ma::product(rotMuv.sliced(u0,uN),Gvv,
                      Tuu.sliced(u0,uN));
          auto Mptr = rotMuv[u0].origin();
          auto Gptr = raw_pointer_cast(Guv[u0].origin());
          for(size_t k=0, kend=(uN-u0)*nv; k<kend; ++k, ++Gptr, ++Mptr)
            (*Gptr) *= (*Mptr);
          if(u0!=uN)
            ma::product(Guv.sliced(u0,uN),rotcXau[0],
                      Qub.sliced(u0,uN));
          comm->barrier();
          if(k0!=kN)
            ma::product(Scu.sliced(k0,kN),Qub,
                      Xcb.sliced(k0,kN));
          // Tub = rotcXau.*Tu
          auto rPptr = rotcXau[0][nu0+u0].origin();
          auto Tuuptr = Tuu.origin()+u0;
          auto Tubptr = Tub[u0].origin();
          for(size_t u_=u0; u_<uN; ++u_, ++Tuuptr)
            for(size_t k=0; k<nel_; ++k, ++rPptr, ++Tubptr)
              (*Tubptr) = (*Tuuptr)*(*rPptr);
          comm->barrier();
          // Jcb = Scu*Tub
          if(k0!=kN)
            ma::product(Scu.sliced(k0,kN),Tub,
                      Jcb.sliced(k0,kN));
          for(int c=k0; c<kN; ++c)
            eloc[1][wi][1] += -0.5*scl*Xcb[c][c];
          for(int c=k0; c<kN; ++c)
            eloc[1][wi][2] += 0.5*scl*scl*Jcb[c][c];
        }

      }
      comm->reduce_in_place_n(eloc.origin(),eloc.num_elements(),std::plus<>(),0);
      if(comm->root()) {
        // add Eref contributions to all configurations
        for(int nd=0; nd<E.size(1); ++nd) {
          auto Ea = E[0][nd];
          auto Eb = E[1][nd];
          for(int wi=0; wi<nwalk; wi++) {
            Ea[wi][1] += eloc[0][wi][1];
            Ea[wi][2] += eloc[0][wi][2];
            Eb[wi][1] += eloc[1][wi][1];
            Eb[wi][2] += eloc[1][wi][2];
          }
        }
      }
      comm->barrier();
* /
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

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 1)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 1)>,
           typename = void>
  void vHS(MatA const& X, MatB&& v, double dt, double a = 1., double c = 0.)
  {
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    vHS(X.partitioned(X.size(0)), v.partitioned(nspin), dt, a, c);
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>>
  void vHS(MatA&& X, MatB&& v, double dt, double a = 1., double c = 0.)
  {
    using ma::T;
    using XType = typename std::decay_t<typename std::decay_t<MatA>::element>;
    using vType = typename std::decay_t<MatB>::element;
    using sfqmc::afqmc::reinterpret_pointer_cast;
    // scale a by sqrt(dt)
    a *= std::sqrt(dt);
    //int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int nwalk   = X.size(1);
    int nchol = ( REAL ? Luv.size(1) : 2 * Luv.size(1) );
    int nu   = Piu.size(1);
    utils::check(Luv.size(0) == nu, "THC::vHS: Shape mismatch.");
    utils::check(X.size(0) == nchol, "THC::vHS: Shape mismatch.");
    utils::check(v.size(0) == nspin * nwalk, "THC::vHS: Shape mismatch: v.size:{} nspin:{} nwalk:{}",
                 v.size(0), nspin, nwalk);
    utils::check(v.size(1) == NMO * NMO, "THC::vHS: Shape mismatch.");
   
    size_t memory_needs = nu * nwalk;
    if (not std::is_same<XType, SPComplexType>::value)
      memory_needs += X.num_elements();
    if (not std::is_same<vType, SPComplexType>::value)
      memory_needs += v.num_elements();

    // calculate how many walkers can be done concurrently
    long Bytes = default_buffer_size_in_MB * 1024L * 1024L;
    // memory_needs = X, v, Tuw
    Bytes -= size_t(memory_needs * sizeof(SPComplexType)); // substract other needs
    Bytes /= size_t(NMO * nu * sizeof(SPComplexType));
    int nwmax = std::min(nwalk, std::max(1, int(Bytes)));
    memory_needs += nwmax * NMO * nu;
    ShmArray<SPComplexType, 1> SM_TMats(iextensions<1u>{memory_needs},
                                        shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());

    size_t cnt(0);
    const_sp_pointer Xptr(nullptr);
    sp_pointer vptr(nullptr);
    // setup origin of vsp and copy_n_cast if necessary
    if (std::is_same<vType, SPComplexType>::value)
    {
      vptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(v.origin()));
    }
    else
    {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(comm->rank()), long(v.num_elements()), long(comm->size()));
      vptr             = make_device_ptr(SM_TMats.origin());
      cnt += size_t(v.num_elements());
      if (std::abs(c) > 1e-12)
        copy_n_cast(make_device_ptr(v.origin()) + i0, iN - i0, vptr + i0);
    }
    // setup origin of Xsp and copy_n_cast if necessary
    if (std::is_same<XType, SPComplexType>::value)
    {
      Xptr = reinterpret_pointer_cast<SPComplexType const>(make_device_ptr(X.origin()));
    }
    else
    {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(comm->rank()), long(X.num_elements()), long(comm->size()));
      copy_n_cast(make_device_ptr(X.origin()) + i0, iN - i0, make_device_ptr(SM_TMats.origin()) + cnt + i0);
      Xptr = make_device_ptr(SM_TMats.origin()) + cnt;
      cnt += size_t(X.num_elements());
    }
    // setup array references
    Array_cref<SPComplexType, 2> Xsp(Xptr, X.extensions());
    Array_ref<SPComplexType, 2> vsp(vptr, v.extensions());

    auto v3d = vsp.partitioned(nspin);
    utils::check(v3d.size(0) == nspin and v3d.size(1) == nwalk and v3d.size(2) == NMO * NMO, 
        "THC::vHS: Internal shape mismatch - oh oh!");

    int u0, uN;
    std::tie(u0, uN) = FairDivideBoundary(comm->rank(), nu, comm->size());
    Array_ref<SPComplexType, 2> Tuw(make_device_ptr(SM_TMats.origin()) + cnt, {nu, nwalk});
    // O[nwalk * nmu * nmu]
    if constexpr(REAL) {
      ma::product(Luv.sliced(u0, uN), Xsp, Tuw.sliced(u0, uN));
    } else {
      // reinterpret as RealType matrices with 2x the columns
      Array_ref<SPRealType, 2> Luv_R(reinterpret_pointer_cast<SPRealType>(make_device_ptr(Luv.origin())),
                                   {Luv.size(0), 2 * Luv.size(1)});
      Array_cref<SPRealType, 2> X_R(reinterpret_pointer_cast<SPRealType const>(Xsp.origin()), {Xsp.size(0), 2 * Xsp.size(1)});
      Array_ref<SPRealType, 2> Tuw_R(reinterpret_pointer_cast<SPRealType>(Tuw.origin()), {nu, 2 * nwalk});
      ma::product(Luv_R.sliced(u0, uN), X_R, Tuw_R.sliced(u0, uN));
    }
    comm->barrier();
    int k0, kN;
    std::tie(k0, kN) = FairDivideBoundary(comm->rank(), NMO, comm->size());
    int iw(0);
    while (iw < nwalk)
    {
      int nw = std::min(nwmax, nwalk - iw);
      for( int is=0, m0=0; is<nspin; ++is, m0+=NMO) {
        if constexpr (REAL) {
          // Qwui[w][u][i] = Piu[i][u] * T[u][w]
          // v[w][i][j] = sum_u Piu[i][u] Qwui[w][u][j] // using batched blas
          Array_ref<SPComplexType, 3> Qwui(Tuw.origin() + Tuw.num_elements(), {nw, nu, NMO});
          ma::element_wise_Aij_Bjk_Ckji(Piu.sliced(m0+k0,m0+kN), 
                                        Tuw.rotated().sliced(iw,iw+nw).unrotated(),
                                        Qwui.rotated(2).sliced(k0,kN).unrotated(2)); 
          comm->barrier();
          Array_ref<SPComplexType, 3> v_(v3d[is][iw].origin(), {nw, NMO, NMO});
          std::vector<decltype(&(Piu.sliced(0, 1)))> vPiu;
          std::vector<decltype(&(Qwui[0]))> vQui;
          std::vector<decltype(&(v_[0].sliced(0, 1)))> vVij;
          vPiu.reserve(nw);
          vQui.reserve(nw);
          vVij.reserve(nw);
          for (int w = 0; w < nw; ++w)
          {
            vPiu.emplace_back(&(Piu.sliced(m0+k0, m0+kN)));
            vQui.emplace_back(&(Qwui[w]));
            vVij.emplace_back(&(v_[w].sliced(k0, kN)));
          }
          ma::BatchedProduct('N', 'N', SPRealType(a), vPiu, vQui, SPRealType(c), vVij);
        } else {
          // Qwiu[w][i][u] = T[u][w] * conj(Piu[i][u])
          // v[w][i][k] = sum_u Qwiu[w][i][u] * Piu[k][u]
          Array_ref<SPComplexType, 2> Qwiu(Tuw.origin() + Tuw.num_elements(), {nw * NMO, nu});
          ma::element_wise_Aij_Bjk_Ckij('C',Piu.sliced(m0+k0,m0+kN), 
                          Tuw.rotated().sliced(iw,iw+nw).unrotated(),
                          Qwiu.partitioned(nw).rotated().sliced(k0,kN).unrotated());
          comm->barrier();
          // v[w][i][j] = sum_u Qwiu[w][i][u] * Piu[j][u]
          Array_ref<SPComplexType, 2> v_(v3d[is][iw].origin(), {nw * NMO, NMO});
          int wk0, wkN;
          std::tie(wk0, wkN) = FairDivideBoundary(comm->rank(), nw * NMO, comm->size());
          ma::product(SPComplexType(SPRealType(a)), Qwiu.sliced(wk0, wkN), T(Piu.sliced(m0,m0+NMO)), 
                      SPComplexType(SPRealType(c)), v_.sliced(wk0, wkN));
        }
      }
      iw += nw;
      comm->barrier();
    }
    if (not std::is_same<vType, SPComplexType>::value)
    {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(comm->rank()), long(v.num_elements()), long(comm->size()));
      copy_n_cast(vsp.origin() + i0, iN - i0, make_device_ptr(v.origin()) + i0);
    }
    comm->barrier();
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 1)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 1)>,
           typename = void>
  void vbias(MatA const& G, MatB&& v, double dt, double a = 1., double c = 0., int k = 0)
  {
    using GType = typename std::decay_t<typename MatA::element>;
    using vType = typename std::decay<MatB>::type::element;
    boost::multi::array_ref<vType, 2, decltype(v.origin())> v_(v.origin(), {v.size(0), 1});
    boost::multi::array_ref<GType const, 2, decltype(G.origin())> G_(G.origin(), {1, G.size(0)});
    vbias(G_, v_, dt, a, c, k);
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>>
  void vbias(MatA const& G, MatB&& v, double dt, double a = 1., double c = 0., int k = 0)
  {
    using GType = typename std::decay_t<typename MatA::element>;
    using vType = typename std::decay<MatB>::type::element;
    using sfqmc::afqmc::reinterpret_pointer_cast;
    // scale a by sqrt(dt)
    a *= std::sqrt(dt);
    if (k > 0)
      APP_ABORT(" Error: THC not yet implemented for multiple references.");
    int nwalk = G.size(0);
    int nu    = Piu.size(1);
    int nchol = ( REAL ? Luv.size(1) : 2 * Luv.size(1) );
    utils::check(v.size(1) == nwalk, "THC::vbias: Size mismatch.");
    utils::check(v.size(0) == nchol, "THC::vbias: Size mismatch.");
    using ma::T;
    int c0, cN;
    std::tie(c0, cN) = FairDivideBoundary(comm->rank(), nchol, comm->size());

    size_t memory_needs = nwalk * nu;
    if (not std::is_same<GType, SPComplexType>::value)
      memory_needs += G.num_elements();
    if (not std::is_same<vType, SPComplexType>::value)
      memory_needs += v.num_elements();
    ShmArray<SPComplexType, 1> SM_TMats(iextensions<1u>{memory_needs},
                                        shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    size_t cnt(0);
    const_sp_pointer Gptr(nullptr);
    sp_pointer vptr(nullptr);
    // setup origin of Gsp and copy_n_cast if necessary
    if (std::is_same<GType, SPComplexType>::value)
    {
      Gptr = reinterpret_pointer_cast<SPComplexType const>(make_device_ptr(G.origin()));
    }
    else
    {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(comm->rank()), long(G.size(0)), long(comm->size()));
      Array_ref<SPComplexType, 2> Gsp(make_device_ptr(SM_TMats.origin()), G.extensions());
      ma::copy_n_cast(G.sliced(i0,iN),Gsp.sliced(i0,iN));
      cnt += size_t(G.num_elements());
      Gptr = make_device_ptr(SM_TMats.origin());
    }
    // setup origin of vsp and copy_n_cast if necessary
    if (std::is_same<vType, SPComplexType>::value)
    {
      vptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(v.origin()));
    }
    else
    {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(comm->rank()), long(v.num_elements()), long(comm->size()));
      vptr             = make_device_ptr(SM_TMats.origin()) + cnt;
      cnt += size_t(v.num_elements());
      if (std::abs(c) > 1e-12)
        copy_n_cast(make_device_ptr(v.origin()) + i0, iN - i0, vptr + i0);
    }
    // setup array references
    Array_cref<SPComplexType, 2> Gsp(Gptr, G.extensions());
    Array_ref<SPComplexType, 2> vsp(vptr, v.extensions());

    if (haj.size() == 1)
    {
      Array_ref<SPComplexType, 2> Guu(make_device_ptr(SM_TMats.origin()) + cnt, {nu, nwalk});
      Guu_from_compact(Gsp, Guu);
      if constexpr (REAL) {
        ma::product(SPRealType(a), T(Luv(Luv.extension(0), {c0, cN})), Guu, SPRealType(c), vsp.sliced(c0, cN));
      } else {	
        // reinterpret as RealType matrices with 2x the columns
        Array_ref<SPRealType, 2> Luv_R(reinterpret_pointer_cast<SPRealType>(make_device_ptr(Luv.origin())),
                                     {Luv.size(0), 2 * Luv.size(1)});
        Array_ref<SPRealType, 2> Guu_R(reinterpret_pointer_cast<SPRealType>(Guu.origin()), {nu, 2 * nwalk});
        Array_ref<SPRealType, 2> vsp_R(reinterpret_pointer_cast<SPRealType>(vsp.origin()), {vsp.size(0), 2 * vsp.size(1)});
        ma::product(SPRealType(a), T(Luv_R(Luv_R.extension(0), {c0, cN})), Guu_R, SPRealType(c), vsp_R.sliced(c0, cN));
      }
    }
    else
    {
      APP_ABORT(" Error: THC not yet implemented for multiple references.");
    }
    if (not std::is_same<vType, SPComplexType>::value)
    {
      copy_n_cast(make_device_ptr(vsp[c0].origin()), vsp.size(1) * (cN - c0), make_device_ptr(v[c0].origin()));
    }
    comm->barrier();
  }

  template<class Mat, class MatB>
  void generalizedFockMatrix([[maybe_unused]] Mat&& G, [[maybe_unused]] MatB&& Fp, [[maybe_unused]] MatB&& Fm)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

*/
  bool distribution_over_cholesky_vectors() const { return false; }
  int number_of_ke_vectors() const { 
    utils::check(_Zuv_.has_value() or _Zuv_rot_.has_value(), "Missing Zuv/Zuv_rot.");
    if(_Zuv_.has_value()) return _Zuv_->extent(0);
    else return _Zuv_rot_->extent(0); 
  }
  int number_of_cholesky_vectors() const { return ( REAL ? _Luv_().extent(1) : 2 * _Luv_().extent(1) ); }

  // transpose=true means G[nwalk][ik], false means G[ik][nwalk]
  bool transposed_G_for_vbias() const { return true; }
  bool transposed_G_for_E() const { return true; }
  // transpose=true means vHS[nwalk][ik], false means vHS[ik][nwalk]
  bool transposed_vHS() const { return true; }

  bool fast_ph_energy() const { return false; }
  // add nspin_in_basis to allow for a spin independent basis too
  bool spin_dependent_vHS() const 
  { return ((walker_type == COLLINEAR) or (walker_type == NONCOLLINEAR)); } 
/*
  nda::array<ComplexType, 2> getHSPotentials() 
  { return nda::array<ComplexType, 2>{}; }

protected:
  // Guu[nu][nwalk]
  template<class MatA, class MatB>
  void Guu_from_compact(MatA const& G, MatB&& Guu)
  {
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    int nu   = int(Piu.size(1));
    int u0, uN;
    std::tie(u0, uN) = FairDivideBoundary(comm->rank(), nu, comm->size());
    int nw           = G.size(0);

    utils::check(G.size(0) == Guu.size(1), "THC::Guu_from_compact: Size mismatch");
    utils::check(G.size(1) == nel * npol * NMO, "THC::Guu_from_compact: Size mismatch");
    utils::check(Guu.size(0) == nu, "THC::Guu_from_compact: Size mismatch");

    ma::fill(Guu, SPComplexType(0.0));

    ComplexType a = (walker_type == CLOSED) ? ComplexType(2.0) : ComplexType(1.0);
    // G3d[w][a][j]
    auto G3d = G.rotated().partitioned(nel).unrotated();

    if(walker_type == COLLINEAR) {

      for( int is=0; is<nspin; is++ ) {
        
        Array<SPComplexType, 3> T1({nw, nelec[is], (uN - u0)},
                   device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
        comm->barrier();
      
        // transposing intermediary to make dot products faster in the next step
        if constexpr (REAL) {
          APP_ABORT("Finish THC<REAL=True>");
        } else {
          ma::productStridedBatched(G3d({0,nw},{is*nup, nup+is*ndown},{0,NMO}),
                                    Piu({is*NMO,(is+1)*NMO}, {u0, uN}), T1);
        }
        // Gwu[w][u] = a * sum_a T1[w][a][u] * cXau[a][u]
        ma::dot('T','T','T',SPComplexType(a),T1,
                            cXau[0]({is*nup, nup+is*ndown},{u0,uN}),
                            SPComplexType(1.0),Guu.sliced(u0,uN));
        comm->barrier();
      
      } // is<nspin

    } else if(walker_type == NONCOLLINEAR) {

      APP_ABORT("THC:: Finish NONCOLLINEAR");

    } else {

      Array<SPComplexType, 3> T1({nw, nup, (uN - u0)},
                 device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      comm->barrier();

      // transposing intermediary to make dot products faster in the next step
      if constexpr (REAL) {
        APP_ABORT("Finish THC<REAL=True>");
      } else {
        ma::productStridedBatched(G3d,Piu({0, NMO}, {u0, uN}), T1);
      }
      // Gwu[w][u] = a * sum_a T1[w][a][u] * cXau[a][u]
      ma::dot('T','T','T',SPComplexType(a),T1,
                          cXau[0]({0,nup},{u0,uN}),
                          SPComplexType(0.0),Guu.sliced(u0,uN));
      comm->barrier();

    } // walker_type

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

  // MAM: Not yet working for non-collinear, need to deal with off-diagonal sectors Gss'uv
  // since this is for energy, only compact is accepted
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
    nda::range M_rng(ip_*NMO,(ip_+1)*NMO);
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nel  = (walker_type == COLLINEAR ? nup+ndown : nup); // NONCOLLINEAR has ndown=0 
    bool has_rot = _Xsiu_rot_.has_value();
    const auto Xiu = ( has_rot ? (*_Xsiu_rot_)()(ispin%nstot,M_rng,all) : 
                                  _Xsiu_()(ispin%nstot,M_rng,all) );
    const auto Yau = ( has_rot ? (*_Ydsau_rot_)()(idet,ispin,p1,nda::range(nelec[ispin]),all) : 
                                 _Ydsau_()(idet,ispin,p1,nda::range(nelec[ispin]),all) );
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
      auto Gwaic = G4d(all,nda::range(ispin*nup,nup+ispin*ndown),range(p2*NMO,(p2+1)*NMO),all);
      // Twav[w][a][v] = sum_j G[w][a][j] X[j][v]
      nda::tensor::contract(Gwaic,"wajc",Xiu,"jv",T4d,"wavc");
    } else {
      auto Gwai = G(all,nda::range(ispin*nup,nup+ispin*ndown),range(p2*NMO,(p2+1)*NMO));
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
        utils::check(false, "finish");
      }
    }
    mpi->comm.barrier();
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
  memory::shared_array<MEM,ComplexType,3> hij;

  // half rotated one body hamiltonian: [ndet][nup+ndn][npol*NMO]
  memory::shared_array<MEM,ComplexType,3> haj;

  // Xsiu[nspin][npol*NMO][nu]
  memory::shared_array<MEM,SPValueType,3> _Xsiu_;

  // Ydsau[ndet][nspin][ipol][nup][nu]
  memory::shared_array<MEM,SPComplexType,5> _Ydsau_;

  // Luv[nu][nv]
  memory::shared_array<MEM,SPValueType,2> _Luv_;

  // Zuv[nu][nv]
  std::optional<decltype(_Luv_)> _Zuv_;

  // Xsiu_rot[nspin][npol*NMO][nu_rot]
  std::optional<decltype(_Xsiu_)> _Xsiu_rot_;

  // Ydsau_rot[ndet][nspin][ipol][nup][nu_rot]
  std::optional<decltype(_Ydsau_)> _Ydsau_rot_; 

  // Zuv_rot[nu_rot][nu_rot]
  std::optional<decltype(_Luv_)> _Zuv_rot_; 

  // v0(i,l) = -0.5 * sum_j <ij|jl>
  memory::shared_array<MEM,ComplexType,3> v0;

  ComplexType E0;

  long default_buffer_size_in_MB = 4L * 1024L;
};

} // namespace afqmc

} // namespace sfqmc

