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

#undef NDEBUG

#include "catch2/catch.hpp"

#include "config.h"
#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/Random.hpp"
#include "utilities/test_common.hpp"

#include "IO/ptree/ptree_utilities.hpp"
#include "IO/app_loggers.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <complex>
#include <utility>

#include "nda/nda.hpp"
#include "utilities/mpi_context.h"

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Walkers/WalkerIO.hpp"

using std::complex;
using std::string;

namespace sfqmc
{
using namespace afqmc;

void myREQUIRE(const double& a, const double& b) { REQUIRE(a == Approx(b)); }

void myREQUIRE(const std::complex<double>& a, const double& b) { REQUIRE(a.real() == Approx(b)); }

void myREQUIRE(const std::complex<double>& a, const std::complex<double>& b)
{
  REQUIRE(a.real() == Approx(b.real()));
  REQUIRE(a.imag() == Approx(b.imag()));
}

template<class M1, class M2>
void check(M1&& A, M2& B)
{
  using element1 = typename std::decay<M1>::type::element;
  using element2 = typename std::decay<M2>::type::element;
  REQUIRE(A.size(0) == B.size(0));
  REQUIRE(A.size(1) == B.size(1));
  for (int i = 0; i < A.size(0); i++)
    for (int j = 0; j < A.size(1); j++)
      myREQUIRE(element1(A[i][j]), element2(B[i][j]));
}

using namespace afqmc;

template<MEMORY_SPACE MEM>
void test_basic_walker_features(std::string wtype)
{
  using Type = std::complex<double>;
  using nda::array;
  using nda::array_view;

  auto& mpi = utils::make_unit_test_mpi_context();

  int NMO = 8, nup = 2, ndown = 2, nwalkers = 10;
  if (wtype == "noncollinear")
  {
    nup = 4;
    ndown = 0;
  }
  AFQMCInfo info;
  info.NMO  = NMO;
  info.nup = nup;
  info.ndown = ndown;
  info.name = "walker";
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("walker_type",wtype);
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, info, rng);
  
  if(wtype != "collinear-ft" and wtype != "noncollinear-ft"){
    int M((wtype == "noncollinear") ? 2 * NMO : NMO);
    int nspin = (wtype == "collinear" ? 2 : 1); 
    array<Type, 3> initA(nspin, M, nup);
    initA() = Type(0.0);
    for (int i = 0; i < nup; i++)
      initA(0,i,i) = Type(0.22);
    if(wtype == "collinear")
      for (int i = 0; i < ndown; i++)
        initA(1,i,i) = Type(0.33);

    wset.resize(nwalkers, initA);

    REQUIRE(wset.size() == nwalkers);
    //Type base(0.0);
    //Type tot_weight(0.0);
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      auto sm = it->template SlaterMatrix<MEM>(Alpha);
      REQUIRE( sm.extent(0) == initA.extent(1) );	
      REQUIRE( sm.extent(1) == nup );	
      REQUIRE(nda::to_host(it->template SlaterMatrix<MEM>(Alpha)) == initA(0,nda::ellipsis{}));
      if( wset.getWalkerType() == COLLINEAR ) { 
        auto smB = it->template SlaterMatrix<MEM>(Beta);
        REQUIRE( smB.extent(0) == initA.extent(1) );	
        REQUIRE( smB.extent(1) == ndown );	
        REQUIRE( nda::to_host(it->template SlaterMatrix<MEM>(Beta)) == initA(1,nda::range::all,nda::range(ndown)));
      }
      it->set_property(WEIGHT,base * 1.0 + 0.5);
      it->set_property(OVLP,base * 1.0 + 0.5);
      it->set_property(E1_,base * 1.0 + 0.5);
      it->set_property(EXX_,base * 1.0 + 0.5);
      it->set_property(EJ_,base * 1.0 + 0.5);
      tot_weight += base * 1.0 + 0.5;
      base += Type(1.0);
      cnt++;
    }
  }
  else{
    int M((wtype == "noncollinear-ft") ? 2 * NMO : NMO);
    int nspin = (wtype == "collinear-ft" ? 2 : 1); 
    array<Type, 3> initU(nspin, M, M);
    array<Type, 2> initD(nspin, M);
    array<Type, 3> initV(nspin, M, M);
    initU() = Type(0.0);
    for (int i = 0; i < M; i++)
      initU(0,i,i) = Type(0.22);
    if(wtype == "collinear-ft")
      for (int i = 0; i < M; i++)
        initU(1,i,i) = Type(0.33);
    initD() = Type(0.0);
    for (int i = 0; i < M; i++)
      initD(0,i) = Type(0.44);
    if(wtype == "collinear-ft")
      for (int i = 0; i < M; i++)
        initD(1,i) = Type(0.55);
    initV() = Type(0.0);
    for (int i = 0; i < M; i++)
      initV(0,i,i) = Type(0.66);
    if(wtype == "collinear-ft")
      for (int i = 0; i < M; i++)
        initV(1,i,i) = Type(0.77);

    wset.resize(nwalkers, initU, initD, initV);

    REQUIRE(wset.size() == nwalkers);
    //int cnt(0);
    //Type base(0.0);
    //Type tot_weight(0.0);
    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      auto umat = it->template UMatrix<MEM>(Alpha);
      REQUIRE( umat.extent(0) == initU.extent(1) );	
      REQUIRE( umat.extent(1) == M );	
      REQUIRE(nda::to_host(it->template UMatrix<MEM>(Alpha)) == initU(0,nda::ellipsis{}));
      if( wset.getWalkerType() == COLLINEAR ) { 
        auto umatB = it->template UMatrix<MEM>(Beta);
        REQUIRE( umatB.extent(0) == initU.extent(1) );	
        REQUIRE( umatB.extent(1) == M );	
        REQUIRE( nda::to_host(it->template UMatrix<MEM>(Beta)) == initU(1,nda::range::all,nda::range(M)));
      }
      it->set_property(WEIGHT,base * 1.0 + 0.5);
      it->set_property(OVLP,base * 1.0 + 0.5);
      it->set_property(E1_,base * 1.0 + 0.5);
      it->set_property(EXX_,base * 1.0 + 0.5);
      it->set_property(EJ_,base * 1.0 + 0.5);
      it->set_property(LOGSCL_UP,base * 1.0 + 0.5);
      it->set_property(LOGSCL_DN,base * 1.0 + 0.5);
      tot_weight += base * 1.0 + 0.5;
      base += Type(1.0);
      cnt++;
    }
  }


  REQUIRE(cnt == nwalkers);
  base = Type(0.0);
  cnt = 0;
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    Type d_(base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(WEIGHT)) == d_);
    REQUIRE(Type(it->get_property(OVLP)) == d_);
    REQUIRE(Type(it->get_property(E1_)) == d_);
    REQUIRE(Type(it->get_property(EXX_)) == d_);
    REQUIRE(Type(it->get_property(EJ_)) == d_);
    if(wtype == "collinear-ft" or wtype == "noncollinear-ft"){
      REQUIRE(Type(it->get_property(LOGSCL_UP)) == d_);
      REQUIRE(Type(it->get_property(LOGSCL_DN)) == d_);
    }
    base += Type(1.0);
    cnt++;
  }
  wset.reserve(20);
  REQUIRE(wset.capacity() == 20);
  base = Type(0.0);
  cnt = 0;
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    REQUIRE(Type(it->get_property(WEIGHT)) == base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(OVLP)) == base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(E1_)) == base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(EXX_)) == base * 1.0 + 0.5);
    REQUIRE(Type(it->get_property(EJ_)) == base * 1.0 + 0.5);
    if(wtype == "collinear-ft" or wtype == "noncollinear-ft"){
      REQUIRE(Type(it->get_property(LOGSCL_UP)) == base * 1.0 + 0.5);
      REQUIRE(Type(it->get_property(LOGSCL_DN)) == base * 1.0 + 0.5);
    }
    base += Type(1.0);
    cnt++;
  }
  for (int i = 0; i < wset.size(); i++)
  {
    Type i_(i);
    REQUIRE(Type(wset[i].get_property(WEIGHT)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(wset[i].get_property(OVLP)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(wset[i].get_property(E1_)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(wset[i].get_property(EXX_)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(wset[i].get_property(EJ_)) == i_ * 1.0 + 0.5);
    if(wtype == "collinear-ft" or wtype == "noncollinear-ft"){
      REQUIRE(Type(wset[i].get_property(LOGSCL_UP)) == i_ * 1.0 + 0.5);
      REQUIRE(Type(wset[i].get_property(LOGSCL_DN)) == i_ * 1.0 + 0.5);
    }
  }
  for (int i = 0; i < wset.size(); i++)
  {
    Type i_(i);
    auto w = wset[i];
    REQUIRE(Type(w.get_property(WEIGHT)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(OVLP)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(E1_)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(EXX_)) == i_ * 1.0 + 0.5);
    REQUIRE(Type(w.get_property(EJ_)) == i_ * 1.0 + 0.5);
    if(wtype == "collinear-ft" or wtype == "noncollinear-ft"){
      REQUIRE(Type(w.get_property(LOGSCL_UP)) == i_ * 1.0 + 0.5);
      REQUIRE(Type(w.get_property(LOGSCL_DN)) == i_ * 1.0 + 0.5);
    }
  }
  REQUIRE(wset.get_target_population() == nwalkers);
  REQUIRE(wset.get_global_target_population() == nwalkers * mpi->comm.size());
  REQUIRE(wset.GlobalPopulation() == nwalkers * mpi->comm.size());
  REQUIRE(wset.GlobalPopulation() == wset.get_global_target_population());
  REQUIRE(wset.NumBackProp() == 0);
  REQUIRE(wset.GlobalWeight() == tot_weight * Type(mpi->comm.size()));

  wset.scaleWeight(2.0);
  tot_weight *= 2.0;
  REQUIRE(wset.GlobalWeight() == tot_weight * Type(mpi->comm.size()));

  std::vector<ComplexType> Wdata;
  wset.processWalkerData(Wdata);
  wset.popControl();
  REQUIRE(wset.GlobalWeight() == Approx(static_cast<RealType>(wset.get_global_target_population())));
  REQUIRE(wset.get_target_population() == nwalkers);
  REQUIRE(wset.get_global_target_population() == nwalkers * mpi->comm.size());
  REQUIRE(wset.GlobalPopulation() == nwalkers * mpi->comm.size());
  REQUIRE(wset.GlobalPopulation() == wset.get_global_target_population());
  REQUIRE(wset.GlobalWeight() == Approx(static_cast<RealType>(wset.get_global_target_population())));
  double nx = (wset.getWalkerType() == NONCOLLINEAR or wset.getWalkerType() == FULLYPOLARIZED ? 1.0 : 2.0);
  for (int i = 0; i < wset.size(); i++)
  {
    auto w = wset[i];
    myREQUIRE(std::exp(nx * wset.getLogOverlapFactor()) * w.get_property(OVLP), w.get_property(E1_));
    REQUIRE(w.get_property(EXX_) ==w.get_property(E1_));
    REQUIRE(w.get_property(EJ_) == w.get_property(E1_));
  }

  if(wtype != "collinear-ft" and wtype != "noncollinear-ft"){
    auto SMs = wset.template SlaterMatrices<MEM>(Alpha);
    REQUIRE( SMs.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR ) { 
      auto SMBs = wset.template SlaterMatrices<MEM>(Beta);
      REQUIRE( SMBs.extent(0) == wset.size() ); 
    }

    // BP
    wset.resize_bp(4,10,2);
    {
      auto F0 = wset.template getFields<MEM>(0);
      auto Fs = wset.template getFields<MEM>();
      memory::array<MEM,ComplexType,2> Fi(F0.shape());
      Fi() = Fs(nda::range::all,0,nda::range::all); 
      wset.storeFields(1,Fi);
      auto WF = wset.template getWeightFactors<MEM>();
      auto WH = wset.template getWeightHistory<MEM>();
      WF() = ComplexType(0.0);
      WH() = ComplexType(0.0);
    }
  }
  else{
    auto UMats = wset.template UMatrices<MEM>(Alpha);
    REQUIRE( UMats.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR_FT ) { 
      auto UMatBs = wset.template UMatrices<MEM>(Beta);
      REQUIRE( UMatBs.extent(0) == wset.size() ); 
    }
    auto DVecs = wset.template DMatrices<MEM>(Alpha);
    REQUIRE( DVecs.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR_FT ) { 
      auto DVecBs = wset.template DMatrices<MEM>(Beta);
      REQUIRE( DVecBs.extent(0) == wset.size() ); 
    }
    auto VMats = wset.template VMatrices<MEM>(Alpha);
    REQUIRE( VMats.extent(0) == wset.size() ); 
    if( wset.getWalkerType() == COLLINEAR_FT ) { 
      auto VMatBs = wset.template VMatrices<MEM>(Beta);
      REQUIRE( VMatBs.extent(0) == wset.size() ); 
    }

    // No BP for finite temperature
  }
  wset.clean();
  REQUIRE(wset.size() == 0);
  REQUIRE(wset.capacity() == 0);
}

template<MEMORY_SPACE MEM>
void test_walker_io(std::string wtype)
{
  using Type = std::complex<double>;
  using nda::array;

  auto& mpi = utils::make_unit_test_mpi_context();

/*
  int NMO = 8, nup = 2, ndown = 2, nwalkers = 10;
  if (wtype == "noncollinear")
  {
    nup = 4;
    ndown = 0;
  }

  AFQMCInfo info;
  info.NMO  = NMO;
  info.nup = nup;
  info.ndown = ndown;
  info.name = "walker";
  int M((wtype == "noncollinear") ? 2 * NMO : NMO);
  array<Type, 2> initA(M, nup);
  array<Type, 2> initB(M, ndown);
  initA() = Type(0.0);
  initB() = Type(0.0);
  for (int i = 0; i < nup; i++)
    initA(i,i) = Type(0.22);
  for (int i = 0; i < ndown; i++)
    initB(i,i) = Type(0.33);
  std::shared_ptr<utils::RandomGenerator_t> rng = std::make_shared<utils::RandomGenerator_t>();

  ptree pt0;
  pt0.put("WalkerSet.name","wset0");
  pt0.put("WalkerSet.walker_type",wtype);
  auto wset = make_WalkerSet<MEM>(mpi, pt0.get_child("WalkerSet"), info, rng);
  wset.resize(nwalkers, initA);

  REQUIRE(wset.size() == nwalkers);
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    auto sm = it->template SlaterMatrix<MEM>(Alpha);
    REQUIRE( sm.extent(0) == initA.extent(0) );
    REQUIRE( sm.extent(1) == initA.extent(1) ); 
    REQUIRE( nda::to_host(it->template SlaterMatrix<MEM>(Alpha)) == initA );
    if( wset.getWalkerType() == COLLINEAR ) {
      auto smB = it->template SlaterMatrix<MEM>(Beta);
      REQUIRE( smB.extent(0) == initB.extent(0) );
      REQUIRE( smB.extent(1) == initB.extent(1) ); 
      REQUIRE( nda::to_host(it->template SlaterMatrix<MEM>(Beta)) == initB );
    }
    it->set_property(WEIGHT,base * 1.0 + 0.1);
    it->set_property(OVLP,base * 1.0 + 0.2);
    it->set_property(E1_,base * 1.0 + 0.3);
    it->set_property(EXX_,base * 1.0 + 0.4);
    it->set_property(EJ_,base * 1.0 + 0.5);
    tot_weight += base * 1.0 + 0.5; // not used?
    base += Type(1.0);
    cnt++;
  }
  REQUIRE(cnt == nwalkers);

  // dump restart file
  {
    h5::file fh5;
    if(mpi->comm.root()) fh5 = h5::file(std::string("dummy_walkers.h5"),'w');
    dumpToHDF5(wset, fh5);
  }
  mpi->comm.barrier();

  {
    h5::file fh5(std::string("dummy_walkers.h5"),'r');
    auto wset2 = make_WalkerSet<MEM>(mpi, pt0.get_child("WalkerSet"), info, rng);
    restartFromHDF5(wset2, nwalkers, fh5, true);
    std::array<walker_data,5> tags = {WEIGHT,OVLP,E1_,EXX_,EJ_};
    for (int i = 0; i < nwalkers; i++)
    {
      CHECK(nda::to_host(wset[i].template SlaterMatrix<MEM>(Alpha)) == nda::to_host(wset2[i].template SlaterMatrix<MEM>(Alpha)));
      for(auto v : tags)
        CHECK(wset[i].get_property(v) == wset2[i].get_property(v));
    }
  }

  mpi->comm.barrier();
  if (mpi->comm.root())
    remove("dummy_walkers.h5");
*/
}

// MAM: Tests are not GPU enabled, fix direct access to GPU memory
TEST_CASE("swset_test_basic", "[shared_wset]")
{
  test_basic_walker_features<HOST_MEMORY>("closed");
  test_basic_walker_features<HOST_MEMORY>("collinear");
  test_basic_walker_features<HOST_MEMORY>("noncollinear");
  test_basic_walker_features<HOST_MEMORY>("fullypolarized");
  test_basic_walker_features<HOST_MEMORY>("collinear-ft");
  test_basic_walker_features<HOST_MEMORY>("noncollinear-ft");
#if defined(ENABLE_DEVICE)
  test_basic_walker_features<DEVICE_MEMORY>("closed");
  test_basic_walker_features<DEVICE_MEMORY>("collinear");
  test_basic_walker_features<DEVICE_MEMORY>("noncollinear");
  test_basic_walker_features<DEVICE_MEMORY>("fullypolarized");
#endif
}
TEST_CASE("walker_io", "[shared_wset]")
{
  test_walker_io<HOST_MEMORY>("closed");
  test_walker_io<HOST_MEMORY>("collinear");
  test_walker_io<HOST_MEMORY>("noncollinear");
  test_walker_io<HOST_MEMORY>("fullypolarized");
#if defined(ENABLE_DEVICE)
  test_walker_io<DEVICE_MEMORY>("closed");
  test_walker_io<DEVICE_MEMORY>("collinear");
  test_walker_io<DEVICE_MEMORY>("noncollinear");
  test_walker_io<DEVICE_MEMORY>("fullypolarized");
#endif
}
} // namespace sfqmc
