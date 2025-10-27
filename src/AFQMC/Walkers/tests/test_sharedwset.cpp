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

  int NMO = 8, NAEA = 2, NAEB = 2, nwalkers = 10;
  if (wtype == "noncollinear")
  {
    NAEA = 4;
    NAEB = 0;
  }
  AFQMCInfo info;
  info.NMO  = NMO;
  info.NAEA = NAEA;
  info.NAEB = NAEB;
  info.name = "walker";
  int M((wtype == "noncollinear") ? 2 * NMO : NMO);
  array<Type, 2> initA(M, NAEA);
  array<Type, 2> initB(M, NAEB);
  initA() = Type(0.0);
  initB() = Type(0.0);
  for (int i = 0; i < NAEA; i++)
    initA(i,i) = Type(0.22);
  for (int i = 0; i < NAEB; i++)
    initB(i,i) = Type(0.33);
  std::shared_ptr<utils::RandomGenerator_t> rng = std::make_shared<utils::RandomGenerator_t>();

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("walker_type",wtype);
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, info, rng);
  wset.resize(nwalkers, initA, initB);

  REQUIRE(wset.size() == nwalkers);
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    auto sm = it->template SlaterMatrix<MEM>(Alpha);
    REQUIRE( sm.extent(0) == initA.extent(0) );	
    REQUIRE( sm.extent(1) == initA.extent(1) );	
    REQUIRE(it->template SlaterMatrix<MEM>(Alpha) == initA);
    if( wset.getWalkerType() == COLLINEAR ) { 
      auto smB = it->template SlaterMatrix<MEM>(Beta);
      REQUIRE( smB.extent(0) == initB.extent(0) );	
      REQUIRE( smB.extent(1) == initB.extent(1) );	
      REQUIRE(it->template SlaterMatrix<MEM>(Beta) == initB);
    }
    *it->weight()  = base * 1.0 + 0.5;
    *it->overlap() = base * 1.0 + 0.5;
    *it->E1()      = base * 1.0 + 0.5;
    *it->EXX()     = base * 1.0 + 0.5;
    *it->EJ()      = base * 1.0 + 0.5;
    tot_weight += base * 1.0 + 0.5;
    base += Type(1.0);
    cnt++;
  }
  REQUIRE(cnt == nwalkers);
  base = Type(0.0);
  cnt = 0;
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    Type d_(base * 1.0 + 0.5);
    REQUIRE(Type(*it->weight()) == d_);
    REQUIRE(Type(*it->overlap()) == base * 1.0 + 0.5);
    REQUIRE(Type(*it->E1()) == base * 1.0 + 0.5);
    REQUIRE(Type(*it->EXX()) == base * 1.0 + 0.5);
    REQUIRE(Type(*it->EJ()) == base * 1.0 + 0.5);
    base += Type(1.0);
    cnt++;
  }

  wset.reserve(20);
  REQUIRE(wset.capacity() == 20);
  base = Type(0.0);
  cnt = 0;
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    REQUIRE(Type(*it->weight()) == base * 1.0 + 0.5);
    REQUIRE(Type(*it->overlap()) == base * 1.0 + 0.5);
    REQUIRE(Type(*it->E1()) == base * 1.0 + 0.5);
    REQUIRE(Type(*it->EXX()) == base * 1.0 + 0.5);
    REQUIRE(Type(*it->EJ()) == base * 1.0 + 0.5);
    base += Type(1.0);
    cnt++;
  }
  for (int i = 0; i < wset.size(); i++)
  {
    Type i_(i);
    REQUIRE(Type(*wset[i].weight()) == i_ * 1.0 + 0.5);
    REQUIRE(Type(*wset[i].overlap()) == i_ * 1.0 + 0.5);
    REQUIRE(Type(*wset[i].E1()) == i_ * 1.0 + 0.5);
    REQUIRE(Type(*wset[i].EXX()) == i_ * 1.0 + 0.5);
    REQUIRE(Type(*wset[i].EJ()) == i_ * 1.0 + 0.5);
  }
  for (int i = 0; i < wset.size(); i++)
  {
    Type i_(i);
    auto w = wset[i];
    REQUIRE(Type(*w.weight()) == i_ * 1.0 + 0.5);
    REQUIRE(Type(*w.overlap()) == i_ * 1.0 + 0.5);
    REQUIRE(Type(*w.E1()) == i_ * 1.0 + 0.5);
    REQUIRE(Type(*w.EXX()) == i_ * 1.0 + 0.5);
    REQUIRE(Type(*w.EJ()) == i_ * 1.0 + 0.5);
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
    myREQUIRE(std::exp(nx * wset.getLogOverlapFactor()) * ComplexType(*w.overlap()), ComplexType(*w.E1()));
    REQUIRE(ComplexType(*w.EXX()) == ComplexType(*w.E1()));
    REQUIRE(ComplexType(*w.EJ()) == ComplexType(*w.E1()));
  }

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
    Fi() = Fs(0,nda::ellipsis{}); 
    wset.storeFields(1,Fi);
    auto WF = wset.template getWeightFactors<MEM>();
    auto WH = wset.template getWeightHistory<MEM>();
    WF() = ComplexType(0.0);
    WH() = ComplexType(0.0);
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

  int NMO = 8, NAEA = 2, NAEB = 2, nwalkers = 10;
  if (wtype == "noncollinear")
  {
    NAEA = 4;
    NAEB = 0;
  }

  AFQMCInfo info;
  info.NMO  = NMO;
  info.NAEA = NAEA;
  info.NAEB = NAEB;
  info.name = "walker";
  int M((wtype == "noncollinear") ? 2 * NMO : NMO);
  array<Type, 2> initA(M, NAEA);
  array<Type, 2> initB(M, NAEB);
  initA() = Type(0.0);
  initB() = Type(0.0);
  for (int i = 0; i < NAEA; i++)
    initA(i,i) = Type(0.22);
  for (int i = 0; i < NAEB; i++)
    initB(i,i) = Type(0.33);
  std::shared_ptr<utils::RandomGenerator_t> rng = std::make_shared<utils::RandomGenerator_t>();

  ptree pt0;
  pt0.put("WalkerSet.name","wset0");
  pt0.put("WalkerSet.walker_type",wtype);
  auto wset = make_WalkerSet<MEM>(mpi, pt0.get_child("WalkerSet"), info, rng);
  wset.resize(nwalkers, initA, initB);

  REQUIRE(wset.size() == nwalkers);
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  for (auto it = wset.begin(); it != wset.end(); ++it)
  {
    auto sm = it->template SlaterMatrix<MEM>(Alpha);
    REQUIRE( sm.extent(0) == initA.extent(0) );
    REQUIRE( sm.extent(1) == initA.extent(1) ); 
    REQUIRE( it->template SlaterMatrix<MEM>(Alpha) == initA );
    if( wset.getWalkerType() == COLLINEAR ) {
      auto smB = it->template SlaterMatrix<MEM>(Beta);
      REQUIRE( smB.extent(0) == initB.extent(0) );
      REQUIRE( smB.extent(1) == initB.extent(1) ); 
      REQUIRE( it->template SlaterMatrix<MEM>(Beta) == initB );
    }
    *it->weight()  = base * 1.0 + 0.1;
    *it->overlap() = base * 1.0 + 0.2;
    *it->E1()      = base * 1.0 + 0.3;
    *it->EXX()     = base * 1.0 + 0.4;
    *it->EJ()      = base * 1.0 + 0.5;
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
    for (int i = 0; i < nwalkers; i++)
    {
      CHECK(wset[i].template SlaterMatrix<MEM>(Alpha) == wset2[i].template SlaterMatrix<MEM>(Alpha));
      CHECK(ComplexType(*wset[i].weight()) == ComplexType(*wset2[i].weight()));
      CHECK(ComplexType(*wset[i].overlap()) == ComplexType(*wset2[i].overlap()));
      CHECK(ComplexType(*wset[i].E1()) == ComplexType(*wset2[i].E1()));
      CHECK(ComplexType(*wset[i].EXX()) == ComplexType(*wset2[i].EXX()));
      CHECK(ComplexType(*wset[i].EJ()) == ComplexType(*wset2[i].EJ()));
    }
  }

  mpi->comm.barrier();
  if (mpi->comm.root())
    remove("dummy_walkers.h5");
}

// MAM: Tests are not GPU enabled, fix direct access to GPU memory
TEST_CASE("swset_test_basic", "[shared_wset]")
{
  test_basic_walker_features<HOST_MEMORY>("closed");
  test_basic_walker_features<HOST_MEMORY>("collinear");
  test_basic_walker_features<HOST_MEMORY>("noncollinear");
  test_basic_walker_features<HOST_MEMORY>("fullypolarized");
}
TEST_CASE("walker_io", "[shared_wset]")
{
  test_walker_io<HOST_MEMORY>("closed");
  test_walker_io<HOST_MEMORY>("collinear");
  test_walker_io<HOST_MEMORY>("noncollinear");
  test_walker_io<HOST_MEMORY>("fullypolarized");
}

} // namespace sfqmc
