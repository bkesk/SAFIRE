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

#include "catch_amalgamated.hpp"

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "Utilities/Random.hpp"

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"
#include "io/ptree/ptree_utilities.hpp"
#include "Utilities/app_loggers.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <complex>

#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
//#include "mpi3/environment.hpp"

//#include "AFQMC/Walkers WalkerSetFactory.hpp"
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

void test_basic_walker_features(bool serial, std::string wtype)
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  arch::INIT(node);
#endif

  using Type = std::complex<double>;

  //assert(world.size()%2 == 0);

  int NMO = 8, NAEA = 2, NAEB = 2, nwalkers = 10;
  if (wtype == "noncollinear")
  {
    NAEA = 4;
    NAEB = 0;
  }

  //auto node = world.split_shared();

  GlobalTaskGroup gTG(world);
  TaskGroup_ TG(gTG, std::string("TaskGroup"), 1, serial ? 1 : gTG.getTotalCores());
  AFQMCInfo info;
  info.NMO  = NMO;
  info.NAEA = NAEA;
  info.NAEB = NAEB;
  info.name = "walker";
  int M((wtype == "noncollinear") ? 2 * NMO : NMO);
  boost::multi::array<Type, 2> initA({M, NAEA}, Type(0.0));
  boost::multi::array<Type, 2> initB({M, NAEB}, Type(0.0));
  for (int i = 0; i < NAEA; i++)
    initA[i][i] = Type(0.22);
  for (int i = 0; i < NAEB; i++)
    initB[i][i] = Type(0.33);
  utils::RandomGenerator_t rng;

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("walker_type",wtype);
  WalkerSet wset(TG, wlk_pt, info, &rng);
  wset.resize(nwalkers, initA, initB);

  REQUIRE(wset.size() == nwalkers);
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  for (WalkerSet::iterator it = wset.begin(); it != wset.end(); ++it)
  {
    auto sm = it->SlaterMatrix(Alpha);
    REQUIRE( (*sm).size(0) == initA.size(0) );	
    REQUIRE( (*sm).size(1) == initA.size(1) );	
    REQUIRE(*it->SlaterMatrix(Alpha) == initA);
    if( wset.getWalkerType() == COLLINEAR ) { 
      auto smB = it->SlaterMatrix(Beta);
      REQUIRE( (*smB).size(0) == initB.size(0) );	
      REQUIRE( (*smB).size(1) == initB.size(1) );	
      REQUIRE(*it->SlaterMatrix(Beta) == initB);
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
  for (WalkerSet::iterator it = wset.begin(); it != wset.end(); ++it)
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
  for (WalkerSet::iterator it = wset.begin(); it != wset.end(); ++it)
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
  REQUIRE(wset.get_TG_target_population() == nwalkers);
  REQUIRE(wset.get_global_target_population() == nwalkers * TG.getNumberOfTGs());
  REQUIRE(wset.GlobalPopulation() == nwalkers * TG.getNumberOfTGs());
  REQUIRE(wset.GlobalPopulation() == wset.get_global_target_population());
  REQUIRE(wset.NumBackProp() == 0);
  REQUIRE(wset.GlobalWeight() == tot_weight * Type(TG.getNumberOfTGs()));

  wset.scaleWeight(2.0);
  tot_weight *= 2.0;
  REQUIRE(wset.GlobalWeight() == tot_weight * Type(TG.getNumberOfTGs()));

  std::vector<ComplexType> Wdata;
  wset.popControl(Wdata);
  REQUIRE(wset.GlobalWeight() == Approx(static_cast<RealType>(wset.get_global_target_population())));
  REQUIRE(wset.get_TG_target_population() == nwalkers);
  REQUIRE(wset.get_global_target_population() == nwalkers * TG.getNumberOfTGs());
  REQUIRE(wset.GlobalPopulation() == nwalkers * TG.getNumberOfTGs());
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

  wset.clean();
  REQUIRE(wset.size() == 0);
  REQUIRE(wset.capacity() == 0);
}

void test_hyperslab()
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  arch::INIT(node);
#endif

  using Type   = std::complex<double>;
  using Matrix = boost::multi::array<Type, 2>;

  int rank = world.rank();

  int nwalk         = 9;
  int nprop         = 7;
  Matrix Data({nwalk, nprop});

  for (int i = 0; i < nwalk; i++)
    for (int j = 0; j < nprop; j++)
      Data[i][j] = i * 10 + rank * 100 + j;

  int nwtot = (world += nwalk);

  hdf_archive dump(world, true);
  if (!dump.create("dummy_walkers.h5", H5F_ACC_EXCL))
  {
    app_error(" Error opening restart file. ");
    APP_ABORT("");
  }
  dump.push("WalkerSet");

  hyperslab_proxy<Matrix, 2> hslab(Data, std::array<int, 2>{nwtot, nprop}, std::array<int, 2>{nwalk, nprop},
                                   std::array<int, 2>{rank * nwalk, 0});
  dump.write(hslab, "Walkers");
  dump.close();
  world.barrier();

  {
    hdf_archive read(world, false);
    if (!read.open("dummy_walkers.h5", H5F_ACC_RDONLY))
    {
      app_error(" Error opening restart file. ");
      APP_ABORT("");
    }
    read.push("WalkerSet");

    Matrix DataIn({nwalk, nprop});

    hyperslab_proxy<Matrix, 2> hslabIn(DataIn, std::array<int, 2>{nwtot, nprop}, std::array<int, 2>{nwalk, nprop},
                                     std::array<int, 2>{rank * nwalk, 0});
    read.read(hslabIn, "Walkers");
    read.close();

    for (int i = 0; i < nwalk; i++)
      for (int j = 0; j < nprop; j++)
      {
        REQUIRE(real(DataIn[i][j]) == i * 10 + rank * 100 + j);
        REQUIRE(imag(DataIn[i][j]) == 0);
      }
  }
  world.barrier();
  if (world.root())
    remove("dummy_walkers.h5");
}

void test_double_hyperslab()
{
  auto world = boost::mpi3::environment::get_world_instance();

  using Type   = std::complex<double>;
  using Matrix = boost::multi::array<Type, 2>;

  int rank = world.rank();

  int nwalk         = 9;
  int nprop         = 3;
  int nprop_to_safe = 3;
  Matrix Data({nwalk, nprop});

  for (int i = 0; i < nwalk; i++)
    for (int j = 0; j < nprop; j++)
      Data[i][j] = i * 10 + rank * 100 + j;

  int nwtot = (world += nwalk);

  hdf_archive dump(world, true);
  if (!dump.create("dummy_walkers.h5", H5F_ACC_EXCL))
  {
    app_error(" Error opening restart file. ");
    APP_ABORT("");
  }
  dump.push("WalkerSet");

  //double_hyperslab_proxy<Matrix,2> hslab(Data,
  hyperslab_proxy<Matrix, 2> hslab(Data, std::array<int, 2>{nwtot, nprop_to_safe},
                                   std::array<int, 2>{nwalk, nprop_to_safe}, std::array<int, 2>{rank * nwalk, 0}); //,

  //                                  std::array<int,2>{nwalk,nprop},
  //                                  std::array<int,2>{nwalk,nprop_to_safe},
  //                                  std::array<int,2>{0,0});
  dump.write(hslab, "Walkers");
  dump.close();
  world.barrier();

  {
    hdf_archive read(world, false);
    if (!read.open("dummy_walkers.h5", H5F_ACC_RDONLY))
    {
      app_error(" Error opening restart file. ");
      APP_ABORT("");
    }
    read.push("WalkerSet");

    //Matrix DataIn({nwalk,nprop});
    Matrix DataIn({nwalk, nprop_to_safe});

    //double_hyperslab_proxy<Matrix,2> hslab(DataIn,
    hyperslab_proxy<Matrix, 2> hslabIn(DataIn, std::array<int, 2>{nwtot, nprop_to_safe},
                                     std::array<int, 2>{nwalk, nprop_to_safe}, std::array<int, 2>{rank * nwalk, 0}); //,
    //                                  std::array<int,2>{nwalk,nprop},
    //                                  std::array<int,2>{nwalk,nprop_to_safe},
    //                                  std::array<int,2>{0,0});
    read.read(hslabIn, "Walkers");
    read.close();

    for (int i = 0; i < nwalk; i++)
    {
      for (int j = 0; j < nprop_to_safe; j++)
      {
        REQUIRE(real(DataIn[i][j]) == i * 10 + rank * 100 + j);
        REQUIRE(imag(DataIn[i][j]) == 0);
      }
      /*
     for(int j=nprop_to_safe; j<nprop; j++) {
       REQUIRE( real(DataIn[i][j]) == 0);
       REQUIRE( imag(DataIn[i][j]) == 0);
     }
*/
    }
  }
  world.barrier();
  if (world.root())
    remove("dummy_walkers.h5");
}

void test_walker_io(std::string wtype)
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());

  using Type = std::complex<double>;

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  arch::INIT(node);
#endif

  //assert(world.size()%2 == 0);

  int NMO = 8, NAEA = 2, NAEB = 2, nwalkers = 10;
  if (wtype == "noncollinear")
  {
    NAEA = 4;
    NAEB = 0;
  }

  //auto node = world.split_shared();

  GlobalTaskGroup gTG(world);
  TaskGroup_ TG(gTG, std::string("TaskGroup"), 1, 1);
  AFQMCInfo info;
  info.NMO  = NMO;
  info.NAEA = NAEA;
  info.NAEB = NAEB;
  info.name = "walker";
  int M((wtype == "noncollinear") ? 2 * NMO : NMO);
  boost::multi::array<Type, 2> initA({M, NAEA}, Type(0.0));
  boost::multi::array<Type, 2> initB({M, NAEB}, Type(0.0));
  for (int i = 0; i < NAEA; i++)
    initA[i][i] = Type(0.22);
  for (int i = 0; i < NAEB; i++)
    initB[i][i] = Type(0.33);
  utils::RandomGenerator_t rng;

  ptree pt0;
  pt0.put("WalkerSet.name","wset0");
  pt0.put("WalkerSet.walker_type",wtype);
  WalkerSet wset(TG, pt0.get_child("WalkerSet"), info, &rng);
  wset.resize(nwalkers, initA, initB);

  REQUIRE(wset.size() == nwalkers);
  int cnt(0);
  Type base(0.0);
  Type tot_weight(0.0);
  for (WalkerSet::iterator it = wset.begin(); it != wset.end(); ++it)
  {
    auto sm = it->SlaterMatrix(Alpha);
    REQUIRE( (*sm).size(0) == initA.size(0) );
    REQUIRE( (*sm).size(1) == initA.size(1) ); 
    REQUIRE(*it->SlaterMatrix(Alpha) == initA);
    if( wset.getWalkerType() == COLLINEAR ) {
      auto smB = it->SlaterMatrix(Beta);
      REQUIRE( (*smB).size(0) == initB.size(0) );
      REQUIRE( (*smB).size(1) == initB.size(1) );
      REQUIRE(*it->SlaterMatrix(Beta) == initB);
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

#if defined(ENABLE_PHDF5)
  hdf_archive dump(world, true);
  {
#else
  hdf_archive dump(world, false);
  if (TG.Global().root())
  {
#endif
    if (!dump.create("dummy_walkers.h5", H5F_ACC_EXCL))
    {
      app_error(" Error opening restart file. ");
      APP_ABORT("");
    }
  }

  // dump restart file
  dumpToHDF5(wset, dump);
  dump.close();

  {
#if defined(ENABLE_PHDF5)
    hdf_archive read(world, true);
    {
#else
    hdf_archive read(world, false);
    if (TG.Global().root())
    {
#endif
      if (!read.open("dummy_walkers.h5", H5F_ACC_RDONLY))
      {
        app_error(" Error opening restart file. ");
        APP_ABORT("");
      }
      else
      {
        read.close();
      }
    }

    WalkerSet wset2(TG, pt0.get_child("WalkerSet"), info, &rng);
    restartFromHDF5(wset2, nwalkers, "dummy_walkers.h5", read, true);
    for (int i = 0; i < nwalkers; i++)
    {
      CHECK(*wset[i].SlaterMatrix(Alpha) == *wset2[i].SlaterMatrix(Alpha));
      CHECK(ComplexType(*wset[i].weight()) == ComplexType(*wset2[i].weight()));
      CHECK(ComplexType(*wset[i].overlap()) == ComplexType(*wset2[i].overlap()));
      CHECK(ComplexType(*wset[i].E1()) == ComplexType(*wset2[i].E1()));
      CHECK(ComplexType(*wset[i].EXX()) == ComplexType(*wset2[i].EXX()));
      CHECK(ComplexType(*wset[i].EJ()) == ComplexType(*wset2[i].EJ()));
    }
  }
  world.barrier();
  if (world.root())
    remove("dummy_walkers.h5");
}

TEST_CASE("swset_test_serial", "[shared_wset]")
{
  setup_loggers(true,2,0);
  test_basic_walker_features(true, "closed");
  test_basic_walker_features(false, "closed");
  test_basic_walker_features(true, "collinear");
  test_basic_walker_features(false, "collinear");
  test_basic_walker_features(true, "noncollinear");
  test_basic_walker_features(false, "noncollinear");
  test_basic_walker_features(true, "fullypolarized");
  test_basic_walker_features(false, "fullypolarized");
}
/*
TEST_CASE("hyperslab_tests", "[shared_wset]")
{
 // test_hyperslab();
  test_double_hyperslab();
}
*/
TEST_CASE("walker_io", "[shared_wset]")
{
  setup_loggers(true,2,0);
  test_walker_io("closed");
  test_walker_io("collinear");
  test_walker_io("noncollinear");
  test_walker_io("fullypolarized");
}

} // namespace sfqmc
