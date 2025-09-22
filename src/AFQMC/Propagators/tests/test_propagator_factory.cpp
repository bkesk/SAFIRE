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

#include "io/ptree/ptree_utilities.hpp"
#include "hdf/hdf_archive.h"
#include "Utilities/Random.hpp"
#include "Memory/device_rng.hpp"
#include "Utilities/app_loggers.h"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "Memory/buffer_managers.h"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"

#include "SparseMatrix/csr_matrix_construct.hpp"
#include "Numerics/ma_blas.hpp"

using std::cerr;
using std::complex;
using std::cout;
using std::endl;
using std::ifstream;
using std::setprecision;
using std::string;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<bool MP>
void propg_fac_shared(boost::mpi3::communicator& world)
{
  if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
  {
    APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
  }
  else
  {
    // Global Task Group
    afqmc::GlobalTaskGroup gTG(world);

    hdf_archive dump;
    if (!dump.open(UTEST_HAMIL, H5F_ACC_RDONLY))
    {
      APP_ABORT(" Error opening integral file in SparseGeneralHamiltonian. ");
    }
    auto format = get_hamiltonian_format(dump,gTG.Global());
    dump.close();

    int NMO, wfn_NMO, NAEA, NAEB;
    NMO = read_nmo_from_hdf(UTEST_HAMIL,format);
    std::tie(wfn_NMO,NAEA, NAEB) = read_info_from_wfn(UTEST_WFN,"any");
    CHECK(NMO == wfn_NMO);
    WALKER_TYPES type         = afqmc::getWalkerType(UTEST_WFN);
    int NPOL                  = (type == NONCOLLINEAR) ? 2 : 1;

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));

    ptree ham_pt;
    ham_pt.put("name","ham0");
    ham_pt.put("system","info0");
    ham_pt.put("filename",UTEST_HAMIL);

    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(gTG, "ham0");

    auto TG   = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());
    int nwalk = 11; // choose prime number to force non-trivial splits in shared routines
    utils::RandomGenerator_t rng;
    auto rng_dev = utils::make_device_rng(777);

    ptree wlk_pt;
    wlk_pt.put("name","wset0");
    if(type == CLOSED) wlk_pt.put("walker_type","closed");
    else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
    else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
    else if(type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
    WalkerSet wset(TG, wlk_pt, InfoMap["info0"], &rng);

    ptree wfn_pt;
    wfn_pt.put("name","wfn0");
    wfn_pt.put("system","info0");
    wfn_pt.put("filename",UTEST_WFN);
    wfn_pt.put("dense_trial","yes");

    WavefunctionFactory WfnFac(InfoMap, MP);
    WfnFac.push("wfn0", wfn_pt);
    Wavefunction& wfn = WfnFac.getWavefunction(TG, TG, "wfn0", type, &ham, 1e-6, nwalk);

    auto initial_guess = WfnFac.getInitialGuess("wfn0");
    REQUIRE(initial_guess.size(0) == 2);
    REQUIRE(initial_guess.size(1) == NPOL * NMO);
    REQUIRE(initial_guess.size(2) == NAEA);
    wset.resize(nwalk, initial_guess[0], initial_guess[1]({0, NPOL * NMO},{0, NAEB}));

    ptree prop_pt;
    prop_pt.put("name","prop0");
    prop_pt.put("system","info0");

    PropagatorFactory PropgFac(InfoMap, MP);
    PropgFac.push("prop0", prop_pt);
    Propagator& prop = PropgFac.getPropagator(TG, "prop0", wfn, &rng_dev);

std::cout<<" serial prop test: " <<NPOL <<" " <<NMO <<" " <<NAEA <<" " <<NAEB <<" " <<nwalk <<std::endl;

    std::cout << setprecision(12);
    wfn.Energy(wset);
    {
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      app_log(1," Initial Energy: {}", (eav / ov).real()); 
    }
    double tot_time = 0;
    RealType dt     = 0.01;
    RealType Eshift = std::abs(ComplexType(*wset[0].overlap()));
    for (int i = 0; i < 4; i++)
    {
      prop.Propagate(2, wset, Eshift, dt, 1);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      tot_time += 2 * dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
    for (int i = 0; i < 4; i++)
    {
      prop.Propagate(4, wset, Eshift, dt, 1);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      tot_time += 4 * dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }

    for (int i = 0; i < 4; i++)
    {
      prop.Propagate(4, wset, Eshift, dt, 2);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      tot_time += 4 * dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
    for (int i = 0; i < 4; i++)
    {
      prop.Propagate(5, wset, Eshift, 2 * dt, 2);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      tot_time += 5 * 2 * dt;
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
      prop.Orthogonalize(wset);
    }
    if(TG.Global().root()) AFQMCTimer.print_all();
  }
}
template<bool MP>
void propg_fac_distributed(boost::mpi3::communicator& world, int ngrp)
{
  if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
  {
    APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
  }
  else
  {
    // Global Task Group
    afqmc::GlobalTaskGroup gTG(world);

    hdf_archive dump;
    if (!dump.open(UTEST_HAMIL, H5F_ACC_RDONLY))
    {
      APP_ABORT(" Error opening integral file in SparseGeneralHamiltonian. ");
    }
    auto format = get_hamiltonian_format(dump,gTG.Global());
    dump.close();

    int NMO, wfn_NMO, NAEA, NAEB;
    NMO = read_nmo_from_hdf(UTEST_HAMIL,format);
    std::tie(wfn_NMO,NAEA, NAEB) = read_info_from_wfn(UTEST_WFN,"any");
    CHECK(NMO == wfn_NMO);
    WALKER_TYPES type         = afqmc::getWalkerType(UTEST_WFN);
    int NPOL                  = (type == NONCOLLINEAR) ? 2 : 1;

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));

    ptree ham_pt;
    ham_pt.put("name","ham0");
    ham_pt.put("system","info0");
    ham_pt.put("filename",UTEST_HAMIL);

    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(gTG, "ham0");

    auto TG     = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());
    auto TGprop = TaskGroup_(gTG, std::string("WfnTG"), ngrp, gTG.getTotalCores());
    //int nwalk = 4; // choose prime number to force non-trivial splits in shared routines
    int nwalk = 11; // choose prime number to force non-trivial splits in shared routines
    utils::RandomGenerator_t rng;
    auto rng_dev = utils::make_device_rng(777);
    Watch Time;

    ptree wlk_pt;
    wlk_pt.put("name","wset0");
    if(type == CLOSED) wlk_pt.put("walker_type","closed");
    else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
    else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
    else if(type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
    WalkerSet wset(TG, wlk_pt, InfoMap["info0"], &rng);

    ptree wfn_pt;
    wfn_pt.put("name","wfn0");
    wfn_pt.put("system","info0");
    wfn_pt.put("filename",UTEST_WFN);

    WavefunctionFactory WfnFac(InfoMap, MP);
    WfnFac.push("wfn0", wfn_pt);
    Wavefunction& wfn = WfnFac.getWavefunction(TGprop, TGprop, "wfn0", type, &ham, 1e-6, nwalk);

    auto initial_guess = WfnFac.getInitialGuess("wfn0");
    REQUIRE(initial_guess.size(0) == 2);
    REQUIRE(initial_guess.size(1) == NPOL * NMO);
    REQUIRE(initial_guess.size(2) == NAEA);
    wset.resize(nwalk, initial_guess[0], initial_guess[1]({0,NPOL*NMO},{0,NAEB}));

    ptree prop_pt;
    prop_pt.put("name","prop0");
    prop_pt.put("system","info0");
    prop_pt.put("nnodes",std::to_string(gTG.getTotalNodes()));

    PropagatorFactory PropgFac(InfoMap, MP);
    PropgFac.push("prop0", prop_pt);
    Propagator& prop = PropgFac.getPropagator(TGprop, "prop0", wfn, &rng_dev);

std::cout<<" dist prop test: " <<NPOL <<" " <<NMO <<" " <<NAEA <<" " <<NAEB <<" " <<nwalk <<std::endl;

    wfn.Energy(wset);
    {
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      app_log(1," Initial Energy: {}", (eav / ov).real());
    }
    double tot_time = 0;
    RealType dt     = 0.01;
    RealType Eshift = std::abs(ComplexType(*wset[0].overlap()));
    Time.reset();
    for (int i = 0; i < 4; i++)
    {
      prop.Propagate(2, wset, Eshift, dt, 1);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      tot_time += 2 * dt;
      prop.Orthogonalize(wset);
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
    }
    for (int i = 0; i < 4; i++)
    {
      prop.Propagate(4, wset, Eshift, dt, 1);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      tot_time += 4 * dt;
      prop.Orthogonalize(wset);
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
    }

    for (int i = 0; i < 4; i++)
    {
      prop.Propagate(4, wset, Eshift, dt, 2);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      tot_time += 4 * dt;
      prop.Orthogonalize(wset);
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
    }
    for (int i = 0; i < 4; i++)
    {
      prop.Propagate(5, wset, Eshift, 2 * dt, 2);
      wfn.Energy(wset);
      ComplexType eav = 0, ov = 0;
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        eav += *it->weight() * (it->energy());
        ov += *it->weight();
      }
      tot_time += 5 * 2 * dt;
      prop.Orthogonalize(wset);
      app_log(1," -- {}  {}  {}",i,tot_time,(eav / ov).real());
    }

    if(TG.Global().root()) AFQMCTimer.print_all();
  }
}

TEST_CASE("propg_fac_shared", "[propagator_factory]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  arch::INIT(node);
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);
  setup_AFQMC_timer();

  propg_fac_shared<false>(world);
  propg_fac_shared<true>(world);
  release_memory_managers();
}
TEST_CASE("propg_fac_distributed", "[propagator_factory]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  int ngrp(world.size());
  arch::INIT(node);
#else
  int ngrp(world.size() / node.size());
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);
  setup_AFQMC_timer();

  propg_fac_distributed<false>(world, ngrp);
  propg_fac_distributed<true>(world, ngrp);
  release_memory_managers();
}

} // namespace sfqmc
