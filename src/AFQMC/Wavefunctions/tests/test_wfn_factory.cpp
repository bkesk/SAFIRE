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

//#undef NDEBUG

#include "catch_amalgamated.hpp"

#include "config.h"
#include "Utilities/AppAbort.hpp"

#include "io/ptree/ptree_utilities.hpp"
#include "hdf/hdf_archive.h"
#include "Utilities/Random.hpp"
#include "Utilities/app_loggers.h"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>

#include "Utilities/Timer.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/readWfn.cpp"
#include "Memory/buffer_managers.h"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

#include "SparseMatrix/csr_matrix_construct.hpp"
#include "Numerics/ma_blas.hpp"

using std::complex;
using std::ifstream;
using std::string;
using ma::real;
using ma::imag;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<bool MP, class Allocator>
void wfn_fac(boost::mpi3::communicator& world)
{
  if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
  {
    APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
  }
  else
  {
    // Global Task Group
    GlobalTaskGroup gTG(world);

    // First strip path of filename.
    std::string base_name = UTEST_WFN.substr(UTEST_WFN.find_last_of("\\/") + 1);
    // Remove file extension.
    std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
    auto file_data       = read_test_results_from_hdf<ComplexType>(UTEST_HAMIL, test_wfn);
    int NMO              = file_data.NMO;
    int NAEA             = file_data.NAEA;
    int NAEB             = file_data.NAEB;
    std::string wfn_type = afqmc::getWavefunctionType(UTEST_WFN);
    WALKER_TYPES type    = afqmc::getWalkerType(UTEST_WFN, wfn_type);
    int nspins           = (type == COLLINEAR) ? 2 : 1;
    int npol             = (type == NONCOLLINEAR) ? 2 : 1;

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));

    ptree ham_pt;
    ham_pt.put("name","ham0");
    ham_pt.put("system","info0");
    ham_pt.put("filename",UTEST_HAMIL);

    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt); 
    Hamiltonian& ham = HamFac.getHamiltonian(gTG, "ham0");

    //auto TG = TaskGroup_(gTG,std::string("WfnTG"),1,1);
    auto TG   = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());
    int nwalk = 11; // choose prime number to force non-trivial splits in shared routines
    utils::RandomGenerator_t rng;

    Allocator alloc_(make_localTG_allocator<ComplexType>(TG));

    ptree wlk_pt;
    wlk_pt.put("name","wset0");
    if(type == CLOSED) wlk_pt.put("walker_type","closed");
    else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
    else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
    else if (type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
    

    ptree wfn_pt;
    wfn_pt.put("name","wfn0");
    wfn_pt.put("system","info0");
    wfn_pt.put("filename",UTEST_WFN);

    WavefunctionFactory WfnFac(InfoMap, MP);
    WfnFac.push("wfn0", wfn_pt);
    Wavefunction& wfn = WfnFac.getWavefunction(TG, TG, "wfn0", type, &ham, 1e-6, nwalk);

    //for(int nw=1; nw<2; nw*=2)
    {
      //nwalk=nw;
      WalkerSet wset(TG, wlk_pt, InfoMap["info0"], &rng);
      auto initial_guess = WfnFac.getInitialGuess("wfn0");
      REQUIRE(initial_guess.size(0) == 2);
      REQUIRE(initial_guess.size(1) == npol * NMO);
      REQUIRE(initial_guess.size(2) == NAEA);

      if (type == COLLINEAR)
        wset.resize(nwalk, initial_guess[0], initial_guess[1](initial_guess.extension(1), {0, NAEB}));
      else
        wset.resize(nwalk, initial_guess[0], initial_guess[0]);

      // Overlap
      wfn.Overlap(wset);

      Watch Time;
      Time.reset();

      wfn.Energy(wset);
      TG.TG_local().barrier();
      if (std::abs(file_data.E0 + file_data.E1 + file_data.E2) > 1e-8)
      {
        for (auto it = wset.begin(); it != wset.end(); ++it)
        {
          REQUIRE(real(ComplexType(*it->E1())) == Approx(real(file_data.E0 + file_data.E1)));
          REQUIRE(real(ComplexType(*it->EXX()) + ComplexType(*it->EJ())) == Approx(real(file_data.E2)));
          REQUIRE(imag(it->energy()) == Approx(imag(file_data.E0 + file_data.E1 + file_data.E2)));
        }
      }
      else
      {
        app_log(1," E: {}", ComplexType(wset[0].energy())); 
        app_log(1," E0+E1: {}", ComplexType(*wset[0].E1()));
        app_log(1," EJ: {}", ComplexType(*wset[0].EJ())); 
        app_log(1," EXX: {}", ComplexType(*wset[0].EXX()));
      }

      auto size_of_G = wfn.size_of_G_for_vbias();
      int Gdim1      = (wfn.transposed_G_for_vbias() ? nwalk : size_of_G);
      int Gdim2      = (wfn.transposed_G_for_vbias() ? size_of_G : nwalk);
      using CMatrix = Matrix_<Allocator>;
      CMatrix G({Gdim1, Gdim2}, alloc_);
      wfn.MixedDensityMatrix_for_vbias(wset, G);

      double dt(0.01);
      auto nCV      = wfn.local_number_of_cholesky_vectors();
      CMatrix X({nCV, nwalk}, alloc_);
      Time.reset();
      wfn.vbias(G, X, dt);
      TG.TG_local().barrier();
      ComplexType Xsum = 0;
      if (std::abs(file_data.Xsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Xsum = 0;
          for (int i = 0; i < X.size(0); i++)
            Xsum += X[i][n];
          REQUIRE(real(ComplexType(Xsum)) == Approx(real(file_data.Xsum)));
          REQUIRE(imag(ComplexType(Xsum)) == Approx(imag(file_data.Xsum)));
        }
      }
      else
      {
        Xsum              = 0;
        ComplexType Xsum2 = 0;
        for (int i = 0; i < X.size(0); i++)
        {
          Xsum += X[i][0];
          Xsum2 += ComplexType(0.5) * X[i][0] * X[i][0];
        }
        app_log(1," Xsum: {}", ComplexType(Xsum));
        app_log(1," Xsum2 (EJ): {}", ComplexType(Xsum2) / dt);
      }

      // spin dependent HS potential?
      // generalize later
      int nx = ( wfn.getHamType() == ModelHamiltonian ? nspins*npol*npol : 1 ); 
      int vdim1 = (wfn.transposed_vHS() ? nwalk : NMO * NMO * nx );
      int vdim2 = (wfn.transposed_vHS() ? NMO * NMO * nx : nwalk );
      CMatrix vHS({vdim1, vdim2}, alloc_);
      Time.reset();
      wfn.vHS(X, vHS, dt);
      TG.TG_local().barrier();
      ComplexType Vsum = 0;
      if (std::abs(file_data.Vsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Vsum = 0;
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS.size(1); i++)
              Vsum += vHS[n][i];
          }
          else
          {
            for (int i = 0; i < vHS.size(0); i++)
              Vsum += vHS[i][n];
          }
          REQUIRE(real(ComplexType(Vsum)) == Approx(real(file_data.Vsum)));
          REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(file_data.Vsum)));
        }
      }
      else
      {
        Vsum = 0;
        if (wfn.transposed_vHS())
        {
          for (int i = 0; i < vHS.size(1); i++)
            Vsum += vHS[0][i];
        }
        else
        {
          for (int i = 0; i < vHS.size(0); i++)
            Vsum += vHS[i][0];
        }
        app_log(1," Vsum: {}", ComplexType(Vsum));
      }
      return;

      // Restarting Wavefunction from file
      ptree wfn_pt2;
      wfn_pt2.put("name","wfn1");
      wfn_pt2.put("system","info0");
      wfn_pt2.put("filename","./dummy.h5");

      WfnFac.push("wfn1", wfn_pt2);
      Wavefunction& wfn2 = WfnFac.getWavefunction(TG, TG, "wfn1", type, nullptr, 1e-6, nwalk);

      WalkerSet wset2(TG, wlk_pt, InfoMap["info0"], &rng);
      //auto initial_guess = WfnFac.getInitialGuess("wfn0");
      REQUIRE(initial_guess.size(0) == 2);
      REQUIRE(initial_guess.size(1) == npol * NMO);
      REQUIRE(initial_guess.size(2) == NAEA);

      if (type == COLLINEAR)
        wset2.resize(nwalk, initial_guess[0], initial_guess[1](initial_guess.extension(1), {0, NAEB}));
      else
        wset2.resize(nwalk, initial_guess[0], initial_guess[0]);

      wfn2.Overlap(wset2);
      for (auto it = wset2.begin(); it != wset2.end(); ++it)
      {
        REQUIRE(real(ComplexType(*it->overlap())) == Approx(1.0));
        REQUIRE(imag(ComplexType(*it->overlap())) == Approx(0.0));
      }

      wfn2.Energy(wset2);
      if (std::abs(file_data.E0 + file_data.E1 + file_data.E2) > 1e-8)
      {
        for (auto it = wset2.begin(); it != wset2.end(); ++it)
        {
          REQUIRE(real(ComplexType(*it->E1())) == Approx(real(file_data.E0 + file_data.E1)));
          REQUIRE(real(*it->EXX() + *it->EJ()) == Approx(real(file_data.E2)));
          REQUIRE(imag(it->energy()) == Approx(imag(file_data.E0 + file_data.E1 + file_data.E2)));
        }
      }
      else
      {
        app_log(1," E: {}", ComplexType(wset[0].energy())); 
        app_log(1," E0+E1: {}", ComplexType(*wset[0].E1()));
        app_log(1," EJ: {}", ComplexType(*wset[0].EJ())); 
        app_log(1," EXX: {}", ComplexType(*wset[0].EXX())); 
      }

      REQUIRE(size_of_G == wfn2.size_of_G_for_vbias());
      wfn2.MixedDensityMatrix_for_vbias(wset2, G);
      REQUIRE(nCV == wfn2.local_number_of_cholesky_vectors());
      wfn2.vbias(G, X, dt);
      Xsum = 0;
      if (std::abs(file_data.Xsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Xsum = 0;
          for (int i = 0; i < X.size(0); i++)
            Xsum += X[i][n];
          REQUIRE(real(ComplexType(Xsum)) == Approx(real(file_data.Xsum)));
          REQUIRE(imag(ComplexType(Xsum)) == Approx(imag(file_data.Xsum)));
        }
      }
      else
      {
        Xsum = 0;
        ComplexType Xsum2(0.0);
        for (int i = 0; i < X.size(0); i++)
        {
          Xsum += X[i][0];
          Xsum2 += ComplexType(0.5) * X[i][0] * X[i][0];
        }
        app_log(1," Xsum: {}", ComplexType(Xsum)); 
        app_log(1," Xsum2 (EJ): {}", ComplexType(Xsum2) / dt);
      }

      wfn2.vHS(X, vHS, dt);
      TG.TG_local().barrier();
      Vsum = 0;
      if (std::abs(file_data.Vsum) > 1e-8)
      {
        for (int n = 0; n < nwalk; n++)
        {
          Vsum = 0;
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS.size(1); i++)
              Vsum += vHS[n][i];
          }
          else
          {
            for (int i = 0; i < vHS.size(0); i++)
              Vsum += vHS[i][n];
          }
          REQUIRE(real(ComplexType(Vsum)) == Approx(real(file_data.Vsum)));
          REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(file_data.Vsum)));
        }
      }
      else
      {
        Vsum = 0;
        if (wfn.transposed_vHS())
        {
          for (int i = 0; i < vHS.size(1); i++)
            Vsum += vHS[0][i];
        }
        else
        {
          for (int i = 0; i < vHS.size(0); i++)
            Vsum += vHS[i][0];
        }
        app_log(1," Vsum: {}", ComplexType(Vsum));
      }

      TG.Global().barrier();
      // remove temporary file
      if (TG.Node().root())
        remove("dummy.h5");
    }
  }
}

template<bool MP, class Allocator>
void wfn_fac_distributed(boost::mpi3::communicator& world, int ngroups)
{

  if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
  {
    APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
  }
  else
  {
    // Global Task Group
    GlobalTaskGroup gTG(world);

    // First strip path of filename.
    std::string base_name = UTEST_WFN.substr(UTEST_WFN.find_last_of("\\/") + 1);
    // Remove file extension.
    std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
    auto file_data       = read_test_results_from_hdf<ComplexType>(UTEST_HAMIL, test_wfn);
    int NMO              = file_data.NMO;
    int NAEA             = file_data.NAEA;
    int NAEB             = file_data.NAEB;
    std::string wfn_type = afqmc::getWavefunctionType(UTEST_WFN);
    WALKER_TYPES type    = afqmc::getWalkerType(UTEST_WFN, wfn_type);
    int npol             = (type == NONCOLLINEAR) ? 2 : 1;
    int nspins           = (type == COLLINEAR) ? 2 : 1;

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));

    ptree ham_pt;
    ham_pt.put("name","ham0");
    ham_pt.put("system","info0");
    ham_pt.put("filename",UTEST_HAMIL);

    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(gTG, "ham0");

    auto TG    = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());
    auto TGwfn = TaskGroup_(gTG, std::string("WfnTG"), ngroups, gTG.getTotalCores());
    int nwalk  = 11; // choose prime number to force non-trivial splits in shared routines
    utils::RandomGenerator_t rng;

    Allocator alloc_(make_localTG_allocator<ComplexType>(TG));

    ptree wlk_pt;
    wlk_pt.put("name","wset0");
    if(type == CLOSED) wlk_pt.put("walker_type","closed");
    else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
    else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
    else if (type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
    WalkerSet wset(TG, wlk_pt, InfoMap["info0"], &rng);

    ptree wfn_pt;
    wfn_pt.put("name","wfn0");
    wfn_pt.put("system","info0");
    wfn_pt.put("filename",UTEST_WFN);

    WavefunctionFactory WfnFac(InfoMap, MP);
    WfnFac.push("wfn0", wfn_pt);
    Wavefunction& wfn = WfnFac.getWavefunction(TGwfn, TGwfn, "wfn0", type, &ham, 1e-6, nwalk);

    auto initial_guess = WfnFac.getInitialGuess("wfn0");
    REQUIRE(initial_guess.size(0) == 2);
    REQUIRE(initial_guess.size(1) == npol * NMO);
    REQUIRE(initial_guess.size(2) == NAEA);

    if (type == COLLINEAR)
      wset.resize(nwalk, initial_guess[0], initial_guess[1](initial_guess.extension(1), {0, NAEB}));
    else
      wset.resize(nwalk, initial_guess[0], initial_guess[0]);

    wfn.Overlap(wset);

    using CMatrix = ComplexMatrix<Allocator>;
    Watch Time;
    Time.reset();
    wfn.Energy(wset);
    TG.TG().barrier();

    if (std::abs(file_data.E0 + file_data.E1 + file_data.E2) > 1e-8)
    {
      for (auto it = wset.begin(); it != wset.end(); ++it)
      {
        REQUIRE(real(ComplexType(*it->E1())) == Approx(real(file_data.E0 + file_data.E1)));
        REQUIRE(real(*it->EXX() + *it->EJ()) == Approx(real(file_data.E2)));
        REQUIRE(imag(it->energy()) == Approx(imag(file_data.E0 + file_data.E1 + file_data.E2)));
      }
    }
    else
    {
      app_log(1," E: {}", ComplexType(wset[0].energy())); 
      app_log(1," E0+E1: {}", ComplexType(*wset[0].E1()));
      app_log(1," EJ: {}", ComplexType(*wset[0].EJ())); 
      app_log(1," EXX: {}", ComplexType(*wset[0].EXX())); 
    }

    auto size_of_G = wfn.size_of_G_for_vbias();
    int Gdim1      = (wfn.transposed_G_for_vbias() ? nwalk : size_of_G);
    int Gdim2      = (wfn.transposed_G_for_vbias() ? size_of_G : nwalk);
    CMatrix G({Gdim1, Gdim2}, alloc_);
    wfn.MixedDensityMatrix_for_vbias(wset, G);

    double dt(0.01);
    auto nCV      = wfn.local_number_of_cholesky_vectors();
    CMatrix X({nCV, nwalk}, alloc_);
    Time.reset();
    wfn.vbias(G, X, dt);
    TG.TG().barrier();

    ComplexType Xsum = 0;
    if (std::abs(file_data.Xsum) > 1e-8)
    {
      for (int n = 0; n < nwalk; n++)
      {
        Xsum = 0;
        if (TGwfn.TG_local().root())
          for (int i = 0; i < X.size(0); i++)
            Xsum += X[i][n];
        Xsum = (TGwfn.TG() += Xsum);
        REQUIRE(real(ComplexType(Xsum)) == Approx(real(file_data.Xsum)));
        REQUIRE(imag(ComplexType(Xsum)) == Approx(imag(file_data.Xsum)));
      }
    }
    else
    {
      Xsum = 0;
      if (TGwfn.TG_local().root())
        for (int i = 0; i < X.size(0); i++)
          Xsum += X[i][0];
      Xsum = (TGwfn.TG() += Xsum);
      app_log(1," Xsum: {}", ComplexType(Xsum)); 
    }

    // vbias must be reduced if false
    if (not wfn.distribution_over_cholesky_vectors())
    {
      boost::multi::array<ComplexType, 2> T({nCV, nwalk});
      if (TGwfn.TG_local().root())
        std::copy_n(X.origin(), X.num_elements(), T.origin());
      else
        std::fill_n(T.origin(), T.num_elements(), ComplexType(0.0, 0.0));
      TGwfn.TG().all_reduce_in_place_n(raw_pointer_cast(T.origin()), T.num_elements(), std::plus<>());
      if (TGwfn.TG_local().root())
        std::copy_n(T.origin(), T.num_elements(), X.origin());
      TGwfn.TG_local().barrier();
    }

    // spin dependent HS potential?
    // generalize later
    int nx = ( wfn.getHamType() == ModelHamiltonian ? nspins*npol*npol : 1 ); 
    int vdim1 = (wfn.transposed_vHS() ? nwalk : NMO * NMO * nx );
    int vdim2 = (wfn.transposed_vHS() ? NMO * NMO * nx : nwalk);
    CMatrix vHS({vdim1, vdim2}, alloc_);
    Time.reset();
    wfn.vHS(X, vHS, dt);
    TG.TG_local().barrier();
    ComplexType Vsum = 0;
    if (std::abs(file_data.Vsum) > 1e-8)
    {
      for (int n = 0; n < nwalk; n++)
      {
        Vsum = 0;
        if (TGwfn.TG_local().root())
        {
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS.size(1); i++)
              Vsum += vHS[n][i];
          }
          else
          {
            for (int i = 0; i < vHS.size(0); i++)
              Vsum += vHS[i][n];
          }
        }
        Vsum = (TGwfn.TG() += Vsum);
        REQUIRE(real(ComplexType(Vsum)) == Approx(real(file_data.Vsum)));
        REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(file_data.Vsum)));
      }
    }
    else
    {
      Vsum = 0;
      if (TGwfn.TG_local().root())
      {
        if (wfn.transposed_vHS())
        {
          for (int i = 0; i < vHS.size(1); i++)
            Vsum += vHS[0][i];
        }
        else
        {
          for (int i = 0; i < vHS.size(0); i++)
            Vsum += vHS[i][0];
        }
      }
      Vsum = (TGwfn.TG() += Vsum);
      app_log(1," Vsum: {}", ComplexType(Vsum));
    }
    return;

    // Restarting Wavefunction from file
    ptree wfn_pt2;
    wfn_pt2.put("name","wfn1");
    wfn_pt2.put("system","info0");
    wfn_pt2.put("filename","./dummy.h5");

    WfnFac.push("wfn1", wfn_pt2);
    Wavefunction& wfn2 = WfnFac.getWavefunction(TG, TG, "wfn1", type, nullptr, 1e-6, nwalk);

    WalkerSet wset2(TG, wlk_pt, InfoMap["info0"], &rng);
    //auto initial_guess = WfnFac.getInitialGuess("wfn0");
    REQUIRE(initial_guess.size(0) == 2);
    REQUIRE(initial_guess.size(1) == npol * NMO);
    REQUIRE(initial_guess.size(2) == NAEA);

    if (type == COLLINEAR)
      wset2.resize(nwalk, initial_guess[0], initial_guess[1](initial_guess.extension(1), {0, NAEB}));
    else
      wset2.resize(nwalk, initial_guess[0], initial_guess[0]);

    wfn2.Overlap(wset2);
    //for(auto it = wset2.begin(); it!=wset2.end(); ++it) {
    //REQUIRE(real(*it->overlap()) == Approx(1.0));
    //REQUIRE(imag(*it->overlap()) == Approx(0.0));
    //}

    wfn2.Energy(wset2);
    if (std::abs(file_data.E0 + file_data.E1 + file_data.E2) > 1e-8)
    {
      for (auto it = wset2.begin(); it != wset2.end(); ++it)
      {
        REQUIRE(real(ComplexType(*it->E1())) == Approx(real(file_data.E0 + file_data.E1)));
        REQUIRE(real(*it->EXX() + *it->EJ()) == Approx(real(file_data.E2)));
        REQUIRE(imag(ComplexType(it->energy())) == Approx(imag(file_data.E0 + file_data.E1 + file_data.E2)));
      }
    }
    else
    {
      app_log(1," E: {}", ComplexType(wset[0].energy())); 
      app_log(1," E0+E1: {}", ComplexType(*wset[0].E1()));
      app_log(1," EJ: {}", ComplexType(*wset[0].EJ())); 
      app_log(1," EXX: {}", ComplexType(*wset[0].EXX())); 
    }

    REQUIRE(size_of_G == wfn2.size_of_G_for_vbias());
    wfn2.MixedDensityMatrix_for_vbias(wset2, G);

    nCV = wfn2.local_number_of_cholesky_vectors();
    wfn2.vbias(G, X, dt);
    Xsum = 0;
    if (std::abs(file_data.Xsum) > 1e-8)
    {
      for (int n = 0; n < nwalk; n++)
      {
        Xsum = 0;
        if (TGwfn.TG_local().root())
          for (int i = 0; i < X.size(0); i++)
            Xsum += X[i][n];
        Xsum = (TGwfn.TG() += Xsum);
        REQUIRE(real(ComplexType(Xsum)) == Approx(real(file_data.Xsum)));
        REQUIRE(imag(ComplexType(Xsum)) == Approx(imag(file_data.Xsum)));
      }
    }
    else
    {
      Xsum = 0;
      if (TGwfn.TG_local().root())
        for (int i = 0; i < X.size(0); i++)
          Xsum += X[i][0];
      Xsum = (TGwfn.TG() += Xsum);
      app_log(1," Xsum: {}", ComplexType(Xsum)); 
    }

    // vbias must be reduced if false
    if (not wfn.distribution_over_cholesky_vectors())
    {
      boost::multi::array<ComplexType, 2> T({nCV, nwalk});
      if (TGwfn.TG_local().root())
        std::copy_n(X.origin(), X.num_elements(), T.origin());
      else
        std::fill_n(T.origin(), T.num_elements(), ComplexType(0.0, 0.0));
      TGwfn.TG().all_reduce_in_place_n(raw_pointer_cast(T.origin()), T.num_elements(), std::plus<>());
      if (TGwfn.TG_local().root())
        std::copy_n(T.origin(), T.num_elements(), X.origin());
      TGwfn.TG_local().barrier();
    }

    wfn2.vHS(X, vHS, dt);
    TG.TG_local().barrier();
    Vsum = 0;
    if (std::abs(file_data.Vsum) > 1e-8)
    {
      for (int n = 0; n < nwalk; n++)
      {
        Vsum = 0;
        if (TGwfn.TG_local().root())
        {
          if (wfn.transposed_vHS())
          {
            for (int i = 0; i < vHS.size(1); i++)
              Vsum += vHS[n][i];
          }
          else
          {
            for (int i = 0; i < vHS.size(0); i++)
              Vsum += vHS[i][n];
          }
        }
        Vsum = (TGwfn.TG() += Vsum);
        REQUIRE(real(ComplexType(Vsum)) == Approx(real(file_data.Vsum)));
        REQUIRE(imag(ComplexType(Vsum)) == Approx(imag(file_data.Vsum)));
      }
    }
    else
    {
      Vsum = 0;
      if (TGwfn.TG_local().root())
      {
        if (wfn.transposed_vHS())
        {
          for (int i = 0; i < vHS.size(1); i++)
            Vsum += vHS[0][i];
        }
        else
        {
          for (int i = 0; i < vHS.size(0); i++)
            Vsum += vHS[i][0];
        }
      }
      Vsum = (TGwfn.TG() += Vsum);
      app_log(1," Vsum: {}", ComplexType(Vsum));
    }

    TG.Global().barrier();
    // remove temporary file
    if (TG.Node().root())
      remove("dummy.h5");
  }
}

TEST_CASE("wfn_fac_sdet", "[wavefunction_factory]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());
  setup_loggers(world.root(),2,2);

#if defined(ENABLE_DEVICE)

  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);

  wfn_fac<false,Alloc>(world);
  wfn_fac<true,Alloc>(world);
  release_memory_managers();
}

TEST_CASE("wfn_fac_distributed", "[wavefunction_factory]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_DEVICE)
  auto node = world.split_shared(world.rank());
  int ngrp(world.size());

  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  auto node   = world.split_shared(world.rank());
  int ngrp(world.size() / node.size());
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);

  wfn_fac_distributed<false,Alloc>(world, ngrp);
  wfn_fac_distributed<true,Alloc>(world, ngrp);
  release_memory_managers();
}


} // namespace sfqmc
