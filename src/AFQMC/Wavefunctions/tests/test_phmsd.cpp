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

#include "hdf/hdf_archive.h"
#include "io/ptree/ptree_utilities.hpp"
#include "Utilities/Random.hpp"
#include "Utilities/app_loggers.h"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>
#include <algorithm>

#include "AFQMC/Wavefunctions/Excitations.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Utilities/readWfn.h"
#include "Memory/buffer_managers.h"


using std::complex;
using std::ifstream;
using std::string;
using std::real;
using std::imag;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<class Allocator>
void test_read_phmsd(boost::mpi3::communicator& world)
{

  if (not file_exists(UTEST_WFN))
  {
    APP_ABORT(" Wavefunction file not found. Run unit test with --wfn /path/to/wfn.dat.");
  }
  else
  {
    // Global Task Group
    GlobalTaskGroup gTG(world);
    auto TG    = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());
    auto TGwfn = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());

    int NMO;
    int nup;
    int ndown;
    std::tie(NMO, nup, ndown) = read_info_from_wfn(UTEST_WFN, "PHMSD");
    WALKER_TYPES walker_type = afqmc::getWalkerType(UTEST_WFN, "PHMSD");
    hdf_archive dump;
    if (!dump.open(UTEST_WFN, H5F_ACC_RDONLY))
      APP_ABORT("Error reading wavefunction file.");
    if (dump.push("Wavefunction", false)<0)
      APP_ABORT(" Error in test_read_phmsd: Group Wavefunction not found. ");
    if (dump.push("PHMSD", false)<0)
      APP_ABORT(" Error in test_read_phmsd: Group PHMSD not found. ");
    int ndets_to_read = -1;
    std::string wfn_type;
    std::vector<PsiT_Matrix> PsiT_MO; 
    std::vector<int> occbuff;
    std::vector<ComplexType> coeffs;
    read_ph_wavefunction_hdf(dump, coeffs, occbuff, ndets_to_read, walker_type, TGwfn.Node(), NMO, nup, ndown, PsiT_MO,
                             wfn_type);
    boost::multi::array_ref<int, 2> occs(raw_pointer_cast(occbuff.data()), {ndets_to_read, nup + ndown});
    ph_excitations<int, ComplexType> abij = build_ph_struct(coeffs, occs, ndets_to_read, TGwfn.Node(), NMO, nup, ndown);
    using std::get;
    auto cit = abij.configurations_begin();
    std::vector<int> configa(nup), configb(ndown);
    // Is it fortuitous that the order of determinants is the same?
    for (int nd = 0; nd < ndets_to_read; nd++, ++cit)
    {
      int alpha_ix = get<0>(*cit);
      int beta_ix  = get<1>(*cit);
      auto ci      = get<2>(*cit);
      abij.get_configuration(0, alpha_ix, configa);
      abij.get_configuration(1, beta_ix, configb);
      std::sort(configa.begin(), configa.end());
      std::sort(configb.begin(), configb.end());
      for (int i = 0; i < nup; i++)
      {
        REQUIRE(configa[i] == occs[nd][i]);
      }
      for (int i = 0; i < ndown; i++)
      {
        REQUIRE(configb[i] == occs[nd][i + nup]);
      }
      REQUIRE(std::abs(coeffs[nd]) == std::abs(ci));
    }
    // Check sign of permutation.
    REQUIRE(abij.number_of_configurations() == ndets_to_read);
  }
}

void getBasicWavefunction(std::vector<int>& occs, std::vector<ComplexType>& coeffs, int NEL)
{
  hdf_archive dump;
  if (!dump.open(UTEST_WFN, H5F_ACC_RDONLY))
    APP_ABORT("Error reading wavefunction file.");
  if (dump.push("Wavefunction", false)<0)
    APP_ABORT(" Error in getBasicWavefunction: Group Wavefunction not found.");
  if (dump.push("PHMSD", false)<0)
    APP_ABORT(" Error in getBasicWavefunction: Group PHMSD not found.");
  std::vector<int> Idata(5);
  if (!dump.readEntry(Idata, "dims"))
    APP_ABORT("Errro reading dims array");
  int ndets = Idata[4];
  occs.resize(ndets * NEL);
  if (!dump.readEntry(occs, "occs"))
    APP_ABORT("Error reading occs array.");
  std::vector<ComplexType> ci_coeffs(ndets);
  if (!dump.readEntry(coeffs, "ci_coeffs"))
    APP_ABORT("Error reading occs array.");
}

// Construct PsiT^{dagger}
template<class Mat>
void getSlaterMatrix(Mat&& SM, boost::multi::array_ref<int, 1>& occs, int NEL)
{
  using T = typename std::decay_t<Mat>::element;
  ma::fill(SM, T(0.0));
  for (int i = 0; i < NEL; i++)
    SM[i][occs[i]] = T(1.0);
}

template<bool MP, class Allocator>
void test_phmsd(boost::mpi3::communicator& world)
{

  if (not file_exists(UTEST_WFN) || not file_exists(UTEST_HAMIL))
  {
    APP_ABORT(" Wavefunction and/or Hamiltonian file not found. Run unit test with --wfn /path/to/wfn.h5 and --hamil /path/to/hamil.h5. ");
  }
  else
  {
    // Global Task Group
    GlobalTaskGroup gTG(world);
    auto TG    = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());
    auto TGwfn = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());
    Allocator alloc_(make_localTG_allocator<ComplexType>(TG));

    int nwalk                 = 1;
    int NMO;
    int nup;
    int ndown;
    std::tie(NMO, nup, ndown) = read_info_from_wfn(UTEST_WFN, "PHMSD");
    // Test overlap.
    //wfn.Overlap(wset);
    WALKER_TYPES type = afqmc::getWalkerType(UTEST_WFN, "PHMSD");
    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));

    int npol = ((type == NONCOLLINEAR) ? 2 : 1);

    ptree ham_pt;
    ham_pt.put("name","ham0");
    ham_pt.put("system","info0");
    ham_pt.put("filename",UTEST_HAMIL);

    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(gTG, "ham0");

    ptree wfn_pt;
    wfn_pt.put("name","wfn0");
    wfn_pt.put("system","info0");
    wfn_pt.put("type","phmsd");
    wfn_pt.put("filename",UTEST_WFN);
    wfn_pt.put("rediag","no");

    WavefunctionFactory WfnFac(InfoMap, MP);
    WfnFac.push("wfn0", wfn_pt);
    Wavefunction& wfn = WfnFac.getWavefunction(TGwfn, TGwfn, "wfn0", type, &ham, 1e-6, nwalk);

    ptree wlk_pt;
    wlk_pt.put("name","wset0");
    if(type==COLLINEAR)
      wlk_pt.put("walker_type","collinear");
    else if(type==NONCOLLINEAR)
      wlk_pt.put("walker_type","noncollinear");
    else if(type==FULLYPOLARIZED)
      wlk_pt.put("walker_type","fullypolarized");
    else
      APP_ABORT(" Error in test_phmsd: Incorrect walker type.");
    utils::RandomGenerator_t rng;
    WalkerSet wset(TG, wlk_pt, InfoMap["info0"], &rng);

    auto initial_guess = WfnFac.getInitialGuess("wfn0");
    REQUIRE(initial_guess.size(0) == 2);
    REQUIRE(initial_guess.size(1) == npol*NMO);
    REQUIRE(initial_guess.size(2) == nup);

    wset.resize(nwalk, initial_guess[0], initial_guess[1](initial_guess.extension(1), {0, ndown}));
    // 1. Test Overlap Explicitly
    // 1.a Get raw occupancies and coefficients from file.
    std::vector<ComplexType> coeffs;
    std::vector<int> buff;
    getBasicWavefunction(buff, coeffs, nup + ndown);
    int ndets = coeffs.size();
    boost::multi::array_ref<int, 2> occs(buff.data(), {ndets, nup + ndown});
    // 1.b Compute overlap of trial wavefunction compotents.
    boost::multi::array<ComplexType, 2> Orbs({npol*NMO, npol*NMO});
    for (int i = 0; i < npol*NMO; i++)
      Orbs[i][i] = ComplexType(1.0);
    boost::multi::array<ComplexType, 2, Allocator> TrialA({nup, npol*NMO}, ComplexType(0.0), alloc_);
    boost::multi::array<ComplexType, 2, Allocator> TrialB({ndown, npol*NMO}, ComplexType(0.0), alloc_);
    auto sdet = wfn.getSlaterDetOperations();
    ComplexType ovlp_sum = ComplexType(0.0);
    ComplexType logovlp(0.0);
    //boost::multi::array<ComplexType,2> GBuff;
    for (int idet = 0; idet < coeffs.size(); idet++)
    {
      // Construct slater matrix from given set of occupied orbitals.
      ComplexType ovlpa, ovlpb = ComplexType(1.0);
      boost::multi::array_ref<int, 1> oa(occs[idet].origin(), {nup});
      getSlaterMatrix(TrialA, oa, nup);
      ovlpa = sdet->Overlap(TrialA, *wset[0].SlaterMatrix(Alpha), logovlp);
      if(type == COLLINEAR) {
        boost::multi::array_ref<int, 1> ob(occs[idet].origin() + nup, {ndown});
        for (int i = 0; i < ndown; i++)
          ob[i] -= NMO;
        getSlaterMatrix(TrialB, ob, ndown);
        ovlpb = sdet->Overlap(TrialB, *wset[0].SlaterMatrix(Beta), logovlp);
      }
      ovlp_sum += ma::conj(coeffs[idet]) * ovlpa * ovlpb;
    }
    wfn.Overlap(wset);

    for (auto it = wset.begin(); it != wset.end(); ++it)
    {
      REQUIRE(std::abs(real(ComplexType(*it->overlap()))) == Approx(std::abs(real(ovlp_sum))));
      REQUIRE(std::abs(imag(ComplexType(*it->overlap()))) == Approx(std::abs(imag(ovlp_sum))));
    }

  }
}

TEST_CASE("test_read_phmsd", "[test_read_phmsd]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);

  test_read_phmsd<Alloc>(world);

  release_memory_managers();
}

TEST_CASE("test_phmsd", "[read_phmsd]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);

  //test_phmsd<Alloc,SlaterDetOperations_serial<Alloc>>(world);
  test_phmsd<false,Alloc>(world);
  test_phmsd<true,Alloc>(world);

  release_memory_managers();
}

} // namespace sfqmc
