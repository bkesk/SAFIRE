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
#include "AFQMC/config.h"

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/test_common.hpp"
#include "utilities/check.hpp"
#include "utilities/h5_utils.hpp"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"
#include "AFQMC/SlaterDeterminantOperations/orthogonalize.hpp"

#include "numerics/sparse/sparse.hpp"

extern std::string UTEST_HAMIL, UTEST_WFN;
namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void ham_ops_basic_serial(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                 std::string hamil_file, std::string wfn_file)
{
  using sfqmc::utils::ARRAY_EQUAL;
  using nda::range;
  using matrix_t = memory::array<MEM,ComplexType,2>;
  auto all = range::all;
  app_log(1, "Running Hamiltonian operations unit test "
    "with:\n  Hamiltonian file: {}\n  wavefunction file: {}", 
    hamil_file, wfn_file
  );
  
  utils::check(utils::file_exists(hamil_file),
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);
  utils::check(utils::file_exists(wfn_file),
               " Wavefunction file not found: {}. \n Run unit test with --wfn /path/to/wfn.h5 ", wfn_file);

  // Determine wavefunction type for test results from wavefunction file name which is
  // has the naming convention wfn_(wfn_type).dat.
  // First strip path of filename.
  std::string base_name = wfn_file.substr(wfn_file.find_last_of("\\/") + 1);
  // Remove file extension.
  std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
  test_wfn = test_wfn.substr(test_wfn.find('_') + 1);

  auto file_data = read_test_results_from_hdf<ComplexType>(hamil_file, test_wfn);
  // finite-T nup <-- ntau, ndown = 0
  auto [NMO,nup,ndown] = read_info_from_wfn(wfn_file, "any");
  utils::check(NMO == file_data.NMO, "Incompatible NMO.");

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  WavefunctionFactory<MEM> WfnFac(InfoMap);
  ptree wfn_pt;
  wfn_pt.put("name", "wfn0");
  wfn_pt.put("system", "info0");
  wfn_pt.put("filename", wfn_file);
  
  WfnFac.push("wfn0", wfn_pt);
  WALKER_TYPES wtype = getWalkerType(wfn_file);
  Wavefunction<MEM>& wfn = WfnFac.getWavefunction(mpi, "wfn0", wtype, &ham, 1);
  
  WalkerSetFactory<MEM> WSetFac(InfoMap);
  ptree wset_pt;
  wset_pt.put("name", "wset0");
  wset_pt.put("system", "info0");
  wset_pt.put("walker_type", walkerTypeToString(wtype));
  WSetFac.push("wset0", wset_pt);
  
  auto rng_wlk = std::make_shared<utils::RandomGenerator_t<>>(2026);
  auto& wset = WSetFac.getWalkerSet(mpi, "wset0", rng_wlk);


  int nel  = (wtype == COLLINEAR or wtype == COLLINEAR_FT) ? (nup+ndown) : nup;
  int npol = (wtype == NONCOLLINEAR or wtype == NONCOLLINEAR_FT) ? 2 : 1;
  int nspin = (wtype == COLLINEAR or wtype == COLLINEAR_FT) ? 2 : 1;
  int ndet = 1;
  int nwalk = 5;
  // time-slice index for finite-T
  int nt = 0;

  wset.resize(nwalk, WfnFac.getInitialGuess("wfn0")());

  // Perturb the initial guess by a deterministic non-trivial sequence and
  // re-orthonormalise.
  std::array nels = {nup, ndown};
  for(int spin = 0; spin < nspin; spin++) {
    nda::array<ComplexType, 1> p_h(nwalk * npol * NMO * nels[spin]);
    for (long k = 0; k < p_h.size(); ++k) {
      double v = 0.01 * (k + 1);
      p_h[k] = ComplexType(v, v * v);
    }
    memory::array<MEM, ComplexType, 3> p(reshape(p_h, nwalk, npol * NMO, nels[spin]));
    auto SM = wset.SlaterMatrices(static_cast<SpinTypes>(spin));
    nda::tensor::add(p, "ijk", SM, "ijk");
    memory::array<MEM, ComplexType, 1> log_detR(nwalk);
    det_ops::orthogonalize(SM, log_detR);
  }

  wfn.runtime_optimization(wset);
  
  // Overlap/GreenFunction
  // optimize HOps evaluation
  wfn.runtime_optimization(wset); 

  // Energy
  wfn.Energy(wset);

  nda::array<ComplexType, 1> E1(nwalk), EJ(nwalk), EXX(nwalk);
  wset.getProperty(E1_, E1);
  wset.getProperty(EJ_, EJ);
  wset.getProperty(EXX_, EXX);
  if (std::abs(file_data.E0 + file_data.E1) > 1e-8) {
    ARRAY_EQUAL(E1, nda::array<ComplexType,1>(nwalk,file_data.E0 + file_data.E1)); 
  } else {
    app_log(1," E1: {} ", E1);
  }

  if (std::abs(file_data.E2) > 1e-8)
  {
    ARRAY_EQUAL(nda::make_regular(EXX + EJ), nda::array<ComplexType,1>(nwalk,file_data.E2)); 
  }
  else
  {
    app_log(1," EJ: {}", EJ);
    app_log(1," EXX: {}", EXX); 
  }

  double dt = 0.01;
  auto nCV  = wfn.number_of_cholesky_vectors();

  {
    nda::array<ComplexType, 1> nMF(2*NMO, ComplexType(1.0));
    memory::array<MEM,ComplexType, 1> vMF(nCV);
    wfn.update_potentials(dt,nMF,vMF,true);
  }

  memory::array<MEM,ComplexType,2> X(nwalk, nCV);
  X() = ComplexType(0.0);
  wfn.vbias(wset, X, dt);
  ComplexType Xsum = 0, Xsum2 = 0;
  auto X_h = nda::to_host(X);
  for (int i = 0; i < nCV; i++)
  {
    Xsum += X_h(0,i);
    Xsum2 += ComplexType(0.5) * X_h(0,i) * X_h(0,i);
  }
  if (std::abs(file_data.Xsum) > 1e-8)
  {
    utils::VALUE_EQUAL(Xsum,file_data.Xsum);
  }
  else
  {
    app_log(1," Xsum: {}", Xsum);
    app_log(1," Xsum2 (EJ): {}", Xsum2 / dt);
  }

  nda::array<ComplexType, 1> X_h_real = nda::real(X_h(0,all)); // cannot use finite imaginary part for vMF
  auto h1 = wfn.getOneBodyPropagatorMatrix(dt, X_h_real);
  CHECK( h1.shape() == std::array<long,3>{nspin,npol*NMO,npol*NMO} );

  auto[vHS_nspin, vHS_npol] = wfn.vHS_dims();
  auto vHS = wfn.vHS(X,dt);
  CHECK( vHS.shape() == std::array<long,4>{vHS_nspin,nwalk,vHS_npol*NMO,NMO} );
  auto vHS_h = nda::to_host(vHS);
  ComplexType Vsum = 0;
  for (int i = 0; i < vHS.extent(2); i++)
    for (int j = 0; j < vHS.extent(3); j++)
      Vsum += vHS_h(0,0,i,j);
  if (std::abs(file_data.Vsum) > 1e-8)
  {
    utils::VALUE_EQUAL(Vsum,file_data.Vsum);
  }
  else
  {
    app_log(1," Vsum: {}", Vsum);
  }

  if (wfn.getHamType() == ModelHamiltonian )
  {
    auto vHS_sparse = wfn.vHS_sparse(X,dt);
    ComplexType Vsum2 = nda::sum(nda::to_host(vHS_sparse(0).values()))/double(nwalk);
    if (std::abs(file_data.Vsum) > 1e-8)
    {
      REQUIRE(real(Vsum2) == Approx(real(file_data.Vsum)));
      REQUIRE(imag(Vsum2) == Approx(imag(file_data.Vsum)));
    }
    else
    {
      app_log(1," Vsum sparse: {}", Vsum2);
    }
  }

/*
  // Generalised Fock matrix. only implemented on Real3IndexFactorization_batched_v2 
#if !defined(ENABLE_DEVICE)
  return;
#endif
  if(HOps.getHamType() != RealDenseFactorized) return;
  // Test Generalised Fock matrix.
  int dm_size;
  CMatrix G2({2 * NMO, NMO}, alloc_);
  dm_size = 2 * NMO * NMO;
  boost::multi::array<ComplexType, 2> Mat({NMO, NMO});
  hdf_archive ref;
  if (!ref.open("G.h5", H5F_ACC_RDONLY))
    APP_ABORT(" Error opening HDF wavefunction file.");
  ref.readEntry(Mat, "Ga");
  for (int i = 0; i < NMO; i++)
  {
    for (int j = 0; j < NMO; j++)
    {
      G2[i][j] = Mat[i][j];
    }
  }
  ref.readEntry(Mat, "Gb");
  for (int i = 0; i < NMO; i++)
  {
    for (int j = 0; j < NMO; j++)
    {
      G2[NMO + i][j] = Mat[i][j];
    }
  }

  //if(wtype==COLLINEAR) {
  //dm_size = 2*NMO*NMO;
  //G2 = CMatrix({2*NMO,NMO});
  //} else if(wtype==CLOSED) {
  //dm_size = NMO*NMO;
  //G2 = CMatrix({NMO,NMO});
  //} else {
  //APP_ABORT("NON COLLINEAR Wavefunction not implemented.");
  //}
  //Ovlp = SDet.MixedDensityMatrix(devPsiT[0],devOrbMat[0], G2.sliced(0,NMO),0.0,false);
  //if(wtype==COLLINEAR) {
  //Ovlp *= SDet.MixedDensityMatrix(devPsiT[1],devOrbMat[1](devOrbMat.extension(1),{0,ndown}), G.sliced(NMO,2*NMO),0.0,false);
  //}
  int nwalk = 1;
  CMatrix Gw2({nwalk, dm_size}, alloc_);
  for (int nw = 0; nw < nwalk; nw++)
  {
    for (int i = 0; i < NMO; i++)
    {
      for (int j = 0; j < NMO; j++)
      {
        // MAM: not sur why this needs a cast!
        Gw2[nw][i * NMO + j]             = ComplexType(G2[i][j]);
        Gw2[nw][NMO * NMO + i * NMO + j] = ComplexType(G2[i + NMO][j]);
      }
    }
  }
  boost::multi::array_ref<ComplexType, 2, pointer> Gw2_(make_device_ptr(Gw2.origin()), {nwalk, dm_size});
  boost::multi::array<ComplexType, 3, Alloc> GFock({2, nwalk, dm_size}, alloc_);
  fill_n(GFock.origin(), GFock.num_elements(), ComplexType(0.0));
  //for(int i = 0; i < nwalk; i++) {
  //std::cout << Gw2_[i][0] << std::endl;
  //}
  //std::cout << "INIT: " << Gw2_[0][0] << " " << Gw2_[0][NMO*NMO] << std::endl;
  HOps.generalizedFockMatrix(Gw2_, GFock[0], GFock[1]);
  //boost::multi::array_ref<ComplexType,2,pointer> GR(make_device_ptr(GFock[0][0].origin()), {NMO,NMO});
  //for(int i = 0; i < nwalk; i++) {
  //std::cout << GFock[0][i][0] << std::endl;
  //}
  //std::cout << "GFOCK: " << GFock[0][0][0] << " " << GFock[1][0][0] << std::endl;
  //std::cout << "Fm: " << std::endl;
  std::fill_n(Mat.origin(), Mat.num_elements(), 0.0);
  ref.readEntry(Mat, "Fha");
  for (int i = 0; i < NMO; i++)
  {
    for (int j = 0; j < NMO; j++)
    {
      if (std::abs(Mat[i][j] - real(ComplexType(GFock[1][0][i * NMO + j]))) > 1e-5)
      {
        std::cout << "DELTAA: " << i << " " << j << " " << Mat[i][j] << " " << real(ComplexType(GFock[1][0][i * NMO + j]))
                  << std::endl;
      }
      //if(std::abs(real(GFock[1][0][i*NMO+j]))>1e-6)
      //std::cout << i << " " << j << " " << real(GFock[1][0][i*NMO+j]) << " " << std::endl;
    }
  }
  //std::cout << "Fp: " << std::endl;
  std::fill_n(Mat.origin(), Mat.num_elements(), 0.0);
  ref.readEntry(Mat, "Fpa");
  for (int i = 0; i < NMO; i++)
  {
    for (int j = 0; j < NMO; j++)
    {
      //std::cout << Mat[i][j] << std::endl;
      //std::cout << Mat[i][j]-real(GFock[0][0][i*NMO+j]) << std::endl;
      if (std::abs(Mat[i][j] - real(ComplexType(GFock[0][0][i * NMO + j]))) > 1e-5)
      {
        std::cout << "DELTAB: " << i << " " << j << " " << Mat[i][j] << " " << real(ComplexType(GFock[0][0][i * NMO + j]))
                  << std::endl;
      }
      //if(std::abs(real(GFock[0][0][i*NMO+j]))>1e-6)
      //std::cout << i << " " << j << " " << real(GFock[0][0][i*NMO+j]) << " " << real(GFock[0][1][i*NMO+j]) << " " << real(GFock[0][2][i*NMO+j]) << std::endl;
    }
  }
*/
}

TEST_CASE("ham_ops_basic_serial", "[hamiltonian_operations]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  if(UTEST_HAMIL!="" and UTEST_WFN!="") {
    app_log(0,"HamiltonianOperations unit testing. Running user provided test:");
    app_log(0," Hamiltonian: {}", UTEST_HAMIL);
    app_log(0," Wavefunction: {}", UTEST_WFN);
    ham_ops_basic_serial<HOST_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);
#if defined(ENABLE_DEVICE)
    ham_ops_basic_serial<DEVICE_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);
#endif
  } else {
    app_log(0,"HamiltonianOperations unit testing. Running standard tests.");
    auto files = utils::get_unit_tests_files(true,true,true,true,true,true);
    for( auto f : files ) {
      try {
        ham_ops_basic_serial<HOST_MEMORY>(mpi,std::get<0>(f),std::get<1>(f));
      } catch (const sfqmc::AppAbortException& e) {
        FAIL_CHECK("APP_ABORT in ham_ops_basic_serial<HOST_MEMORY>(" << std::get<0>(f) << "): " << e.what());
      }
#if defined(ENABLE_DEVICE)
      try {
        ham_ops_basic_serial<DEVICE_MEMORY>(mpi,std::get<0>(f),std::get<1>(f));
      } catch (const sfqmc::AppAbortException& e) {
        FAIL_CHECK("APP_ABORT in ham_ops_basic_serial<DEVICE_MEMORY>(" << std::get<0>(f) << "): " << e.what());
      }
#endif
    }
  }
}


} // namespace sfqmc
