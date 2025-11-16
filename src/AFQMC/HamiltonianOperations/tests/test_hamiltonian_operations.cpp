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

#include "catch2/catch.hpp"

#include "config.h"
#include "AFQMC/config.h"

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/test_common.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"

#include "numerics/sparse/sparse.hpp"

//#include "AFQMC/SlaterDeterminantOperations/

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

template<bool MP, MEMORY_SPACE MEM>
void ham_ops_basic_serial(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                 std::string hamil_file, std::string wfn_file)
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using sfqmc::utils::ARRAY_EQUAL;
  using nda::range;
  using matrix_t = memory::array<MEM,ComplexType,2>;
  auto all = range::all;
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
  auto file_data       = read_test_results_from_hdf<ComplexType>(hamil_file, test_wfn);

  int NMO              = file_data.NMO;
  int nup              = file_data.NAEA;
  int ndown            = file_data.NAEB;

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  WALKER_TYPES wtype;
  int ndets_to_read = 1;
  nda::array<ComplexType,1> ci;
  h5::file file(wfn_file,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  // do we want PHMSD wfns here?
  h5::group ngrp = wgrp.open_group("NOMSD");
  getCommonInput(ngrp, NMO, nup, ndown, ndets_to_read, ci, wtype);
  int nel  = (wtype == COLLINEAR) ? (nup+ndown) : nup;
  int npol = (wtype == NONCOLLINEAR) ? 2 : 1;
  int nspin = (wtype == COLLINEAR) ? 2 : 1;
  int ndet = 1;
  int nwalk = 5;

  //[ndet][nspin](nel,npol*NMO)
  auto psi = read_nomsd_wavefunction<MEM>(ngrp,ndet,wtype,NMO,nup,ndown);
  utils::check(psi.shape() == std::array<long,2>{ndet,nspin}, "Shape mismatch.");

  memory::array<MEM,ComplexType,3> OrbMat(nwalk,npol*NMO,nel);
  {
    nda::array<ComplexType,2> T(npol*NMO,nup);
    nda::h5_read(ngrp,"Psi0_alpha",T);
    for(int i=0; i<nwalk; ++i)
      OrbMat(i,all,range(nup)) = T();
    if (wtype == COLLINEAR) {
      T() = ComplexType(0.0);
      auto Odn = T(all,range(ndown));
      nda::h5_read(ngrp,"Psi0_beta",Odn);
      for(int i=0; i<nwalk; ++i)
        OrbMat(i,all,range(nup,nel)) = T();
    }
  }
  auto HOps=ham.getHamiltonianOperations<MEM,MP>(wtype, mpi, psi);

  memory::array<MEM,ComplexType,3> G(nwalk, nel, npol * NMO);
  memory::array<MEM,ComplexType,1> ovlp(nwalk,ComplexType(0.0)); 
  
  // Overlap/GreenFunction
  det_ops::MixedDensityMatrix(psi(0,0),OrbMat(all,all,range(nup)),G(all,range(nup),all),ovlp);
  if (wtype == COLLINEAR)
    det_ops::MixedDensityMatrix(psi(0,1),OrbMat(all,all,range(nup,nel)),G(all,range(nup,nel),all),ovlp);
  ARRAY_EQUAL(ovlp,nda::array<ComplexType,1>(nwalk,ComplexType(0.0)));
  // 2d views and transposed copies just in case
  auto G2d = nda::reshape(G,std::array<long,2>{nwalk,nel*npol * NMO});
  memory::array<MEM,ComplexType,2> Gt2d(nel*npol * NMO, nwalk);
  if constexpr (MEM==HOST_MEMORY)
    Gt2d = nda::transpose(G2d);
  else
    nda::tensor::elementwise(G2d,"ij",Gt2d,"ji");

  // Energy
  memory::array<MEM,ComplexType,2> Eloc(nwalk, 3);
  if(HOps.transposed_G_for_E())
  {
    HOps.energy(Eloc, G2d, 0);
  } else {
    HOps.energy(Eloc, Gt2d, 0);
  }
  auto eloc_h = nda::to_host(Eloc);
  if (std::abs(file_data.E0 + file_data.E1) > 1e-8) {
    ARRAY_EQUAL(eloc_h(all,0), nda::array<ComplexType,1>(nwalk,file_data.E0 + file_data.E1)); 
  } else
    app_log(1," E1: {} ", eloc_h(0,0));
  if (std::abs(file_data.E2) > 1e-8)
  {
    ARRAY_EQUAL(eloc_h(all,1), nda::array<ComplexType,1>(nwalk,file_data.E2)); 
  }
  else
  {
    app_log(1," EJ: {}", eloc_h(0,2));
    app_log(1," EXX: {}", eloc_h(0,1)); 
    app_log(1," ETotal: {}", eloc_h(0,0)+eloc_h(0,1)+eloc_h(0,2)); 
  }

  double dt = 0.01;
  auto nCV  = HOps.number_of_cholesky_vectors();

  memory::array<MEM,SPComplexType,2> X(nCV, nwalk);
  X() = SPComplexType(0.0);
  if (HOps.transposed_G_for_vbias())
  {
    HOps.vbias(G2d, X, dt);
  } else {
    HOps.vbias(Gt2d, X, dt);
  }
  ComplexType Xsum = 0, Xsum2 = 0;
  auto X_h = nda::to_host(X);
  for (int i = 0; i < nCV; i++)
  {
    Xsum += X_h(i,0);
    Xsum2 += ComplexType(0.5) * X_h(i,0) * X_h(i,0);
  }
  if (std::abs(file_data.Xsum) > 1e-8)
  {
    REQUIRE(real(Xsum) == Approx(real(file_data.Xsum)));
    REQUIRE(imag(Xsum) == Approx(imag(file_data.Xsum)));
  }
  else
  {
    app_log(1," Xsum: {}", Xsum);
    app_log(1," Xsum2 (EJ): {}", Xsum2 / dt);
  }

  auto vHS_dims = HOps.vHS_dims();
  auto vHS = HOps.vHS(X,dt);
  REQUIRE( vHS.shape() == std::array<long,4>{vHS_dims[0],nwalk,vHS_dims[1]*NMO,NMO} );
  auto vHS_h = nda::to_host(vHS);
  ComplexType Vsum = 0;
  for (int i = 0; i < vHS.extent(2); i++)
    for (int j = 0; j < vHS.extent(3); j++)
      Vsum += vHS_h(0,0,i,j);
  if (std::abs(file_data.Vsum) > 1e-8)
  {
    REQUIRE(real(Vsum) == Approx(real(file_data.Vsum)));
    REQUIRE(imag(Vsum) == Approx(imag(file_data.Vsum)));
  }
  else
  {
    app_log(1," Vsum: {}", Vsum);
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

  ham_ops_basic_serial<false,HOST_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);
  ham_ops_basic_serial<true,HOST_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);
#if defined(ENABLE_DEVICE)
  ham_ops_basic_serial<false,DEVICE_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);
  ham_ops_basic_serial<true,DEVICE_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);
#endif
}

} // namespace sfqmc
