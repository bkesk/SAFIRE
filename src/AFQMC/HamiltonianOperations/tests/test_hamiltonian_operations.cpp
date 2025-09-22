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

#include "Utilities/app_loggers.h"
#include "io/ptree/ptree_utilities.hpp"
#include "hdf/hdf_archive.h"
#include "hdf/hdf_multi.h"

/*
#undef APP_ABORT
#define APP_ABORT(x)             \
  {                              \
    std::cout << x << std::endl; \
    throw;                       \
  }
*/

#include <string>
#include <vector>
#include <complex>
#include <iomanip>

#include "AFQMC/config.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "SparseMatrix/csr_hdf5_readers.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "Memory/buffer_managers.h"

#include "SparseMatrix/csr_matrix_construct.hpp"
#include "Numerics/ma_blas.hpp"

using std::cerr;
using std::complex;
using std::cout;
using std::endl;
using std::ifstream;
using std::setprecision;
using std::string;
using ma::real;
using ma::imag;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<bool MP, class Alloc>
void ham_ops_basic_serial(boost::mpi3::communicator& world)
{
  using pointer = device_ptr<ComplexType>;

  if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
  {
    APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
  }
  else
  {
    // Global Task Group
    afqmc::GlobalTaskGroup gTG(world);

    // Determine wavefunction type for test results from wavefunction file name which is
    // has the naming convention wfn_(wfn_type).dat.
    // First strip path of filename.
    std::string base_name = UTEST_WFN.substr(UTEST_WFN.find_last_of("\\/") + 1);
    // Remove file extension.
    std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
    auto file_data       = read_test_results_from_hdf<ComplexType>(UTEST_HAMIL, test_wfn);
    int NMO              = file_data.NMO;
    int NAEA             = file_data.NAEA;
    int NAEB             = file_data.NAEB;

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));

    ptree ham_pt;
    ham_pt.put("name","ham0");
    ham_pt.put("system","info0");
    ham_pt.put("filename",UTEST_HAMIL);
 
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(gTG, "ham0");

    using CMatrix = ComplexMatrix<Alloc>;
    hdf_archive dump;
    if (!dump.open(UTEST_WFN, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening HDF wavefunction file.");
    dump.push("Wavefunction", false);
    dump.push("NOMSD", false);
    std::vector<int> dims(5);
    if (!dump.readEntry(dims, "dims"))
      APP_ABORT(" Error in getCommonInput(): Problems reading dims. ");
    WALKER_TYPES WTYPE(initWALKER_TYPES(dims[3]));
    //    int walker_type = dims[3];
    int NEL  = (WTYPE == CLOSED) ? NAEA : (NAEA + NAEB);
    int NPOL = (WTYPE == NONCOLLINEAR) ? 2 : 1;
    //    WALKER_TYPES WTYPE = CLOSED;
    //    if(walker_type==1) WTYPE = COLLINEAR;
    //    if(walker_type==2) WTYPE = NONCOLLINEAR;

    auto TG = TaskGroup_(gTG, std::string("DummyTG"), 1, gTG.getTotalCores());
    Alloc alloc_(make_localTG_allocator<ComplexType>(TG));
    std::vector<PsiT_Matrix> PsiT;
    PsiT.reserve(2);
    dump.push(std::string("PsiT_0"));
    PsiT.emplace_back(csr_hdf5::HDF2CSR<PsiT_Matrix, shared_allocator<ComplexType>>(dump, gTG.Node()));
    if (WTYPE == COLLINEAR)
    {
      dump.pop();
      dump.push(std::string("PsiT_1"));
      PsiT.emplace_back(csr_hdf5::HDF2CSR<PsiT_Matrix, shared_allocator<ComplexType>>(dump, gTG.Node()));
    }

    dump.pop();
    boost::multi::array<ComplexType, 3> OrbMat({2, NPOL * NMO, NAEA});
    {
      boost::multi::array<ComplexType, 2> Psi0({NPOL * NMO, NAEA});
      dump.readEntry(Psi0, "Psi0_alpha");
      OrbMat[0] = Psi0;
      if (WTYPE == COLLINEAR)
      {
        dump.readEntry(Psi0, "Psi0_beta");
        OrbMat[1] = Psi0;
      }
    }
    dump.close();
    hdf_archive dummy;
    auto HOps=ham.getHamiltonianOperations<MP>(WTYPE, PsiT, TG, TG, dummy);

    // Calculates Overlap, G
// NOTE: Make small factory routine!
#if defined(ENABLE_DEVICE)
    SlaterDetOperations SDet(SlaterDetOperations_serial<ComplexType, DeviceBufferManager>{NPOL * NMO, NAEA, DeviceBufferManager{}});
#else
    SlaterDetOperations SDet(SlaterDetOperations_shared<ComplexType>(NPOL * NMO, NAEA));
#endif

    boost::multi::array<ComplexType, 3, Alloc> devOrbMat(OrbMat, alloc_);
    std::vector<local_csr_Matrix<ComplexType>> devPsiT(move_vector<local_csr_Matrix<ComplexType>>(std::move(PsiT)));

    CMatrix G({NEL, NPOL * NMO}, alloc_);
    ComplexType Ovlp = SDet.MixedDensityMatrix(devPsiT[0], devOrbMat[0], G.sliced(0, NAEA), 0.0, true);
    if (WTYPE == COLLINEAR)
    {
      Ovlp *= SDet.MixedDensityMatrix(devPsiT[1], devOrbMat[1](devOrbMat.extension(1), {0, NAEB}),
                                      G.sliced(NAEA, NAEA + NAEB), 0.0, true);
    }
    REQUIRE(real(Ovlp) == Approx(1.0));
    REQUIRE(imag(Ovlp) == Approx(0.0));

    boost::multi::array<ComplexType, 2, Alloc> Eloc({1, 3}, alloc_);
    {
      int nc = 1, nr = NEL * NPOL * NMO;
      if (HOps.transposed_G_for_E())
      {
        nr = 1;
        nc = NEL * NPOL * NMO;
      }
      boost::multi::array_ref<ComplexType, 2, pointer> Gw(make_device_ptr(G.origin()), {nr, nc});
      HOps.energy(Eloc, Gw, 0, TG.getCoreID() == 0);
    }
    Eloc[0][0] = (TG.Node() += ComplexType(Eloc[0][0]));
    Eloc[0][1] = (TG.Node() += ComplexType(Eloc[0][1]));
    Eloc[0][2] = (TG.Node() += ComplexType(Eloc[0][2]));
    if (std::abs(file_data.E0 + file_data.E1) > 1e-8)
    {
      REQUIRE(real(ComplexType(Eloc[0][0])) == Approx(real(file_data.E0 + file_data.E1)));
      REQUIRE(imag(ComplexType(Eloc[0][0])) == Approx(imag(file_data.E0 + file_data.E1)));
    }
    else
      app_log(1," E1: {} ", ComplexType(Eloc[0][0]));
    if (std::abs(file_data.E2) > 1e-8)
    {
      REQUIRE(real(Eloc[0][1] + Eloc[0][2]) == Approx(real(file_data.E2)));
      REQUIRE(imag(Eloc[0][1] + Eloc[0][2]) == Approx(imag(file_data.E2)));
    }
    else
    {
      app_log(1," EJ: {}", ComplexType(Eloc[0][2]));
      app_log(1," EXX: {}", ComplexType(Eloc[0][1]));
      app_log(1," ETotal: {}", ComplexType(Eloc[0][0] + Eloc[0][1] + Eloc[0][2]));
    }

    double dt = 0.01;
    auto nCV      = HOps.local_number_of_cholesky_vectors();

    CMatrix X({nCV, 1}, ComplexType(0.0), alloc_);
    {
      int nc = 1, nr = NEL * NPOL * NMO;
      if (HOps.transposed_G_for_vbias())
      {
        nr = 1;
        nc = NEL * NPOL * NMO;
      }
      boost::multi::array_ref<ComplexType, 2, pointer> Gw(make_device_ptr(G.origin()), {nr, nc});
      HOps.vbias(Gw, X, dt);
    }
    TG.TG_local().barrier();
    ComplexType Xsum = 0, Xsum2 = 0;
    for (int i = 0; i < X.size(0); i++)
    {
      Xsum += X[i][0];
      Xsum2 += ComplexType(0.5) * X[i][0] * X[i][0];
    }
    if (std::abs(file_data.Xsum) > 1e-8)
    {
      REQUIRE(real(Xsum) == Approx(real(file_data.Xsum)));
      REQUIRE(imag(Xsum) == Approx(imag(file_data.Xsum)));
    }
    else
    {
      app_log(1," Xsum: {}", ComplexType(Xsum));
      app_log(1," Xsum2 (EJ): {}", ComplexType(Xsum2) / dt);
    }

    int vdim1 = (HOps.transposed_vHS() ? 1 : NMO * NMO);
    int vdim2 = (HOps.transposed_vHS() ? NMO * NMO : 1);
    CMatrix vHS({vdim1, vdim2}, alloc_);
    TG.TG_local().barrier();
    HOps.vHS(X, vHS, dt);
    TG.TG_local().barrier();
    ComplexType Vsum = 0;
    if (HOps.transposed_vHS())
    {
      for (int i = 0; i < vHS.size(1); i++)
        Vsum += vHS[0][i];
    }
    else
    {
      for (int i = 0; i < vHS.size(0); i++)
        Vsum += vHS[i][0];
    }
    if (std::abs(file_data.Vsum) > 1e-8)
    {
      REQUIRE(real(Vsum) == Approx(real(file_data.Vsum)));
      REQUIRE(imag(Vsum) == Approx(imag(file_data.Vsum)));
    }
    else
    {
      app_log(1," Vsum: {}", ComplexType(Vsum));
    }
 
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

    //if(WTYPE==COLLINEAR) {
    //dm_size = 2*NMO*NMO;
    //G2 = CMatrix({2*NMO,NMO});
    //} else if(WTYPE==CLOSED) {
    //dm_size = NMO*NMO;
    //G2 = CMatrix({NMO,NMO});
    //} else {
    //APP_ABORT("NON COLLINEAR Wavefunction not implemented.");
    //}
    //Ovlp = SDet.MixedDensityMatrix(devPsiT[0],devOrbMat[0], G2.sliced(0,NMO),0.0,false);
    //if(WTYPE==COLLINEAR) {
    //Ovlp *= SDet.MixedDensityMatrix(devPsiT[1],devOrbMat[1](devOrbMat.extension(1),{0,NAEB}), G.sliced(NMO,2*NMO),0.0,false);
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
  }
}

TEST_CASE("ham_ops_basic_serial", "[hamiltonian_operations]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)

  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);
  ham_ops_basic_serial<false,Alloc>(world);
  ham_ops_basic_serial<true,Alloc>(world);
  release_memory_managers();
}

} // namespace sfqmc
