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
    
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "utilities/test_common.hpp"
  
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"
#include "numerics/sparse/sparse.hpp"
    
#include <string>
#include <vector>
#include <complex> 
#include <iomanip>
#include <random>

#include "AFQMC/config.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Walkers/WalkerSetFactory.hpp"
#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
//#include "AFQMC/Estimators/BackPropagatedEstimator.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"

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

template<MEMORY_SPACE MEM>
void reduced_density_matrix(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file)
{
  using sfqmc::utils::ARRAY_EQUAL;
  using nda::range;
  auto all = range::all;
  utils::check(utils::file_exists(hamil_file),
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);
  utils::check(utils::file_exists(wfn_file),
               " Wavefunction file not found: {}. \n Run unit test with --wfn /path/to/wfn.h5 ", wfn_file);

  auto[NMO,nup, ndown] = read_info_from_wfn(UTEST_WFN, "any");
  utils::check(NMO == read_nmo_from_hdf(hamil_file), "NMO differ between hamil and wfn files.");

  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev = std::make_shared<utils::RandomGenerator_t<MEM>>(utils::make_rng<MEM>(777));

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  auto& ham = HamFac.getHamiltonian(mpi, "ham0");

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file);
  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("system","info0");
  if(type == CLOSED) wlk_pt.put("walker_type","closed");
  else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
  else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
  else if(type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);

  int nspin            = (type == COLLINEAR) ? 2 : 1;
  int npol             = (type == NONCOLLINEAR) ? 2 : 1;
  int nel              = (type == COLLINEAR) ? nup+ndown : nup;

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("dense_trial",true);

  int nwalk = 11; 
  WavefunctionFactory<MEM> WfnFac(InfoMap);
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);

  ptree prop_pt;
  prop_pt.put("name","prop0");
  prop_pt.put("system","info0");

  PropagatorFactory<MEM> PropgFac(InfoMap);
  PropgFac.push("prop0", prop_pt);
  auto& prop = PropgFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  auto initial_guess = WfnFac.getInitialGuess("wfn0");
  REQUIRE(initial_guess.shape() == std::array<long,3>{nspin,npol*NMO,nup});
  wset.resize(nwalk, initial_guess);

  using EstimPtr = std::shared_ptr<EstimatorBase>;
  std::vector<EstimPtr> estimators;
/*
    ptree est_pt;
    est_pt.put("name","back_propagation");
    est_pt.put("nsteps",1); // deprecated
    est_pt.put("block_size",1); // deprecated
    est_pt.put("measure_interval_multiplier", 2); // test with single value
    est_pt.put("path_restoration","no");
    est_pt.put("OneRDM.nskip_output",0);
    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt));
    bool impsamp = true;
    estimators.emplace_back(
        static_cast<EstimPtr>(std::make_shared<BackPropagatedEstimator>(TG, InfoMap["info0"], "none", est_pt,
                                                                        type, wset, wfn, prop, impsamp)));

    // generate P1 with dt=0
    prop.generateP1(0.0, wset.getWalkerType());

    std::string file = create_test_hdf(UTEST_WFN, UTEST_HAMIL);
    std::ofstream out;
    dump.create(file);
    dump.open(file);
    for (int iblock = 0; iblock < 10*afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL; iblock++)
    {
      wset.advanceBPPos();
      estimators[0]->accumulate_block(iblock*0.01,wset);
      estimators[0]->print(out, dump, wset);
    }
    dump.close();
    boost::multi::array<ComplexType, 1> read_data(boost::multi::iextensions<1u>{2 * NMO * NMO});

    ComplexType denom;
    hdf_archive reader;
    // Read from a particular block.
    if (!reader.open(file, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening estimates.h5. ");
    reader.read(read_data, "Observables/BackPropagated/FullOneRDM/Average_0/one_rdm_000000004");
    reader.read(denom, "Observables/BackPropagated/FullOneRDM/Average_0/denominator_000000004");
    // Test EstimatorHandler eventually.
    //int nup_READ, ndown_READ, NMO_READ, WALKER_TYPE_READ;
    //reader.read(nup_READ, "Metadata/nup");
    //REQUIRE(nup_READ==nup);
    //reader.read(ndown_READ, "Metadata/ndown");
    //REQUIRE(ndown_READ==ndown);
    //reader.read(NMO_READ, "Metadata/NMO");
    //REQUIRE(NMO_READ==NMO);
    //reader.read(WALKER_TYPE_READ, "Metadata/WALKER_TYPE");
    //REQUIRE(WALKER_TYPE_READ==type);
    reader.close();
    // Test the RDM. Since no back propagation has been performed the RDM should be
    // identical to the mixed estimate.
    if (type == CLOSED)
    {
      REQUIRE(read_data.num_elements() >= NMO * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {NMO, NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(nup));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == COLLINEAR)
    {
      REQUIRE(read_data.num_elements() >= 2 * NMO * NMO);
      boost::multi::array_ref<ComplexType, 3> BPRDM(read_data.origin(), {2, NMO, NMO});
      ma::scal(1.0 / denom, BPRDM[0]);
      ma::scal(1.0 / denom, BPRDM[1]);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[0][i][i] + BPRDM[1][i][i];
      REQUIRE(trace.real() == Approx(nup + ndown));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, 2 * NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 3, pointer> G(Gw.origin(), {2, NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == FULLYPOLARIZED) // Kyle: be careful here; I'm not sure that I have a reference!
    {
      REQUIRE(read_data.num_elements() >= NMO * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {NMO, NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(nup));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else
    {
      APP_ABORT(" NONCOLLINEAR Wavefunction found.");
    }

    ptree est_pt2;
    est_pt2.put("name","back_propagation");
    est_pt2.put("nsteps",1); // deprecated
    est_pt2.put("block_size",1); // deprecated
    
    std::vector<int> nback_prop_interval_multipliers = {1, 2, 3};
    ptree temp_tree;
    for (const auto& value : nback_prop_interval_multipliers) {
        ptree item;
        item.put("", value); // empty key for the value
        temp_tree.push_back(std::make_pair("", item));
    }
    est_pt2.add_child("measure_interval_multiplier", temp_tree); // test with a vector
    est_pt2.put("path_restoration","no");
    est_pt2.put("OneRDM.nskip_output",0);
    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt2));

    std::vector<EstimPtr> estimators2;
    estimators2.emplace_back(
        static_cast<EstimPtr>(std::make_shared<BackPropagatedEstimator>(TG, InfoMap["info0"], "none", est_pt2,
                                                                        type, wset, wfn, prop, impsamp)));

    dump.open(file);
    for (int iblock = 0; iblock < 6*afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL; iblock++)
    {
      wset.advanceBPPos();
      estimators2[0]->accumulate_block(iblock*0.01,wset);
      estimators2[0]->print(out, dump, wset);
    }
    dump.close();

    if (!reader.open(file, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening estimates.h5. ");
    else
      app_log(1," Successfully opened {}", file);
    reader.read(read_data, "Observables/BackPropagated/FullOneRDM/Average_1/one_rdm_000000002");
    reader.read(denom, "Observables/BackPropagated/FullOneRDM/Average_1/denominator_000000002");
    reader.close();

    if (type == CLOSED)
    {
      REQUIRE(read_data.num_elements() >= NMO * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {NMO, NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(nup));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == COLLINEAR)
    {
      REQUIRE(read_data.num_elements() >= 2 * NMO * NMO);
      boost::multi::array_ref<ComplexType, 3> BPRDM(read_data.origin(), {2, NMO, NMO});
      ma::scal(1.0 / denom, BPRDM[0]);
      ma::scal(1.0 / denom, BPRDM[1]);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[0][i][i] + BPRDM[1][i][i];
      REQUIRE(trace.real() == Approx(nup + ndown));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, 2 * NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 3, pointer> G(Gw.origin(), {2, NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == FULLYPOLARIZED) // Kyle: be careful here; I'm not sure that I have a reference!
    {
      REQUIRE(read_data.num_elements() >= NMO * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {NMO, NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(nup));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else
    {
      APP_ABORT(" NONCOLLINEAR Wavefunction found.");
    }

    if (!reader.open(file, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening estimates.h5. ");
    reader.read(read_data, "Observables/BackPropagated/FullOneRDM/Average_2/one_rdm_000000002");
    reader.read(denom, "Observables/BackPropagated/FullOneRDM/Average_2/denominator_000000002");
    reader.close();
  }
*/
}

TEST_CASE("reduced_density_matrix", "[estimators]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  reduced_density_matrix<HOST_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);

#if defined(ENABLE_DEVICE)
  reduced_density_matrix<DEVICE_MEMORY>(mpi,UTEST_HAMIL,UTEST_WFN);
#endif
}

} // namespace sfqmc
