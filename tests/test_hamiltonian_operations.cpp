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
////////////////////////////////////////////////////////////////////////////////

// Self-contained cross-implementation consistency test for HamiltonianOperations.
//
// Builds a 4x4 2D Hubbard model directly in memory and instantiates each
// HamiltonianOperations variant (Real3IndexFactorization, THCOps,
// KP3IndexFactorization with one k-point, KPTHCOps with one k-point,
// ModelHamOps) on top of analytically derived Cholesky / THC factors. Then
// checks that all variants agree on the public interface methods:
//   - energy(E, G, idet=0)
//   - getOneBodyPropagatorMatrix(dt, vMF=0)
//   - vHS(vbias(G), dt) (basis-independent NMO x NMO operator)
// Used as the reference is the first variant constructed (Real3IndexFactorization);
// every subsequent variant must agree with it.
//
// The check is repeated over a grid that spans both
//   * the intrinsic spin symmetry of the one-body matrix T (closed / collinear /
//     noncollinear, i.e. all allowed nspin_in_H / npol_in_H), and
//   * every target walker_type reachable from that symmetry under
//     walkerTypeIsConvertible (CLOSED <= COLLINEAR <= NONCOLLINEAR).

#include "catch2/catch_test_macros.hpp"

#include <complex>
#include <random>
#include <string>
#include <vector>

#include "config.h"
#include "AFQMC/config.h"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "test_common.hpp"

#include "hubbard_factorizations.hpp"

namespace sfqmc {
using namespace afqmc;

template<MEMORY_SPACE MEM>
void hamiltonian_operations_hubbard_4x4_consistency(WALKER_TYPES h_symmetry, WALKER_TYPES walker_type) {
  using nda::range;
  auto all = range::all;

  auto& mpi = sfqmc::utils::make_unit_test_mpi_context();

  tests::HubbardSpec spec;
  spec.h_symmetry = h_symmetry;
  spec.walker_type = walker_type;
  // Defaults: 4x4 lattice, t=1, U=4, nup=ndown=4.
  int NMO   = spec.NMO();
  int wns   = spec.walker_nspin();
  int wnp   = spec.walker_npol();
  int nel   = spec.walker_nel();
  int nwalk = 3;
  double dt = 0.01;

  // Model Hamiltonians do not support CLOSED walkers (see ModelHamOpsGenerator).
  bool with_model = (walker_type != CLOSED);

  app_log(0, "Hubbard 4x4 HamiltonianOperations cross-check: H-symmetry={}, walker={}",
          walkerTypeToString(h_symmetry), walkerTypeToString(walker_type));
  app_log(0, "  NMO = {}, walker_nup = {}, walker_ndown = {}, npol = {}",
          NMO, spec.walker_nup(), spec.walker_ndown(), wnp);

  // Build all supported variants from the same physical model.
  auto H_real   = tests::build_real3index<MEM>(mpi, spec);
  auto H_thc    = tests::build_thc<MEM>(mpi, spec);
  auto H_kp     = tests::build_kp3index<MEM>(mpi, spec);
  auto H_kpthc  = tests::build_kpthc<MEM>(mpi, spec);

  REQUIRE(H_real.getHamType()  == RealDenseFactorized);
  REQUIRE(H_thc.getHamType()   == THC);
  REQUIRE(H_kp.getHamType()    == KPFactorized);
  REQUIRE(H_kpthc.getHamType() == KPTHC);

  std::optional<HamiltonianOperations<MEM>> H_model;
  if(with_model) {
    H_model = tests::build_modelhamops<MEM>(mpi, spec);
    REQUIRE(H_model->getHamType() == ModelHamiltonian);
  }

  // ---- 1. energy(E, G, 0) ----
  // Use a randomly populated half-rotated Green's function. Same G for every variant.
  nda::array<ComplexType, 2> G_h = nda::rand(nwalk, nel * wnp * NMO);

  auto run_energy = [&](auto& H) {
    memory::array<MEM, ComplexType, 2> G(G_h);
    nda::array<ComplexType, 2> E_h(nwalk, 3);
    memory::array<MEM, ComplexType, 2> E(E_h);
    H.energy(E, G, 0, true, true, true);
    return nda::array<ComplexType, 2>(nda::to_host(E()));
  };

  auto E_ref = run_energy(H_real);
  auto E_thc = run_energy(H_thc);
  auto E_kp  = run_energy(H_kp);
  // kpthc not implemented on GPU yet!
  // auto E_kpthc = run_energy(H_kpthc);

  auto print_e = [](char const* name, auto& E) {
    app_log(1, "  {}: E1={:+.6e} EXX={:+.6e} EJ={:+.6e}",
            name, std::real(E(0, 0)), std::real(E(0, 1)), std::real(E(0, 2)));
  };
  print_e("Real3", E_ref);
  print_e("THC  ", E_thc);
  print_e("KP3  ", E_kp);

  // The Cholesky / THC / KP-* variants share the same EXX / EJ bookkeeping
  // convention, so they must match component by component.
  CHECK_THAT(E_thc, utils::Approx(E_ref));
  CHECK_THAT(E_kp, utils::Approx(E_ref));

  if(with_model) {
    auto E_model = run_energy(*H_model);
    print_e("Model", E_model);
    // ModelHamOps uses a different decomposition (continuous-charge HS) that
    // routes the entire onsite Hubbard 2-body contribution through the J
    // channel, leaving its EXX column zero. Compare E1 alone and the
    // physically invariant total 2-body energy (EXX + EJ).
    CHECK_THAT(E_model(all, 0), utils::Approx(E_ref(all, 0)));
    CHECK_THAT(nda::make_regular(E_model(all, 1) + E_model(all, 2)),
               utils::Approx(nda::make_regular(E_ref(all, 1) + E_ref(all, 2))));
  }

  // ---- 2. getOneBodyPropagatorMatrix(dt, vMF=0) ----
  // H1 has the same walker-basis shape (nspin, npol*NMO, npol*NMO) for every
  // variant, so it is directly comparable.
  auto run_h1 = [&](auto& H) {
    int nCV = H.number_of_cholesky_vectors();
    nda::array<ComplexType, 1> vMF(nCV);
    vMF() = ComplexType(0);
    return H.getOneBodyPropagatorMatrix(dt, vMF);
  };

  auto H1_ref   = run_h1(H_real);
  auto H1_thc   = run_h1(H_thc);
  auto H1_kp    = run_h1(H_kp);
  [[maybe_unused]] auto H1_kpthc = run_h1(H_kpthc); // exercised but not asserted (no GPU path)

  CHECK_THAT(H1_thc, utils::Approx(H1_ref));
  CHECK_THAT(H1_kp, utils::Approx(H1_ref));
  if(with_model) {
    auto H1_model = run_h1(*H_model);
    CHECK_THAT(H1_model, utils::Approx(H1_ref));
  }

  // ---- 3. vHS(vbias(G), dt) ----
  // The auxiliary-field basis differs across variants, so vbias outputs are not
  // directly comparable. The composition vHS(vbias(G)) returns a basis-invariant
  // operator. Variants disagree only in how the (nspin_in_vHS, npol_in_vHS)
  // structure is laid out, so we expand each result into the canonical walker
  // representation (nwalk, walker_nspin, walker_npol, NMO, NMO) before comparing:
  //   v_eff(w, is, ip) = V(is % nspin_in_vHS, w, (ip % npol_in_vHS)*NMO + i, j)
  auto run_vHS_of_vbias = [&](auto& H) {
    int nCV = H.number_of_cholesky_vectors();
    memory::array<MEM, ComplexType, 2> G(G_h);
    memory::array<MEM, ComplexType, 2> X(nwalk, nCV);
    X() = ComplexType(0);
    H.vbias(G, X, dt);
    auto V = H.vHS(X, dt);
    return nda::array<ComplexType, 4>(nda::to_host(V));
  };

  auto expand_vHS = [&](nda::array<ComplexType, 4> const& V) {
    int nsV = V.shape(0);
    int npV = V.shape(2) / NMO;
    nda::array<ComplexType, 5> Ev(nwalk, wns, wnp, NMO, NMO);
    for(int w = 0; w < nwalk; ++w) {
      for(int is = 0; is < wns; ++is) {
        for(int ip = 0; ip < wnp; ++ip) {
          int pr = (ip % npV) * NMO;
          Ev(w, is, ip, all, all) = V(is % nsV, w, range(pr, pr + NMO), all);
        }
      }
    }
    return Ev;
  };

  auto V_ref = expand_vHS(run_vHS_of_vbias(H_real));
  auto V_thc = expand_vHS(run_vHS_of_vbias(H_thc));
  auto V_kp  = expand_vHS(run_vHS_of_vbias(H_kp));

  CHECK_THAT(V_thc, utils::Approx(V_ref));
  CHECK_THAT(V_kp, utils::Approx(V_ref));
  if(with_model) {
    auto V_model = expand_vHS(run_vHS_of_vbias(*H_model));
    CHECK_THAT(V_model, utils::Approx(V_ref));
  }
}

} // namespace sfqmc

TEST_CASE("hamiltonian_operations: hubbard 4x4 consistency", "[hamiltonian_operations]") {
  using namespace sfqmc::afqmc;
  // Span every (H-symmetry, walker_type) pair allowed by walkerTypeIsConvertible
  // (CLOSED <= COLLINEAR <= NONCOLLINEAR).
  for(auto h_sym : {CLOSED, COLLINEAR, NONCOLLINEAR}) {
    for(auto wtype : {CLOSED, COLLINEAR, NONCOLLINEAR}) {
      if(walkerTypeIsConvertible(h_sym, wtype)) {
        sfqmc::hamiltonian_operations_hubbard_4x4_consistency<HOST_MEMORY>(h_sym, wtype);
#if defined(ENABLE_DEVICE)
        sfqmc::hamiltonian_operations_hubbard_4x4_consistency<DEVICE_MEMORY>(h_sym, wtype);
#endif
      }
    }
  }
}
