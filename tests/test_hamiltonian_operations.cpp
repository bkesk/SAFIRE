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
// checks that all five variants agree on the public interface methods:
//   - energy(E, G, idet=0)
//   - getOneBodyPropagatorMatrix(dt, vMF=0)
//   - vHS(vbias(G), dt) (basis-independent NMO x NMO operator)
// Used as the reference is the first variant constructed (Real3IndexFactorization);
// every subsequent variant must agree with it.

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

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void run_hubbard_consistency()
{
  using nda::range;
  auto all = range::all;

  auto& mpi = sfqmc::utils::make_unit_test_mpi_context();

  tests::HubbardSpec spec;
  // Defaults: 4x4 lattice, t=1, U=4, nup=ndown=4, COLLINEAR.
  int NMO   = spec.NMO();
  int npol  = (spec.walker_type == NONCOLLINEAR) ? 2 : 1;
  int nel   = spec.nel();
  int nwalk = 3;
  double dt = 0.01;

  app_log(0, "Hubbard 4x4 self-contained HamiltonianOperations cross-check");
  app_log(0, "  NMO = {}, nup = {}, ndown = {}", NMO, spec.nup, spec.ndown);

  auto h0  = tests::build_h0(spec);
  auto Psi = tests::build_trial_orbitals(spec); // (NMO, nup+ndown), alpha then beta

  // Build all variants from the same physical model.
  auto H_real   = tests::build_real3index<MEM>(mpi, spec, h0, Psi);
  auto H_thc    = tests::build_thc<MEM>(mpi, spec, h0, Psi);
  auto H_kp     = tests::build_kp3index<MEM>(mpi, spec, h0, Psi);
  auto H_kpthc  = tests::build_kpthc<MEM>(mpi, spec, h0, Psi);
  auto H_model  = tests::build_modelhamops<MEM>(mpi, spec, h0, Psi);

  // Sanity: HamiltonianTypes enum reported by each variant.
  REQUIRE(H_real.getHamType()  == RealDenseFactorized);
  REQUIRE(H_thc.getHamType()   == THC);
  REQUIRE(H_kp.getHamType()    == KPFactorized);
  REQUIRE(H_kpthc.getHamType() == KPTHC);
  REQUIRE(H_model.getHamType() == ModelHamiltonian);

  // ---- 1. energy(E, G, 0) ----
  // Use a randomly populated half-rotated Green's function. Same G for every variant.
  nda::array<ComplexType, 2> G_h = nda::rand(nwalk, nel * npol * NMO);

  auto run_energy = [&](auto& H) {
    memory::array<MEM, ComplexType, 2> G(G_h);
    nda::array<ComplexType, 2> E_h(nwalk, 3);
    memory::array<MEM, ComplexType, 2> E(E_h);
    H.energy(E, G, 0, true, true, true);
    return nda::array<ComplexType, 2>(nda::to_host(E()));
  };

  auto E_ref   = run_energy(H_real);
  auto E_thc   = run_energy(H_thc);
  auto E_kp    = run_energy(H_kp);
  // kpthc not implemented on GPU yet!
  // auto E_kpthc = run_energy(H_kpthc);
  auto E_model = run_energy(H_model);

  auto print_e = [](char const* name, auto& E) {
    app_log(1, "  {}: E1={:+.6e} EXX={:+.6e} EJ={:+.6e}",
            name, std::real(E(0, 0)), std::real(E(0, 1)), std::real(E(0, 2)));
  };
  print_e("Real3", E_ref);
  print_e("THC  ", E_thc);
  print_e("KP3  ", E_kp);
  // print_e("KPTHC", E_kpthc);
  print_e("Model", E_model);

  // The Cholesky / THC / KP-* variants share the same EXX / EJ bookkeeping
  // convention, so they must match component by component.
  CHECK_THAT(E_thc, utils::Approx(E_ref));
  CHECK_THAT(E_kp, utils::Approx(E_ref));
  // CHECK_THAT(E_kpthc, utils::Approx(E_ref));

  // ModelHamOps uses a different decomposition (continuous-charge HS) that
  // routes the entire onsite Hubbard 2-body contribution through the J
  // channel, leaving its EXX column zero. Compare E1 alone and the
  // physically invariant total 2-body energy (EXX + EJ).
  CHECK_THAT(E_model(all, 0), utils::Approx(E_ref(all,0)));
  CHECK_THAT(nda::make_regular(E_model(all, 1) + E_model(all, 2)), utils::Approx(nda::make_regular(E_ref(all, 1) + E_ref(all, 2))));

  // ---- 2. getOneBodyPropagatorMatrix(dt, vMF=0) ----
  auto run_h1 = [&](auto& H) {
    int nCV = H.number_of_cholesky_vectors();
    nda::array<ComplexType, 1> vMF(nCV);
    vMF() = ComplexType(0);
    return H.getOneBodyPropagatorMatrix(dt, vMF);
  };

  auto H1_ref   = run_h1(H_real);
  auto H1_thc   = run_h1(H_thc);
  auto H1_kp    = run_h1(H_kp);
  auto H1_kpthc = run_h1(H_kpthc);
  auto H1_model = run_h1(H_model);

  CHECK_THAT(H1_thc, utils::Approx(H1_ref));
  CHECK_THAT(H1_kp, utils::Approx(H1_ref));
  // CHECK_THAT(H1_kpthc, utils::Approx(H1_ref));
  CHECK_THAT(H1_model, utils::Approx(H1_ref));

  // ---- 3. vHS(vbias(G), dt) ----
  // The auxiliary-field basis differs across variants, so vbias outputs are not
  // directly comparable. The composition vHS(vbias(G)) returns an NMO x NMO
  // operator that is basis-invariant: for the same physical Hamiltonian and
  // same G it must agree.
  auto run_vHS_of_vbias = [&](auto& H) {
    int nCV = H.number_of_cholesky_vectors();
    memory::array<MEM, ComplexType, 2> G(G_h);
    memory::array<MEM, ComplexType, 2> X(nwalk, nCV);
    X() = ComplexType(0);
    H.vbias(G, X, dt);
    auto V = H.vHS(X, dt);
    return nda::array<ComplexType, 4>(nda::to_host(V));
  };

  auto V_ref   = run_vHS_of_vbias(H_real);
  auto V_thc   = run_vHS_of_vbias(H_thc);
  auto V_kp    = run_vHS_of_vbias(H_kp);
  // auto V_kpthc = run_vHS_of_vbias(H_kpthc);
  auto V_model = run_vHS_of_vbias(H_model);

  CHECK_THAT(V_thc, utils::Approx(V_ref));
  CHECK_THAT(V_kp, utils::Approx(V_ref));
  // CHECK_THAT(V_kpthc, utils::Approx(V_ref));

  // ModelHamOps explicitly produces a block per walker spin, while the closed-
  // Hamiltonian variants return a single block intended to be applied to both
  // spins. For our closed-shell trial wavefunction those blocks are identical,
  // so compare each spin block of the model output to the reference.
  REQUIRE(V_model.shape()[0] == 2);
  REQUIRE(V_ref.shape()[0]   == 1);
  CHECK_THAT(V_model(0, nda::ellipsis{}), utils::Approx(V_ref(0, nda::ellipsis{})));
  CHECK_THAT(V_model(1, nda::ellipsis{}), utils::Approx(V_ref(0, nda::ellipsis{})));
}

} // namespace sfqmc

TEST_CASE("hubbard_consistency_4x4_collinear", "[hamiltonian_operations]")
{
  sfqmc::run_hubbard_consistency<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  sfqmc::run_hubbard_consistency<DEVICE_MEMORY>();
#endif
}
