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

// Helpers for the self-contained HamiltonianOperations cross-implementation test.
//
// Builds a 2D Hubbard model directly in memory and produces every
// HamiltonianOperations variant of the same physical Hamiltonian via its
// lowest-level constructor. The Cholesky / THC tensors are hand-derived from
// the diagonal structure of the onsite Hubbard interaction.

#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AFQMC/config.h"
#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"
#include "AFQMC/HamiltonianOperations/KP3IndexFactorization.hpp"
#include "AFQMC/HamiltonianOperations/KPTHCOps.hpp"
#include "AFQMC/HamiltonianOperations/ModelComponents/ModelComponent.hpp"
#include "AFQMC/HamiltonianOperations/ModelComponents/SparseEnergy.hpp"
#include "AFQMC/HamiltonianOperations/ModelHamOps.hpp"
#include "AFQMC/HamiltonianOperations/Real3IndexFactorization.hpp"
#include "AFQMC/HamiltonianOperations/THCOps.hpp"
#include "AFQMC/Hamiltonians/ModelHamOpsGenerator.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "nda/nda.hpp"
#include "numerics/shared_array/shared_array.hpp"
#include "numerics/sparse/sparse.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"

namespace sfqmc::afqmc::tests
{

struct HubbardSpec
{
  int Lx{4};
  int Ly{4};
  RealType t{1.0};
  RealType U{4.0};
  int nup{4};
  int ndown{4};
  WALKER_TYPES walker_type{COLLINEAR};

  int NMO() const { return Lx * Ly; }
  int nel() const { return nup + ndown; }
  int nel_for_spin(int s) const { return s == 0 ? nup : ndown; }
  int spin_offset(int s) const { return s == 0 ? 0 : nup; }
};

/// Build the kinetic 1-body matrix for a 2D Hubbard model with PBC.
inline nda::array<RealType, 2> build_h0(HubbardSpec const& s)
{
  int N = s.NMO();
  nda::array<RealType, 2> h(N, N);
  h() = RealType(0);
  for (int rx = 0; rx < s.Lx; ++rx)
  {
    for (int ry = 0; ry < s.Ly; ++ry)
    {
      int i  = rx * s.Ly + ry;
      int jx = ((rx + 1) % s.Lx) * s.Ly + ry;
      int jy = rx * s.Ly + ((ry + 1) % s.Ly);
      h(i, jx) -= s.t;
      h(jx, i) -= s.t;
      h(i, jy) -= s.t;
      h(jy, i) -= s.t;
    }
  }
  return h;
}

/// Build a deterministic concatenated trial Slater matrix of shape
/// (NMO, nup+ndown). The orthonormal NMO x NMO basis is the QR factor of a
/// Hilbert matrix; alpha gets the first nup columns, beta the first ndown.
/// For closed-shell (nup == ndown) the two spin blocks are identical.
inline nda::array<RealType, 2> build_trial_orbitals(HubbardSpec const& s)
{
  int NMO = s.NMO();
  nda::array<RealType, 2> M(NMO, NMO);
  for (int i = 0; i < NMO; ++i)
    for (int j = 0; j < NMO; ++j)
      M(i, j) = RealType(1) / RealType(i + j + 1);
  nda::array<RealType, 1> tau(NMO);
  // Following test_phmsd.cpp: act on transpose so M ends up with orthonormal columns.
  nda::lapack::geqrf(nda::transpose(M), tau);
  nda::lapack::gqr(nda::transpose(M), tau);

  nda::array<RealType, 2> Psi(NMO, s.nel());
  for (int spin = 0; spin < 2; ++spin)
  {
    int offset = s.spin_offset(spin);
    int nel_s  = s.nel_for_spin(spin);
    Psi(nda::range::all, nda::range(offset, offset + nel_s)) =
        M(nda::range::all, nda::range(nel_s));
  }
  return Psi;
}

/// View of the spin-`spin` block of a concatenated Psi.
inline auto psi_spin(HubbardSpec const& s, nda::array<RealType, 2> const& Psi, int spin)
{
  return Psi(nda::range::all, nda::range(s.spin_offset(spin), s.spin_offset(spin) + s.nel_for_spin(spin)));
}

namespace detail
{

/// Allocate a shared_array matching the host tensor's shape and copy into
/// it. Done on a single rank, then synced through the comm barrier.
template<MEMORY_SPACE MEM, typename T, int N>
auto to_shared(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
               nda::array<T, N> const& src)
{
  auto sa = memory::make_shared_array<MEM, T, N>(mpi, src.shape());
  if (mpi->node_comm.root()) sa() = src();
  mpi->comm.barrier();
  return sa;
}

/// Half-rotated 1-body matrix haj of shape (1, nup+ndown, NMO):
///   haj[0, a, k] = sum_i Psi(i, a) * h0(i, k).
/// Computed as Psi^T * h0 in real arithmetic and then promoted to complex.
inline nda::array<ComplexType, 3> host_haj(HubbardSpec const& s,
                                           nda::array<RealType, 2> const& h0,
                                           nda::array<RealType, 2> const& Psi)
{
  long NMO = s.NMO();
  long nel = s.nel();
  nda::array<RealType, 2> haj_2d(nel, NMO);
  nda::blas::gemm(nda::transpose(Psi), h0, haj_2d);
  nda::array<ComplexType, 3> haj(1, nel, NMO);
  haj() = ComplexType(0);
  haj(0, nda::range::all, nda::range::all) = haj_2d;
  return haj;
}

/// Diagonal exchange counterterm v0[..., i, i] = -0.5 U for the closed
/// Hubbard onsite ERI. `Rank` is 3 for shape (1, NMO, NMO) or 4 for
/// (1, 1, NMO, NMO).
template<typename T, int Rank>
nda::array<T, Rank> host_vexx_diagonal(long NMO, RealType U)
{
  static_assert(Rank == 3 || Rank == 4, "vexx is either (1,NMO,NMO) or (1,1,NMO,NMO)");
  std::array<long, Rank> shape{};
  shape.fill(1);
  shape[Rank - 2] = NMO;
  shape[Rank - 1] = NMO;
  nda::array<T, Rank> v(shape);
  v() = T(0);
  nda::diagonal(nda::reshape(v, std::array<long, 2>{NMO, NMO})) = T(-0.5) * T(U);
  return v;
}

/// Trivial Brillouin-zone tables for the single-kpoint lift (nkpts=1).
struct SingleQTables
{
  nda::array<int, 2> nocc;     // (nspin, nkpts) — occupations per spin per k
  nda::array<int, 1> minusq;   // q -> index of -q
  nda::array<int, 2> qk_to_k2; // (q, k) -> k+q index
  nda::array<int, 1> qmap;     // q -> index in symmetric-q list (KP3IndexFactorization)
};

inline SingleQTables make_single_q_tables(HubbardSpec const& s)
{
  return SingleQTables{
      .nocc     = nda::array<int, 2>{{s.nup}, {s.ndown}},
      .minusq   = nda::array<int, 1>{0},
      .qk_to_k2 = nda::array<int, 2>{{0}},
      .qmap     = nda::array<int, 1>{0},
  };
}

} // namespace detail

/// Real3IndexFactorization for 2D Hubbard, closed Hamiltonian, COLLINEAR walker.
/// Cholesky vectors are diagonal: L_n[i,j] = sqrt(U) δ_ni δ_nj, n in [0, NMO).
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_real3index(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s,
    nda::array<RealType, 2> const& h0,
    nda::array<RealType, 2> const& Psi)
{
  long NMO     = s.NMO();
  long nCV     = NMO;
  RealType sqU = std::sqrt(s.U);

  nda::array<ComplexType, 3> hij_h{nda::reshape(h0, std::array<long, 3>{1, NMO, NMO})};
  auto haj_h  = detail::host_haj(s, h0, Psi);
  auto vexx_h = detail::host_vexx_diagonal<RealType, 3>(NMO, s.U);

  // Likn(1, NMO, NMO, nCV) with L_n[i,j] = sqU δ_ni δ_nj: only (n, n, n) is nonzero.
  nda::array<RealType, 4> Likn_h(1, NMO, NMO, nCV);
  Likn_h() = RealType(0);
  for (long n = 0; n < nCV; ++n)
    Likn_h(0, n, n, n) = sqU;

  // Lnak[s] = sqrt(U) * Psi_s(n,a) * delta_{n,k}, shape (1, 1, nCV, nel_s, NMO).
  // For each (spin, a), fill the (n, k) diagonal of the (nCV, NMO) slice with sqU * Psi_s(:, a).
  nda::array<memory::shared_array<MEM, ComplexType, 5>, 1> Lnak(2);
  for (int spin = 0; spin < 2; ++spin)
  {
    long nel_s = s.nel_for_spin(spin);
    auto Psi_s = psi_spin(s, Psi, spin);
    nda::array<ComplexType, 5> Lnak_h(1, 1, nCV, nel_s, NMO);
    Lnak_h() = ComplexType(0);
    for (long a = 0; a < nel_s; ++a)
      nda::diagonal(Lnak_h(0, 0, nda::range::all, a, nda::range::all)) =
          sqU * Psi_s(nda::range::all, a);
    Lnak(spin) = detail::to_shared<MEM>(mpi, Lnak_h);
  }

  return HamiltonianOperations<MEM>(Real3IndexFactorization<MEM>(
      mpi, s.walker_type, NMO, s.nup, s.ndown,
      detail::to_shared<HOST_MEMORY>(mpi, hij_h),
      detail::to_shared<MEM>(mpi, haj_h),
      detail::to_shared<MEM>(mpi, Likn_h),
      std::move(Lnak),
      detail::to_shared<HOST_MEMORY>(mpi, vexx_h),
      ComplexType(0)));
}

/// THCOps for 2D Hubbard with REAL=true, closed Hamiltonian, COLLINEAR walker.
/// X[s,i,u] = δ_{iu}, L_uv = sqrt(U) δ_uv (factorized), Z_uv = U δ_uv (Coulomb).
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_thc(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s,
    nda::array<RealType, 2> const& h0,
    nda::array<RealType, 2> const& Psi)
{
  long NMO     = s.NMO();
  long nu      = NMO;
  RealType sqU = std::sqrt(s.U);

  nda::array<ComplexType, 3> hij_h{nda::reshape(h0, std::array<long, 3>{1, NMO, NMO})};
  auto haj_h  = detail::host_haj(s, h0, Psi);
  auto vexx_h = detail::host_vexx_diagonal<ComplexType, 3>(NMO, s.U);

  // X(1, NMO, nu) = identity on the orbital block.
  nda::array<RealType, 3> X_h(1, NMO, nu);
  X_h() = RealType(0);
  nda::diagonal(X_h(0, nda::range::all, nda::range::all)) = RealType(1);

  // Y(1, 2, 1, nup, nu): Y[0, spin, 0, a, u] = Psi_s(u, a). Last-but-one axis
  // is nup; the beta block uses ndown columns and any trailing entries
  // (when ndown < nup) stay zero.
  nda::array<ComplexType, 5> Y_h(1, 2, 1, s.nup, nu);
  Y_h() = ComplexType(0);
  for (int spin = 0; spin < 2; ++spin)
  {
    auto Psi_s = psi_spin(s, Psi, spin);
    Y_h(0, spin, 0, nda::range(Psi_s.extent(1)), nda::range::all) = nda::transpose(Psi_s);
  }

  // Factorized Coulomb L = sqrt(U) I (vHS uses this).
  nda::array<RealType, 2> L_h(nu, nu);
  L_h() = RealType(0);
  nda::diagonal(L_h) = sqU;

  // Coulomb matrix Z = L L^T = U I (energy uses this).
  nda::array<RealType, 2> Z_h(nu, nu);
  Z_h() = RealType(0);
  nda::diagonal(Z_h) = s.U;

  return HamiltonianOperations<MEM>(THCOps<MEM, /*REAL=*/true>(
      mpi, s.walker_type, NMO, s.nup, s.ndown,
      detail::to_shared<HOST_MEMORY>(mpi, hij_h),
      detail::to_shared<MEM>(mpi, haj_h),
      detail::to_shared<MEM>(mpi, X_h),
      detail::to_shared<MEM>(mpi, Y_h),
      detail::to_shared<MEM>(mpi, L_h),
      std::optional{detail::to_shared<MEM>(mpi, Z_h)},
      /*X_rot=*/std::nullopt,
      /*Y_rot=*/std::nullopt,
      /*Z_rot=*/std::nullopt,
      detail::to_shared<HOST_MEMORY>(mpi, vexx_h),
      ComplexType(0)));
}

/// KP3IndexFactorization for 2D Hubbard, single-kpoint lift (nkpts=1, nbnd=NMO).
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_kp3index(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s,
    nda::array<RealType, 2> const& h0,
    nda::array<RealType, 2> const& Psi)
{
  long NMO      = s.NMO();
  long nbnd     = NMO;
  long nkpts    = 1;
  long nCV      = NMO;
  long q0       = 0;
  RealType sqU  = std::sqrt(s.U);
  long nocc_max = std::max(s.nup, s.ndown);

  nda::array<ComplexType, 4> hij_h{nda::reshape(h0, std::array<long, 4>{1, 1, NMO, NMO})};
  auto haj_h  = detail::host_haj(s, h0, Psi);
  auto vexx_h = detail::host_vexx_diagonal<ComplexType, 4>(NMO, s.U);
  auto bz     = detail::make_single_q_tables(s);

  // LQ[0][0, 0, 0, i, j, n] = sqU δ_{i,n} δ_{j,n}: only (n, n, n) is nonzero.
  nda::array<ComplexType, 6> LQ_h(1, 1, nkpts, nbnd, nbnd, nCV);
  LQ_h() = ComplexType(0);
  for (long n = 0; n < nCV; ++n)
    LQ_h(0, 0, 0, n, n, n) = ComplexType(sqU);

  // Lank / Lbnk: half-rotated Cholesky in K-point form. For real diagonal L
  // and a single q, both have the same per-element value, just different
  // index-name conventions; here both reduce to setting the (n, k) diagonal
  // of the (nCV, NMO) slice for each (spin, a) to sqU * Psi_s(:, a).
  nda::array<ComplexType, 6> Lank_h(1, 2, nkpts, nocc_max, nCV, NMO);
  nda::array<ComplexType, 6> Lbnk_h(1, 2, nkpts, nocc_max, nCV, NMO);
  Lank_h() = ComplexType(0);
  Lbnk_h() = ComplexType(0);
  for (int spin = 0; spin < 2; ++spin)
  {
    auto Psi_s = psi_spin(s, Psi, spin);
    for (long a = 0; a < Psi_s.extent(1); ++a)
    {
      auto col = sqU * Psi_s(nda::range::all, a);
      nda::diagonal(Lank_h(0, spin, 0, a, nda::range::all, nda::range::all)) = col;
      nda::diagonal(Lbnk_h(0, spin, 0, a, nda::range::all, nda::range::all)) = col;
    }
  }

  nda::array<memory::shared_array<MEM, ComplexType, 6>, 1> LQ(nkpts);
  nda::array<memory::shared_array<MEM, ComplexType, 6>, 1> Lank(nkpts);
  nda::array<memory::shared_array<MEM, ComplexType, 6>, 1> Lbnk(nkpts);
  LQ(0)   = detail::to_shared<MEM>(mpi, LQ_h);
  Lank(0) = detail::to_shared<MEM>(mpi, Lank_h);
  Lbnk(0) = detail::to_shared<MEM>(mpi, Lbnk_h);

  return HamiltonianOperations<MEM>(KP3IndexFactorization<MEM>(
      mpi, s.walker_type, nbnd, q0, std::move(bz.nocc), std::move(bz.minusq),
      std::move(bz.qk_to_k2), std::move(bz.qmap),
      detail::to_shared<HOST_MEMORY>(mpi, hij_h),
      detail::to_shared<MEM>(mpi, haj_h),
      std::move(LQ), std::move(Lank), std::move(Lbnk),
      detail::to_shared<HOST_MEMORY>(mpi, vexx_h),
      ComplexType(0)));
}

/// KPTHCOps for 2D Hubbard, single-kpoint lift (nkpts=1, nbnd=NMO).
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_kpthc(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s,
    nda::array<RealType, 2> const& h0,
    nda::array<RealType, 2> const& Psi)
{
  long NMO     = s.NMO();
  long nu      = NMO;
  long nkpts   = 1;
  long q0      = 0;
  RealType sqU = std::sqrt(s.U);

  nda::array<ComplexType, 4> hij_h{nda::reshape(h0, std::array<long, 4>{1, 1, NMO, NMO})};
  auto haj_h  = detail::host_haj(s, h0, Psi);
  auto vexx_h = detail::host_vexx_diagonal<ComplexType, 4>(NMO, s.U);
  auto bz     = detail::make_single_q_tables(s);

  nda::array<ComplexType, 4> X_h(1, nkpts, NMO, nu);
  X_h() = ComplexType(0);
  nda::diagonal(X_h(0, 0, nda::range::all, nda::range::all)) = ComplexType(1);

  nda::array<ComplexType, 6> Y_h(1, 2, 1, nkpts, s.nup, nu);
  Y_h() = ComplexType(0);
  for (int spin = 0; spin < 2; ++spin)
  {
    auto Psi_s = psi_spin(s, Psi, spin);
    Y_h(0, spin, 0, 0, nda::range(Psi_s.extent(1)), nda::range::all) = nda::transpose(Psi_s);
  }

  // L = sqrt(U) I (factorized), Z = U I (Coulomb), single q-block.
  nda::array<ComplexType, 3> L_h(1, nu, nu);
  nda::array<ComplexType, 3> Z_h(1, nu, nu);
  L_h() = ComplexType(0);
  Z_h() = ComplexType(0);
  nda::diagonal(L_h(0, nda::range::all, nda::range::all)) = ComplexType(sqU);
  nda::diagonal(Z_h(0, nda::range::all, nda::range::all)) = ComplexType(s.U);

  return HamiltonianOperations<MEM>(KPTHCOps<MEM>(
      mpi, s.walker_type, NMO, s.nup, s.ndown, nkpts, q0,
      std::move(bz.nocc), std::move(bz.minusq), std::move(bz.qk_to_k2),
      detail::to_shared<HOST_MEMORY>(mpi, hij_h),
      detail::to_shared<MEM>(mpi, haj_h),
      detail::to_shared<MEM>(mpi, X_h),
      detail::to_shared<MEM>(mpi, Y_h),
      detail::to_shared<MEM>(mpi, L_h),
      std::optional{detail::to_shared<MEM>(mpi, Z_h)},
      /*X_rot=*/std::nullopt,
      /*Y_rot=*/std::nullopt,
      /*Z_rot=*/std::nullopt,
      detail::to_shared<HOST_MEMORY>(mpi, vexx_h),
      ComplexType(0)));
}

/// Subclass that promotes the protected helpers needed to assemble a
/// ModelHamOps directly from in-memory sparse matrices, without an HDF5
/// backing file. We pick `shift_1body=true` so the -0.5 U diagonal
/// counterterm lands in H1, matching the Real3 / THC / KP* convention.
class HubbardModelHamOpsAccess : public ModelHamOpsGenerator
{
public:
  explicit HubbardModelHamOpsAccess(AFQMCInfo const& info)
      : ModelHamOpsGenerator(info, dummy_pt(), ComplexType(0), ComplexType(0))
  {}

  using ModelHamOpsGenerator::addComponent;
  using ModelHamOpsGenerator::find_occupied_pairs;
  using ModelHamOpsGenerator::make_SparseEnergy;

private:
  static ptree dummy_pt()
  {
    ptree pt;
    pt.put("name", "test_hubbard");
    pt.put("filename", "/dev/null");
    pt.put("shift_1body", true);
    return pt;
  }
};

/// ModelHamOps for 2D Hubbard, COLLINEAR walker, with onsite-U through the
/// continuous-charge HS.
template<MEMORY_SPACE MEM>
HamiltonianOperations<MEM> build_modelhamops(
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
    HubbardSpec const& s,
    nda::array<RealType, 2> const& h0,
    nda::array<RealType, 2> const& Psi)
{
  long NMO   = s.NMO();
  using csrM = math::sparse::csr_matrix<RealType, HOST_MEMORY, int, int>;

  // hij: closed h0 lifted to collinear (2*NMO, NMO).
  csrM hij_closed({long(NMO), long(NMO)}, NMO);
  for (long i = 0; i < NMO; ++i)
    for (long j = 0; j < NMO; ++j)
      if (std::abs(h0(i, j)) > 1e-12)
        hij_closed.add({int(i), int(j)}, h0(i, j));
  hij_closed.remove_empty_spaces();
  csrM hij = math::sparse::closed_to_collinear(hij_closed);

  // Onsite Hubbard U on (2*NMO, NMO): opposite-spin diagonal in the upper
  // block; same-spin onsite vanishes by Pauli.
  csrM U_combined({long(2 * NMO), long(NMO)}, 1);
  for (long i = 0; i < NMO; ++i)
    U_combined.add({int(i), int(i)}, s.U);
  U_combined.remove_empty_spaces();

  // ModelHamOpsGenerator consumes per-HS-channel U/J vectors in fixed slots:
  // [0]=continuous_charge, [1]=continuous_spin, [2]=discrete_charge,
  // [3]=discrete_spin. Onsite-U through continuous_charge populates only [0].
  std::vector<csrM> Uvec;
  Uvec.reserve(4);
  Uvec.emplace_back(std::move(U_combined));
  for (int i = 1; i < 4; ++i)
    Uvec.emplace_back(csrM({long(2 * NMO), long(NMO)}, 0));
  std::vector<csrM> Jvec;
  Jvec.reserve(3);
  for (int i = 0; i < 3; ++i)
    Jvec.emplace_back(csrM({long(NMO), long(NMO)}, 0));

  AFQMCInfo info{"hubbard_test", int(NMO), s.nup, s.ndown};
  HubbardModelHamOpsAccess gen(info);

  nda::array<long, 1> n2IJ =
      gen.find_occupied_pairs<RealType>(s.walker_type, Uvec, Jvec);

  using map_t = std::unordered_map<long, int>;
  map_t IJ2n;
  IJ2n.reserve(n2IJ.size());
  for (long n = 0; n < n2IJ.size(); ++n)
    IJ2n.insert(std::make_pair(n2IJ[n], int(n)));

  // Ordering is load-bearing: addComponent only reads its U/J inputs whereas
  // make_SparseEnergy moves them. Building the component first keeps Uvec[0]
  // valid for the SparseEnergy call.
  std::vector<ModelComponent<MEM, /*REAL=*/true>> Hams;
  Hams.reserve(1);
  gen.addComponent<MEM, /*REAL=*/true, RealType, map_t>(
      s.walker_type, ContinuousChargePropagator, mpi, Uvec[0], Jvec[0], Hams,
      n2IJ, IJ2n);
  utils::check(Hams.size() == 1, "Expected exactly one ModelComponent for onsite Hubbard U.");

  auto ET = gen.make_SparseEnergy<MEM, /*REAL=*/true, RealType>(
      mpi, s.walker_type, hij, Uvec[0], Jvec[0], ComplexType(0));

  // PsiC(1, 2, NMO, nup): per-spin Slater matrices, sliced from concatenated Psi.
  nda::array<ComplexType, 4> PsiC_h(1, 2, NMO, s.nup);
  PsiC_h() = ComplexType(0);
  for (int spin = 0; spin < 2; ++spin)
  {
    auto Psi_s = psi_spin(s, Psi, spin);
    PsiC_h(0, spin, nda::range::all, nda::range(Psi_s.extent(1))) = Psi_s;
  }

  return HamiltonianOperations<MEM>(ModelHamOps<MEM, /*REAL=*/true>(
      mpi, s.walker_type, s.nup, s.ndown,
      detail::to_shared<MEM>(mpi, PsiC_h),
      std::move(ET), std::move(Hams), n2IJ));
}

} // namespace sfqmc::afqmc::tests
