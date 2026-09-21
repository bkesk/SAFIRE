# CLAUDE.md — SAFIRE

Orientation for AI agents working in this repository. Verify specifics against the
source before relying on them; this file records intent and known deviations, not a
guarantee of the current state.

## What this is

**SAFIRE** (**S**tochastic **A**uxiliary-**F**ields for **I**nte**R**acting **E**lectrons)
is a flexible, high-performance **C++20** implementation of the **auxiliary-field quantum
Monte Carlo (AFQMC)** method. The core lives in [src/](src/); a set of Python tools that
support it lives in [utils/](utils/).

Portions are derived from the QMCPACK project (see [README.md](README.md) and `LICENSES/`).

## Build

CPU-only:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j10
```
GPU (CUDA): add `-DENABLE_CUDA=ON`. Submodules must be initialized first. Key deps: MPI,
HDF5 (parallel recommended), Boost, BLAS/LAPACK, MKL (CPU sparse); `nda`, `spdlog`,
`cpptrace`, `cxxopts`, `Catch2` are fetched automatically. See [README.md](README.md) for
full details. Top-level target is the `safire` executable; library is `safire_lib`.
Unit tests are built when `BUILD_UNIT_TESTS=ON` (default) — see **Testing** below.

## Repository layout

```
src/
  main.cpp
  AFQMC/        ← the core AFQMC engine (the "onion", see below)
  numerics/     ← tensor/sparse/fft math, device kernels
  IO/           ← input parsing (ptree)
  arch/         ← architecture-specific code (CUDA)
  utilities/    ← MPI context, RNG, timers, HDF5 helpers, parser, etc.
  to_be_removed/← legacy code slated for deletion — do not build on it
tests/          ← Catch2 unit tests + unit_test_files/ reference data
utils/          ← Python support tooling (see "Python tooling" below)
```

## The AFQMC "onion" architecture

The classes under [src/AFQMC/](src/AFQMC/) are designed as composable **layers** forming an
onion. The rule:

> **A layer may hold references to layers BELOW it (deeper in the onion), never above.**
> A layer may ask questions deeper into the onion, but must not reach back up.

Composition is often hidden behind `std::variant` / type-erasure wrappers, so to check what
a layer actually holds you must look at the **concrete** implementation classes, not just the
variant wrapper.

### The layers, bottom (innermost) → top (outermost)

| # | Layer | Directory | Role / what it composes (downward refs are OK) |
|---|-------|-----------|-----------------------------------------------|
| 1 | SlaterDeterminantOperations | `SlaterDeterminantOperations/` | Bottom. **Free function templates only** (`det_ops::` namespace) — no classes, no state. |
| 2 | Walkers | `Walkers/` | `WalkerSetBase` etc. Holds only numeric/tensor/config/MPI state. No cross-layer members. |
| 3 | HamiltonianOperations | `HamiltonianOperations/` | Variant over `THCOps`, `KPTHCOps`, `KP3IndexFactorization`, `Real3IndexFactorization`, `ModelHamOps`. Holds only numeric/tensor + same-layer `ModelComponents/`. |
| 4 | Hamiltonians | `Hamiltonians/` | **A factory for HamiltonianOperations.** `THCHamiltonian`, `KPTHCHamiltonian`, `RealDenseHamiltonian`, `KPFactorizedHamiltonian`, `ModelHamOpsGenerator`. *Produces* `HamiltonianOperations` via `getHamiltonianOperations(...)` (return value, not a stored member). |
| 5 | Wavefunctions | `Wavefunctions/` | Variant over `NOMSD`, `PHMSD`, `NOMSD_FT`. Each concrete class holds one `HamiltonianOperations<MEM> HamOp` (layer 3 — downward, OK). |
| 6 | Propagators | `Propagators/` | `AFQMCBasePropagator` (concrete), `Propagator` (variant), `PropagatorFactory`. Holds `Wavefunction*` (layer 5, OK). Clean — all refs downward. |
| 7 | Estimators | `Estimators/` | `EstimatorHandler` + `EstimatorBase` implementations (energy, mixed, back-propagated) and `Observables/`. Hold `Wavefunction*` (layer 5) and `Propagator<MEM>*` (layer 6) — both downward, OK. |
| 8 | Drivers | `Drivers/` | Top. `AFQMCDriver`, `DriverFactory` (and legacy `CSAFQMCDriver`). May reference anything below. |

This ordering was confirmed by auditing the actual composed members of the concrete classes
(2026-06). **All layers (1–8) hold references strictly downward — no violations.**

> **Why Propagators sits below Estimators.** The dependency between these two layers is
> strictly one-directional: **Estimators depends on Propagators, never the reverse**
> (Propagators holds zero references to Estimators), and at construction the propagator is
> built first and passed *into* the `EstimatorHandler`
> ([DriverFactory.cpp:298,341](src/AFQMC/Drivers/DriverFactory.cpp#L298)). The back-propagation
> estimators ([BackPropagatedEstimator.hpp](src/AFQMC/Estimators/BackPropagatedEstimator.hpp),
> [BPWithTimeEvolvedOperators.hpp](src/AFQMC/Estimators/BPWithTimeEvolvedOperators.hpp)) hold a
> `Propagator<MEM>* prop0` to re-propagate walkers backward/forward in time
> (`prop0->BackPropagate(...)`, `prop0->PropagateOperators(...)`) — a legal downward reference.

### Known hierarchy violations

No member-level violations are currently known, and no upward source (`#include`)
dependencies are currently known. The composed-member audit (2026-06) found every concrete
class references only layers below it.

> Violations may still exist, especially in object **setup/factory** paths, that this audit
> did not surface. Treat the above as known-but-not-exhaustive. If you discover one, document
> it here.

### Legacy / stale code to avoid

- [src/to_be_removed/](src/to_be_removed/) — slated for deletion; do not build new work on it.
- `CSAFQMCDriver` ([CSAFQMCDriver.h](src/AFQMC/Drivers/CSAFQMCDriver.h)) is **non-templated**
  and uses the old non-`<MEM>` type names; it appears stale relative to the templated layers.
  Prefer `AFQMCDriver<MEM>`.

## Testing

Unit tests are **Catch2**-based, live in [tests/](tests/), and are registered with **CTest**
(built when `BUILD_UNIT_TESTS=ON`, the default). Run them from a configured build tree:

```bash
ctest --output-on-failure        # full suite
ctest -R "<regex>"               # subset, e.g. -R "phmsd|hamiltonian_operations"
ctest -N                         # list registered tests without running
```

What's there:
- **`test_afqmc`** — a single Catch2 binary aggregating all AFQMC cases, compiled from the
  `tests/test_*.cpp` sources (hamiltonian / wavefunction / propagator / driver factories,
  estimators, phmsd, observables, one-body, sdet ops, …). Each `ctest` entry such as
  `phmsd_components: ph excited energy (real dense cholesky)` is a Catch2 case *inside* this
  binary, registered under the `afqmc` label.
- Standalone numerics binaries: `test_const_shared_array`, `test_sparse`, `test_nda_functions`,
  `test_math_product`.
- Shared fixtures/helpers: [tests/catch_main.cpp](tests/catch_main.cpp),
  [tests/test_common.hpp](tests/test_common.hpp), [tests/test_utils.hpp](tests/test_utils.hpp),
  [tests/hubbard_factorizations.hpp](tests/hubbard_factorizations.hpp).
- Reference data the tests read lives under
  [tests/unit_test_files/](tests/unit_test_files/), in per-system subdirectories
  (e.g. `hubbard/`, `Ne_cc-pvdz/`, `Li_noncollinear/`, `finiteT/`).

Notes:
- Some cases are launched under MPI (`mpiexec -n N --oversubscribe <binary>`); the AFQMC
  Catch2 cases run the binary directly.
- To iterate on one case, run the binary with the Catch2 case name, e.g.
  `<build>/tests/bin/test_afqmc "phmsd: compute"`.
- The factory build tests (`propagator_factory: build`, `driver_factory: build`) are the
  slowest cases — expect them to dominate a full run.
- Machine-specific build/test commands (toolchain, already-configured build trees) may be
  kept in a git-ignored `CLAUDE.local.md`.

## Python tooling (`utils/`)

[utils/](utils/) holds Python tools supporting the C++ code: the `afqmctools` package
(Hamiltonian/wavefunction generation, observable & scalar analysis, format converters) plus
CLI scripts in [utils/cli/](utils/cli/). Install via `pip install .` from `utils/`
(see [utils/README.md](utils/README.md)).

**These tools were added ad-hoc and contain redundancies and outdated functions.**
[utils/README.md](utils/README.md) already separates **Supported CLI Tools** from
**Deprecated CLI Tools** (e.g. `aimbes_to_afqmc` is deprecated in favor of
`aimbes_to_2nd_quant`) — treat that list as the current source of truth for the CLI scripts.

> ⚠️ **This section is incomplete.** A thorough audit of which `afqmctools` *library*
> modules/functions are current vs. redundant/outdated has **not** been done yet and is
> deferred to a future session. Until then, do not assume a given utility function is the
> preferred one — verify, and prefer what the README marks as supported.

**`utils/AutoHF/` is an independent package** (its own git history) and is intentionally
**out of scope** for this file. Do not document or treat it as part of SAFIRE here.
