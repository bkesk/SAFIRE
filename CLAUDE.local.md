# CLAUDE.local.md — local build & run (this machine)

Machine-specific, **git-ignored**. Generic/portable instructions live in [CLAUDE.md](CLAUDE.md)
and [README.md](README.md). The commands below are **verified working** on this machine
(2026-06): full build of `safire` + `test_afqmc` and `ctest` 27/27 passing.

## Toolchain (recorded in the configured build trees)

- C++ compiler: `/opt/homebrew/opt/llvm/bin/clang++` (Homebrew LLVM — **not** Apple Clang)
- C compiler: `/opt/homebrew/opt/llvm/bin/clang`
- MPI: `/opt/homebrew/bin/mpicxx` / `mpiexec` (open-mpi 5.0.1), OpenMP via libomp
- CMake: `/opt/homebrew/bin/cmake`
- Language standard in use: **C++23**; `ENABLE_CUDA=OFF`, `BUILD_UNIT_TESTS=ON`

## Build trees (already configured — prefer incremental)

Two configured trees exist; both use the **Unix Makefiles** generator:

- `build/Debug`   — `CMAKE_BUILD_TYPE=Debug`  (matches root `compile_commands.json`)
- `build/Release` — `CMAKE_BUILD_TYPE=Release`

Incremental build (default to Debug for dev):
```bash
cd build/Debug
make -j8 safire_lib test_afqmc safire
```
Output binaries:
- main app:  `build/Debug/bin/safire`
- unit tests: `build/Debug/tests/bin/test_afqmc` (single Catch2 binary holding all AFQMC cases)

## Configure from scratch (only if a tree is missing/stale)

```bash
cd /Users/keskridge/software/SAFIRE
git submodule update --init --recursive            # AutoHF etc.
cmake -S . -B build/Debug -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang
cmake --build build/Debug -j8
```
(Most deps — nda, spdlog, cpptrace, cxxopts, Catch2, tblis — are fetched automatically.)

## Run the tests

From the build tree, via ctest:
```bash
cd build/Debug
ctest --output-on-failure                 # full suite: 27 tests, ~3.5 min
ctest -R "phmsd|hamiltonian_operations"   # subset by regex
ctest -N                                  # list tests without running
```
Notes:
- Some tests are launched under MPI (`mpiexec -n 4 --oversubscribe <binary>`); the AFQMC
  Catch2 cases run the binary directly.
- Slowest cases: `propagator_factory: build` (~87 s), `driver_factory: build` (~49 s).
- Run Catch2 cases directly for a tight loop, e.g.:
  ```bash
  build/Debug/tests/bin/test_afqmc "phmsd_components: ph excited energy (real dense cholesky)"
  ```

## Run the app

`safire` takes one positional input file (`.json` or `.xml`) and is an MPI program:
```bash
mpiexec -n 4 build/Debug/bin/safire input.json
# options: --verbosity {0..3}  --debug {0..3}  --compute {cpu|gpu|default}  -h/--help
```
- Generate input JSON with the Python tool `write_afqmc_json` (see [utils/README.md](utils/README.md)).
- Example inputs live under `tests/unit_test_files/`, e.g.
  `tests/unit_test_files/hubbard/4x4squareU4/afqmc.json`.

## How to explain ideas to users

This applies to documentation, tutorials, examples, docstrings, comments, warning and error messages.

- the users are reasonably smart; don't over explain common ideas
- the users are physics domain experts; no need to teach them physics, just how to do their physics
    with this software.
- the so-called "emdash" (i.e. `This is a sentence - sometimes AI interjects using an emdash - that was interrupted` ) 
    is bad! no one wants them, do not use them. Use regular punctuation instead.
- avoid explaining the same idea multiple times in the same place
