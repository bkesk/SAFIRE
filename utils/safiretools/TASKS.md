# safiretools — Prototype Build Task List

Status: living checklist. Tracks work not yet done. Read [DESIGN.md](DESIGN.md) first — every
task here implements a decision recorded there; this file doesn't re-justify anything, it just
sequences the build. Check items off as they land; add new ones if scope shifts, but changes to
*decisions* belong in DESIGN.md, not here.

Ordered bottom-up through the dependency stack (each phase only depends on earlier phases).
Within a phase, tasks can go in parallel unless noted.

## Phase 0 — ✅ Package scaffolding

- [x] Create the `safiretools/` package skeleton (empty `__init__.py` + subpackage dirs per the
      DESIGN.md tree: `analysis/`, `hamiltonian/`, `hamiltonian/model/`, `wavefunction/`,
      `convert/`, `qe/`).
- [x] Add `safiretools` as a build target in `utils/pyproject.toml` alongside (not yet replacing)
      `afqmctools`/`stats`, so both can build during the migration.
- [x] Decide + set up a test layout (e.g. `utils/tests/safiretools/` mirroring the package tree) —
      there's no existing test suite for `afqmctools`/`stats` to carry over, so this starts fresh.
      Decision: `utils/tests/safiretools/` mirrors `safiretools/` module-for-module (e.g.
      `tests/safiretools/hamiltonian/model/test_lattice.py` for
      `safiretools/hamiltonian/model/lattice.py`); subdirectories are created as later phases add
      their tests rather than pre-created empty. `tests/safiretools/conftest.py` is the root for
      shared safiretools fixtures, separate from the afqmctools-oriented root `tests/conftest.py`.
      No `pyproject.toml` changes needed — `testpaths = ["tests"]` already covers the new dir.

## Phase 1 — ✅ Foundation (dependency-free modules)

- [x] `types.py`: `SpinSymm(IntEnum)` — `CLOSED=1, COLLINEAR=2, NONCOLLINEAR=3`. Test: values match
      `WALKER_TYPES` in `src/AFQMC/config.h` on this branch.
- [x] `hdf5.py`: generic HDF5 read/write primitives, consolidating `utils/io.py`'s generic bits,
      `rhonk.py`'s vendored copy, and `stats/config_h5.py` (drop `pytables`, use `h5py` only).
      Also fixed a latent bug found while writing `test_hdf5.py`: `add_dataset`'s overwrite branch
      (`fh5[name] = value` when `name` already exists) raised `OSError` in h5py instead of
      overwriting — never hit in existing call sites since they only write fresh files. Now does
      `del fh5[name]` first.
- [x] `stats.py`: `mean_and_error(samples, axis=0)` (replaces `stat_h5.py::me2d`,
      `scalar_dat.py::single_column_from_array`/`error`, the `scipy.stats.sem` fallback in
      `analysis/average.py`), `reblock()` (smaller trailing block, not discard-and-raise), and the
      autocorrelation-time estimator (numba dropped for one dot product per lag — same O(n·*l*)
      cost and same numerics to 1.2e-15; see DESIGN.md "Statistics core"). Handle complex samples
      as independent real/imaginary error bars. Detect
      degenerate/constant data via an upfront variance check, not post-hoc NaN replacement.
  - [x] Test: `mean_and_error` against a known-mean synthetic series (real and complex).
  - [x] Test: `reblock` on a sample count that doesn't evenly divide — confirm smaller trailing
        block, not an exception.
  - [x] Test: constant-input series is detected up front, not silently turned into an error of 1.0.

Tests live in `tests/safiretools/test_types.py`, `test_hdf5.py`, `test_stats_core.py` (named
`test_stats_core.py`, not `test_stats.py`, to avoid colliding with the existing
`tests/test_stats.py` for the old `afqmctools`/`stats` packages — neither test tree uses
`__init__.py`, so pytest identifies modules by basename alone; user chose the rename over adding
`__init__.py` files). All pass; ran the full non-functional suite afterward and confirmed no
regressions (4 pre-existing failures in `test_autohf.py`/`test_lattice_models.py`, unrelated to
this phase — a scipy-version issue in code this phase didn't touch).

## Phase 2 — ✅ Lattice

- [x] `hamiltonian/model/lattice.py`: port `afqmctools/systems/lattice.py` → `Lattice` ABC +
      `SquareLattice`/`TriangularLattice`/`HoneycombLattice`/`KagomeLattice`, with
      `Lattice.from_dict(params)` replacing the free-function `get_lattice(params)`.
      `CustomLattice` was ported too (user call), and is *not* a redundant alias for the built-in
      types: it is the one type whose geometry the caller defines. See
      **"Unit-cell geometry belongs to the lattice type"** in DESIGN.md.
  - [x] Fix: `a1`/`a2`/`basis` overrides silently discarded for the built-in lattice types are now
        **rejected, not applied** — see DESIGN.md **Unit-cell geometry belongs to the lattice type**.
  - [x] Geometry is immutable after construction, per the same DESIGN.md section.
        `_set_unitcell()` is the single private installer, copying before freezing so a caller's
        array is never frozen out from under them; its only callers are `__init__` and `build()`'s
        `cyl_mode` reshaping. `build()` raises `RuntimeError` if called twice — it would
        otherwise re-double a `cyl_mode` basis and append a second copy of every site.
  - [x] Fix: `_neighbor_distance_map`'s `min_distance` param is passed by `__init__` but doesn't
        exist on the method signature — add it (currently untested/unhit `TypeError` if non-`None`).
  - [x] Fix: `_is_allowed_site` defined twice back-to-back with identical bodies — remove the
        duplicate.
  - [x] Remove dead rotation-group neighbor-generation code (`ROTATION_GROUP`, `_rotations()`,
        `_check_add_image_neighbors`) — docstring claims it's live, it isn't. `ROTATION_ANGLES` and
        the `_2d_rotation()` helper went with it (their only use was building `ROTATION_GROUP`).
  - [x] Remove superseded `_is_valid_image_old` (unresolved "check against next version" TODO).
  - [x] No standalone `to_hdf5`/`from_hdf5` on `Lattice` — confirmed, with a correction to the
        premise: today **no** lattice state reaches HDF5 at all, embedded or otherwise.
        `utils/io.py::write_model_hamiltonian` writes only the Hamiltonian's matrices/metadata; a
        lattice is reconstructed from the `params` dict, which round-trips through
        `write_model_params`/`read_input_params` as **TOML**. Whether `LatticeHamiltonian`'s HDF5
        file should start embedding lattice geometry is therefore an open Phase 3 decision, not a
        behavior to preserve.
  - [x] `cyl_mode` guard: `ValueError` unless `_type == 'triangular'` (DESIGN.md **Basis
        validation**). This rejects `CustomLattice` too, even one given triangular vectors; loosen
        the check if that turns out to matter.
  - [x] Fix the standing BUG note on `CustomLattice` about large basis vectors with a real check —
        `build()` calls `_validate_basis()`. See DESIGN.md **Basis validation** for the two failure
        cases and the rule. Validated against a brute-force oracle over 660 (size, offset, boundary)
        combinations at L>=3: 455 admitted, all agreeing with the oracle on both the
        nearest-neighbor distance and the pair count; 0 false negatives, 0 false positives.
  - [x] Re-export `Lattice` (only the ABC, not concrete subclasses) from top-level `__init__.py`
        — pulled forward from Phase 8 so the docs can show the intended public import
        (`from safiretools import Lattice`) instead of a deep path that Phase 8 would have to
        re-edit. `safiretools/__init__.py` now exports exactly this one name.
- [x] Test: `from_dict` handles `a1`/`a2` correctly for each lattice type (regression test for the
      fix above). Reworded from "round-trips overrides" to match the decision above: each built-in
      type rejects each of `a1`/`a2`/`basis` from both `from_dict` and its constructor, tolerates
      them present-but-`None`, and reports geometry matching its own `_unitcell()`; `CustomLattice`
      is the one type where they round-trip, checked all the way through to site positions.
- [x] Test: one geometry/neighbor-map regression case per lattice type against known-good output
      (guards the dead-code removal not changing live behavior).

- [x] Point docs, tutorials and examples at the new `Lattice` (see **Doc migration** below).

Tests live in `tests/safiretools/hamiltonian/model/test_lattice.py` (137 cases, all passing; full
non-functional suite afterward: 412 passed, 0 failed). Equivalence with the old implementation was
checked with a throwaway harness comparing `afqmctools.systems.lattice.get_lattice` against
`Lattice.from_dict` over 42 cases (4 lattice types × 3 sizes × 3 boundary combinations, plus twist,
default-type, and `cyl_mode` XC/YC cases), fingerprinting site count, positions, coordinates, `a1`/
`a2`/`basis`/`A`, the full `_dist_map`, and the n=1..3 direct and image neighbor pair sets including
phases, relative positions, and supercell shifts — plus `get_directed_pairs` and the
unbuilt-accessor errors. Everything matched except the `nb`/`num_sublattice` fix noted below.

### Changes beyond the listed fixes

Structural, chosen to fix the *cause* of the `a1`/`a2` bug rather than the symptom (user call):

- `Lattice` is a real ABC now, with an abstract `_unitcell() -> (a1, a2, basis)`. The built-in
  subclasses have no `__init__` at all — they only declare `_type` and that hook. The base has the
  single constructor, and it takes **no geometry parameters and no `**kwargs`**, so the
  "assign `self.a1` before `super().__init__()`, then let `**kwargs` swallow the caller's value"
  pattern that caused the bug is gone, geometry arguments cannot reach a built-in type even by
  accident, and an unknown keyword now raises `TypeError`.
- `CustomLattice` is the sole subclass with its own `__init__`: it takes required keyword-only
  `a1`/`a2` plus optional `basis`, stashes them, and returns them from `_unitcell()`.
- Constructor arguments after `L` are keyword-only. The old base and subclass signatures disagreed
  on positional order (`metric` vs `build` in slot 2), so positional calls were already ambiguous.
- `TriangularLattice`'s "does not fully support basis; proceed with caution" warning is gone with
  the `_custom_basis_supported` flag that gated it — a basis can no longer be passed to that type
  at all. The underlying caveat (large basis vectors break direct/image neighbor computation) moved
  to `CustomLattice`'s docstring, which is now the only place it can be hit.
- `nb`/`num_sublattice` are read-only properties over `_nb` instead of copies made in `__init__`.
  They used to go stale: `build()` doubles the basis under `cyl_mode='XC'/'YC'` and updates only
  `_nb`, so a cylinder-mode lattice reported `num_sublattice == 1` with 2 sites per cell (this is
  the sole intentional difference the equivalence harness reports; `utils/visualize.py` is the one
  caller and was mis-plotting cylinder-mode lattices as a result). `_set_unitcell()` keeps `_nb` in
  step with the basis by construction.
- The `cyl_mode` warning text in `build()` now reads `(new a1=... new a2=...)` instead of
  `(new self.a1=... new self.a2=...)`, since the block works on locals before installing them.
- The eager `_neighbor_distance_map(distance=0., min_distance=...)` call in `__init__` is gone. It
  ran before `build()`, when `self.sites` is empty, so it could only ever raise — first `TypeError`
  on the missing parameter, and `ValueError` even once that was added. `_dist_map_min_distance` is
  stored and applied lazily when the distance map is first built. For the semantics chosen, see
  DESIGN.md **`min_distance` filters the neighbor-distance map but keeps the self-distance**.

Convention alignment per DESIGN.md, behavior-preserving:

- Progress `print()`s (`"computing and storing Nth-nearest neighbors"`, `"Computing distance matrix
  of lattice"`, `"Reading lattice site positions from Lattice"`, `"Removed distance matrix"`) →
  `logger.debug`. `warnings.warn` calls are left as warnings.
- `raise Warning(...)` in `_build_nth_neighbors`/`_build_nth_image_neighbors` → `RuntimeError`.
  Both are unreachable through the public getters, which only call them on a cache miss.
- `valid_L()` no longer asserts `len(L.shape) == 2` for array-like `L`. That assert fired for the
  obvious numpy input (`np.array([4, 4])`, whose `shape` is `(2,)`), so `L` as an array was broken;
  it is now a plain 2-element length check. Also drops a bare `assert` used for validation.
- `get_directed_pairs`'s `if not is_image / elif is_image / else: append(-1)` collapsed to
  `if/else`. The `-1` branch was unreachable; a comment records that the invalid-site encoding
  exists but is never produced while both boundaries are assumed periodic.
- Unused `itertools` import dropped; 2-space-indented method bodies normalized to 4.

### Doc migration

All 16 lattice-using doc sources now build lattices with `from safiretools import Lattice` +
`Lattice.from_dict(...)`: `docs/examples/models/{01..08}`, `docs/tutorials/models/{03..08}`, and
`docs/snippets/01_setting_up/{11_custom_terms,12_rashba_soc}/run.py`. `docs/_build/` is generated
output and was left alone.

Mostly a mechanical `get_lattice(` → `Lattice.from_dict(` swap, but four content changes were
needed because doc code relied on the bug this phase fixed:

- **Four blocks passed a `basis` with a fixed-geometry `type`** and so were silently building a
  different lattice than they described: `tutorials/.../03_setting_up_a_lattice` (the "Lattice
  basis" exercise, the water-molecule example, and the Lieb-lattice example) and
  `examples/models/08_lieb_lattice_emery` (the opening Lieb/Emery cell). All four are genuinely
  custom lattices and now say `type='custom'` with explicit `a1`/`a2`. Note the "Lattice basis"
  exercise had also been passing *1-dimensional* basis vectors (`[[0.0],[0.5]]`), which numpy
  broadcast into both components — directly contradicting the prose two paragraphs above it.
- **The "Lattice basis" prose documented the old behavior** ("If a basis is provided ... it will be
  ignored"). Rewritten to state the geometry-belongs-to-the-type rule, with a runnable cell showing
  the error and a pointer to the custom lattice.
- **`Lattice.from_dict()`'s parameter list** in that tutorial now documents `a1`/`a2`/`basis` as
  custom-only and `cyl_mode` as triangular-only; neither restriction was mentioned before.
- **`examples/models/08_lieb_lattice_emery`'s two direct `lat.CustomLattice(...)` calls** became
  `Lattice.from_dict(dict(type='custom', ...))`, which also drops its `lat.PBCBoundary` use.
  DESIGN.md deliberately does not re-export concrete subclasses, so `from_dict` is the public path
  and avoids a deep import in user-facing docs.

Reference docs: `docs/afqmctools/lattice_models.rst` had an empty "Lattice Class" stub; it now
documents the new class, states the geometry rule, links the tutorial, and autodocs
`safiretools.hamiltonian.model.lattice`. `docs/afqmctools/api/afqmctools.systems.rst` was left
pointing at the old module, which still exists — it goes away in Phase 9 with the package.

Verification (`nb_execution_mode = "off"`, so a docs build does **not** execute these cells and
would not have caught breakage):

- A harness executed each doc file's `{code-cell}` blocks cumulatively in one namespace, notebook
  style: **36 lattice-touching blocks ran clean, 0 lattice failures.** The 3 remaining failures in
  lattice-touching blocks are pre-existing and unrelated — two are the `AFQMC_EXEC`-unset
  `Path(None)` crash in `tutorial_utils` (the Phase 8 item), one needs a `qmc.s000.stat.h5` from an
  actual AFQMC run.
- A second harness confirmed the new `Lattice` drives every un-ported `afqmctools` consumer the
  docs feed it to — `visualize.plot_lattice` (all 5 lattice types, plus `nth_neighbor=2`),
  `HamiltonianBuilder.from_input`, direct `builder.nth_neighbor_hopping`/`onsite_hubbard`/
  `finalize`, `write_model_hamiltonian`, `get_directed_pairs` + `write_pair_correlators`,
  `free_electron`, and `get_planewaves`. Nothing type-checks `Lattice` with `isinstance`, so the
  duck-typed hand-off works; `basis` becoming a tuple is unobserved because no consumer reads it.
- Full sphinx build succeeds. It surfaced one new warning — `Lattice` became an ambiguous
  cross-reference target once both modules were autodoc'd — fixed by fully qualifying the type in
  `afqmctools/wavefunction/free_electron.py`'s `lattice` parameter docstring (the only edit made to
  afqmctools library code in this phase). Warning count back to the pre-change baseline of 19.

## Phase 3 — ✅ Hamiltonian

- [x] `hamiltonian/base.py`: `Hamiltonian` ABC — `to_hdf5(path) -> None` /
      `from_hdf5(path) -> Hamiltonian` (classmethod, dispatches to concrete subclass); `spin_symm`
      as a plain attribute, not a subclass axis. Dispatch goes through a public
      `hamiltonian_format(path)`, which replaces `converter.py::read_hamil_type` and returns one of
      `model` / `dense` / `kpoint` / `thc` / `kpoint_coqui`. Subclasses implement
      `_read_hdf5(path, fmt)` rather than overriding `from_hdf5`, so dispatch stays in one place, and
      the reader classes are imported lazily so that reading one format never pulls in another's
      dependencies (`import safiretools` touches neither pyscf nor mpi4py).
- [x] `hamiltonian/model/builder.py`: port `afqmctools/hamiltonian/model/builder.py`'s
      `HamiltonianBuilder`, keep public.
  - [x] Fix `force_herm` diagonal-zeroing bug for its two real call sites (`tband` in
        `nth_neighbor_hopping`, `epsilon_band` in `onebody_onsite`): reconstruct as
        `np.triu(M,0) + np.triu(M,1).conj().T`, not `np.triu(M,1)` alone. **The fix was
        unreachable before it could be applied**: both call sites take a *boolean* parameter also
        named `force_herm`, which shadows the imported function, so `force_herm(tband, ...)` would
        have raised `TypeError: 'bool' object is not callable`. The function is now
        `force_hermitian`; the boolean parameter keeps its documented name. Both call sites share a
        new `_band_matrix()` helper.
  - [x] Keep the *separate*, correct upper-triangular convention for U1/U2/J (diagonal genuinely
        unused there) as its own helper — do not merge with the `force_herm` fix above. It is now an
        explicit `upper_triangular_band_matrix(value, nbands, include_diagonal=False)`, which
        `_build_intrasite_band_matrix`/`_build_intersite_band_matrix` collapse into: they differed
        only in `combinations` vs `combinations_with_replacement`.
  - [x] Replace bare `builder.hamiltonian`/`builder.lattice` attribute access with explicit
        `builder.get_hamiltonian()` / `builder.get_lattice()` accessor methods. The backing
        attributes are private (`_hamiltonian`/`_lattice`).
  - [x] Test: `force_herm` fix preserves the diagonal on a synthetic non-Hermitian `t`/`epsilon`
        matrix (regression test for the bug).
  - [x] Test: U1/U2/J upper-triangular helper still zeroes/ignores the diagonal as intended (guards
        against accidentally applying the `force_herm` fix to this path).
- [x] `hamiltonian/model/lattice_hamiltonian.py`: `LatticeHamiltonian(Hamiltonian)`. `from_dict()`
      wraps `HamiltonianBuilder` internally for the common case. `HamiltonianComponent` moved here
      too (from `ham_class.py`), since it is part of what the container holds.
- [x] `hamiltonian/molecular.py`: `MolecularHamiltonian(Hamiltonian)`, ported from
      `hamiltonian/mol.py`. Factories are `.from_pyscf(scf_data, ...)` (was `generate_hamiltonian`)
      and `.from_integrals(hcore, chol=/eri=, ...)` (was `process_generic_hamiltonian`).
      `modified_cholesky_direct` came along from `utils/linalg.py` — this is its only live consumer
      in the ported scope.
- [x] `hamiltonian/periodic.py`: `PeriodicHamiltonian(Hamiltonian)`, merging `kpoint.py` +
      `supercell.py` — dedupe the ~90% duplicated Cholesky-solver code confirmed during the audit.
      One `PeriodicCholesky` class with a `kp_sym` flag (user call), see **Periodic merge** below,
      and one on-disk format: the supercell case is emitted as a Γ-point k-point Hamiltonian
      (`nkpts=1`), see **Supercell output format**.
  - [x] Test: k-point and supercell paths both still produce known-good output post-merge
        (regression, not new coverage).
- [x] `hamiltonian/fcidump.py`: FCIDUMP external-format I/O, ported as-is.
- [x] Each concrete `Hamiltonian` subclass implements its own `to_hdf5`/`from_hdf5` (formats
      genuinely differ by domain, per DESIGN.md). For models this required splitting the shared
      `Uij` dataset back into `U`/`U1`/`U2` on read — see DESIGN.md **A model file's `Uij` splits
      back into `U`/`U1`/`U2` exactly**. Verified by re-finalizing and rewriting: the second file is
      dataset-for-dataset identical to the first.

Tests live in `tests/safiretools/hamiltonian/` (`test_base.py`, `test_molecular.py`,
`test_periodic.py`, `test_fcidump.py`, `model/test_builder.py`,
`model/test_lattice_hamiltonian.py`) — 336 safiretools cases in total, all passing on both
python 3.12 and 3.13 environments. Full non-functional suite afterward: 600 passed, 1 failed —
`tests/test_hamiltonian.py::TestSupercell::test_modified_cholesky`, which is **pre-existing and
unrelated** (it fails standalone on a pristine collection, in both environments; the assertion
counts Cholesky entries above `1e-10` and gets 6047 rather than the recorded 6229, i.e. numerical
noise in un-ported afqmctools code against current scipy/numpy).

### Equivalence with afqmctools

Checked with throwaway harnesses comparing the ported code against the original, both in memory and
byte-for-byte on the written HDF5/FCIDUMP files:

- **Builder / model Hamiltonian** — 13 cases (Hubbard, attractive U, t-t', Hubbard-Kanamori
  multiband, extended V, Heisenberg, AFM/FM/charge pinning, twist, honeycomb, triangular
  next-nearest, closed-shell, HST override), fingerprinting every term's shape, `model_type`,
  `hst_type`, `spin_symm` and sorted nonzero entries, plus every dataset of the written file.
  **13/13 match**, with the single documented dtype difference below.
- **Molecular** — `modified_cholesky_direct`, `chunked_cholesky`, `transform_cholesky`,
  `freeze_core` all reproduce the originals exactly; `write_hamil_mol`'s output matches
  `MolecularHamiltonian.from_pyscf(...).to_hdf5(...)` dataset for dataset, and the result is still
  readable by afqmctools' own `read_hamiltonian`.
- **Periodic** — `generate_grid_shifts`, `construct_qk_maps`, `get_ortho_ao` and `setup_basis_map`
  match exactly; the supercell Cholesky solver reproduces `supercell.Cholesky.run` to `1e-9`
  (shape `(4, 64, 53)` on the diamond 2x1x1 fixture); the assembled Γ-supercell `L` matches the old
  supercell path's own scatter **bitwise**; and `write_hamil_kpoints`'s output matches
  `PeriodicHamiltonian.write_from_pyscf` dataset for dataset **both serially and under
  `mpiexec -n 2`**, including the per-rank scratch-file merge and its cleanup.
- **FCIDUMP** — `write_fcidump` is byte-identical across `sym=1/4/8`, real/complex, parenthesized
  and spinor variants; `write_fcidump_kpoint` is byte-identical for uniform `nmo_pk`;
  `read_fcidump`, `read_fcidump_header`, `check_sym` (exhaustively over a 4-orbital index space)
  and `h1_spat2spin` all agree.

### Changes beyond the listed fixes

Bugs found while porting. Each changes behavior, so each is listed rather than folded in silently.

- **`real_valued` was inverted** (DESIGN.md **Known bug: inverted `real_valued` flag**), so a real
  Hamiltonian now writes rank-1 real `data_`. **This is the one difference in the equivalence check
  above.** The C++ side already accepts both (`ModelHamOpsGenerator.cpp:393` takes rank 1 or 2) and
  the rank-1 path is exercised by `tests/unit_test_files/hubbard/4x4squareU4/afqmc.h5`; the file
  also halves in size.
- **Inter-site U1/U2/J dropped half their band pairs** (user-spotted). `_build_intersite_band_matrix`
  used `itertools.combinations_with_replacement`, i.e. `m <= n` only. That is right for an *onsite*
  term, where the band matrix is kroneckered with the identity over sites and `m > n` maps below the
  combined diagonal (so the executable discards it anyway). It is wrong inter-site: there the site
  factor is already strictly upper (`I < J`), so band pair `(m, n)` lands at
  `(I*nbands + m, J*nbands + n)`, which is above the combined diagonal for **every** `(m, n)` — the
  whole band matrix is in the range SAFIRE reads. `(m, n)` is band `m` on site `I` interacting with
  band `n` on site `J`, a different interaction from `(n, m)`, so restricting to `m <= n` silently
  omitted the rest. On two sites with two bands the `s0b1`–`s1b0` interaction was missing outright.
  Split into `onsite_band_matrix` (strictly upper) and `intersite_band_matrix` (full).
  **Scope of the change:** only `nbands > 1` *and* `nth_neighbor > 0` — every other combination is
  bit-identical, which is exactly why the equivalence harness (13/13, still passing) never caught it.
  Measured against afqmctools: `nbands=2` gains `n(n-1)/2 = 1` band pair per bond, `nbands=3` gains 3.
- **`fm_pinning` read `afm_pin_type`.** A copy-paste slip that made the `fm_pin_type` input key
  dead. It now reads `fm_pin_type`.
- **`hubbard_U1_density_density` crashed on a band matrix.** Its `if not U1:` guard raises
  `ValueError: truth value of an array ... is ambiguous` for the `(nbands, nbands)` input its own
  docstring documents. Dropped — the `np.allclose(U1, 0.0)` check two lines below already covered
  the scalar case it was written for.
- **`write_dense`'s hcore truncation.** It chose hcore's dtype with
  `numpy.all(numpy.iscomplex(hcore))`, which is false for any Hermitian matrix (real diagonal), so a
  genuinely complex hcore was silently written as `numpy.real(hcore)`. Now decided with
  `numpy.any`. Note that `real_chol=False` still writes a file afqmctools' own reader rejects
  (`ComplexIntegrals` conflicts with the real hcore) — pre-existing, and preserved; see DESIGN.md
  **Hamiltonian on-disk formats**.
- **`FileHandler`'s per-rank scratch name.** It built `"rank" + str(rank) + "_" + filename` from the
  *full path*, producing `rank1_/abs/path/ham.h5` — unopenable unless the caller happened to pass a
  bare filename in the working directory. Split into a `rank_filename()` helper that prefixes only
  the basename.
- **k-point Cholesky over-allocation.** `KPCholesky` allocated its buffers at `maxvecs * nmo_tot`
  while its loop could never exceed `maxvecs * nmo_max`, wasting a factor of `nkpts` on
  `cholvecs[nkk, nij, maxvecs]`. The buffer now matches the loop bound; the factorization is
  unchanged.
- **`write_fcidump_kpoint`'s offsets.** `np.cumsum(nmo_pk) - nmo_pk[0]` is only correct when every
  k-point has the same orbital count; it is `np.cumsum(nmo_pk) - nmo_pk` now. No behavior change for
  uniform `nmo_pk`, which is the only case the old tests cover.
- **`SpinSymm` string round trip.** `utils/io.py` wrote `'closed'` but its reader looked up
  `'close'`, so a closed-shell Hamiltonian's `spin_type` could never be read back.
  `SpinSymm.from_input` accepts both and `label` always writes `'closed'`.

Dropped as dead or unreachable:

- `spindep` — a `_parse_ham_input` key set as an attribute on the Hamiltonian and never read
  anywhere in the repo.
- `write_rhoG_kpoints` / `write_rhoG_supercell` — **stubbed** per DESIGN.md Scope. Why neither
  could run: `write_rhoG_kpoints` references an undefined `nelec`, and `write_rhoG_supercell` calls
  a bare `quit()` on line 145 before touching undefined `Xocc`, `ngs` and `hamil_file`.
- `PartitionOld` — an exact duplicate of `Partition(kp_sym=False)`.
- `_ZeroComponent` — already self-described as deprecated, returned `None`.
- A duplicated `opposite_twists` warning block in `nth_neighbor_hopping` (identical, back to back).
- The tautological `assert` in `_add_interaction` checking that a sign split reconstructs its input.

Convention alignment per DESIGN.md, behavior-preserving: `print()` → `logging`; `sys.exit()` in the
Cholesky solvers → `RuntimeError`/`ValueError`; `Partition`'s nproc/nkpts check → `ValueError`;
`construct_qk_maps`' four `sys.exit()` paths → `ValueError`.

### Periodic merge

The decision — one `PeriodicCholesky` with a `kp_sym` flag, and the two entry points — is in
DESIGN.md **Periodic Cholesky: one solver, one flag**. What the merge actually came to: the
factorization loop (pivot broadcast, new-column evaluation, projection, residual update, global
pivot selection) is shared verbatim, and only four things differ, each a small method:

| | `kp_sym=True` | `kp_sym=False` |
|---|---|---|
| `momentum_blocks()` | one block per `Q <= kminus[Q]` | a single `None` block |
| `k_pairs(block)` | `(k1, QKToK2[Q][k1])` | `(n2k1[k], n2k2[k])` |
| `_global_row(k1,k2)` | `k1` | `k1*nkpts + k2` |
| `_conserves_momentum` | always true | `k3 == kconserv[k1,k2,k4]` |

plus a `_done_index` that drops the redundant `k4` axis when `kp_sym` (within a momentum block `k4`
follows from `k3`). `run()` is a generator yielding one finished block at a time, so the k-point
path still streams to disk rather than holding every `L_Q` at once.

`generate_orbital_products` turned out to be *identical* between the two originals once expressed
over a `(k1, k2)` pair list — `kpoint.py` derived the pairs from `QKToK2[Q]` and `supercell.py` from
`n2k1`/`n2k2`, and the bodies were otherwise the same.

A single `_supercell_layout()` keeps the flag confined to the solver: it collapses a `kp_sym=False`
result's k-point axis into one combined basis and hands back the same constructor arguments any
k-point Hamiltonian takes.

Of the two entry points, `from_pyscf` raises for `comm.size > 1` with a pointer to
`write_from_pyscf`, which is the production path and the direct replacement for
`write_hamil_kpoints`. A test asserts their outputs are identical.

### Supercell output format (user call)

The decision — **a supercell Hamiltonian is the Γ point of the supercell** — and its consequences
are in DESIGN.md **Hamiltonian on-disk formats**. What that means concretely here: `nkpts=1`,
`nmo_pk=[nmo_tot]`, `QKTok2=[[0]]`, `MinusK=[0]`, `KPoints=[[0,0,0]]`, one `H1_kp0` of shape
`(nmo_tot, nmo_tot, 2)`, and one `L0` of shape `(1, nmo_tot**2 * nchol, 2)`.
`write_cholMat_singleWriter`, `write_h1`, `write_cholesky`, `write_info` and `write_one_body` are
not ported; they only ever produced the removed sparse format. Two knock-on cleanups:
`hamiltonian_format` loses the `periodic_dense` case, and `periodic.py` no longer imports anything
from `molecular.py`.

Verified against the C++ that `nkpts=1` is a legal k-point Hamiltonian:
`KPFactorizedHamiltonian.cpp` resolves `Q0_index` (`minusq(0)==0` and `qk_to_k2(0,0)==0`), the
`NMOPerKP` consistency loop is vacuous, `NMO == nbnd*nkpts` holds, and `L{Q}` is read as
`(nkpts, nbnd*nbnd*nchol)` complex with "normalization (1/sqrt(nkpts)) assumed to be included".
That normalization is the same `1/sqrt(nkpts_original)` factor the old sparse supercell writer
already applied (`write_cholMat_singleWriter`'s `fct`), so nothing changes numerically.

**Equivalence check:** the Γ layout was compared against the old supercell path by scattering
`supercell.Cholesky.run`'s output with afqmctools' own index expression from
`write_cholMat_singleWriter` (`ik2n[i,k1]*nmo_tot + ik2n[j,k2]`). Same `nchol` (103 on the diamond
2x1x1 fixture), same `(256, 103)` shape, **exact bitwise match**.

Parallel writing of a supercell factorization raises `NotImplementedError` — a settled decision,
not a gap; see DESIGN.md **Periodic Cholesky: one solver, one flag**.

### Lattice metadata in the model HDF5 file (resolves the Phase 2 open question)

Per DESIGN.md **Hamiltonian on-disk formats**, `LatticeHamiltonian.to_hdf5` writes the lattice's
metadata — not the `Lattice` object — under `Hamiltonian/ModelHamiltonian/Lattice`:

| dataset | shape | notes |
|---|---|---|
| `type` | scalar string | `'square'`, `'honeycomb'`, `'custom'`, … |
| `L` | `(ndim,)` int64 | `[L1, L2]` today, `[L1, …, LN]` later |
| `boundaries` | `(ndim,)` string | `'open'` / `'pbc'`, per axis |
| `twist` | `(ndim,)` float64 | boundary phase per axis, radians |
| `lattice_vectors` | `(ndim, ndim)` float64 | rows are `a1`, `a2`, … |
| `basis` | `(nb, ndim)` float64 | |
| `cyl_mode` | scalar string | `'none'` when unset |

`nbands` is written alongside it, since `dims[3]` only records `nsites * nbands` and the two cannot
otherwise be separated on read. `LatticeHamiltonian.lattice_params` flattens the metadata back into
today's `L1`/`L2`/`boundary1`/`boundary2` keys, so `Lattice.from_dict(ham.lattice_params)` rebuilds
the lattice — tested against site positions for every lattice type including `custom`. All of this
is additive: the AFQMC executable ignores groups it does not read.

## Phase 3b — ✅ Hamiltonian doc migration

- [x] Point docs, tutorials and examples at the new `HamiltonianBuilder` / `LatticeHamiltonian`
      (`builder.hamiltonian` -> `builder.get_hamiltonian()`, `io.write_model_hamiltonian(...)` ->
      `hamiltonian.to_hdf5(...)`, `MolecularHamiltonian.from_pyscf` in place of `write_hamil_mol`).
      Deferred out of Phase 3 because Phase 3's task list carried no doc-migration item; done here
      as its own phase.

**35 doc sources migrated**, in four groups:

- **Model / lattice** (13): `docs/examples/models/{01..08}`, `docs/tutorials/models/{04..08}`,
  plus `tutorials/models/overview.rst`.
- **Snippets** (13): `docs/snippets/01_setting_up/{01..09}/readme.rst`,
  `{11_custom_terms,12_rashba_soc}/run.py`, and the 11 `input*.toml` files.
- **Molecular** (13): `docs/tutorials/molecules/{03,05}`, `docs/examples/molecules/{01..08,tbd_*}`,
  `docs/installation.rst`, and the three `01_hello_*` tutorials (unused imports).
- **Reference**: `docs/afqmctools/lattice_models.rst` — the empty "Hamiltonian Builder" stub is
  filled in and a new "Serialization" section documents the `to_hdf5`/`from_hdf5` pair and
  `hamiltonian_format`; autodocs `safiretools.hamiltonian.base`, `.model.builder` and
  `.model.lattice_hamiltonian`. `docs/afqmctools/api/*.rst` still point at the old modules, which
  still exist — they go away in Phase 9 with the package (same call Phase 2 made).

### The blocker this phase had to clear (user call)

Three `isinstance(source, Hamiltonian)` gates rejected a `safiretools.LatticeHamiltonian`, and
~20 doc sources hand the Hamiltonian object straight to `free_electron(source=...)` or
`AutoHFHamiltonian(...)`. All three now accept either type — see DESIGN.md
**Accepting a `LatticeHamiltonian` in afqmctools and AutoHF**:

| file | note |
|---|---|
| `afqmctools/wavefunction/free_electron.py:127` | raised `ValueError` before |
| `afqmctools/inputs/from_autohf.py:65` | silently skipped the conversion before |
| `AutoHF/autohf/hamiltonian.py` | **separate submodule/repo** — needs its own commit |

### Changes beyond the mechanical swap

- **`nelec` moved into the `hamiltonian` input block.** It is Hamiltonian state now
  (`LatticeHamiltonian` records it, `to_hdf5` writes it into `dims`), not a write-time argument, so
  the 11 snippet TOML files carry `nelec` under `[hamiltonian]` instead of `[misc_params]`, and the
  Python doc sources put it in the params dict or pass it to `HamiltonianBuilder`. This also removed
  every `io.read_input_params(...)["misc_params"]["nelec"]` read from the docs. Verified harmless to
  the afqmctools path, which ignores unknown `hamiltonian` keys.
- **`HamiltonianComponent` is gone from the docs entirely** (user call): there is no reason for a
  user to build a component by hand. `snippets/.../11_custom_terms` and
  `examples/models/08_lieb_lattice_emery` now use `builder.custom_one_body(...)` — which also lets
  them drop the hand-stacking of the spin sectors, since the build step shapes the matrix itself —
  and the custom `Uij` component in `08` became `builder.onsite_hubbard(U)` with a per-site `U`
  array, which is exactly the documented site-dependent input convention. `HamiltonianComponent` is
  therefore **not** re-exported at the top level.
- **`to_hdf5()` now replaces the Hamiltonian, not the file** — see the follow-up section below.
  The docs therefore keep a Hamiltonian and a wavefunction in one file, in whatever order reads
  best, and the reference docs' "Serialization" section says so.
- **Dead kwargs removed from the molecular examples.** `walker_type=` and `with_soc=` were never
  parameters of `write_hamil_mol`, so every call site passing them raised `TypeError` as written:
  `04_V-fully_polarized` (`walker_type='fullypolarized'` — a spin symmetry that no longer exists;
  dropped, since the polarization lives in the wavefunction and the dense format records only the
  polarization count), `05_Li2_frozen_core` (`walker_type="closed"` -> the real `spin_symm="closed"`)
  and `06_Pb-spin-orbit` (`walker_type='noncollinear', with_soc=True` -> the real
  `spin_symm='noncollinear'`; SOC comes from `load_from_pyscf_chk_mol(soc_type='ecp')`, and
  `ortho_ao=True` was added to the second call, which omitted it and would now raise).
- **Build steps take their amplitude by keyword in the docs** (`nth_neighbor_hopping(t=...)`,
  `onsite_hubbard(U=...)`, `rashba_soc(rashba_lambda=...)`), which the decorator fix below made
  possible.
- **Typos/bugs the migration removed by construction**: `io.write_model_hamiltion` (7 snippet call
  sites plus `examples/models/08` and the reference page — the function never existed);
  `snippets/.../07`'s use of `io.` with no `io` import; `read_hamiltonian`/`write_hamiltonian_generic`
  imported but never called in 5 sources; and `lattice_models.rst`'s Python params dict missing the
  `nbands = 2` that its own TOML equivalent had.
- **`docs/tutorials/solids/04_computing_observables` deliberately stays on afqmctools' reader.** Its
  provided `hamil.h5` is in **CoQuí** format (`Interaction/Vq0`), which `hamiltonian_format()`
  identifies as `kpoint_coqui` but has no safiretools reader, so `Hamiltonian.from_hdf5` raises
  `NotImplementedError`. A comment in the cell records why.
- Prose that named afqmctools as the provider of the Hamiltonian framework now says safiretools;
  prose about the still-afqmctools CLI (`make_model_ham`) and about `write_pair_correlators` was
  left alone. A blanket package rename belongs to Phase 9.

### Verification

- **Doc-cell harness** (`nb_execution_mode = "off"`, so a sphinx build does *not* execute these
  cells and would not catch breakage): each doc's `{code-cell}` / `code-block:: python` blocks run
  cumulatively in one namespace, notebook style, and every block is classified by whether it touches
  the Hamiltonian API. Run against both the migrated tree and a pristine `git archive HEAD` baseline.
  On the identical 23-file set: **Hamiltonian-touching blocks that run clean went 18 -> 36, and
  failing ones 22 -> 18**, while cascading non-Hamiltonian failures fell 60 -> 29 (downstream cells
  stop dying on a Hamiltonian cell that never worked). Over the full 29-file sweep including the
  molecular docs: 47 clean. Every remaining failure was checked against the baseline and is
  pre-existing or a harness artifact — see **Known pre-existing breakage** below.
- **Syntax**: all 254 code blocks / scripts across the 35 migrated sources compile.
- **Molecular equivalence, byte-for-byte on the written HDF5**: `write_hamiltonian_generic(...)` vs
  `MolecularHamiltonian.from_integrals(...).to_hdf5(...)` — **dataset-for-dataset identical**;
  `write_hamil_mol(...)` vs `MolecularHamiltonian.from_pyscf(...).to_hdf5(...)` — **identical**, both
  plain and with `cas=`. Passing the true `nelec` (which the docs now do) changes exactly one
  dataset, `Hamiltonian/dims`, from `(0,0)` to the real electron counts.
- **Sphinx**: full build succeeds with the **identical 9-warning set as the pristine baseline**.
  Autodoc'ing `safiretools.hamiltonian.base`/`.model.builder`/`.model.lattice_hamiltonian` alongside
  the still-present afqmctools modules first introduced 6 ambiguous-cross-reference warnings
  (`Hamiltonian` x2, `Lattice` x4); fixed by qualifying the type in the four safiretools
  `lattice : Lattice` fields and the two afqmctools `Hamiltonian` fields. `free_electron`'s `source`
  docstring now also documents that it accepts a `LatticeHamiltonian`, which the gate change made
  true.
- **Tests**: 349 safiretools cases pass; full non-functional suite 613 passed, 1 failed —
  `tests/test_hamiltonian.py::TestSupercell::test_modified_cholesky`, the same pre-existing,
  unrelated failure Phase 3 recorded.

### Follow-up fixes (user call, done)

Three issues this phase surfaced were fixed rather than deferred. Each has regression tests in
`tests/safiretools/` (24 new cases; suite now 637 passed, same 1 pre-existing failure).

**1. `to_hdf5()` replaces the Hamiltonian, not the whole file.** It opened with mode `'w'`, so a
Hamiltonian written into a file that already held a wavefunction destroyed it — and writing both
into one file is the common case. Now every writer opens in append mode and deletes an existing
`Hamiltonian` group first, via `open_for_hamiltonian()` / `clear_hamiltonian()` in
`hamiltonian/base.py`. A SAFIRE input file holds at most one Hamiltonian and at most one
wavefunction, so replacing is the right semantics, and afqmctools' `write_wfn` already used exactly
this pattern for `Wavefunction` — the two are now symmetric, and order no longer matters.
Covers `LatticeHamiltonian`, `MolecularHamiltonian`, `PeriodicHamiltonian` and
`PeriodicHamiltonian.write_from_pyscf`; the per-rank Cholesky scratch files stay truncated, since
they only ever hold one run's partial blocks.

*Verified*: wavefunction survives a Hamiltonian written before or after it, for all three
subclasses; rewriting leaves no stale `ModelComponent_k` groups and reads back as the new
Hamiltonian; the file is created when absent. Under `mpiexec` at 1, 2 and 4 ranks the streaming
writer preserves a wavefunction, drops a stale Hamiltonian, and still produces its per-rank scratch
files. The `phdf=True` branch is reasoned about but **not executed**: this environment's `h5py` was
built without MPI support, which is pre-existing (the same call fails identically before the
change). The collective `del` is issued by every rank, which is what HDF5 requires.

**2. The build-step decorators are signature-preserving.** `skip_empty_params` and
`iterate_nth_order` declared `wrapper(self, params, *args, **kwargs)`, which renamed the first
parameter of every decorated step to `params`, so none of the documented amplitude names worked as
keywords: `builder.onsite_hubbard(U=4.0)` raised
`TypeError: ... missing 1 required positional argument: 'params'`. Both now bind the call against
the wrapped function's real signature (`inspect.signature(...).bind`, defaults applied, amplitude
taken by name), so every documented name works positionally *and* by keyword, and
`inspect.signature` reports the true signature for autodoc. Pre-existing, inherited from afqmctools.

This also unblocks **`rashba_soc` with per-order hopping**, which recursed with
`rashba_soc(rashba_lambda=..., t=tval, ...)` and so could never run for a list-valued `t` — the bug
behind `snippets/.../12_rashba_soc`. That snippet now builds and writes its Hamiltonian.

*Verified*: all nine decorated steps accept their documented keyword; keyword and positional calls
produce identical terms; zero/`None` amplitudes still skip; signatures report the real names.

**3. The dense reader told real from complex by rank, not by the trailing axis.** `_from_complex`
recognized SAFIRE's interleaved-complex layout as "the last axis has length 2", which is ambiguous
for a real `(2, 2)` `hcore` (`nmo == 2`) or a real Cholesky matrix with `nchol == 2`; those were
silently read as complex, and `MolecularHamiltonian._read_hdf5` then failed with
``hcore must be a square matrix, got shape (2,)``. `_to_complex` appends the axis, so the two
layouts differ in *rank* — which `dims` already pins down. Both helpers now take the expected real
rank (default 2, right for `hcore` and for the Cholesky matrix) and raise on anything else.
`periodic.py`'s `_deinterleave` got the same treatment for symmetry, though the k-point format is
complex throughout so it was never reachable there.

*Verified*: `nmo == 2` and `nchol == 2` now round-trip exactly; complex data is still recognized;
larger cases (`nmo` = 10, 18) unchanged; a malformed rank raises.

### Follow-up: API surface (user call, done)

Two decisions taken while reviewing the above; the rule behind them is recorded in DESIGN.md under
**"The top-level import surface defines what is user-facing"**.

**4. The interleaved-complex helpers are defined once, in `hdf5.py`** (DESIGN.md **Complex arrays on
disk**). There were **four** copies of the forward function and three of the backward one —
`hdf5.py`'s public pair (Phase 1, uncalled by anything in safiretools), plus private copies in
`molecular.py`, `periodic.py` (byte-identical bodies) and `lattice_hamiltonian.py`. All three
private copies are gone and every call site imports from `safiretools.hdf5`. `hdf5.py`'s old
version unconditionally viewed as complex and `ravel()`ed, which is why it carried a `shape=`
argument, now dropped in favour of `real_ndim`.

*Verified*: real and complex model Hamiltonians round-trip with ``data_`` at rank 1 and 2
respectively; the dense byte-equivalence checks against afqmctools still come out dataset-for-dataset
identical; new helper-level tests in `test_hdf5.py` cover both ranks, the ambiguous real shapes and
a rejected rank.

**5. The FCIDUMP I/O is user-facing** — **superseded by Phase 4h**, which moved these entry points
onto the Hamiltonian classes and took all six names back off the public surface. Recorded as it
stood — and re-exported at the top level: `read_fcidump`,
`read_fcidump_header`, `write_fcidump`, `write_fcidump_kpoint`, plus `h1_spat2spin`/`h2_spat2spin` —
those two because `write_fcidump_kpoint`'s own `NotImplementedError` tells the user to call them, and
a user-facing error message may not name something the user cannot import. The format internals
(`fcidump_header`, `check_sym`, `fmt_integral`) stay dev-facing. `tutorials/molecules/03` now says
`from safiretools import read_fcidump` instead of the deep module path.

`hamiltonian_format()` was **not** promoted, so the reference prose no longer names it; whether it
should be is queued in DESIGN.md **Future changes** along with shortening the four deep-path
docstrings, both of which need the top-level API page that Phase 8 owns.

### Known pre-existing breakage, confirmed against the baseline and left alone

Out of scope here (wavefunction is Phase 4, autoHF interop Phase 7), but all verified to fail
identically before this phase:

- `free_electron(..., output=...)` — `output` is not a parameter of `free_electron`. Breaks all 8
  `snippets/01_setting_up` readmes and both `run.py` snippets at the wavefunction step; their
  Hamiltonian steps now run clean.
- `snippets/.../06_twist_angle` references `input1.toml` / `input_charge.toml`; the directory holds
  `input.toml` and `input2.toml`.
- `examples/models/{02,03}` call `autohf.solver.lattice_hf`, which does not exist (`lattice_hf` is
  defined in `autohf/__init__.py`, and is itself deprecated there in favour of `solve_hf`).
- `examples/models/08`'s "Old Scratch work" Lieb cell passes a full `(nsites, nsites)` matrix as
  `t`, which is not a supported hopping input.
- `tutorial_utils` raises on `Path(None)` when `AFQMC_EXEC` is unset (the Phase 8 item).
- `docs/tutorials/models/04` documents the `make_model_ham` CLI, which DESIGN.md drops. Phase 6/9.

## Phase 4 — ✅ Wavefunction

- [x] `wavefunction/base.py`: `Wavefunction` ABC — `spin_symm`, `nelec`, `nmo`; implements
      `to_hdf5`/`from_hdf5` once (not per-subclass — format is identical across domains). Dispatch
      goes through a public `wavefunction_format(path)` returning `nomsd`/`phmsd`, mirroring
      `hamiltonian_format`; subclasses implement `_write_payload`/`_read_payload` rather than
      overriding the pair, so the shared header (`dims`, `ci_coeffs`, `Psi0_alpha`/`Psi0_beta`) is
      written in one place. `open_for_wavefunction`/`clear_wavefunction` mirror the Hamiltonian
      pair, so `to_hdf5` replaces the wavefunction rather than the file. The orthonormality
      helpers (`modified_gram_schmidt`, `is_orthonormal`) live here too.
- [x] `wavefunction/io.py`: native SAFIRE HDF5 schema read/write, built on `hdf5.py` (Phase 1),
      shared by both concrete subclasses below.
- [x] `wavefunction/nomsd.py`: `NOMSDWavefunction(Wavefunction)` (coeffs + per-determinant orbital
      matrices), with classmethods `.from_free_electron()`, `.from_pyscf()`, `.from_pbc_scf()`
      dispatching to the implementation modules below. `infer_spin_symm` replaces
      `_get_slater_type`.
- [x] `wavefunction/phmsd.py`: `PHMSDWavefunction(Wavefunction)` (coeffs + occa/occb), with
      `.from_dice()` dispatching to `dice.py`, plus `.from_pyscf_cas()` and `.from_pbc_scf()` —
      see **Classmethods beyond the listed set**.
- [x] `wavefunction/free_electron.py`: `from_free_electron()` implementation; `model.py`'s legacy
      duplicate (`make_free_elec`/`write_free_electron_wfn`) retired. **The AutoHF variational-energy
      call is gone entirely** (user call): it is the wrong pathway for measuring an energy, and it
      would have made `wavefunction/` depend on the AutoHF interop that Phase 7 owns.
      `measure_evar`, `measure_spin` and `return_autohf` go with it.
- [x] `wavefunction/pyscf.py`: `from_pyscf()` implementation, ported from `wavefunction/mol.py`.
      `write_cas_wfn` came along as `from_pyscf_cas()`, and `ci_wavefunction`/`read_cas_meta`
      were ported out of `afqmctools/utils/pyscf_utils.py` (as `ci_expansion`/`read_cas_meta`) so
      that nothing in safiretools imports upward into afqmctools.
- [x] `wavefunction/pbc.py`: `from_pbc_scf()` implementation. Returns whichever representation the
      occupancies call for; the two classmethods pick their `ndet_max` and validate the result.
      `write_wfn_pbc_old` and `slater_gto2mo` are not ported — see **Dropped**.
- [x] `wavefunction/dice.py`: `from_dice()` implementation, split out of
      `wavefunction/converter.py`, with proper typed exceptions (not the current error handling).
      One entry point detects HDF5 vs. text output. `read_wavefunction`/`read_nomsd_hdf5`/
      `read_phmsd_hdf5`/`orbs_from_dset` from the same module are replaced by
      `Wavefunction.from_hdf5`.
  - [x] Test: length-based dispatch today (`len(wfn)==2` vs `==3` in `write_wfn`) is fully replaced
        by the `NOMSDWavefunction`/`PHMSDWavefunction` type split — no remaining ad hoc length
        checks anywhere in the ported code. Checked both as a source scan over
        `safiretools/wavefunction/*.py` and behaviorally: a bare matrix or an occupation array
        handed to `NOMSDWavefunction` raises rather than being reinterpreted.
- [x] Test: one round-trip test per construction path (`from_free_electron`, `from_pyscf`,
      `from_pbc_scf`, `from_dice`) through `to_hdf5`/`from_hdf5`. `from_pyscf_cas` has one too.
- [x] Re-export `Wavefunction`, `NOMSDWavefunction` and `PHMSDWavefunction` from the top-level
      `__init__.py` (pulled forward from Phase 8, as Phase 2 did for `Lattice`).

Tests live in `tests/safiretools/wavefunction/` (`test_wavefunction_base.py`, `test_io.py`,
`test_nomsd.py`, `test_phmsd.py`, `test_free_electron.py`, `test_pyscf.py`, `test_pbc.py`,
`test_dice.py`) — 199 new cases, 583 safiretools cases in total, all passing. Full non-functional
suite afterward: 877 passed, 1 failed — `tests/test_hamiltonian.py::TestSupercell::
test_modified_cholesky`, the same **pre-existing, unrelated** failure Phase 3 and 3b recorded.
`test_wavefunction_base.py` is named that way, not `test_base.py`, because
`tests/safiretools/hamiltonian/test_base.py` already exists and neither test tree uses
`__init__.py` — the same collision Phase 1 hit with `test_stats_core.py`.

> Verification ran on the python 3.12 environment only. The 3.13 environment on this machine can
> no longer collect the suite (no `toml`, no `pyscf`), so the dual-version check Phase 3 did was
> not repeated.

### The fully-polarized decision (user call)

The decision is in DESIGN.md **Spin-symmetry enum**: no beta electrons is `COLLINEAR` with
`ndown == 0`, written with zero-width beta blocks. afqmctools instead wrote `dims[3] = 4` with
single-polarization blocks. The executable's readers require the beta blocks to exist for a
collinear file (`getInitialGuess` opens `Psi0_beta` whenever `nspin_in_guess == 2`, and
`read_nomsd_wavefunction` opens `PsiT_{2*idet+1}`).

The Python side is complete per the call. At the time it was written afqmctools could not build the
corresponding molecular case at all (`_make_slater`'s fourth branch indexed `mo_coeffs[0,:,:]`,
raising `IndexError` for the 2-D `mo_coeff` an ROHF reference has), and a regression test recorded
that; the branch is gone from afqmctools too after the sync described below, so
`test_afqmctools_could_not_build_that_case` was **removed** — see **Reconciled with the C++ sync**.

> **The C++ side agrees, as of the sync with `main` (2026-09).** A collinear walker set accepts an
> empty beta sector, and `tests/test_polarized_consistency.cpp` writes the same layout safiretools
> does. Details in **Reconciled with the C++ sync** at the end of this phase.

### Equivalence with afqmctools

Checked with throwaway harnesses comparing the ported code against the original, dataset for
dataset on the written HDF5 files. Every comparison below is **exact** unless noted.

- **`write_wfn` itself** — 5 NOMSD cases (closed 1- and 2-determinant, collinear 1- and
  3-determinant, noncollinear) and 2 PHMSD cases (occupation-only, and with two orbital
  references): **6/7 exact**, the seventh differing in exactly one dataset, `type`, which is the
  documented afqmctools bug below.
- **Free electron** — 3 cases (4x4 Hubbard collinear, 4x1 two-band Kanamori noncollinear, and a
  polarized 4x4 with `nelec = (5, 0)`): the two integer-occupancy cases match dataset for dataset;
  the polarized case matches on every dataset afqmctools writes (`dims[3]` aside) and adds the
  zero-width beta blocks described above.
- **Molecular** — `write_wfn_mol` vs `NOMSDWavefunction.from_pyscf(...).to_hdf5(...)` matches
  dataset for dataset for a closed-shell Ne/STO-3G RHF and an open-shell O/STO-3G ROHF reference.
  `write_cas_wfn` vs `PHMSDWavefunction.from_pyscf_cas(...)` agrees on `dims`, `occs` and
  `ci_coeffs` for a Ne/6-31G CASSCF(4o,4e).
- **Periodic** — `write_wfn_pbc` vs `NOMSDWavefunction.from_pbc_scf(...)` matches for a diamond
  2x1x1 KRKS closed-shell reference, the same data presented as collinear, and a collinear
  reference with two partially occupied bands (`ndet_max = 1`). At `ndet_max = 4` the PHMSD output
  matches on every dataset except `type`, again the documented bug. **Psi0 agrees to 1e-8, not
  exactly**: afqmctools applied its 1e-8 sparsification threshold to the whole determinant array in
  place and so thresholded the dense `Psi0` too, which safiretools does not — the threshold exists
  to sparsify the stored `PsiT` blocks, and perturbing the initial walker is a side effect.
- **Dice** — `read_dice_ascii_wavefunction` and `read_dice_h5_wavefunction` agree on coefficients,
  `occa`, `occb`, `nmo` and the electron counts; the one difference is the `nmo` fix below.
- **End to end** — `build/Release/bin/safire` runs to completion on a safiretools-written
  Hamiltonian *and* wavefunction sharing one file, for a collinear 4x4 Hubbard model at half
  filling and a noncollinear 4x1 two-band Kanamori model.

### Bugs found while porting

Each changes behavior, so each is listed rather than folded in silently.

- **`write_phmsd`'s `type` under-counted its references.** It wrote `fh5['type'] = 1` whenever *any*
  orbital matrix was passed, then wrote `PsiT_0` *and* `PsiT_1` when `nb > 0`. The executable reads
  `type == 1` as "one reference" (`PsiT.resize(1)` in `read_ph_wavefunction_hdf`) and takes the
  combined RHF/GHF branch, so the beta reference was written and then silently ignored. `type` now
  counts the references actually written — 0, 1 or 2 — and `PHMSDWavefunction.nreferences` is the
  in-memory equivalent. **This is the single dataset the PHMSD equivalence checks disagree on.**
- **`write_phmsd` crashed on a single reference.** With `orbmat=(A, None)` — which is exactly what
  `write_wfn_pbc` passes for a closed-shell reference — the `if nb > 0` branch dereferenced the
  `None` beta matrix. A `None` entry now simply does not count as a reference.
- **`read_dice_ascii_wavefunction` reported `nmo` one too large for complex output.** It computed
  `nmo = len(data) - 2` *before* deciding whether the coefficient occupied one column or two, so
  the imaginary part was counted as an orbital. It is `len(fields) - first_occupation` now.
- **`free_electron` mutated its caller's parameter dict**, injecting `twist` into
  `source["lattice"]` in place. `from_free_electron` copies first, per DESIGN's no-mutation
  convention; a regression test pins both behaviors.
- **`_get_slater_type` read an equal-population collinear matrix as closed-shell.** It tested
  `nelec[0] == nelec[1]` alone, so a `(nmo, nup+ndown)` UHF-style matrix with `nup == ndown` was
  classified `CLOSED` and its beta columns dropped. `infer_spin_symm` uses the column count, which
  separates the two unambiguously.
- **A spin-independent one-body term produced an empty beta channel.** `_collinear_free_elec`
  sliced `Hfree[:Nmo,:]` / `Hfree[Nmo:,:]` unconditionally, so an `(nmo, nmo)` one-body term gave a
  `(0, nmo)` beta block. `_collinear_blocks` uses such a term for both channels, and raises for any
  other shape.
- **`_reoccupy` raised `TypeError` on `ndet_max=None`** — the value afqmctools' own
  `pyscf_to_afqmc` CLI passes when `-n` is not given — via `min(len(occs_a), None)`. None now means
  "every determinant the degeneracy allows".
- **A partially occupied closed-shell reference failed deep inside.** `reoccupy`'s non-UHF branch
  divides the occupancies by two, which is a list of per-k arrays in the partial-occupancy case
  (`TypeError`), and the `nocc = 2*nocc` doubling is only right for integer occupancies anyway.
  `from_pbc_scf` now raises a `ValueError` naming the limitation and pointing at the collinear
  presentation; both behaviors are pinned by tests.
- **`rediag_fock`'s association matters.** `X^H (F X)` and `(X^H F) X` differ in the last bits, and
  that is enough to rotate the eigenvectors of a degenerate subspace by ~1e-6. The port keeps
  afqmctools' grouping, which is what makes the periodic equivalence checks exact. Noted in the
  function, since it looks like a stylistic detail and is not.

Dropped as dead, unreachable, or superseded:

- `write_wfn_pbc_old` — superseded by `write_wfn_pbc`, which differs only in reading `walker_type`
  through the enum coercion instead of comparing it raw.
- `slater_gto2mo` — no callers anywhere, and unable to run as written: its `NONCOLLINEAR` branch
  falls off the end returning `None`, and its `'mol'` path reads `kwargs['cell']`. Recorded here
  rather than ported; the same conversion is what `from_pyscf`'s `_transform_slater` does.
- `check_orthonormality` (the `wfn[0]`-indexing variant) — superseded by `is_orthonormal`, which
  works on one Slater matrix and treats a zero-column block as orthonormal instead of raising
  inside `np.max`.
- `_SlaterType`, `_slater_enum_map`, `_slater2dims`, `_get_slater_type`, `_is_closed`,
  `_is_collinear`, `_is_noncollinear`, `_is_fully_polarized` — replaced by `SpinSymm` plus
  `infer_spin_symm`, per DESIGN's **Spin-symmetry enum**.
- `print_eigenvalues`' `verbose < 2` truncation — the MO energy table is logged at DEBUG now, up to
  the highest occupied orbital.
- `generate_orbitals`' `foll_mo_b = full_mo_b` — a typo'd no-op assignment in a branch whose
  `full_mo_b` is never read.

Convention alignment per DESIGN.md, behavior-preserving: `print()` → `logging` throughout
(including the free-electron shell-filling report, which lost its `verbose` parameter);
`sys.exit()` in `generate_orbitals` and `determine_occupancies` → `ValueError`; every `assert(0)`
in the Dice readers → `ValueError` with a message a caller can act on.

### Classmethods beyond the listed set

DESIGN lists `.from_dice()` as `PHMSDWavefunction`'s only factory. Two more were needed, and both
are genuinely particle-hole constructions rather than new domains:

- **`PHMSDWavefunction.from_pyscf_cas()`** — `write_cas_wfn`'s replacement. It has real callers
  (`docs/examples/molecules/03_n2-phmsd`, `07_3d_TMO_benchmark`, `tbd_H2O_charge_density`, and
  `tests/workflows/test_neon.py`), and DESIGN's rule is that caller-count is never on its own a
  reason to drop something.
- **`PHMSDWavefunction.from_pbc_scf()`** (user call) — `write_wfn_pbc` produces a particle-hole
  expansion, not a NOMSD, whenever bands are partially occupied and more than one determinant is
  asked for, so the periodic path genuinely spans both representations. The machinery is shared in
  `pbc.py`; `NOMSDWavefunction.from_pbc_scf` forces `ndet_max=1` (always a single determinant,
  which is what afqmctools' default did), and `PHMSDWavefunction.from_pbc_scf` raises a
  `ValueError` naming the other classmethod when the occupancies are integer.

### Design choices worth recording

All of these are now in DESIGN.md and are not repeated here: **A wavefunction's determinants carry
one block per independent spin channel** (covers both the `dets` shape and
`nelec`/`nelec_on_disk`), **Writing never orthonormalizes**, and — in **Public API patterns** — the
default-`psi0` warning being NOMSD-specific.

Producer-side note for the first: `_make_slater`, `_collinear_free_elec`,
`_noncollinear_free_elec` and `create_wavefunction` all already built exactly that shape. For the
second: afqmctools raised the "UHF initial walker" warning from `write_nomsd` only, which is where
the `_warn_about_default_psi0` hook comes from.

### Preserved, not endorsed

`_reoccupy`'s ascending `argsort` keeps the **least** probable determinants — carried over verbatim
and **queued for a decision** in DESIGN.md **Things we might change**, since fixing it changes
numerics on a path with established behavior and would break the `ndet_max=4` periodic equivalence
check.

### Reconciled with the C++ sync (2026-09)

Phases 1–4 were written against a C++ tree that predated the removal of the fourth spin symmetry,
and several notes here described that older tree. Re-checked against the synced C++; the format
needed no change, and one obsolete test was removed.

**What the C++ now does, and how safiretools lines up:**

| checked | C++ | safiretools |
|---|---|---|
| spin-symmetry values | `WALKER_TYPES` is `CLOSED=1`, `COLLINEAR=2`, `NONCOLLINEAR=3` ([config.h:48](../../src/AFQMC/config.h#L48)) | `SpinSymm` has the same three, same values ✅ |
| a legacy `dims[3] == 4` file | `initWALKER_TYPES(4)` aborts with `Unsupported walker type dims[3]==4. A wavefunction with no beta electrons is COLLINEAR (dims[3]==2) with ndown=0; regenerate it that way.` | `SpinSymm.from_input(4)` raises `ValueError: Unknown spin symmetry 4: supported values are [1, 2, 3] …` ✅ neither names the removed type |
| polarized NOMSD layout | `derive_polarized_wfn` in [tests/test_polarized_consistency.cpp](../../tests/test_polarized_consistency.cpp) writes `dims = [NMO, nup, 0, 2, 1]`, `Psi0_alpha (NMO, nup)`, `Psi0_beta (NMO, 0)`, `PsiT_0 (nup, NMO)` CSR, `PsiT_1 (0, NMO)` CSR | byte-for-byte the same shapes out of `write_header`/`write_orbitals` ✅ |
| empty beta sector at runtime | supported: `walker::SlaterMatrix(Beta)` returns a zero-width view, zero-extent beta ops are skipped where they trap on GPU, and `polarized: consistency with up-only noncollinear` pins the numerics | the `ndown == 0` decision no longer has a C++ caveat ✅ |
| collinear beta blocks still required | `getInitialGuess` reads `Psi0_beta` for any `COLLINEAR` file, at its in-file width; `read_nomsd_wavefunction` opens `PsiT_{2*idet+1}` and `check_shape`s it against `ndown` | zero-width blocks are written, not omitted ✅ |
| PHMSD spin symmetry | `read_ph_wavefunction_hdf` still rejects `CLOSED` and `NONCOLLINEAR` ([readWfn.cpp:126](../../src/AFQMC/Utilities/readWfn.cpp#L126)) | `PHMSDWavefunction` is always `COLLINEAR` ✅ |
| PHMSD `type` | reads 0, 1 **and 2**, resizing `PsiT` to 2 for `type == 2` ([readWfn.cpp:140](../../src/AFQMC/Utilities/readWfn.cpp#L140)) | Phase 4's fix — `type` counts the references actually written ✅ |

**The one change this needed.** `tests/safiretools/wavefunction/test_pyscf.py::TestFromPyscf::
test_afqmctools_could_not_build_that_case` asserted `IndexError` from
`afqmctools.wavefunction.mol.generate_wavefunction` for a frozen core that empties the beta channel.
afqmctools' fourth `_make_slater` branch is gone too, so `_make_slater_collinear` now handles the
2-D ROHF `mo_coeff` and the call succeeds — the test failed on premise, not on safiretools code, and
was **deleted**. `test_a_frozen_core_can_empty_the_beta_channel` right above it still covers the
safiretools behavior.

**Two C++/doc fixes came out of this pass** (user call), since neither could be made from the Python
side:

- **`initWALKER_TYPES`'s legacy-file abort no longer names the removed type**
  ([config.h:66](../../src/AFQMC/config.h#L66)). It still says what to do — regenerate as
  `COLLINEAR` with `ndown=0` — so nothing actionable was lost, and the two implementations now agree
  on saying nothing about the old value. Two comments in
  `tests/test_polarized_consistency.cpp` that called `COLLINEAR`/`ndown=0` "the target that replaces
  the legacy FULLYPOLARIZED" were reworded the same way.
- **`docs/user_manual/wavefunctions.rst` documented `/Wavefunction/PHMSD/type` as "integer 0/1"**,
  which was stale against *both* sides: the executable reads 0, 1 and 2, and Phase 4 made the writer
  emit a real reference count. It now documents all three values and what each writes.

## Phase 4b — ✅ Wavefunction doc migration

Deferred out of Phase 4 the way Phase 3b was out of Phase 3 (user call): Phase 4's task list
carried no doc-migration item, and the wavefunction API appears in ~35 doc sources.

- [x] Point docs, tutorials and examples at the new `NOMSDWavefunction` / `PHMSDWavefunction`
      (`free_electron(...)` -> `NOMSDWavefunction.from_free_electron(...)` + `.to_hdf5(...)`,
      `write_wfn_mol(...)` -> `NOMSDWavefunction.from_pyscf(...).to_hdf5(...)`,
      `write_cas_wfn(...)` -> `PHMSDWavefunction.from_pyscf_cas(...).to_hdf5(...)`,
      `write_wfn_pbc(...)` -> `NOMSDWavefunction.from_pbc_scf(...)`,
      `read_wavefunction(...)` -> `Wavefunction.from_hdf5(...)`).
      `write_wfn_pbc` turned out to have **no doc callers at all** — the periodic wavefunction path
      is untouched by the docs, so that mapping was unused.
- [x] The AutoHF variational-energy report that `free_electron(measure_evar=True)` printed has no
      replacement in safiretools. **Decided (user call): split by whether the number is consumed.**
      The 8 `snippets/01_setting_up` readmes only *printed* it, so their sample-output block and the
      "Internally, the AutoHF Hartree-Fock code is used..." paragraph are dropped and replaced by a
      pointer to `:ref:`setup_ex_9``, which already shows the explicit `lattice_hf` route.
      `examples/models/{01,07}` *consume* `return_autohf`'s results downstream (charge/spin-density
      plots), so they gain an explicit
      `lattice_hf(AutoHFHamiltonian(source=...), settings=dict(ansatz='SD', steps=-1, ...),
      initial_guess=wfn.dets[0], suppress_logo=True)` call — the same arguments `free_electron`
      passed internally, and the same style `snippets/09` and `examples/models/08` already use.
      This uses AutoHF directly and so does **not** pre-empt the Phase 7 safiretools interop.
- [x] `free_electron(..., output=...)` — `output` was never a parameter, so this breaks all 8
      `snippets/01_setting_up` readmes and both `run.py` snippets at the wavefunction step today
      (recorded under Phase 3b's **Known pre-existing breakage**). The migration removes it by
      construction, since serialization is `to_hdf5(path)`. **All 10 now run clean** (0/10 -> 9/10;
      the one exception is snippet 06's orphan block, below).
- [x] `docs/examples/molecules/04_V-fully_polarized` was left on afqmctools' `write_wfn_mol` with a
      comment saying it waits on a C++ walker item — the same treatment
      `docs/tutorials/solids/04_computing_observables` got in Phase 3b.
      **That reason expired with the C++ sync, and the example is ported now** — see
      **Phase 4e** below.

**37 doc sources migrated**, in five groups:

- **Snippets** (12): `snippets/01_setting_up/{01..08}/readme.rst` and
  `{11_custom_terms,12_rashba_soc}/{run.py,output.txt}`.
- **Model examples** (6): `examples/models/{01,02,03,05,07,08}`.
- **Model tutorials** (2): `tutorials/models/{06_writing_a_trial_wavefunction,08_computing_observables}`.
- **Molecular examples** (9): `examples/molecules/{01_O-atom (ham.py + index.rst), 03_n2-phmsd,
  05_Li2_frozen_core (script + index.rst), 06_Pb-spin-orbit, 07_3d_TMO_benchmark,
  08_local_embedding, tbd_H2O_charge_density}`, plus the comment in `04_V-fully_polarized`.
- **Molecular / solids tutorials + reference** (8): `tutorials/molecules/{03,04,05}`, the three
  `01_hello_*` tutorials (unused `write_wfn` imports dropped), and `docs/afqmctools/afqmctools.rst`
  (prose that still credited afqmctools with the lattice/Hamiltonian/wavefunction tooling).

No new reference page was added: no reference doc referenced the old wavefunction API in prose, and
the `docs/afqmctools/api/*.rst` autodoc stubs go away in Phase 9 with the package. Autodoc'ing the
wavefunction classes belongs with the top-level API page **Phase 8** owns (DESIGN.md
**Future changes**).

### Changes beyond the mechanical swap

- **`snippets/.../06_twist_angle`'s first block could never have run.** It read
  `input_params["lattice"]` without ever defining `input_params`, and named `input1.toml` in a
  directory holding `input.toml`/`input2.toml`. Both had to be fixed to write a working
  `from_free_electron` call, so the block now does `input_params = toml.load("input.toml")` and
  passes `twist=input_params["lattice"].get("twist", None)`. **The explicit `twist=` is load-bearing,
  not decoration** — see DESIGN.md **Things we will change**, where the underlying
  `DEFAULT_TWIST` override is queued for a fix. (Verified that a string twist —
  `["0 Pi", "7/8 Pi"]` — survives `np.asarray` into `Lattice._parse_twist`.)
- **`tutorials/models/06`'s GHF cell built `sd_ghf` and then wrote `sd_uhf`.** afqmctools accepted
  the mismatched matrix silently; `NOMSDWavefunction` validates the shape against `nelec`/`spin_symm`
  and raises, so the cell now passes `sd_ghf` as it always intended. The multi-Slater cell below it
  also rebound the name `sd_uhf` to a GHF-shaped array; it is `sd_multi` now.
- **`tutorials/models/06`'s `init=` section became `psi0`.** `write_wfn(init=...)` took an array with
  an extra leading axis; `Wavefunction.psi0` is a sequence with **one Slater matrix per independent
  spin channel**, which is a different shape, so the paragraph was rewritten rather than renamed and
  a short illustrative snippet added.
- **`examples/models/{01,07}` build the twisted Hamiltonian explicitly.** `free_electron` rebuilt it
  internally from the parameter dict, which was invisible; now that the same Hamiltonian must also
  reach `AutoHFHamiltonian`, the docs build it with `Lattice.from_dict(dict(lattice_params,
  twist=twist))` + `HamiltonianBuilder.from_input(...)`. This makes visible the point
  `examples/models/01`'s own prose was already making ("the Hamiltonian will typically need to be
  rebuilt"). `examples/models/07` gained a comment recording that its trial wavefunction is
  deliberately built from the *un-pinned* Hamiltonian.
- **`examples/models/02`'s trial-density loop dropped a reshape trick.**
  `wfn.reshape(N_sites, na, 2, order='F')` (to split the collinear determinant into spin blocks)
  became `alpha, beta = trial.spin_blocks(0)`, which is the public accessor for exactly that. The
  two are equivalent — the Fortran-order reshape maps column `j = p + na*s` to `(p, s)` — but only
  one of them says so.
- **`examples/models/08`'s annealing loop uses `orthonormalize()`.** It called
  `modified_gram_schmidt` directly, which is not on safiretools' top-level import surface;
  `NOMSDWavefunction.orthonormalize()` is the public equivalent, so no promotion was needed
  (DESIGN.md **The top-level import surface defines what is user-facing**).
- **The two `run.py` snippets enable logging and their `output.txt` was regenerated from a real
  run.** Their recorded output was the old `print()` progress plus the AutoHF energy; under DESIGN's
  `print()` -> `logging` convention a standalone script prints nothing by default. Each script now
  opens with `logging.basicConfig(level=logging.INFO, format="%(message)s")` and `output.txt` is the
  actual captured output (term-by-term build log, eigenvalues, shell filling).
- **`tbd_H2O_charge_density`'s unused `write_cas_wfn` import** was swapped to `PHMSDWavefunction`
  rather than deleted — it is a stub example, and the import marks where the author intended to use
  it. The three `01_hello_*` tutorials' unused `write_wfn` imports *were* deleted, matching Phase 3b.

### Verification

- **Doc-cell harness** (`nb_execution_mode = "off"`, so a sphinx build does *not* execute these and
  would not catch breakage): each doc's `{code-cell}` blocks — and each `.rst`'s
  `.. code-block:: python` — run cumulatively in one namespace, notebook style, in a scratch cwd,
  with each block classified by whether it touches the wavefunction API. Run against both the
  migrated tree and a pristine `git archive HEAD` baseline over the same 31 sources.
  **Wavefunction-touching blocks that run clean went 40 -> 51, and failing ones 22 -> 8, with zero
  blocks that passed on the baseline failing after the migration.** Plain ```` ```python ```` fences
  in markdown are *not* executed (myst-nb runs only `{code-cell}`), so the harness skips them too.
  Two environment fixes were needed for the numbers to mean anything: `AFQMC_EXEC` must be set or
  `tutorial_utils` raises `Path(None)` at import and takes every downstream cell with it (the
  Phase 8 item), and `utils/AutoHF` must be ahead of the venv's installed `autohf`, which predates
  the Phase 3b `LatticeHamiltonian` shim.
  A follow-up run with each doc's `files/` directory staged raises
  `tutorials/molecules/03` to **6/6 on both trees** (its FCIDUMP block only lacked its data file).
- **The `examples/models/02` read path was checked directly**, since the harness cannot reach it
  through the doc: `Wavefunction.from_hdf5` on the free-electron `afqmc.h5` the doc itself writes
  returns a `NOMSDWavefunction` whose `spin_blocks(0)` stacks to `(2, 40, 16)` — exactly the
  2 x nsites x nelec array the downstream `greens_1body(o)` wants. The loop's later iterations fail
  only because `afqmc_1.h5` contains a `Hamiltonian` and no `Wavefunction`, the pre-existing
  `autohf.solver.lattice_hf` cell having never run.
- **End to end through the executable**: `examples/molecules/01_O-atom` was run in full — `scf/scf.py`
  (PySCF ROHF + UHF), then the migrated `ham/ham.py`, then `build/Release/bin/safire` on its
  `afqmc/afqmc.json`. The Hamiltonian and the wavefunction land in one `afqmc.h5` and read back as
  `MolecularHamiltonian` (nmo 14) and `NOMSDWavefunction` (collinear, nelec (5, 3), dets
  `(1, 14, 8)`); SAFIRE loads both, reports `Wavefunction type: NOMSD`, and runs to completion at
  E ≈ -74.83 Ha against a UHF reference of -74.792 Ha.
- **Sphinx**: full build succeeds with the **identical 9-warning set as the pristine baseline**.
- **Tests**: 584 safiretools cases pass. This phase changed no library code.

### Could not be checked, and why

- `examples/molecules/07_3d_TMO_benchmark/06_TMO_benchmark.py` — a benchmark that runs CASSCF on
  transition-metal oxide dimers; it does not complete in a reasonable interactive budget. Its
  `PHMSDWavefunction.from_pyscf_cas(...).to_hdf5(...)` call is the same one
  `examples/molecules/03_n2-phmsd` exercises and that Phase 4 covers with tests.
- `examples/models/02_stripes`' final two wavefunction blocks — they consume `afqmc_{Ueff}.h5` files
  written by a cell that calls `autohf.solver.lattice_hf`, which does not exist (pre-existing,
  recorded under Phase 3b). The read itself is fine: it was checked directly (above), and
  `tutorials/models/08` performs the same `Wavefunction.from_hdf5` on an `autohf_to_afqmc`-written
  file and passes.
- `examples/models/08_lieb_lattice_emery`'s "Old Scratch work" section — pre-existing breakage
  (it passes a full `(nsites, nsites)` matrix as `t`), recorded under Phase 3b. Migrated off the old
  API but still not runnable.
- `snippets/.../06_twist_angle`'s **second** code block — a copy-paste orphan from snippet 05: it
  reads `input_charge.toml`, a file that exists only in `05_pinning_field`. Pre-existing; migrated
  in place and its stale sample output removed, but left otherwise alone since deleting the block is
  a content decision, not part of this migration. **Flagged for a decision.**
- `docs/examples/molecules/04_V-fully_polarized` — not migrated in this phase (see above); done in
  **Phase 4e**. Sphinx also reported a pre-existing `literalinclude` typo there (`index.rst`
  included `input/setup.py` where the directory is `inputs/`), which the user fixed during
  Phase 4c — the docs build is 8 warnings now rather than the 9 this phase was measured against.
  The matching **prose** typo was fixed in Phase 4e.

## Phase 4c — ✅ Base-class factory dispatch (Hamiltonian half later reverted — see Phase 4g)

Every construction factory is reachable from the ABC, which picks the concrete subclass
(user call). See DESIGN.md **Where a factory lives** for the decision, the
dispatch keys, and why `Wavefunction` needs this more than `Hamiltonian` does. **Subclass factories
stay** — they are aliases, plus two genuine narrowing forms. Docs are deliberately **not** part of
this phase; they are Phase 4d.

> ⚠️ **The `Hamiltonian` half of this phase was reverted in Phase 4g** (team call, 2026-09-09).
> The `Wavefunction` half stands as written below. Read the `Hamiltonian` checkboxes here as a
> record of what was built and then undone, not of the current code.

Both `from_hdf5` methods already dispatched and needed no work.

- [x] `Wavefunction`: added `from_free_electron`, `from_pyscf`, `from_pyscf_cas`, `from_dice`,
      `from_pbc_scf`, each delegating to the free function in
      `wavefunction/{free_electron,pyscf,pbc,dice}.py` via an import inside the method body. Real
      signatures, not `**kwargs` — there is no clash.
  - [x] `Wavefunction.from_pbc_scf` passes `ndet_max` straight through and returns whichever
        representation the occupancies call for; it does **not** force or validate. It keeps the
        implementation's own `ndet_max=1` default, so the expansion stays opt-in.
  - [x] Test: each returns the documented concrete type and agrees with the subclass alias.
  - [x] Test: `Wavefunction.from_pbc_scf` returns `NOMSDWavefunction` for integer occupancies and
        `PHMSDWavefunction` for partially occupied bands with `ndet_max > 1`.
- [x] `Hamiltonian`: added `from_dict` (lattice), `from_integrals` (molecular), `write_from_pyscf`
      (periodic) as fixed-domain delegations, and `from_pyscf` as the one real dispatcher.
  - [x] `Hamiltonian.from_pyscf` dispatches on **key presence**: `'cell'` -> `PeriodicHamiltonian`,
        `'mol'` -> `MolecularHamiltonian`. Not `isinstance` — `pyscf.pbc.gto.Cell` subclasses
        `gto.Mole`, so a type test reports a periodic cell as molecular. Neither key, or both,
        raises `ValueError` naming what it found.
  - [x] It forwards `**kwargs` rather than merging the two signatures, so a molecular call passing
        `kpoint_symmetry=` raises `TypeError` from the concrete classmethod instead of being
        silently ignored. Docstring carries both parameter lists.
  - [x] Test: a molecular and a periodic `scf_data` each reach the right subclass and agree with the
        subclass factory.
  - [x] Test: neither key / both keys raise `ValueError`; an unknown keyword raises `TypeError`
        naming the real parameter.
- [x] Kept every subclass factory *call* working, and its tests. Two are genuine subclass
      definitions rather than inherited aliases, and their docstrings say so:
      `NOMSDWavefunction.from_pbc_scf` forces `ndet_max=1`, and `PHMSDWavefunction.from_pbc_scf`
      raises when a single determinant would describe the system exactly.
- [x] Re-exported nothing new — `Hamiltonian` and `Wavefunction` were already top-level.

### The aliases are inherited, not duplicated

The one design refinement over the plan above, now recorded in DESIGN.md
**Where a factory lives**. Concretely for this phase: the four fixed-answer
subclass classmethods (`NOMSDWavefunction.from_free_electron`/`.from_pyscf`,
`PHMSDWavefunction.from_dice`/`.from_pyscf_cas`) were **deleted**, not kept alongside a base copy,
and a test pins that by asserting
``NOMSDWavefunction.from_pyscf.__func__ is Wavefunction.from_pyscf.__func__`` and that the name does
not appear in the subclass's own ``__dict__``.

The guard message a mismatched call now gets:
``ValueError: from_free_electron always builds a NOMSDWavefunction, which is not a
PHMSDWavefunction; call it on NOMSDWavefunction or on the dispatching Wavefunction`` — better than
the `AttributeError` it used to give.

The same trick does **not** work on `Hamiltonian`: a base `from_dict` inherited by the subclass that
implements it would recurse, so the base delegates and forwards `**kwargs` instead.

### Verification

- 32 new cases (`tests/safiretools/hamiltonian/test_base.py::TestFactoryDispatch` and
  `::TestFactoryDispatchPyscf`, `wavefunction/test_wavefunction_base.py::TestFactoryDispatch`,
  `wavefunction/test_pbc.py::TestBaseClassDispatch`). **616 safiretools cases pass**, up from 584.
- Full non-functional suite: **880 passed, 1 failed** —
  `tests/test_hamiltonian.py::TestSupercell::test_modified_cholesky`, the same **pre-existing,
  unrelated** failure Phases 3, 3b and 4 recorded.
- The Phase 4b doc-cell harness was re-run over the 30 migrated sources (excluding the slow TMO
  benchmark): **zero regressions**, wavefunction-touching blocks 51 -> 52 clean. The docs still call
  the subclass forms — which now resolve to the inherited base methods — so this also confirms the
  aliases behave identically.

## Phase 4d — ✅ Base-class factory doc migration

Split out of Phase 4c the way 3b and 4b were split out of 3 and 4 (user call).

- [x] Point the doc factory call sites at the base class: `NOMSDWavefunction.from_free_electron(...)`
      -> `Wavefunction.from_free_electron(...)`, `NOMSDWavefunction.from_pyscf(...)` ->
      `Wavefunction.from_pyscf(...)`, `PHMSDWavefunction.from_pyscf_cas(...)` ->
      `Wavefunction.from_pyscf_cas(...)`, `MolecularHamiltonian.from_pyscf(...)` /
      `PeriodicHamiltonian.write_from_pyscf(...)` -> `Hamiltonian.…`, and
      `LatticeHamiltonian.from_dict(...)` -> `Hamiltonian.from_dict(...)`.
      Roughly 20 of the sources Phase 4b migrated, plus the Hamiltonian sites from Phase 3b.
      `MolecularHamiltonian.from_integrals(...)` -> `Hamiltonian.from_integrals(...)` came along —
      Phase 4c put it on the ABC too, and `tutorials/molecules/03` is built around it.
      `PeriodicHamiltonian.write_from_pyscf` turned out to have **no doc callers at all**, the same
      way `write_wfn_pbc` did in Phase 4b, so that mapping was unused.
- [x] **Hand-construction sites keep the concrete class** and must not be swept up:
      `NOMSDWavefunction(coeffs=, dets=, …)` and `PHMSDWavefunction(coeffs=, occa=, occb=, …)` take
      genuinely different data (`examples/models/05`, `examples/molecules/08`,
      `tutorials/models/06`, `tutorials/molecules/{03,04,05}`). The constructor is deliberately
      **not** dispatched — see DESIGN.md.
- [x] Reuse Phase 4b's doc-cell harness against a pristine baseline; the bar is again zero blocks
      that passed before and fail after. **0 regressions** — see **Verification**.

**31 doc sources migrated**, in five groups. 63 call sites and prose mentions changed hands
(27 `from_free_electron`, 23 `from_pyscf`, 8 `from_integrals`, 3 `from_pyscf_cas`,
2 `from_dict`), plus 41 import lines:

- **Snippets** (10): `snippets/01_setting_up/{01..08}/readme.rst` and
  `{11_custom_terms,12_rashba_soc}/run.py`. Uniform — import, call, and the "accepts a `twist`
  keyword argument" sentence.
- **Model examples / tutorials** (6): `examples/models/{01,02,03,07,08}`,
  `tutorials/models/06_writing_a_trial_wavefunction`.
- **Molecular examples** (11): `examples/molecules/{01_O-atom (ham.py + index.rst),
  02_B_atom_SHCI_trial_wfn, 03_n2-phmsd, 04_V-fully_polarized/inputs/setup.py,
  05_Li2_frozen_core (script + index.rst), 06_Pb-spin-orbit, 07_3d_TMO_benchmark,
  08_local_embedding, tbd_H2O_charge_density}`.
- **Molecular tutorials + installation** (3): `tutorials/molecules/{03,05}`, `docs/installation.rst`.
- **Reference** (1): `docs/afqmctools/lattice_models.rst` — see below.

### Which classes stayed

Six sources still name a concrete class, and each is deliberate:

| source | why |
|---|---|
| `examples/models/05_multi_slater_trial` | `NOMSDWavefunction(coeffs=, dets=, …)` — hand-construction |
| `examples/models/08_lieb_lattice_emery` | both: `Wavefunction.from_free_electron` **and** the annealing loop's `NOMSDWavefunction(...)`, so it imports both names |
| `examples/molecules/08_local_embedding` | `Hamiltonian.from_pyscf` + `NOMSDWavefunction(...)` |
| `tutorials/models/06_writing_a_trial_wavefunction` | the whole first half is hand-construction; only the free-electron section moved |
| `tutorials/molecules/03_writing_a_hamiltonian` | `Hamiltonian.from_integrals`/`.from_pyscf` + `NOMSDWavefunction(...)` |
| `tutorials/molecules/{04,05}` | `04` is *entirely* hand-construction (`NOMSDWavefunction` ×4, `PHMSDWavefunction` ×1) and was not touched at all; `05` keeps its `NOMSDWavefunction(...)` |

`examples/molecules/04_V-fully_polarized/inputs/setup.py` moved its **Hamiltonian** call to
`Hamiltonian.from_pyscf` but stays on afqmctools' `write_wfn_mol`, for the C++ walker reason
Phase 4b recorded; the NOTE comment naming the eventual replacement is now
`Wavefunction.from_pyscf`.

`docs/tutorials/solids/04_computing_observables` is untouched — it never used these factories, and
is still on afqmctools' CoQuí reader (DESIGN.md **Future changes**).

### Changes beyond the mechanical swap

- **`docs/afqmctools/lattice_models.rst` gained a "Construction" section.** Rewriting its
  `LatticeHamiltonian.from_dict()` example as `Hamiltonian.from_dict()` left the reference page
  showing an ABC building a lattice model with nothing saying why that works. The new section lists
  the four `Hamiltonian` factories, states that `from_pyscf` dispatches on the `'cell'`/`'mol'` key
  while the rest have one answer each, and records that the subclass form is the *same* method and
  that asking a subclass for a Hamiltonian it cannot produce raises `ValueError`. This is the one
  addition beyond the three bullets above.
- **`examples/molecules/tbd_H2O_charge_density`'s unused import** — a stub example whose import
  marks where the author intended to write a CAS wavefunction (Phase 4b swapped it to
  `PHMSDWavefunction` rather than deleting it). It is `Wavefunction` now, since
  `from_pyscf_cas` is the call it was standing in for.
- **`examples/models/02_stripes` dropped a name from its import.** It imported
  `NOMSDWavefunction, Wavefunction` — the first for `from_free_electron`, the second for
  `from_hdf5`. One name covers both now.
- Import lists kept alphabetical order where they already had it, so
  `HamiltonianBuilder, Lattice, NOMSDWavefunction, SpinSymm` became
  `HamiltonianBuilder, Lattice, SpinSymm, Wavefunction`.
- `examples/models/08_lieb_lattice_emery`'s "Old Scratch work" cell keeps an unused
  `NOMSDWavefunction` import. It is neither a call site nor a hand-construction, and the section is
  pre-existing broken (Phase 3b), so it was left alone.

### Verification

- **Doc-cell harness**, rebuilt to Phase 4b's design (`nb_execution_mode = "off"`, so a sphinx build
  does *not* execute these cells): each source's `{code-cell}` fences / `.. code-block:: python`
  directives run cumulatively in one namespace, in a per-source scratch cwd seeded with the doc
  unit's own data files, with each block flagged for whether it names a migrated class. Run over 29
  sources on both trees.

  | tree | blocks | clean | API-touching | API clean |
  |---|---|---|---|---|
  | baseline | 184 | 129 | 54 | 48 |
  | current | 184 | 129 | 54 | 48 |

  **0 regressions and 0 newly-passing blocks — the two trees agree block for block.** That is the
  expected result: Phase 4c's aliases are the *same function object*, so this migration can only
  break something by getting an import wrong.

  > **The baseline is the pre-Phase-4d working tree, not `git archive HEAD`.** Phases 3b/4b/4c are
  > still uncommitted on this branch, so `HEAD` predates the migration these edits build on. The
  > baseline was reconstructed by inverting exactly the Phase 4d edits
  > (`scratchpad/revert_4d.py`, an explicit per-file table that aborts if any replacement no longer
  > matches), and the reconstruction was audited by diffing it against the working tree: **every
  > changed line is one of the intended swaps and nothing else.**

- **Every remaining failure is pre-existing and identical on both trees.** The API-touching ones:
  snippet `06_twist_angle`'s orphan `input_charge.toml` block (Phase 4b flagged it for a decision);
  `examples/models/02`'s `autohf.solver.lattice_hf` cell and the `afqmc_1.h5` read downstream of it;
  `01_O-atom/ham/ham.py` and `04_V-fully_polarized/inputs/setup.py`, which need `../scf/*.chk` from a
  PySCF script the harness does not run; `tutorials/molecules/05`'s missing `qmc.s001.stat.h5`. The
  non-API ones are missing AFQMC output files, `examples/models/08`'s "Old Scratch work" matrix `t`,
  and `08_local_embedding`'s `import embedding` — all recorded under Phase 3b/4b.
- **Syntax**: all 244 code blocks / scripts compile, across the 41 doc sources currently modified on
  this branch (this phase's 31 plus the earlier phases').
- **Static import check**: over those same 41, every `X.method(...)` naming a safiretools
  class resolves to an import in that same source — **0 unimported names**. This is what covers the
  two sources the harness does not execute: `examples/molecules/07_3d_TMO_benchmark` (a
  transition-metal-oxide CASSCF benchmark, excluded as too slow, same as Phase 4c) and
  `afqmctools/lattice_models.rst` (illustrative fragments, never runnable).
- **Sphinx**: full build succeeds with **8 warnings, the same set as before this phase** — the
  `sphinx.ext.apidoc` setup warning, an afqmctools docstring block-quote, three missing
  `snippets/02_running_afqmc` include files, the `tbd_H2O_charge_density` orphan-toctree warning, and
  the jupytext-unavailable notice. No new ambiguous cross-references: `Hamiltonian` and
  `Wavefunction` appear only in prose and code blocks here, not in autodoc type fields.
- This phase changed **no library code**, so the test suite is unchanged from Phase 4c (616
  safiretools cases).

## Phase 4e — ✅ Port the polarized vanadium example

The last molecular doc source still calling afqmctools for a wavefunction. Phase 4b left it alone
because the C++ walker setup did not accept an empty beta sector; the sync with `main` removed that
reason (see **Reconciled with the C++ sync** in Phase 4), so it is ported now.

- [x] `docs/examples/molecules/04_V-fully_polarized/inputs/setup.py`:
      `write_wfn_mol(scf_data=…, filename=fout, cas=…)` ->
      `Wavefunction.from_pyscf(scf_data=…, cas=…).to_hdf5(fout)`, the base-class factory route
      Phase 4d put every other molecular example on. The stale `NOTE` comment and the
      `from afqmctools.wavefunction.mol import write_wfn_mol` line go with it; the example now
      imports `Hamiltonian, Wavefunction` from `safiretools` and nothing else from afqmctools except
      `load_from_pyscf_chk_mol`, matching `01_O-atom/ham/ham.py` and `05_Li2_frozen_core`.
      A comment records why the trial comes out collinear with `ndown == 0`.
- [x] `index.rst`: step 2's prose pointer `input/setup.py` -> `inputs/setup.py` (the
      `literalinclude` half was fixed in Phase 4c), plus one sentence saying that a trial with no
      beta electrons is collinear with `ndown == 0`, which is why `afqmc.json` asks for
      `"walker_type": "collinear"` and the beta blocks are zero width.

### Verification — the one doc source that was run all the way through

Unlike the Phase 4b/4d harness runs, this example was executed end to end from a clean scratch copy,
which is exactly what it needed and could not get before:

1. **`scf/scf.py`** — PySCF ROHF + CASCI(32o,3e) on the V atom. `CASCI E = -942.902126835175`,
   `E(CI) = -3.53525252060444`, `S^2 = 3.75` (the quartet).
2. **`inputs/setup.py`** (ported) — writes Hamiltonian and wavefunction into one `afqmc.h5`, which
   reads back as `MolecularHamiltonian` (`nmo` 32) and `NOMSDWavefunction`
   (collinear, `nelec = (3, 0)`, `dets (1, 32, 3)`, `psi0` blocks `(32, 3)` and `(32, 0)`).
   On disk: `Wavefunction/NOMSD/dims = [32, 3, 0, 2, 1]`, `Psi0_beta (32, 0, 2)`,
   `PsiT_0/dims = [3, 32, 3]`, `PsiT_1/dims = [0, 32, 0]` — the layout
   `tests/test_polarized_consistency.cpp`'s `derive_polarized_wfn` writes.
3. **`build/Release/bin/safire`** under `mpiexec -n 4` on the example's own `afqmc.json` (steps cut
   to 60, 10 walkers/rank for the smoke test) — runs to completion, 0 vbias and 0 eloc clamp hits.
   The mixed energy falls monotonically from -942.8868 at τ = 0.05 to -942.8957 at τ = 0.30,
   heading from the ROHF reference toward the CASCI value. Not a converged number, but the right
   trajectory from a trial the executable accepts.

**Equivalence with afqmctools:** the ported wavefunction was compared dataset for dataset against
`write_wfn_mol` writing into a copy of the same file. **13 of 14 datasets are bit-identical**,
including both zero-width beta blocks — afqmctools writes `Psi0_beta (32, 0, 2)` and
`PsiT_1/dims = [0, 32, 0]` too, now that its own fourth branch is gone. The one difference is
`Psi0_alpha`, at **1.3e-15 max**: afqmctools applies its 1e-8 sparsification threshold to the whole
determinant array *in place* and so thresholds the dense `Psi0` as well, which safiretools does not.
That is the difference Phase 4 already documented for the periodic path, appearing here for the same
reason; no entry above the threshold differs (0 entries where safiretools has `> 1e-8` and
afqmctools has 0).

### Three `index.rst` content fixes (user call, done)

Flagged during the port and then fixed on request, since all three were wrong rather than merely
dated:

- **The closing line pointed at two things that do not exist** — "See ``run.sh`` for execution
  details. The ``dice/`` directory contains an alternative trial wavefunction generation approach
  using selected CI." There is no `run.sh` and no `dice/` here. It now names the real batch script,
  `inputs/run_safire_cpu.sh`, and cross-references
  :doc:`../02_B_atom_SHCI_trial_wfn/06_SHCI_trial_wavefunction` for the selected-CI trial —
  the same `:doc:` target `examples/molecules/03_n2-phmsd` already links to.
- **Step 1's reference energies did not match what `scf/scf.py` prints**, and step 1 disagreed with
  step 5 on the CASCI total (-942.901099 vs -942.902141). Running the script gives
  **ROHF -942.884910**, **CASCI(32o,3e) -942.902127**, **E(CI) -3.535253** (`S^2 = 3.75`), so the
  ROHF energy was off by 0.012 Ha — a *different SCF solution*, not a typo. Steps 1 and 5 now both
  carry the measured CASCI value, quoted to 6 decimals rather than 12. **The recorded AFQMC energy,
  -942.9017(2), was left as is** — it is a production 7000-step/64-rank result, and it still agrees
  with the corrected CASCI reference to 4.3e-4 Ha, which is the point step 5 makes. See
  **`scf.py` now pins the ROHF solution** below for how the number is enforced rather than hoped for.
- **Step 4 credited `scalar_stats` to afqmctools.** Dropped the attribution rather than moving it to
  safiretools: `scalar_stats` is the one CLI safiretools keeps, so the neutral phrasing is correct
  both today and after **Phase 6**. The identical sentence survives in
  `examples/molecules/01_O-atom/index.rst`, `examples/molecules/05_Li2_frozen_core/index.rst` and
  `tutorials/molecules/01_hello_safire`, which were out of scope here — a one-line change in each
  when Phase 6 lands.

### `scf.py` now pins the ROHF solution (user call, done)

Correcting step 1's energies exposed the real problem: nothing in the example *guaranteed* which
ROHF solution it got, so the doc had drifted onto a different one. Rather than document that as
variability, `scf/scf.py` now enforces it.

- **`solve_stable_rohf(mol, chkfile, max_rotations=5)`** replaces the bare
  `scf.ROHF(mol).newton().kernel()`. It runs an **internal** stability analysis
  (`mf.stability(return_status=True)`), and while unstable rotates the orbitals along the reported
  instability and re-solves from there (`mf.kernel(mo, mf.mo_occ)`), raising if still unstable after
  `max_rotations`. It also raises if the stable solution did not converge.
  Only internal stability is checked — that is the "genuine minimum vs. saddle point" question.
  External stability would test the symmetry-breaking ROHF -> UHF rotations, which would change what
  this example is (an ROHF orbital basis for a polarized trial), so it is deliberately not used.
- **The total energy is checked** against `ROHF_ENERGY = -942.884909528` at
  `ROHF_ENERGY_TOL = 1e-4`, and raises naming both numbers if it misses. The tolerance sits two
  orders of magnitude below the 0.012 Ha gap to the competing solution and well above run-to-run
  noise. A single HF check guards the whole chain, since CASCI and the AFQMC Hamiltonian are
  deterministic given the orbitals.

> **`newton()` needs `dm0=` spelled out, or the MOs passed positionally.** `SecondOrderROHF.kernel`
> is `(mo_coeff=None, mo_occ=None, dm0=None)`, not the plain solver's `(dm0=None, **kwargs)`, so the
> `mf.run(dm1)` idiom from PySCF's own stability examples would silently pass a density matrix as
> `mo_coeff`. This code passes the rotated MOs positionally instead, which is what the newton solver
> wants anyway.

**Verified:**

- The solution the script already found, **-942.884909528, is internally stable**
  (`lowest eigs of H = [1.6e-06, 8.4e-06, 9.5e-02]`, all positive), and the plain non-newton solver
  lands on the same energy and is also stable. So **the doc's old -942.873080 was the wrong
  solution, not this one** — the stability loop is a guard that passes on the first check rather
  than a fix that moves the answer.
- `scf.py` exits 0, reporting ROHF -942.8849095280832 and CASCI -942.902126835175
  (`E(CI) = -3.53525252060444`, `S^2 = 3.75`) — the numbers now in `index.rst`.
- **The guard fires.** With `ROHF_ENERGY` temporarily set to the doc's old value in a scratch copy,
  the script raises `RuntimeError: ROHF total energy -942.884909528 Ha differs from the expected
  stable solution -942.873080273 Ha by more than 0.0001 Ha. …` — i.e. it would have caught the
  original mistake.
- **The whole chain re-runs from the new `rohf.chk`**: `setup.py` writes the same
  `dims = [32, 3, 0, 2, 1]` / `Psi0_beta (32, 0, 2)` / `PsiT_1/dims = [0, 32, 0]` wavefunction, and
  `mpiexec -n 4 safire` reproduces the earlier trajectory value for value
  (-942.892082 / -942.893479 / -942.895685 at tau = 0.20 / 0.25 / 0.30), confirming the checkpoint
  holds the same solution as before.

Step 1's note in `index.rst` was rewritten accordingly: instead of warning that results may drift,
it says the script runs the stability analysis and checks the energy, so a run that lands elsewhere
fails loudly.

Step 1's prose also claimed `scf/scf.py` is run "to perform ROHF … and to generate a trial
wavefunction for AFQMC." It writes only `rohf.chk`; the trial comes from `inputs/setup.py` in
step 2. Corrected to say the orbitals are saved to `rohf.chk` and the trial is built from them in
step 2.

## Phase 4g — ✅ Revert the `Hamiltonian` base-class factories

Team call (2026-09-09): keep only the **truly shared** factory on `Hamiltonian` — `from_hdf5`,
whose target comes from the file rather than the call site — and define each domain factory on the
subclass that produces it. Undoes the `Hamiltonian` half of Phase 4c and its Phase 4d docs.
**`Wavefunction` is unchanged**: there the representation often is not knowable at the call site, so
base-class dispatch stays. See DESIGN.md **Where a factory lives**.

- [x] `hamiltonian/base.py`: deleted `from_dict`, `from_integrals`, `from_pyscf` and
      `write_from_pyscf`, plus the `_check_domain` guard they shared — exactly the 137 lines Phase 4c
      added. `from_hdf5`/`_read_hdf5` untouched.
  - [x] Nothing to restore on the subclasses: Phase 4c added base copies but never deleted
        `LatticeHamiltonian.from_dict`, `MolecularHamiltonian.from_integrals`/`.from_pyscf` or
        `PeriodicHamiltonian.from_pyscf`/`.write_from_pyscf`, unlike the `Wavefunction` half.
  - [x] The guard is gone because it is now unnecessary, not because the check was dropped: a
        sibling's factory raises `AttributeError` for not existing rather than `ValueError` for
        refusing. Closes Phase 4c's known `from_pyscf` cross-call gap by construction.
- [x] Deleted `TestFactoryDispatch` and `TestFactoryDispatchPyscf` from
      `tests/safiretools/hamiltonian/test_base.py` — 16 cases, all of them testing only the removed
      dispatch. Subclass-factory coverage was already in `test_molecular.py`, `test_periodic.py` and
      `model/test_lattice_hamiltonian.py`, so no coverage was lost. 632 → 616 collected, all pass.
- [x] Docs: 13 source files back onto the subclass factories (`Hamiltonian.from_pyscf` →
      `MolecularHamiltonian.from_pyscf`, `.from_dict` → `LatticeHamiltonian`, `.from_integrals` →
      `MolecularHamiltonian`), imports updated to match. `Hamiltonian.from_hdf5` left alone
      throughout — 34 call sites, all still correct.
  - [x] `docs/afqmctools/lattice_models.rst`: the **Hamiltonian Builder** wording restored verbatim
        to its pre-Phase-4d text. Its **Construction** section, which Phase 4d added, was **kept and
        rewritten** to map each factory to its own class rather than deleted — it is a useful
        reference and nothing in the revert argues against having it.
  - [x] `docs/_build/` deliberately untouched (generated output; regenerates from source).

## Phase 4h — ✅ FCIDUMP I/O onto the Hamiltonian classes

User call (2026-09-10): a FCIDUMP is a Hamiltonian on disk, so it is read and written through the
class that holds those integrals, like every other format. The six re-exported free functions come
off the public surface entirely (**"Methods only"**, user call), leaving
`hamiltonian/fcidump.py` dev-facing. Supersedes the Phase 3b follow-up item 5 below. See DESIGN.md
**FCIDUMP is an external format the Hamiltonian classes own**.

- [x] `MolecularHamiltonian.from_fcidump(path, symmetry=None, cholesky_tol=1e-6, spin_symm=None,
      verbose=False)` — reads the file, then goes through `from_integrals`, so the two-electron
      integrals are Cholesky-decomposed exactly as an `eri` tensor is. `nelec` comes from the
      `NELEC`/`MS2` header; `spin_symm` defaults to `CLOSED` for `MS2 == 0` and `COLLINEAR`
      otherwise, and has to be passed explicitly for a spinor-basis file, which is not
      self-describing.
  - [x] **The `(0, 1, 3, 2)` transpose is load-bearing** and is now inside the factory: FCIDUMP
        holds `(ik|jl)` at `[i, k, j, l]`, the decomposition needs the hermitian pair matrix
        `{(ik), (lj)}`. Identical for real integrals, meaningless for complex ones — verified at
        `min eig = -24` vs `-5e-8`, ERI error `5.8e+03` vs `7.4e-07` on a random 3-orbital complex
        case. It was in `tutorials/molecules/03` as a bare `np.transpose` and is not lost with the
        two-step recipe it belonged to.
- [x] `MolecularHamiltonian.to_fcidump(path, tol, ctol, sym, cplx, paren, use_spinor)` — instance
      state supplies `hcore`/`chol`/`enuc`/`nmo`/`nelec`, and `chol_is_eri` is gone since the class
      always holds a factorization. Rejects `use_spinor` on an already-spin-orbital Hamiltonian, and
      rejects a spatial-basis Cholesky matrix beside a spin-orbital `hcore` (which `__init__`
      allows, and a FCIDUMP cannot express).
- [x] `PeriodicHamiltonian.to_fcidump(...)` plus `_chol_all_momenta()`, which reconstructs each
      unstored `-Q` block (`L^{-Q}_{k1}[i,j,n] = conj(L^{Q}_{k2}[j,i,n])`, `k2 = qk_to_k2[-Q, k1]`)
      and fills in its `nchol_pk` entry. Without it the dict of half the momentum transfers cannot
      even be handed to the writer. Not new behavior — afqmctools' reader did the same expansion in
      `get_kpoint_chol`.
  - [x] Uneven `nmo_pk` now raises instead of writing garbage, since the writer walks the combined
        index as though every k-point carried `nmo_max` orbitals. `write_fcidump_kpoint` itself is
        left alone.
  - [x] **No `PeriodicHamiltonian.from_fcidump`** — a FCIDUMP records no k-point structure, so
        there is nothing to read back into. The sibling factory is simply absent, as in Phase 4g.
- [x] `hamiltonian/fcidump.py`: module docstring says it is dev-facing and names the four entry
      points; `write_fcidump_kpoint`'s spinor `NotImplementedError` no longer names
      `h1_spat2spin`/`h2_spat2spin` (now unimportable) and points at `use_spinor=False`. Otherwise
      untouched — same functions, same signatures, same behavior.
- [x] `__init__.py`: the six FCIDUMP names dropped from the imports and from `__all__`.
- [x] Tests: `TestFcidump` in `test_molecular.py` (8 cases: real and complex round trips, the
      header-derived spin symmetry, an explicit override, the spinor basis, and the two rejections)
      and in `test_periodic.py` (5 cases: byte-identical to `write_fcidump_kpoint` on a full
      momentum set, the `-Q` reconstruction, a missing partner, uneven `nmo_pk`, the spinor raise).
      `test_fcidump.py` is unchanged — it imports the deep path, which is still where the format
      lives. 636 collected, all pass.
- [x] Docs: `tutorials/molecules/03`'s Python FCIDUMP section is one `from_fcidump(...).to_hdf5(...)`
      call instead of `read_fcidump` → transpose → `from_integrals`; its `read_fcidump()` and
      `from_integrals()` subsections collapse into one (`from_integrals` is already documented
      earlier in the same tutorial). Verified against the tutorial's own `files/H2_FCIDUMP`:
      `nmo=2`, `nelec=(1,1)`, `nchol=3`, `enuc=0.714285714285714`, and the HDF5 round-trips. The
      file now also records `nelec`, which the old recipe left at `(0, 0)` by not passing it.
- [x] The CLI scripts (`afqmc_to_fcidump`, `fcidump_to_afqmc`, `fcidump_spat2spin`) still call
      **afqmctools**, not safiretools, so nothing there had to move. They go with the rest of the
      CLI surface at Phase 6/9.

## Phase 4i — ✅ Gap-filling before Phase 5

Twelve items the user listed after Phase 4h, mostly simplifications. The three that had more than
one reasonable reading were settled up front (user calls, recorded below); the rest were
straightforward.

### 1. The PySCF checkpoint loaders, and `from_pyscf` taking a path

- [x] `convert/pyscf.py` (user call — over a top-level module): `load_pyscf_chk_mol`,
      `load_pyscf_chk`, `as_scf_data`, `is_periodic_chk`, `determine_spin_symm`,
      `canonical_orthogonalization`, `ecp_soc`. Ported from
      `afqmctools/utils/pyscf_utils.py`'s `load_from_pyscf_chk_mol`/`load_from_pyscf_chk` plus
      `utils/linalg.py::get_ortho_ao_mol` and `pyscf_utils.get_ecp_soc`. Nothing in safiretools
      imports upward into afqmctools any more.
- [x] **Every `from_pyscf` takes a checkpoint path *or* a loaded mapping**, through `as_scf_data`:
      `MolecularHamiltonian.from_pyscf`, `PeriodicHamiltonian.from_pyscf`,
      `Wavefunction.from_pyscf`, `Wavefunction.from_pbc_scf`. The first argument is `source` rather
      than `scf_data` now. Both forms are needed: the path makes the simple case one call, and the
      mapping is what lets one load with non-default options (`soc_type='ecp'`) serve both the
      Hamiltonian and the wavefunction.
- [x] **The basis and the wavefunction are separate arguments.** `Wavefunction.from_pyscf(source,
      basis=None)` — `source` is the solution the wavefunction is built *from*, `basis` the solution
      whose orbitals it is expressed *in*. Renamed from `basis_scf_data`, and both accept a path.
- [x] `is_periodic_chk` tells the two checkpoint kinds apart by whether the serialized molecule
      carries lattice vectors, and `as_scf_data` raises when a factory is handed the wrong kind.
      An `isinstance` test could not: `pbc.gto.Cell` subclasses `gto.Mole`.
- [x] Doc-facing prose in DESIGN.md: **Reading a PySCF checkpoint: `convert/pyscf.py`**.
- [x] **Follow-up (user-spotted): the periodic loader's k-point normalization was wrong.**
      `_periodic_occupancies` branched on `isinstance(mo_occ, list)` to decide the layout, which is
      not a property of the layout at all: PySCF returns one array when the k-points share an
      orbital count and a list of arrays when they do not, independently of whether there is a
      k-point axis or a spin axis. Replaced by `_per_kpoint(values, nkpts, entry_ndim, name)`, which
      keys on **nesting depth** (`_nesting_depth`) and returns one entry per k-point; `single_kpt`
      is gone, since `nkpts` from the k-point array already says it, and `_reshape_single_kpt` with
      it. `mo_occ`, `mo_energy`, `mo_coeff` and `fock` all go through the one normalizer, with
      `entry_ndim` 1 for the vectors and 2 for the matrices.
  - [x] **Fixes a layout afqmctools could not read at all**: a single-k-point calculation
        (`pbc.scf.RHF(cell, kpt=...)`) stores every quantity with no k-point axis, and afqmctools
        inferred the spin symmetry from `mo_coeff[0]`'s shape — a *row* when the axis is absent —
        raising ``Unable to determine a valid Slater determinant type``. A regression test records
        both halves.
  - [x] Added a self-consistency check across the four quantities (user-adjacent, flagged):
        `scf/fock` is written by hand after the SCF in these workflows, so it can carry a spin
        structure the solution does not have. That used to surface as an `IndexError` inside
        `_generate_orbitals`; it is a `ValueError` naming the disagreement now.
  - [x] Verified against afqmctools over every layout that loader can read — KRKS 2x1x1 and 1x1x1
        (both `ortho_ao`), KUHF 2x1x1, single-k UHF: **5/5 match** on `kpts`, `hcore`, `fock`,
        `nmo_pk` and `X`. 18 new test cases (669 safiretools cases; full suite 938 passed, same 1
        pre-existing failure).

### 2-12. The simplifications

- [x] **`rediag_fock` inlined** (`wavefunction/pbc.py`) — a one-line `np.linalg.eigh` at its single
      call site. The `X^H (F X)` association survives as a comment, since it is load-bearing: the
      other grouping rotates a degenerate subspace's eigenvectors by ~1e-6.
- [x] **`orthonormalize=` gone from every factory**, and the factories no longer call
      `orthonormalize()`. PySCF's orbitals are orthonormal in the basis they are expressed in, and
      the free-electron and periodic paths occupy eigenvectors of a Hermitian matrix.
      `orthonormalize()` stays as an explicit public method.
- [x] **The PBC multi-determinant route is gone** (user call: CoQuí is the supported route for
      solids). `PHMSDWavefunction.from_pbc_scf`, `ndet_max`, the combination enumeration in
      `_determine_occupancies` and the `probabilities`/`coeffs` bookkeeping all go. A metallic
      reference still gives a single determinant over the **leading configuration** — the
      lowest-indexed partially occupied bands, which is `itertools.combinations`' first element and
      so is bit-identical to what the old `refdet=0` path produced.
  - This also retires the *Things we might change* entry about the expansion keeping the **least**
    probable determinants: the code that did it is gone.
- [x] **The `afqmctools` editorial comments are gone** — nine of them, in `types.py`,
      `hamiltonian/{molecular,fcidump}.py`, `hamiltonian/model/{builder,lattice_hamiltonian}.py` and
      `wavefunction/dice.py`. Each was rewritten to state what the code *does*, not what the old
      code did wrong; the two load-bearing ones kept their justification (the `'close'` spelling
      alias, and why `numpy.any` rather than `numpy.all` decides `hcore`'s dtype) without the
      history.
- [x] **`infer_spin_symm` deleted** rather than moved (user call was to consider replacing it with
      something data-driven). It was circular: `make_slater` is *driven by* the reference's spin
      symmetry, so re-reading that symmetry off the resulting matrix's shape could only ever return
      what it was told. `from_pyscf` now uses the symmetry `determine_spin_symm` established when
      the checkpoint was read.
- [x] **The overlap's condition number replaces the orthonormality check** (user call: warn, do not
      raise). `overlap_condition_number` in `wavefunction/slater.py`,
      `warn_if_ill_conditioned` in `wavefunction/io.py`, limit `CONDITION_MAX = 1/sqrt(eps)` ≈
      6.7e7. Both degenerate cases are pinned by tests: an **all-zero matrix is `inf`**, and a
      matrix with **no columns is 1.0** (the zero-width beta block of a polarized wavefunction).
      `Wavefunction._warn_if_not_orthonormal` and `_slater_matrices` go with it, so the check now
      lives where the bytes are produced and names the dataset.
  - The point of the change: conditioning is what AFQMC actually needs, since it inverts the
    overlap. A random Slater matrix is not orthonormal and is perfectly usable, and now writes
    silently.
- [x] **`write_header` infers `nmo` and `nelec`** from `psi0` and the spin symmetry, through the new
      `io.header_dims`. Both parameters are gone from the signature; `Wavefunction.to_hdf5` is the
      one call site in the package, and the four in `test_io.py` were updated.
- [x] **The sparse eigensolver is gone from `free_electron.py`**: `_one_body_eigenstates`,
      `use_dense`, `SHELL_BUFFER`, `num_eigenvals` and the `scipy.sparse` imports. `spl.eigh` at
      both call sites, over `to_dense(one_body)` — the whole spectrum is wanted, since every orbital
      is a filling candidate, so a sparse solver could never pay off, and `eigsh` mishandles the
      complex matrices a twisted lattice always produces.
- [x] **MPI is gone from `hamiltonian/periodic.py`** (user call). Deleted: `Partition`,
      `fair_share`, `bisect`, `FileHandler`, `rank_filename`, `_SerialComm`, `_merge_rank_files`,
      `_write_kpoint_basics`, `_write_kpoint_block`, `write_from_pyscf`, the `phdf` parallel-HDF5
      branch, and the pivot `Allgather`/`Bcast`/`barrier`. `PeriodicCholesky` owns the full k-point
      and orbital-pair ranges (`nkk`/`nij`), `_pick_pivot`+`_pivot_owner`+`_largest_residual`
      collapse into one `_largest_residual`, and `from_pyscf` + `to_hdf5` is the single entry point.
      **1540 -> 1038 lines.** With `ij0 == 0` and `ijN == nij` the guards and offset arithmetic in
      `generate_orbital_products`, `generate_diagonal`, `_supercell_layout` and `_kpoint_block` all
      fall away — the last becomes a single `reshape`.
- [x] **`modified_gram_schmidt` is a reduced QR decomposition.** More accurate and faster than the
      explicit loop, and the same Householder path LAPACK uses. The sign convention is pinned so
      `R`'s diagonal is real and non-negative, which makes `Q` unique and keeps the function
      **idempotent on an already orthonormal input** — the explicit loop was too, and
      `orthonormalize()` relies on it.

### Verification

- **Loaders, against afqmctools' originals**: 5 cases (Ne RHF, O ROHF, O UHF, diamond KRKS with
  `ortho_ao` both ways) agree on every shared key. `walker_type` is now a `SpinSymm` determined
  from the data rather than an `_SlaterType` inferred from shapes, and agrees in every case.
- **Path vs mapping**: `MolecularHamiltonian.from_pyscf` and `Wavefunction.from_pyscf` produce
  dataset-for-dataset identical files either way. `Wavefunction.from_pyscf` still matches
  afqmctools' `write_wfn_mol` exactly.
- **A wavefunction in another solution's basis**: a UHF solution written into an ROHF-basis file,
  both in one HDF5 file.
- **End to end**: `build/Release/bin/safire` runs to completion on a safiretools-written 4x4 Hubbard
  Hamiltonian + free-electron wavefunction, through the new `write_header` inference and the
  condition check. (The input JSON schema has changed with the C++ sync — `sweeps`,
  `n_walkers_per_mpi_task`, `measure_interval_multiplier` — which is worth knowing before Phase 6
  touches `execution.py`.)
- **Tests**: 650 safiretools cases pass, up from 636. Full non-functional suite: 920 passed, 1
  failed — `tests/test_hamiltonian.py::TestSupercell::test_modified_cholesky`, the same
  pre-existing, unrelated failure Phase 3 recorded.
- New test module `tests/safiretools/convert/test_convert_pyscf.py` (26 cases). Named with the
  `convert_` prefix because `tests/safiretools/wavefunction/test_pyscf.py` already exists and
  neither test tree uses `__init__.py` — the same collision Phase 1 hit with `test_stats_core.py`
  and Phase 4 with `test_wavefunction_base.py`.

### Not done, and why

- **Doc migration.** ~11 doc sources still call
  `afqmctools.utils.pyscf_utils.load_from_pyscf_chk_mol` and pass the mapping positionally, which
  still works — the mapping form is supported and `SpinSymm.from_input` accepts afqmctools'
  `_SlaterType`. Nothing is broken, but nothing shows the new one-call form either. Its own phase,
  like 4b and 4d.
- **`write_rhoG`** kept its `NotImplementedError` and lost only its `comm`/`phdf` parameters.

## Phase 4j — ✅ Wrap up docs for phase 4

- [x] **Doc migration.** The 11 doc sources that called
      `afqmctools.utils.pyscf_utils.load_from_pyscf_chk_mol` now show the one-call route wherever the
      default load suffices: the factory takes the checkpoint path. **No doc source imports from
      `afqmctools.utils.pyscf_utils` any more.**

**Eight sources take the path directly** — `examples/molecules/{01_O-atom/ham/ham.py,
02_B_atom_SHCI_trial_wfn,03_n2-phmsd (the PEC loop),04_V-fully_polarized,05_Li2_frozen_core,
06_Pb-spin-orbit (both wavefunction blocks)}`, `tutorials/molecules/{03_writing_a_hamiltonian,
05_computing_observables}`. Five blocks keep the mapping, because a path cannot express what they
do; each says why in a comment beside the load:

| source | why the mapping |
|---|---|
| `examples/molecules/03_n2-phmsd` (first block) | `base='mcscf'` — the CASSCF orbital basis |
| `examples/molecules/07_3d_TMO_benchmark` | `base='mcscf'` |
| `examples/molecules/06_Pb-spin-orbit` (x2) | `soc_type='ecp'` — SOC integrals |
| `examples/molecules/08_local_embedding` | replaces `scf_data['mo_coeff']` with a Foster-Boys localized basis — the point of the example |
| `examples/molecules/tbd_H2O_charge_density` | reads `scf_data['mo_coeff']` |

`04_V-fully_polarized` used the mapping only to compute an **unused** `ncore` from `mol.nelec` and
to dump every `scf_data` key; both went with the switch to the path form.

### The loaders stay dev-facing; a PySCF example may name the deep path (user call)

`load_pyscf_chk_mol` was briefly re-exported at the top level, on the reading that DESIGN.md's
"user-facing docs show only top-level imports" rule forces a promotion. **Reverted.** The five
blocks above say `from safiretools.convert.pyscf import load_pyscf_chk_mol`, and that is the one
acknowledged exception to the rule: each already imports PySCF itself, so a `convert.pyscf` import
sits naturally in the code around it, and promoting the loader would put on the public surface a
function that nothing in the API takes or returns. Recorded in DESIGN.md under **Reading a PySCF
checkpoint**. `safiretools/__init__.py` is therefore unchanged by this phase.

**Rejected alternative: pass-through kwargs.** Giving `from_pyscf` a `base=`/`soc_type=`/`mo_coeff=`
set would let every doc use a path, but it becomes ambiguous exactly where it is needed most —
`Wavefunction.from_pyscf(ghf_chk, basis=rohf_chk, soc_type='ecp')` has two checkpoints and one
`soc_type` — and `08_local_embedding` would need a new orbital-override argument on top.

### Stale keyword arguments removed along the way

Phase 4i renamed `scf_data`/`basis_scf_data` to `source`/`basis`, so these calls raised `TypeError`
as written: `MolecularHamiltonian.from_pyscf(scf_data=...)` in `01_O-atom` and `05_Li2_frozen_core`,
and `Wavefunction.from_pyscf(scf_data=..., basis_scf_data=...)` in `01_O-atom`,
`04_V-fully_polarized`, `05_Li2_frozen_core`, `06_Pb-spin-orbit` (x2) and
`03_writing_a_hamiltonian`. All are positional or `basis=` now. `03_writing_a_hamiltonian` drops
`basis=` entirely, since it passed the same solution twice and `basis` defaults to `source`.

Prose: `02_B_atom_SHCI_trial_wfn` and `03_n2-phmsd` no longer credit afqmctools with the Hamiltonian
framework or with the loader; the CLI prose is still afqmctools' and was left alone (Phase 6/9).
`tbd_H2O_charge_density`'s setup cell dropped an unused loader import.

### Verification

- **All 62 edited code blocks / scripts compile.**
- **Ran the migrated `03_writing_a_hamiltonian` block end to end** (H2/STO-3G RHF): the
  path-form `MolecularHamiltonian.from_pyscf` and the one-argument
  `Wavefunction.from_pyscf` both write into one `afqmc.h5` — `Hamiltonian/dims` `[0 0 0 2 1 1 0 3]`,
  `Wavefunction/NOMSD/dims` `[2 1 1 1 1]` — and read back as `MolecularHamiltonian` /
  `NOMSDWavefunction`.
- **Tests**: 668 safiretools cases pass; `safiretools/__init__.py` is untouched, so the
  public surface is byte-identical to Phase 4i's.

## Phase 5 — Statistics / analysis domain layer

- [ ] `analysis/metadata.py`: one typed metadata accessor (`walker_type: SpinSymm`, `taus`, etc.),
      replacing ad hoc `get_metadata` dict access.
- [ ] `analysis/equilibration.py`: `Teq -> Neq`, ported from `analysis/common.py::_Neq_from_Teq`
      **with the formula fixed**: `delta_tau = taus.max()` (`= max_nback_prop * dt`), not
      `taus[1]-taus[0]`. Keep the `Teq`/`nequil` knob as a real feature (post-hoc trim for samples
      that slip past the C++-side `equil_multiplier` cut), not dead config.
  - [ ] Test: reproduces the bug on a synthetic multi-level `taus` array (assert old formula and
        new formula diverge), then confirms the fixed formula matches
        `max_nback_prop * dt`.
- [ ] `analysis/rdm.py`: 1-RDM extraction/averaging, ported from `analysis/rdm.py` +
      `stat_h5.py::afobs` (the narrow, live path only — broader RDM pipeline stays deferred per
      DESIGN.md Scope).
  - [ ] Fix `check_1rdm_convergence`'s quadrature-sum indexing bug (currently uses only one of the
        two BP-average endpoints' errors).
  - [ ] Test: quadrature-sum fix uses both endpoints' errors on a synthetic two-endpoint case.
- [] This is a sensitive phase; a human needs to beta test this to make sure that things are reasonable; set up a suite of cases to check the observables with; this should be entire workflows in chemistry, solids, and lattice models.

## Phase 6 — Execution & CLI

- [ ] `execution.py` (name/location placeholder — see Open Questions in DESIGN.md; do not treat the
      module name as final): redesigned AFQMC run-config JSON generator, ported from
      `inputs/from_hdf.py`. Redesign scope: apply the classmethod-factory / typed-exception /
      no-mutation conventions from DESIGN.md; do not carry over `inputs/energy.py`-style
      validation (dropped, see DESIGN.md Scope — flagged for the future observables library).
- [ ] `scalar_stats` CLI entry point: port only this one, raising typed exceptions internally and
      confining `sys.exit()`/argument-error handling to the CLI entry function itself.
- [ ] Drop `energy_stats` alias and every other CLI entry point.
- [ ] Update `utils/pyproject.toml` console-script entries accordingly (see Phase 9).

## Phase 7 — External interop

- [ ] `qe/`: relocate `qe_tools.py` + `qe_utils.py` — fix bugs, preserve existing behavior (these
      have external callers, per [[feedback-library-external-callers]] equivalent note in
      DESIGN.md). Drop `qe_driver.py` outright (currently unimportable, references a nonexistent
      module, superseded).
- [ ] `convert/autohf.py`: port AutoHF interop, harden the currently-fragile unguarded import.
- [ ] `convert/pyscf.py`: thin orchestration layer calling `hamiltonian.molecular` +
      `wavefunction.pyscf` (`NOMSDWavefunction.from_pyscf`) — no logic beyond wiring the two
      together.
- [ ] `observables/rhonk.py`: keep close to current shape (external callers depend on it); replace
      its vendored HDF5 helpers with calls into the shared `hdf5.py` (Phase 1) — behavior-preserving
      dedup only, no functional changes.
- [ ] Confirm AIMBES interop (`aimbes_utils.py`, `aimbes_to_2nd_quant`/`aimbes_to_afqmc`) is *not*
      ported — leave a one-line pointer comment to its current location in case support returns.

## Phase 8 — Top-level API surface

- [ ] `__init__.py`: finish the curated flat re-exports. `__all__` there is the definition of the
      public surface, so names go in as each phase lands rather than being tracked as a second list
      here — `mean_and_error`, from Phase 1's `stats.py`, is the one still missing. Concrete
      `Lattice` subclasses, `HamiltonianComponent` and the FCIDUMP format internals are
      deliberately excluded (DESIGN.md **The top-level import surface defines what is
      user-facing**).
      Started in Phase 2: `Lattice` was exported early so the migrated docs could show the real
      public import.
- [ ] Audit for the `AFQMC_EXEC` import-time `RuntimeError` (currently fires in
      `tutorial_utils/helper.py`) — convert to a lazy check that only fires when something actually
      invokes SAFIRE, not on package import.
  - [ ] Test: `import safiretools` succeeds with `AFQMC_EXEC` unset.
- [ ] Confirm no module in the public import path (`__init__.py`'s transitive imports) pulls in
      `logging.Logger` monkey-patching (drop the current `afqmctools/__init__.py` patch) or any
      always-on progress-printing side effect.

## Phase 9 — Packaging / dependency cleanup

- [ ] `utils/pyproject.toml`: drop `pytables` dependency (only reason was `stats/config_h5.py`,
      rewritten onto `h5py` in Phase 1).
- [ ] Merge the `LATTICE_HF` optional-dependency group into `AUTOHF` (confirmed exact duplicate).
- [ ] Remove/fix broken or stale console-script entries found during the original audit:
      `wfn_to_hdf5` (points at a nonexistent `cli.wfn_to_hdf5:main`), `coqui_to_*` naming mismatch
      vs. README, duplicate `scalar_stats`/`energy_stats` entries (superseded by Phase 6).
- [ ] Once `safiretools` covers everything in DESIGN.md's Scope, remove `afqmctools` and `stats`
      from `utils/pyproject.toml` and delete the old packages — **do this only after** confirming
      with the user, since these are library packages with external callers (per
      [[feedback-library-external-callers]]) and removal is a breaking change for them, not just an
      in-repo cleanup.
- [ ] Update `utils/README.md`'s Supported/Deprecated CLI Tools list to reflect the new, much
      smaller CLI surface (`scalar_stats` only).

## Deferred / blocked — do not resolve unilaterally

- `execution.py`'s final naming/location, and whether it gets its own CLI entry point beyond
  `scalar_stats` — explicitly deferred to a broader team discussion (see DESIGN.md Open Questions).
  Phase 6 above should proceed under the placeholder name; renaming later is a mechanical follow-up,
  not a blocker to prototyping.
- The broader RDM/observable pipeline (2RDM, generalized Fock matrix, spin-spin, pair-correlation,
  EKT/NOON) and `inputs/energy.py`-style validation — out of scope for this task list entirely; they
  belong to the future C++-spanning observables rewrite.
