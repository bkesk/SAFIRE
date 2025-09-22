# Hubbard-Kanamoir Hamiltonian

This example is deprecated and will be removed. See `docs/examples`.

From the reference: PRB 99, 235142 (2019)
Table I:

The AFQMC energy of the H-K Hamiltonian on a 6x1 lattice (periodic in the long direction only)
and using U=2.0, U1 = 1.5, U2 = 1.0, and J = 0.5 is -3.774(3) and exacti diagonalization gives
-3.773268.
The AFQMC energy from the PRB above *might* have been computed using GHF walkers / Noncollinear.

Using QMC-FI/AFQMC, with the run parameters in the sample input below, we get -3.772(2) - Collinear walkers

sample input:

```json
{
  "afqmc": {
    "project": {
      "id": "qmc",
      "series": 0,
      "mixed_precision": false
    },
    "execute": {
      "walker_set": {
        "walker_type": "COLLINEAR"
      },
      "wavefunction": {
        "filename": "afqmc.h5"
      },
      "timestep": 0.01,
      "steps": 10000,
      "n_walkers": 10,
      "estimator": {
        "name": "energy",
        "overwrite": true,
        "print_components": true
      }
    }
  }
}
```

