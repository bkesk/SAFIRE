---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3
---

+++ {"id": "jFoBNSA0D-an"}

# Hubbard model on a honeycomb lattice

## Set up the Hamiltonian

```{code-cell}
:id: G8BN6a14EL83

# setup scratch dir
from pathlib import Path
from tutorial_utils import run_afqmc, get_scratch_dir

home = Path.home()
scratch_rootdir = home / ".scratch"
scratch_rootdir.mkdir(exist_ok=True)
scratch_dir = get_scratch_dir("honeycomb_hubbard",scratch_rootdir)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 877
id: djut2ei2D6d-
outputId: 024de36e-f9af-4009-c2a7-9981d6c01a39
---
# set up Hamiltonain
# setup the Hamiltonian
import numpy as np

from afqmctools.systems.lattice import get_lattice
from afqmctools.utils.visualize import plot_lattice
from afqmctools.hamiltonian.model.director import HamiltonianDirector
import afqmctools.utils.io as io

lattice = get_lattice(
    params=dict(
        L1 = 4,
        L2 = 4,
        boundary1 = "pbc",
        boundary2 = "pbc",
        type = "honeycomb"
    )
)

# plot the lattice for reality checks!
plot_lattice(lattice)

# define Hamiltonian parameters
hamiltonian_params = {
    'U': 6.0
}

params = {
    'hamiltonian': hamiltonian_params
}

# make the Hamiltonian
hamiltonian = HamiltonianDirector(
    source=params,
    lattice=lattice
).build()

# save for AFQMC
io.write_model_hamiltonian(hamiltonian,fname=scratch_dir/"afqmc.h5")
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: bYkD4o3sHwP9
outputId: 877b0a3a-869b-49cb-a0b2-c13233a2baa8
---
# run autoHF and save wavefunction as trial wavefunction
from autohf import lattice_hf
from autohf.hamiltonian import AutoHFHamiltonian
from afqmctools.inputs.from_autohf import autohf_to_afqmc

# Typically, we want to use a Ueff to generate a trial wavefunction
#    since HF's critical U is lower than the correct critical U
Ueff = 3.0

# define effective Hamiltonian parameters
hamiltonian_params = {
    'U': Ueff
}

params = {
    'hamiltonian': hamiltonian_params
}

# make the Hamiltonian
effective_hamiltonian = HamiltonianDirector(
    source=params,
    lattice=lattice
).build()

# convert afqmctools Hamitlonian to AutoHFHamiltonian
autohf_hamiltonian = AutoHFHamiltonian(effective_hamiltonian)

# set the number of electrons!!
nelec=(16-1,16-1)

settings = dict(
    ansatz = 'SD_ROT',
    numSteps = 100,
    nelec = nelec,
    numTrials = 100,
    seed = 42
)

results = lattice_hf(hamiltonian=autohf_hamiltonian,settings=settings)

autohf_to_afqmc(results,output_fname=scratch_dir/"afqmc.h5")
```

+++ {"id": "vYmHYe-nIm_Q"}

## Write input file

For more information about the input file, see the [Understanding the input file](../../../tutorials/models/02_understanding_the_input_file/02_understanding_the_input_file.html) tutorial.

```{code-cell}
:id: pm-N_EFxInNK

from afqmctools.inputs.from_hdf import write_json

# write input file
afqmc_execution_options = {
    "timestep": 0.01,
    "steps": 10000,
    "accumlate_interval": 10,           # in units of steps
    "measure_interval": 10,             # in units of steps
    "population_control_interval" : 10, # in units of steps
    "walker_ortho_interval" : 10 ,      # in units of steps
    "n_walkers_per_mpi_task": 100,
    "seed" : 42
}

write_json(
    scratch_dir/ "afqmc.json",
    fwfn0=scratch_dir / "afqmc.h5",
    exec_opts=afqmc_execution_options
)
```

+++ {"id": "2umfVnQDJcfa"}

## run AFQMC

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: xvJVj1DGJeW0
outputId: 2127c777-b4a8-4b14-b74e-08f4aab97659
---
# run AuxiliaryFields
run_afqmc(
    run_dir=scratch_dir,
    run_mode="local_cpu",
    input_file="afqmc.json",
    np=16,             # number of MPI tasks
    output_file=None   # optionally direct output to file instead of here
)
```

+++ {"id": "9YE2JAdlJesB"}

## Analyze the results

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 904
id: eADuTKxlJgfx
outputId: 14caad1a-1465-4d39-bfaa-07d91d0d92b7
---
# analyze results
from stats.scalar_dat import analyze_scalar_data

_ = analyze_scalar_data(dict(
    fname = scratch_dir/"qmc.s000.scalar.dat",
    xaxis = "time",
    nequil = 5,
    trace = True
))
```

```{code-cell}
:id: 3tywEbD9Ow2u

















































```
