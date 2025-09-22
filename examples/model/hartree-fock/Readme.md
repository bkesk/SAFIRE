# Hartree-Fock toml example

The following is an example script to use the `afqmctools` and `autohf` setup to create initial input files for the SAFIRE code. 
By running 

```bash
# Don't forget to load environment variables and venvs etc
$ python generate_input.py input.toml
```
After, `afqmc.h5` and `afqmc.json` will be generated, ready to use with the compiled C++ code. 

As an alternative to the provided script, one can also use the Hamiltonian builder, e.g.

```python
from afqmctools.systems import lattice as lat
import afqmctools.hamiltonian.model.builder as ham
from afqmctools.wavefunction.model import write_free_electron_wfn, write_wfn
import afqmctools.utils.io as io

from jax import config
config.update("jax_enable_x64", True)
from autohf.solver import lattice_hf

lattice_dims = (4,4)
nelec = [8,8]
outputname = "afqmc.h5"
lattice = lat.SquareLattice(
            L=lattice_dims,
            axis1_boundary=lat.PBCBoundary,
            axis2_boundary=lat.PBCBoundary,
            #twist=["1 Pi","1 Pi"]
            )

builder = ham.HamiltonianBuilder(
          lattice=lattice,
          spin_symm="collinear",
              )
# add standard Hubbard terms
builder.nth_neighbor_hopping(1.0)
builder.onsite_hubbard(4)
builder.finalize()


io.write_model_hamiltion(hamiltonian=builder.hamiltonian,
                         fname=outputname,nelec=nelec,)

### Can now write free electron trial
# write_free_electron_wfn(fname,nelec=(8,8),)
### Or do AutoHF

hf_settings = dict(
    # adjust as needed
    numSteps = 200,
    output = outputname,
    opt_method="lbfgs",
    ansatz="SD_ROT",
    nelec = nelec,
    numTrials = 8,
    seed = 0,
    gpu = False,
    noncollinear = False,
  
)
data = lattice_hf(
    hamiltonian=builder.hamiltonian,
    lattice=lattice,
    settings=hf_settings,
    # jaxoptargs = dict(linesearch="hager-zhang"), # as an alternative
)
```
