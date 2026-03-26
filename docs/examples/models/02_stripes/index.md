---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "af3003d3-6a86-4f3b-aedd-d2fcc19a22bc"}

# Stripes and the square Hubbard Model

Author: Ryan Levy

```{code-cell} ipython3
:id: 9b2b92e5-990e-47bf-b235-3a37edbf4a0d

import os
# This are in case you have a GPU enabled JAX but are running on CPU.
os.environ['JAX_PLATFORMS'] = 'cpu'
# If you want to use GPU enabled Jax in multiply notebooks with the same
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# use x64 numbers
from jax import config
config.update("jax_enable_x64", True)
```

```{code-cell} ipython3
:id: 6883d177-8f0e-4e90-bea0-54b6c711cad6

import matplotlib.pyplot as plt
import numpy as np
```

```{code-cell} ipython3
:id: f6e937d2-fde5-4f1b-afd9-6afb3798244a

# all the imports used later in the tutorial, but put here for convenience

import afqmctools.systems.lattice as lat
import afqmctools.utils.visualize as vis
import afqmctools.utils.io as io
import afqmctools.hamiltonian.model.director as ham

from afqmctools.wavefunction.converter import read_wavefunction
from afqmctools.wavefunction.model import write_free_electron_wfn
from afqmctools.analysis.rdm import average_afqmc_rdm
```

```{code-cell} ipython3
:id: a332abc9-fc0a-4774-b316-6016060e0091

from afqmctools.observables.greens import greens_1body
import afqmctools.observables.spin as spobs
```

```{code-cell} ipython3
:id: 8c9febda-fb02-489c-8c4f-422a343b80a2

import autohf
```

+++ {"editable": true, "id": "083d8f36-02de-4281-a56c-7cd3d5add077"}

## Physics Background

In the paper _Stripes and spin-density waves in the doped two-dimensional Hubbard model: Ground state phase diagram_ [1] we see that for the nearest-neighbor square Hubbard model, the ground state may have a stable staggered stripe ground state order

<div>
<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/models/ex02_stripes/fig9.png" width="500 px"/>
</div>

We'll explore how to see this with AFQMC and the role of the trial wave function

+++ {"editable": true, "id": "8dc800f8-72bf-4591-911f-7d99b692c8f4"}

---
[1] Hao Xu, Hao Shi, Ettore Vitali, Mingpu Qin, and Shiwei Zhang Phys. Rev. Research 4, 013239 – Published 28 March 2022

+++ {"id": "e8420a88-5a19-4b32-996d-21b3d85aa58c"}

## Obtaining the Spin density

Let's look at a 10x4 system with 0.2 doping at $U=6$ with a staggered AFM pinning field at the edges. The pinning breaks symmetry in order to understand the response of the system to the applied field.

When confronted with a new problem, the first thing we generally do is use a non-interacting (free electron) wave function as the trial. Let's do that now

```{code-cell} ipython3
:id: 9d8005b9-ae0e-42e9-9d87-7fa2d233c43c

U = 6.0
lattice_dims = (10,4)

lattice_params = {
    'L1' : lattice_dims[0],
    'L2' : lattice_dims[1],
    'boundary1' : 'Open',
    'boundary2' : 'PBC'
}

import afqmctools.systems.lattice as lat

lattice = lat.get_lattice(params=lattice_params)

Ne = int(lattice.N_sites*(1-0.2))//2
nelec = (Ne,Ne)
nelec
```

```{code-cell} ipython3
vis.plot_lattice(lattice)
```

+++ {"id": "a02dd178-90bb-463e-b5d6-75fccb288272"}

The Hamiltonian will be simple,

$$
H &= H_t + H_U + H_{pin} \\
H_t &= -t \sum_{\langle i j\rangle \sigma} c^\dagger_{i\sigma} c_{j\sigma} \\
H_U &= U\sum_i n_{i\uparrow}n_{i\downarrow}\\
H_{pin} &= \sum_{i\in \text{edge}} (-1)^{i_x+i_y} h_i S^z_i
$$

and we have a convenience function to add the edge pinning as well. We'll assume $t=1$, $h_i=0.5$ just like in the paper

```{code-cell} ipython3
---
id: 53474a16-1c9e-4c9a-9d49-7022c8a659f1
outputId: 731af8f0-e234-49d5-cfc5-d38fb5483845
colab:
  base_uri: https://localhost:8080/
---
builder = ham.HamiltonianBuilder(
          lattice=lattice,
          spin_symm="collinear" # we have no spin-flip terms
              )
# add standard Hubbard terms
builder.nth_neighbor_hopping(1.0)
builder.onsite_hubbard(U)
builder.afm_pinning(0.5)
builder.finalize()
```

```{code-cell} ipython3
:id: f94e92dd-9802-46ea-a1db-0baffb7efbb4

io.write_model_hamiltonian(builder.hamiltonian, "afqmc.h5",
                        nelec=nelec,spin_symm="collinear")
```

```{code-cell} ipython3
---
id: 5db7dbbb-0484-480c-8857-71986b9da4c4
outputId: a0cc1270-40df-46ca-fc5a-094aae93aa2b
colab:
  base_uri: https://localhost:8080/
---
# get a trial wavefunction: First, let's try a free-electron (i.e. non-interacting) wavefunction
from afqmctools.wavefunction.model import write_free_electron_wfn
write_free_electron_wfn(
    hamiltonian_fname="afqmc.h5",
    nelec=nelec
)
```

+++ {"id": "ec2d1ea0-fc24-45f6-a799-5e9d14e39639"}

Now we'll write the file to run AFQMC. There are some better tools than this but let's hard code some things for now

```{code-cell} ipython3
:id: 866c943b-b811-40b9-a8a5-e257e30bda24

def getInputJson(series,wf_fname,h_fname=None):
    if h_fname is None:
        h_fname = wf_fname
    return f'''{{
  "afqmc": {{
    "project": {{
      "id": "qmc",
      "series": {series},
      "mixed_precision": false
    }},
    "execute": {{
      "walker_set": {{
        "walker_type": "COLLINEAR"
      }},
      "wavefunction": {{
        "filename": "{wf_fname}"
      }},
      "hamiltonian": {{
        "filename": "{h_fname}"
      }},
      "timestep": 0.005,
      "steps": 20000,
      "accumlate_interval": 2,
      "measure_interval": 2,
      "population_control_interval" : 2,
      "walker_ortho_interval" : 2,
      "n_walkers_per_mpi_task": 540,
      "estimator": {{
        "name": "energy",
        "overwrite": true,
        "print_components": true
      }},
      "estimator": {{
        "name":"back_propagation",
        "path_restoration":true,
        "extra_path_restoration":true,
        "bp_walker_ortho_interval":2,
        "nsteps":500,
        "naverages":1,
        "equil":2000,
        "onerdm" : {{
          "name":"onerdm"
	      }}
      }}
    }}
  }}
}}
'''
```

```{code-cell} ipython3
---
id: b343c03f-ab1d-490a-a421-86a44864ecf4
outputId: 200baf2a-73d2-4f0e-d713-4cc0c39f277a
colab:
  base_uri: https://localhost:8080/
---
print(getInputJson(0,"afqmc.h5"))
```

```{code-cell} ipython3
:id: 6e930735-29d0-4dc0-838d-475283cc6bb4

with open("afqmc.json","w") as f:
    lines = getInputJson(0,"afqmc.h5")
    f.write(lines)
```

```{code-cell} ipython3
:id: 59c74aca-3a04-4677-b6ab-fa1e1d61d366
:outputId: df9d0f96-bb53-4ec7-a6cc-42462bd0a27e

!sbatch --wait run_afqmc.sh afqmc.json
```

```{code-cell} ipython3
---
id: 96ec3330-42f6-4b14-af45-7e18a5225f7f
outputId: 8c812e11-d1f3-454d-82c4-0322455b50bb
colab:
  base_uri: https://localhost:8080/
---
!mpirun -np 12 $AFQMC_EXEC afqmc.json
```

```{code-cell} ipython3
:id: e8e41aff-0a7a-49fd-bbe8-821dc49fc7a8
:outputId: 2091d2fd-7735-4d06-ef08-73f9d31f4ae8

!scalar_stats qmc.s000.scalar.dat -s time -t -e 10.0 --savefig energy_fe.png
```

+++ {"id": "711d86ce-c4f5-4b17-88b6-4f11c90a4f79"}

<img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/models/ex02_stripes/energy_fe.png" width="800px" />

+++ {"id": "e5e2892a-59fe-412e-9d8b-42f42c05ff79"}

### Analysis

We'll now analyze the spin density of the output given the free electron trial

```{code-cell} ipython3
:id: 08eaa53d-eed0-4540-9bb8-08ea5aeab4c0

def staggeredSigns(Nx,Ny):
  signs = np.concatenate([(-1)**(np.arange(Nx)+1),(-1)**(np.arange(Nx))]*(Ny//2)).reshape(Ny,Nx).T.flatten()
  return -signs

def averageOverCols(Sz,Nx,Ny):
    return (Sz*staggeredSigns(*lattice.L)).reshape(*lattice_dims).mean(1)
```

```{code-cell} ipython3
:id: 0a52fe29-2397-4436-b84d-48b95ac5f777

rho_avg, delta_rho = average_afqmc_rdm(rdm_file="qmc.s000.stat.h5")
```

```{code-cell} ipython3
:id: 6c223f77-f3f4-42a5-bacb-a0f84a868f2b

Xs,Ys,Zs = spobs.local_spin(np.vstack(rho_avg[0]),"collinear").real

delta_Sz = np.sqrt(delta_rho[0,0].diagonal()**2+delta_rho[0,1].diagonal()**2 )
# We're collinear
assert np.allclose(Xs,0)
assert np.allclose(Ys,0)
```

```{code-cell} ipython3
:id: 79ed1667-cd75-451d-b68a-5897ca697206
:outputId: fb38354c-9921-4f09-86ef-eb21cf200940

plt.imshow(Zs.reshape(*lattice.L).T)
plt.colorbar()
plt.show()

plt.imshow(delta_Sz.reshape(*lattice.L).T)
plt.colorbar()
plt.show()
```

```{code-cell} ipython3
:id: 1f80b790-9e93-4c4a-b043-f2aea22a0e11

# DMRG data of the form: site Sz Nup Ndn
SzDmrg = np.loadtxt("data_10x4_Ne16U6.0_15360.dat",usecols=(1,))
```

```{code-cell} ipython3
:id: 3790867c-321a-4e47-84f8-e82a1a407d07


```

```{code-cell} ipython3
:id: d2e9c723-0471-404c-acb5-30e862b7905f
:outputId: a53dc884-c7ab-43d4-9b39-1ad5c3e23957

# now we'll average over the columns

avg_delta_Sz = np.sqrt((delta_Sz.reshape(*lattice.L)**2).sum(1))

plt.errorbar(np.arange(lattice.L[0]),averageOverCols(Zs,*lattice.L),yerr=avg_delta_Sz,
             fmt='o-',label="AFQMC Free Electron")
plt.plot(-averageOverCols(SzDmrg,*lattice.L),label="DMRG",c='k')

plt.xlabel("Col")
plt.ylabel("Staggered Sz")
plt.legend()

# plt.ylim(-0.3,0.2)
plt.show()
```

+++ {"id": "e9fbcc21-8216-4f25-a64e-54f84dc4023c"}

## Self Consistency w/ Hartree-Fock

The idea behind self-consistent AFQMC is to have the trial wave function match as much as possible the output of AFQMC. [cite me]

For this model, we can do this by introducing an effective Hubbard model, where $U$ is replaced with $U_{eff}$. By solving Hartree-Fock (mean field theory) for this effective model and using the result as a trial wave function, we can scan through different $U_{eff}$ to find the one that matches the best.

We'll use AutoHF to explore creating a trial wave function for AFQMC. Let's do a few between $U_{eff}=1$ and $U_{eff}=4$

_Note:_ The Hartree-Fock code is faster if you use a GPU

```{code-cell} ipython3
:id: ee2b575c-7d27-4a1d-a9b9-e9b02a682f8f
:outputId: 743ad05d-6d2a-4eba-ec3c-92e78bcbe2e3

for Ueff in [1,2,3,4]:

    builder_eff = ham.HamiltonianBuilder(
              lattice=lattice,
              spin_symm="collinear"
                  )
    # add standard Hubbard terms
    builder_eff.nth_neighbor_hopping(1.0)
    builder_eff.onsite_hubbard(Ueff) # We want our HF solver to solve Ueff not U
    builder_eff.afm_pinning(0.5)
    builder_eff.finalize()

    # NOTICE: we must be careful here! We can either keep around afqmc.h5
    # which has the original Hamiltonian (U=6) and keep the wf and its hamiltonian together
    # or we have one file with afqmc_Ueff which has both the wf and builder.hamiltonian
    io.write_model_hamiltonian(builder_eff.hamiltonian, f"afqmc_{Ueff}.h5",
                            nelec=nelec,spin_symm="collinear")

    hf_settings = dict(
        numSteps = 2000,
        output = f"afqmc_{Ueff}.h5", # change file name!
        opt_method="lbfgs",
        ansatz="SD_ROT",
        nelec = nelec,
        numTrials = 8,
        seed = 1,
        noncollinear = False
    )
    data = autohf.solver.lattice_hf(
        hamiltonian=builder_eff.hamiltonian,
        lattice=lattice,
        settings=hf_settings,
    )
    print(data["E_final"])

    with open(f"afqmc_{Ueff}.json","w") as f:
        lines = getInputJson(Ueff,f"afqmc_{Ueff}.h5","afqmc.h5")
        f.write(lines)
    print(f"Written {Ueff}")
```

```{code-cell} ipython3
:id: 3d68db2c-e000-4c4f-8b48-2f4bd5836749

for Ueff in [1,2,3,4]:
    !sbatch --wait run_afqmc.sh afqmc_{Ueff}.json
```

```{code-cell} ipython3
:id: 9ac5439b-daa4-4cf4-a248-e2466acf32e5

rhos, deltas = [],[]

for Ueff in [0,1,2,3,4]:
    rho_avg, delta_rho = average_afqmc_rdm(rdm_file=f"qmc.s00{Ueff}.stat.h5")
    rhos.append(rho_avg)
    deltas.append(delta_rho)
```

```{code-cell} ipython3
:id: 7f7e1dc8-4378-4da3-a77a-8e2638ed3c0e

for Ueff,rho_avg,delta_rho in zip([0,1,2,3,4],rhos,deltas):
    Xs,Ys,Zs = spobs.local_spin(np.vstack(rho_avg[0]),"collinear").real

    delta_Sz = np.sqrt(delta_rho[0,0].diagonal()**2+delta_rho[0,1].diagonal()**2 )
    # We're collinear
    assert np.allclose(Xs,0)
    assert np.allclose(Ys,0)

    fig,axes = plt.subplots(1,2,figsize=(12,4))
    p = axes[0].matshow(Zs.reshape(*lattice.L).T)
    plt.colorbar(p,ax= axes[0])


    p =axes[1].matshow(delta_Sz.reshape(*lattice.L).T)
    plt.colorbar(p,ax= axes[1])
    if Ueff==0:
        fig.suptitle(f"Free Electron")
    else:
        fig.suptitle(f"{Ueff=}")
    plt.show()
```

```{code-cell} ipython3
:id: 283e9c2b-2cac-44e7-80d6-42225a466de9

# now we'll average over the columns

for Ueff,rho_avg,delta_rho in zip([0,1,2,3,4],rhos,deltas):
    Xs,Ys,Zs = spobs.local_spin(np.vstack(rho_avg[0]),"collinear").real

    delta_Sz = np.sqrt(delta_rho[0,0].diagonal()**2+delta_rho[0,1].diagonal()**2 )
    avg_delta_Sz = np.sqrt((delta_Sz.reshape(*lattice.L)**2).sum(1))
    if Ueff==0:
        label = "AFQMC Free Electron"
    else:
        label = f"AFQMC W/ HF Ueff={Ueff}"
    plt.errorbar(np.arange(lattice.L[0]),averageOverCols(Zs,*lattice.L),yerr=avg_delta_Sz,
                 fmt='o-',label=label)
plt.plot(-averageOverCols(SzDmrg,*lattice.L),'x-',label="DMRG",c='k')

plt.xlabel("Col")
plt.ylabel("Staggered Sz")
plt.legend()

# plt.ylim(-0.3,0.2)
plt.show()
```

+++ {"id": "8445abd3-e472-4406-9c2a-9ca3a8c8d3a3"}

## Self-Consistent Condition

We'll now measure the error between the trial wave function's observable and the output from AFQMC

```{code-cell} ipython3
:id: eff15403-1be8-4f08-8ed8-9ccf212e1d9f

trial_rhos = []
for Ueff in Ueffs:
    if Ueff==0:
        fname = "afqmc.h5"
    else:
        fname = f"afqmc_{Ueff}.h5"
    (coeffs,wfn), psi0, (na, nb),spintype = read_wavefunction(fname)
    # We assume spin balance below
    o = wfn.reshape(lattice.N_sites,na,2,order='F').real
    o = np.stack([o[:,:,0],o[:,:,1]])

    rdm = greens_1body(o)
    trial_rhos.append(rdm)
```

```{code-cell} ipython3
:id: 2926392c-64b7-4c5b-be5a-d289adc38ff5

# now we'll average over the columns
Ueffs = [0,1,2,3,4]
for Ueff,trial_rho in zip(Ueffs,trial_rhos):
    Xs,Ys,Zs = spobs.local_spin(np.vstack(trial_rho),"collinear").real


    avg_delta_Sz = np.sqrt((delta_Sz.reshape(*lattice.L)**2).sum(1))
    if Ueff==0:
        label = "Free Electron"
    else:
        label = f"HF W/ HF Ueff={Ueff}"
    plt.errorbar(np.arange(lattice.L[0]),
                 averageOverCols(Zs,*lattice.L),yerr=avg_delta_Sz,
                 fmt='o--',label=label)
plt.plot(-averageOverCols(SzDmrg,*lattice.L),'x-',label="DMRG",c='k')

plt.xlabel("Col")
plt.ylabel("Staggered Sz")
plt.legend()

# plt.ylim(-0.3,0.2)
plt.show()
```

```{code-cell} ipython3
:id: 61bcaf57-b7bd-4671-801f-31ab5d6684e1

Ueffs = [0,1,2,3,4]

dmrg_sz = -averageOverCols(SzDmrg,*lattice.L)

xs,ys,yerrs = [],[],[]
for Ueff,rho_avg,delta_rho,trial_rho in zip(Ueffs,rhos,deltas,trial_rhos):
    XsT,YsT,ZsT = spobs.local_spin(np.vstack(trial_rho),"collinear").real
    Xs,Ys,Zs = spobs.local_spin(np.vstack(rho_avg[0]),"collinear").real

    afqmc_sz = averageOverCols(Zs,*lattice.L)
    trial_sz = averageOverCols(ZsT,*lattice.L)


    delta_Sz = np.sqrt(delta_rho[0,0].diagonal()**2+delta_rho[0,1].diagonal()**2 )
    avg_delta_Sz = np.sqrt((delta_Sz.reshape(*lattice.L)**2).sum(1))

    xs.append(Ueff)
    ys.append(np.mean(np.abs(afqmc_sz-dmrg_sz)))
    yerrs.append(np.mean(avg_delta_Sz**2))

plt.errorbar(xs,ys,yerr=yerrs,fmt='o-',label="AFQMC",capsize=6)
plt.xticks(Ueffs)
plt.xlabel("Ueff")
plt.ylabel(r"$\langle |$DMRG $S_z - $AFQMC $S_z|\rangle$",fontsize=12)
plt.legend()
plt.grid()

# plt.ylim(-0.3,0.2)
plt.show()


xs,ys,yerrs = [],[],[]
for Ueff,rho_avg,delta_rho,trial_rho in zip(Ueffs,rhos,deltas,trial_rhos):
    XsT,YsT,ZsT = spobs.local_spin(np.vstack(trial_rho),"collinear").real
    Xs,Ys,Zs = spobs.local_spin(np.vstack(rho_avg[0]),"collinear").real

    afqmc_sz = averageOverCols(Zs,*lattice.L)
    trial_sz = averageOverCols(ZsT,*lattice.L)


    delta_Sz = np.sqrt(delta_rho[0,0].diagonal()**2+delta_rho[0,1].diagonal()**2 )
    avg_delta_Sz = np.sqrt((delta_Sz.reshape(*lattice.L)**2).sum(1))

    xs.append(Ueff)
    ys.append(np.mean(np.abs(afqmc_sz-trial_sz)))
    yerrs.append(np.mean(avg_delta_Sz**2))

plt.errorbar(xs,ys,yerr=yerrs,fmt='o-',label="AFQMC",capsize=6)
plt.xticks(Ueffs)
plt.xlabel("Ueff")
plt.ylabel(r"$\langle |$Trial $S_z - $AFQMC $S_z|\rangle$",fontsize=12)
plt.legend()
plt.grid()

# plt.ylim(-0.3,0.2)
plt.show()
```

+++ {"id": "9e472093-a34e-4083-bb1a-3e04e07f19d2"}

So we can see that somewhere around $U_{eff}\approx 3$
would have the lowest error for self consistency.
Despite the difference between DMRG and AFQMC being on average 0.01, there is a clear optimal trial wave function to use. For fine tuning between $U_{eff}\in [2,4]$, we'll need more sophisticated methods than presented here.

In our original study, we found the optimal $U_{eff} \approx 2.77$

+++ {"id": "672d9ba4-f659-466d-be57-8fa082fcfb0e"}

## Your Turn

### Part 1 – Recreate Fig 1.

+++ {"editable": true, "id": "0ac955ea-ab2d-49fb-b9dd-9a0fafdaf159"}

<div>
    <img src="https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/models/ex02_stripes/fig1.png" width="300px" />
</div>

+++ {"editable": true, "id": "27958cd8-4035-4720-82b8-c163b1a89e99"}

Let's recreate figure 1 from Xu et al, by changing

- Lattice size 10 × 4 to 20 × 4
- Doping 0.2 to 0.1 hole doping
- $U = 6$ ✓

+++ {"id": "edc3a318-40a1-40cb-86bf-b85ae366fc9b"}

### Part 2 - Try different settings

Lets use this as a test bed to explore how AFQMC responds to different settings. Some questions to consider

- Use $U_{eff} = U$ or the "bare U" Hartree-Fock trial. How does the quality of the results change?
- We compared spin densities, what about hole density? This is the effectively the same as comparing charge density.
- Use a trial with a small number of steps so that the solver doesn't converge. How do the results change? How could you tell this trial isn't good?
- The free electron trial isn't a valid RHF state (why?). What would be the corresponding RHF solution? What does that output look like?
- Explore the parameters of backpropagation, how does the 1rdm change as a function of `nsteps`? you can include `naverages` to see the results converge

