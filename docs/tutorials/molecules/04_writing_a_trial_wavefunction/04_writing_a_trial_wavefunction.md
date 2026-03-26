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

+++ {"id": "P8_OXxiRE_dA"}

# Writing a Trial Wavefunction


<b>Goal:</b>
Become acquainted with how to write a Trial wavefunction to the SAFIRE HDF5 format.




## What you will learn

1.  How to write a single Slater determinant trial wavefunction given the Slater Matrix
2.  How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices </font>
3.  How to write a configuration-interaction (CI) type trial wavefunction given CI coefficients and a corresponding occupancy strings </font>



+++

## ▶️ Run the cell below to setup the tutorial

```{code-cell} ipython3
# Run me (shift+enter or click the play button) to setup the tutorial!
from pathlib import Path

# simple setup
from tutorial_utils import run_afqmc, get_scratch_dir

#TODO: update this to a good directory for scratch files
home = Path.home() / ".scratch"
scratch_dir = get_scratch_dir("04_mols_writing_a_trial", home)
```

+++ {"id": "vt9PbuQhpw4u"}



## Introduction




In general, the trial wavefunction in AFQMC is a linear combination of Slater determinants,


$$
|\Psi_T \rangle = \sum^{N_{det}}_n C_n | \Phi_n \rangle,
$$


where

$C_n$

is a complex-valued coefficient,
and $|\Phi_n\rangle$ are Slater determinants which are not necessarily
orthogonal to each other.
Of course, each Slater determinant consists of some set of single-particle orbitals, $\{ \psi_p \}$, such that,
$$
\psi_{p} = \sum_i \bar{C}_{ip} \phi_i,
$$
where $\{\phi_i\}$ are the chosen orthonormal basis
set orbitals.
Slater determinants can either be represented explicitly as a Slater matrix,  



$$
\Phi ≐ \begin{bmatrix}
    \bar{C}_{00} &\bar{C}_{01} & \bar{C}_{02} & \dots  & \bar{C}_{0N} \\
    \bar{C}_{10} & \bar{C}_{11} & \bar{C}_{12} & \dots  & \bar{C}_{1N} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \bar{C}_{M0} & \bar{C}_{M1} & \bar{C}_{M2} & \dots  & \bar{C}_{MN}
\end{bmatrix},
$$



where $M,N$ are the number of basis functions and electrons, respectively,
or in terms of an "occupation vector", which is simply a list of orbital indices which should be occupied.
    





SAFIRE implements two basic types of trial wavefunction.


1. <b>"particle-hole" multi-Slater determinant (ph-mSD) trial wavefunctions</b>
which is a configuration interaction-like wavefunction
where  $\langle \Phi_n |\Phi_m\rangle = \delta_{nm} \forall n,m$. In this case,
the Slater determinants are specified in terms of a set of orbitals, and a list
of $C_n$ and corresponding occupancy vectors. 

2. <b> non-orthogonal multi-slater Determinant (NOMSD) trial wavefunction </b>
where **strictly** $\langle \Phi_n |\Phi_m\rangle \neq 0 \forall n,m$.
In this case, the Slater determinants are specified as a list
of $C_n$ and corresponding Slater matrices. 




We'll explore each of these wavefunction types in more detail below.

<!--
<div class="alert alert-block alert-warning">
<b>Important:</b>
When using NOMSD trial wavefunctions, it is important
that each Slater determinant has a non-zero overlap with all
other Slater determinants to avoid singular values.
Use ph-mSD for orthogonal expansions of Slater determinants.
</div>

-->


+++ {"id": "RXzexuh_sQyL"}



## Single Slater determinant trial wavefunction

We've already seen examples of writing a single Slater determinant trial wavefunction in previous tutorials.
Here, we will explain the code blocks that you have already seen which we have reproduced below.

Technically, we have been writing single Slater determinant trial wavefunctions as a non-orthogonal multi-Slater determinant (NOMSD) expansion with only a single determinant.
To accomplish this, we wrote down a set of orbitals in the chosen orthonormal orbital basis as,

```python
import numpy as np

orbitals = np.array(
    [[1.0,0.0],
    [0.0,1.0]]
)
```

where we are using the RHF orbitals as a basis to represent the RHF orbitals.

Next, we constructed the Slater matrix, $\Phi_{ip}$, and wrote down a trivial expanstion coefficient with,

```python
phi_0 = np.array([
    orbitals[:, :number_of_electrons[0]]
])
C_0 = 1.0
```

Finally, we write this to an HDF5 file in the SAFIRE format using the `write_wfn()` function as,

```python
from afqmctools.wavefunction.mol import write_wfn

wfn = ( np.array([C_0]), phi_0)

write_wfn(
    filename=scratch_dir/ "rhf_wavefunction.h5",
    wfn=wfn,
    walker_type="rhf",
    nelec=number_of_electrons,
    norb=number_of_orbitals
)
```

Note that we write the "wavefunction" as a tuple of length two where the first entry is a 1-dimensional array with the CI coefficient, and the second entry is a 3 dimensional array where the first index corresponds to the Slater determinant index, the second index corresponds to the basis set index, and the third index correpsonds to the electron index.

We put all of this together in the following code block.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 385
id: locAjSqZsYSI
outputId: b67f9653-4cfe-4ef4-ad16-1b9b5502444f
---
import numpy as np

from afqmctools.wavefunction.mol import write_wfn

number_of_electrons = (1,1)
number_of_orbitals = 2

orbitals = np.array(
    [[1.0,0.0],
    [0.0,1.0]]
)

phi_0 = np.array([
    orbitals[:, :number_of_electrons[0]]
])
C_0 = 1.0

wfn = ( np.array([C_0]), phi_0)

write_wfn(
    filename=scratch_dir/ "rhf_wfn.h5",
    wfn=wfn,
    walker_type="rhf",
    nelec=number_of_electrons,
    norb=number_of_orbitals
)
```

we can use the `h5dump` utility to inspect the contents of the file that we just generated. 
`h5dump -n [filename]` will show us which datasets exist, and we can use `h5dump -d [dataset] [filename]` to
view the actual data.

Note that the NOMSD format is a compressed sparse row (CSR) format.
See the wavefunction formats in the [user manual](https://users.flatironinstitute.org/~beskridge/safire/user_manual/wavefunctions.html) for
details on the expected data.

```{code-cell} ipython3
!h5dump -n {scratch_dir}/rhf_wfn.h5
```

```{code-cell} ipython3
!h5dump -d /Wavefunction/NOMSD/PsiT_0/data_ {scratch_dir}/rhf_wfn.h5
```

```{code-cell} ipython3
!h5dump -d /Wavefunction/NOMSD/PsiT_0/jdata_ {scratch_dir}/rhf_wfn.h5
```

+++ {"id": "VJfGuez4rsE-"}



## NOMSD Trial Wavefunction


In the previous example, we learned how to specify a trial wavefunctional with a single Slater determinant. Here, we will see how to we can write a suitable non-orthogonal multi-Slater trial wavefunction for the AFQMC code.

Similarly to the single Slater determinant case, we begin by specifying a set of orbitals. In this case, however, we chose a set of non-orthognal orbitals represented in a common basis:

```python
import numpy as np

orbitals = np.array(
    [[1.0,0.7071],
    [0.0,0.7071]]
)
```
Note that each column of the `orbitals` object are normalized, but they are not orthogonal to each other.

Next, we constuct two determinants with a single electron each. The first determinant will be specified by the orbital, whereas the second orbital will be used in the other determinant.

```python
phi_nomsd = np.array([
    orbitals[:, :number_of_electrons[0]], # Representing the first determinant, with the first orbital occupied
    orbitals[:, 1:number_of_electrons[0]], # Representing the second determinant, with the second orbital occupied
])
C_0 = 1.0
C_1 = 1.0
```

The dimensions of `phi_nomsd` should be (N$_\mathrm{dets}$,N$_\mathrm{basis}$,N$_\mathrm{elec}$), as previously indicated in the single determinant case.

Finally, we write this to an HDF5 file in the SAFIRE format using the `write_wfn()` function as,

```python
from afqmctools.wavefunction.mol import write_wfn

wfn = ( np.array([C_0,C_1]), phi_nomsd)

write_wfn(
    filename=scratch_dir/ "NOMSD_wfn.h5",
    wfn=wfn,
    walker_type="rhf",
    nelec=number_of_electrons,
    norb=number_of_orbitals
)
```

We put all of this together in the code block below.

```{code-cell} ipython3
:id: 8Y0sDP2wr1vk

number_of_electrons = (1,1)
number_of_orbitals = 2

orbitals = np.array(
    [[1.0,0.7071],
    [0.0,0.7071]]
)

phi_nomsd = np.array([
    orbitals[:, :number_of_electrons[0]],                                                # Representing the first determinant, with the first orbital occupied
    orbitals[:, number_of_electrons[0]:number_of_electrons[0] + number_of_electrons[1]], # Representing the second determinant, with the second orbital occupied
])
C_0 = 1.0
C_1 = 1.0

wfn = ( np.array([C_0,C_1]), phi_nomsd)

write_wfn(
    filename=scratch_dir/ "NOMSD_wfn.h5",
    wfn=wfn,
    walker_type="rhf",
    nelec=number_of_electrons,
    norb=number_of_orbitals
)
```

```{code-cell} ipython3
!h5dump -n {scratch_dir}/NOMSD_wfn.h5
```

notice that there are two Slater determinants in the HDF5 file! We note that
, in the collinear format, the spin up and spin down Slater determinants are stored as separate data sets with even (odd) indices
corresponding to up (down) sector Slater determinants.

```{code-cell} ipython3
!h5dump -d /Wavefunction/NOMSD/PsiT_0/data_ {scratch_dir}/NOMSD_wfn.h5
```

```{code-cell} ipython3
!h5dump -d /Wavefunction/NOMSD/PsiT_1/data_ {scratch_dir}/NOMSD_wfn.h5
```

+++ {"id": "YtBpL-q8rlGT"}



## ph-mSD Trial Wavefunctions.

In the previous sections, we learned (i) how to construct trial wavefunctions with a single Slater determinant, (ii) how to construct non-orthogonal multi-Slater determinant trial wavefunctions (NOMSD). Here, we will look at how to use CI-like expansions, also called particle-hole multi Slater (ph-mSD) wavefunctions, as trial wavefunctions for AFQMC.

In contrast with the NOMSD case where each Slater determinant had its own set of molecular orbitals, ph-mSD wavefunctions are characterized by a single set of molecular orbitals. Orthogonal Slater Determinants are generated by occupying different orbitals contained in the set. For instance, let's take the following set of orbitals $ \{\phi_g, \phi_u \}$, with $\langle \phi_g | \phi_u \rangle = 0 $, and construct two Slater determinants
$$
|\Psi_0 \rangle = a_{g,\downarrow}^\dagger a_{g,\uparrow}^\dagger |0\rangle
$$
and
$$
|\Psi_1 \rangle = a_{u,\downarrow}^\dagger a_{u,\uparrow}^\dagger |0\rangle
$$

You can verify that these two Slater determinants are also orthogonal to each other $\langle \Psi_0 | \Psi_1 \rangle = 0$. Taking $|\Psi_0 \rangle$ to be our reference state, we can re-express the other determinant as
$$
| \Psi_1 \rangle = a_{u,\downarrow}^\dagger a_{u,\uparrow}^\dagger a_{g,\downarrow} a_{g,\uparrow} |\Psi_0 \rangle
$$
where we explicitly create holes (through the $a_{g,\downarrow} a_{g,\uparrow}$ operators) and particles (through the $a_{u,\downarrow}^\dagger a_{u,\uparrow}^\dagger$) operators on top of a reference state.

For the tooling side of the AFQMC, to write a ph-mSD trial wavefunction we need to specify an array with the occupied orbitals in each Slater determinant. $α$ and $β$ electrons should be specified separately. Finally, we need to provide an array containing the normalized weights/coefficients associated with each determiant in our trial wavefunction.
Combining these ingredients (array of normalized coefficients, array of occupied $α$ orbitas and array of occupied $β$ orbitas) as a tuple objects allows us to write the ph-mSD wavefunction as an hdf5 file for the AFQMC code

In our example, if we enumerate the orbitals in set starting with 0, we have:

```{code-cell} ipython3
:id: 9u7rGYAqrlrV

import numpy as np

number_of_electrons = (1,1)
number_of_orbitals = 2

# Specify an array containing the normalized coefficients for each
# Slater Determinant.
Cn = np.array([0.9, np.sqrt(1 - 0.9**2)])

# Specify arrays for the α and β occupied orbitals for Ψ0.
occa_0 = np.array([0])
occb_0 = np.array([0])

# Specify arrays for the α and β occupied orbitals for Ψ1.
occa_1 = np.array([1])
occb_1 = np.array([1])

# Combine the α and β occupied arrays for each determinant in a single object
# for each spin channel.
occa = np.array((occa_0, occa_1))
occb = np.array((occb_0, occb_1))

# Create a tuple object with the coefficients, α and β occupied arrays
# to write our ph-mSD trial wavefunction in the appropriate hdf5 format.
wfn = ( Cn, occa, occb )


write_wfn(
    filename=scratch_dir/ "PHMSD_wfn.h5",
    wfn=wfn,
    walker_type="uhf",
    nelec=number_of_electrons,
    norb=number_of_orbitals
)
```

```{code-cell} ipython3
!h5dump -n {scratch_dir}/PHMSD_wfn.h5
```

here, the intersting information is the orbital occupancies in `/Wavefunction/PHMSD/occs` and
the CI coefficients in `/Wavefunction/PHMSD/ci_coeffs`.

```{code-cell} ipython3
!h5dump -d /Wavefunction/PHMSD/occs {scratch_dir}/PHMSD_wfn.h5
```

```{code-cell} ipython3
!h5dump -d /Wavefunction/PHMSD/ci_coeffs {scratch_dir}/PHMSD_wfn.h5
```

+++ {"id": "FoovlUVcFqxX"}

## Summary



In this tutorial, you became acquainted with how to write a Trial wavefunction to the SAFIRE HDF5 format.





### What you learned



1. How to write a single Slater determinant trial wavefunction given the Slater Matrix
2. How to write a single Slater determinant given orbtial occupancies
3. How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices
4. How to write a configuration-interaction (CI) type trial wavefunction given CI coefficients and a corresponding occupancy strings





```{code-cell} ipython3
:id: qVBQ6CAhF3Xn


```
