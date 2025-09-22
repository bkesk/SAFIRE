This example is deprecated and will be removed. See `docs/examples`.

In this example, we will show how to generate the AFQMC input for the
one-band Hubbard model.

.. math::
  :label: eq-hubarrd-4x4

  H = -t \sum\limits_{<i,j>} c^\dagger_{i\sigma} c_{j\sigma} + U \sum\limits_i n_{i\uparrow} n_{i\downarrow}

For simplicity, we set up the half-filled 4x4 square lattice at `U/t=4`
with nearest-neighbor hopping.
There are 16 single-particle orbitals (one per lattice site) in the basis.
To run AFQMC, we need to write the hamiltonian and a trial wavefunction in
this basis.

The one-body part of the hamiltonian encodes square lattice connectivity under
periodic boundary conditions.

.. code-block:: python

  lat = (4, 4)
  neighbors = np.array([
    [ 0, -1],
    [-1,  0],
    [ 1,  0],
    [ 0,  1],
  ])
  h1 = hubbard_hop(lat, neighbors)
  h1 = -t*h1

Without the -t, the 1-body hamiltonian looks like

.. code-block:: python

  [[0 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0]
   [1 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0]
   [0 1 0 1 0 0 1 0 0 0 0 0 0 0 1 0]
   [1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1]
   [1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0]
   [0 1 0 0 1 0 1 0 0 1 0 0 0 0 0 0]
   [0 0 1 0 0 1 0 1 0 0 1 0 0 0 0 0]
   [0 0 0 1 1 0 1 0 0 0 0 1 0 0 0 0]
   [0 0 0 0 1 0 0 0 0 1 0 1 1 0 0 0]
   [0 0 0 0 0 1 0 0 1 0 1 0 0 1 0 0]
   [0 0 0 0 0 0 1 0 0 1 0 1 0 0 1 0]
   [0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 1]
   [1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1]
   [0 1 0 0 0 0 0 0 0 1 0 0 1 0 1 0]
   [0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 1]
   [0 0 0 1 0 0 0 0 0 0 0 1 1 0 1 0]]

The two-body part is only non-zero for on-site opposite-spin interaction.

.. code-block:: python

  # uniform on-site U
  nsite = len(h1)
  Uij = U*np.diag(np.ones(nsite))

To obtain an unrestricted Hartree-Fock (UHF) trial wavefunction, we will use PySCF's custom hamiltonian feature

.. code-block:: python

  def get_veff(mol, dm, *args):
    j_a = np.diag(np.einsum('ii->i', dm[0]) * U)
    k_a = np.diag(np.einsum('ii->i', dm[0]) * U)
    j_b = np.diag(np.einsum('ii->i', dm[1]) * U)
    k_b = np.diag(np.einsum('ii->i', dm[1]) * U)
    j = j_a + j_b
    veff_a = j-k_a
    veff_b = j-k_b
    return veff_a, veff_b
  
  mf = scf.UHF(mol)
  mf.get_hcore = lambda *args: h1
  mf.get_ovlp = lambda *args: np.eye(len(h1))
  mf.get_veff = get_veff

The UHF calculation is implemented in `scf/scf.py`.
Once the scf calculation is finished, we can extract all the necessary
information for the AFQMC run from the chkfile using `ham/ham.py`

.. code-block:: python

  from afqmctools.hamiltonian.hubbard import write_hubbard
  chkfile = '../scf/n4,4-U4.00-ne8,8.h5'
  nelec = read_nelec(chkfile)

  # write Hamiltonian
  h1 = read_hcore(chkfile)
  U = read_hubbard(chkfile)
  # uniform on-site U
  norb = len(h1)
  Uij = U*np.diag(np.ones(norb))
  write_hubbard('afqmc.h5', h1, Uij, nelec)

The wavefunction is also written to the same file in the same script

.. code-block:: python

  from afqmctools.wavefunction.mol import write_wfn
  # write Wavefunction
  mo_coeff = read_mo_coeff(chkfile)

  nup, ndn = nelec
  ci = np.array([1.0+0j])
  wfn = np.zeros((len(ci), norb, nup+ndn), dtype=np.complex128)
  wfn[0,:,:nup] = mo_coeff[0][:,:nup]
  wfn[0,:,nup:] = mo_coeff[1][:,:ndn]

  uhf = True
  write_wfn('afqmc.h5', (ci, wfn), uhf,
                    nelec, norb, verbose=True)

To create the AFQMC input file, use `write_afqmc_json.py`.

.. code-block:: bash

   write_afqmc_json.py -i afqmc.h5 -b 400


Next, invoke the qmcapp executable. A sample Slurm runscript is given below,
where `AFQMC_PATH = "[path to afqmc code build]"` should be set to the build 
directory.

.. code-block:: bash

  #!/bin/bash -l
  #SBATCH --constraint=ib
  #! Number of MPI ranks (= tasks for Slurm)
  #SBATCH --ntasks=160
  #SBATCH --time=4:00:00

  module purge

  export AFQMC_PATH = "[path to afqmc code build]"
  source $AFQMC_PATH/env.bash 

  export AFQMC_EXEC=$AFQMC_PATH/bin/qmcapp

  # Launch MPI code...
  srun --cpu-bind=cores $AFQMC_EXEC --filenames afqmc.json


Finally, the AFQMC results can be analyzed as described in `examples/analysis/energy`.
