This example is deprecated and will be removed. See `docs/examples`.

In this example we will show how to format a casscf trial wavefunction.

The first step is to run a CASSCF calculation using the scf.py script.
We set up the :math:`N_2` dimer to replicate the calculations from Al-Saidi
et al J. Chem. Phys. 127, 144101 (2007).
They find a CASSCF energy of -108.916484 Ha, and a ph-AFQMC energy of -109.0884(7) Ha with
a 97 determinant CASSCF trial.

.. code-block:: python

    mol = gto.M(atom=[['N', (0, 0, 0)], ['N', (0, 0, 3.0)]],
                basis='cc-pvdz',
                unit='Bohr')
    rhf = scf.RHF(mol)
    rhf.chkfile = chkfile
    rhf.kernel()

    ncas = 12
    nactive = (3, 3)
    nmo = rhf.mo_coeff.shape[-1]
    mc = mcscf.CASSCF(rhf, ncas, nactive)
    mc.chkfile = chkfile
    mc.kernel()

Then, we can use ham.py to write the hamiltonian and write the wavefunction.
To write the CAS wavefunction, we first unpack the ci coefficients matrix

.. code-block:: python

    ci, occa, occb = zip(*fci.addons.large_ci(mc.ci, ncas, nactive,
                         tol=tol, return_strs=False))

Next we reinsert the frozen core as the AFQMC simulation is not run using an active space:

.. code-block:: python

    core = [i for i in range(mc.ncore)]
    occa = [numpy.array(core + [o + mc.ncore for o in oa]) for oa in occa]
    occb = [numpy.array(core + [o + mc.ncore for o in ob]) for ob in occb]

Next we need to generate the one- and two-electron integrals. Note that we need to use the
CASSCF MO coefficients to rotate the integrals.

.. code-block:: python

    scf_data = load_from_pyscf_chk_mol(chkfile, 'mcscf')
    write_hamil_mol(scf_data, 'afqmc.h5', 1e-5, verbose=True)

Finally we can write the wavefunction to the same HDF5 file for convenience.
The wavefunction can have its own HDF5 file.

.. code-block:: python

    ci = numpy.array(ci, dtype=numpy.complex128)
    uhf = True # UHF always true for CI expansions.
    write_wfn('afqmc.h5', (ci, occa, occb), uhf, mol.nelec, nmo)

To create the AFQMC input file, use `write_afqmc_json.py`

.. code-block:: bash

   write_afqmc_json.py -i afqmc.h5

**deprecated** Note the ``rediag`` option in wavefunction
is necessary if the CI code used uses a different convention for ordering creation and
annihilation operations when defining determinant strings.

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
