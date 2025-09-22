In this example, we will show how to generate the AFQMC input from a pbc
pyscf KRHF calculation for a 2x2x2 supercell of diamond.

First run the pyscf calculation using the scf.py script.

In addition to the standard pyscf calculation, we need to use the orthoAO.py
script to add some more information to the chkfile

.. code-block:: python

  from afqmctools.utils.linalg import get_ortho_ao
  hcore = mf.get_hcore()
  fock = (hcore + mf.get_veff())
  X, nmo_per_kpt = get_ortho_ao(cell,kpts)
  with h5py.File(chkfile, 'a') as fh5:
    fh5['scf/hcore'] = hcore
    fh5['scf/fock'] = fock
    fh5['scf/orthoAORot'] = X
    fh5['scf/nmo_per_kpt'] = nmo_per_kpt

storing the fock matrix, core Hamiltonian and rotation matrix to the
orthogonalised AO basis. This is currently required for running PBC AFQMC calculations.

Notice, this workflow is constructed for clarity not speed.
If possible, concatenating orthoAO.py at the end of scf.py is more efficient because

.. code-block:: python

  mf.get_veff()

can be an expensive operation.

Once the above scripts are run, we can use the `pyscf_to_afqmc.py` script
to generate the necessary AFQMC input file.

.. code-block:: bash

  mpirun -n 8 pyscf_to_afqmc.py -i ../scf/chkfile.h5 -o afqmc.h5 -a -t 1e-5 -v

The `-a` flag instructs the converter to use orthogoanlized AOs as basis.
We also accelerate their calculation using MPI.
If there are more tasks than kpoints, then
the number of tasks must be an integer multiple of the number of kpoints.

This will generate a Hamiltonian file that is identical to the molecular
calculations. This is because we have not exploited k-point symmetry.

Adding the `-kp` flag allows the converter to exploit k-point symmetry

.. code-block:: bash

  mpirun -n 8 pyscf_to_afqmc.py -i ../scf/chkfile.h5 -o afqmc.h5 -a -kp -t 1e-5 -v

This will allow AFQMC obtain the same results with less memory and time.

To create the AFQMC input file, use `write_afqmc_json.py`

.. code-block:: bash

 write_afqmc_json.py -i afqmc.h5

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
