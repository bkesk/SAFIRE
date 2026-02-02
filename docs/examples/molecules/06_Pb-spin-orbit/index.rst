.. _example_molecule_pb_spin_orbit:

Lead Atom with Spin-Orbit Coupling
===================================

This example shows how to handle spin-orbit coupling effects in AFQMC calculations on a heavy element (lead).

**System**: Pb atom  
**Basis**: Augmented cc-pVDZ with effective core potential (ECP)  
**Special Feature**: Spin-orbit coupling treatment using PySCF ECP integration

Files
-----

**SCF with Spin-Orbit Coupling** (``scf/scf.py``):

.. literalinclude:: scf/scf.py
   :language: python

**Hamiltonian and Wavefunction Generation** (``ham/ham.py``):

This script generates Hamiltonians for both scalar-relativistic and spin-orbit coupling treatments:

.. literalinclude:: ham/ham.py
   :language: python

The script writes two HDF5 files: one for scalar-relativistic calculations (``afqmc_sf.h5``) and one with spin-orbit coupling (``afqmc_soc.h5``).

Running the Example
-------------------

This example demonstrates treating spin-orbit coupling (SOC) in heavy elements:

1. **Generate trial wavefunctions**: Execute ``scf/scf.py`` to run multiple SCF calculations:
   
   - ROHF calculation for orbital basis
   - GHF calculation for scalar-relativistic treatment
   - GHF calculation with SOC using an effective core potential (ECP) that includes relativistic effects and spin-orbit coupling

2. **Create AFQMC inputs**: Execute ``ham/ham.py`` to generate two separate Hamiltonians:
   
   - ``afqmc_sf.h5``: Scalar-relativistic (spin-free) Hamiltonian using ``walker_type='ghf'``
   - ``afqmc_soc.h5``: Hamiltonian with spin-orbit coupling using ``with_soc=True``

3. **Run SAFIRE**: Execute separate AFQMC calculations with the HDF5 files:

   .. code-block:: bash

      # Scalar-relativistic calculation
      mpirun -n 32 safire --filenames afqmc_sf.h5
      
      # Spin-orbit coupling calculation
      mpirun -n 32 safire --filenames afqmc_soc.h5

   Comparing the two results shows the magnitude of SOC effects on the Pb atom energy.

See ``run.sh`` for automated execution. The ``clean_ex.sh`` script removes generated files for a fresh run.
