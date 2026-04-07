.. _setup_ex_11:

Adding Custom Terms to the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This advanced example covers adding a manually-constructed one-body term to a
lattice model Hamiltonian by directly invoking the HamiltonianBuilder.

.. literalinclude:: run.py

If the sample Python script is run above with the sample input file, the following
should be present at the end of the output.

.. literalinclude:: output.txt

See the examples in :ref:`run_afqmc_exs` for how to run AFQMC.
