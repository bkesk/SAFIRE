.. _setup_ex_12:

Adding Rashba SOC to the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example covers adding Rashba SOC to the Hamiltonian.
Currently, this is only supported when working directly with the Hamiltonian Builder
in Python without using an inputfile.

.. literalinclude:: run.py

If the sample Pyhton script is run above with the sample input file, the following
should be present at the end of the output.

.. literalinclude:: output.txt

See the examples in :ref:`run_afqmc_exs` for how to run AFQMC.
