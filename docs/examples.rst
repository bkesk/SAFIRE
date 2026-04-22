.. _user_examples:

Examples
========

This section provides practical, hands-on examples to help you get the most out of SAFIRE.
Whether you want to see how a complete workflow unfolds or just need a quick reminder of how 
to perform a single task, you'll find both here. 

.. _walkthrough_examples:

Walkthroughs
============

Step-by-step walkthroughs that carry you from start to finish through a complete process. 
These show how multiple features fit together in real-world scenarios, 
making them useful for understanding context and best practices.
The walkthroughs are organized based on their application domain.


.. _examples_molecules:

Quantum Chemistry / Molecules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide the following worked examples in which we
go through the entire workflow. 
The quantum chemistry workflow requires some external 
quantum chemistry code to generate integrals and a trial wavefunction.
For the convenience, we use
PySCF since it can be directly invoked within
interactive Python notebooks; however, using the information
in the tutorials, these same calculations can be performed starting
from other quantum chemistry codes so long as they can write a FCIDUMP,
and can print wavefunction information. Both of these features are
ubiquitous in modern quantum chemistry codes.

.. toctree::
   :numbered:
   :glob:
   :maxdepth: 1

   /examples/molecules/0*/*

.. 3. :doc:`Charge density of the water molecule </examples/molecules/tbd_H2O_charge_density/index>`

.. _examples_solids:

Ab initio solids
~~~~~~~~~~~~~~~~

For ab initio solids, we provide the following worked examples. In conjunction with SAFIRE, we use Quantum Espresso to perform
DFT calculations, and Coquí to generate a Hamiltonian and write trial wavefunctions.


.. toctree::
   :numbered:
   :glob:
   :maxdepth: 1

   /examples/solids/0*/*

.. _examples_models:

Lattice models
~~~~~~~~~~~~~~

For lattice models, we provide the following worked examples, where we use autoHF to perform
HF calculations, and afqmctools to generate Hamiltonians and write trial wavefunctions.


.. toctree::
   :numbered:
   :glob:
   :maxdepth: 1

   /examples/models/0*/*

.. _snippet_examples:

Snippets
========

Concise, targeted examples that focus on one specific action, option, or setting.
Snippets are designed as quick references — you’ll find exactly how to do it without extra explanation.

.. toctree::
  :maxdepth: 1

  snippets/01_setting_up/setup_examples
  snippets/02_running_afqmc/afqmc_examples
  snippets/03_analyzing_results/analysis_examples


