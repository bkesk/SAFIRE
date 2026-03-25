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

Quantum Chemistry / Molecules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/molecules/index.rst


Ab initio solids
~~~~~~~~~~~~~~~~

We use Quantum Espresso to perform
DFT calculations, and Coquí to generate a Hamiltonian and write trial wavefunctions.

1. `Charge density in Si - CoQuí interface <examples/solids/01_Si_density_coqui_interface/index.html>`_
2. `Carbon diamond PySCF interface <examples/solids/02_C_diamond_pyscf_interface/index.html>`_
3. `Solid Na momentum distribution <examples/solids/03_Na_momentum_distribution/index.html>`_
4. `Bulk modulous of MnO <examples/solids/04_NaCl_bulk_modulus/index.html>`_
5. `Band gap of LiH <examples/solids/05_LiH_band_gap/index.html>`_
6. `Magnetization of ? <examples/solids/06_magnetization/index.html>`_


Lattice models
~~~~~~~~~~~~~~

We use autoHF to perform
HF calculations, and afqmctools to generate Hamiltonians and write trial wavefunctions.

1. `4x4 Hubbard Model with open shell <examples/models/01_4x4_hubbard_model/index.html>`_
2. `Stripe Ordering <examples/models/02_stripes/index.html>`_
3. `Hubbard at half-filling <examples/models/03_half_filling_hubbard/index.html>`_
4. `pair correlation functions <examples/models/04_pair_correlators/index.html>`_
5. `Multi-Slater determinant trial wavefunction <examples/models/05_multi_slater_trial/index.html>`_
6. `Hubbard on a Honeycomb lattice <examples/models/06_honeycomb_hubbard/index.html>`_
7. `Hubbard with t-prime <examples/models/07_tprime_hubbard/index.html>`_
8. `Emery Model <examples/models/08_lieb_lattice_emery/index.html>`_


.. _snippet_examples:

Snippets
========

Concise, targeted examples that focus on one specific action, option, or setting.
Snippets are designed as quick references — you’ll find exactly how to do it without extra explanation.

.. toctree::
  :maxdepth: 1
  :caption: Examples:

  snippets/01_setting_up/setup_examples
  snippets/02_running_afqmc/afqmc_examples
  snippets/03_analyzing_results/analysis_examples


