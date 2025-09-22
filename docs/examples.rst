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

The quantum chemistry workflow requires some external 
quantum chemistry code to generate integrals and a trial wavefunction.
For the convenience, we use
PySCF since it can be directly invoked within
interactive Python notebooks; however, using the information
in the tutorials, these same calculations can be performed starting
from other quantum chemistry codes so long as they can write a FCIDUMP,
and can print wavefunction information. Both of these features are
ubiquitous in modern quantum chemistry codes.

1. `Potential energy curve of the nitrogen dimer <https://colab.research.google.com/drive/1mGHyp2hxF-upogHeI_uTSlyegSyqimpu#scrollTo=HAXNMGv0fmdE>`_
2. `Electron Affinity of the lead atom <https://colab.research.google.com/drive/1RUJ1buSrK8rBjreHA0oaPfOGPBuF3PAp#scrollTo=8rI2HhjiZRWO>`_
3. `Charge density of the water molecule <https://colab.research.google.com/drive/1uiF2R6CE_cSQ7m9uj2R2aS0KB1Un3Bn_>`_ 
4. `B atom : SHCI trial wavefunction <https://colab.research.google.com/drive/1gC5CtD4Kw8PXFuPKAWuQReCEwzVQrvvd#scrollTo=UwuNT7DqI_rN>`_
5. `Local embedding <https://colab.research.google.com/drive/1571aeCThqCVBlQeuXIASTbGRmD5BNlu8>`_
6. `Ionization Potential <https://colab.research.google.com/drive/1We4jEkm_Vx8RLhGiXP16wGT6gw2WaM84>`_
7. `3d-Transition Metal diatomics <https://colab.research.google.com/drive/1aLP4sl0Xe0ZvKlJl1jCOikYx_g6G6UEO>`_


Ab initio solids
~~~~~~~~~~~~~~~~

We use Quantum Espresso to perform
DFT calculations, and Coquí to generate a Hamiltonian and write trial wavefunctions.

1. `Charge density in Si - CoQuí interface <https://colab.research.google.com/drive/1V2qDFA5PJkM2xeliDLsKuTrWR1FpqTal>`_
2. `Carbon diamond PySCF interface <https://colab.research.google.com/drive/172tZtwLFCHI-bW6eIVXuLRgnJqPun8z6>`_
3. `Solid Na momentum distribution <https://colab.research.google.com/drive/1qNBPcToh58UJPeg9Qx6YxcIS6RPL5a2o>`_
4. `Bulk modulous of MnO <https://colab.research.google.com/drive/1oJRRBTptrW_66NRWs7CdQix68Qgz9WuX>`_
5. `Band gap of LiH <https://colab.research.google.com/drive/17x0SnUEhi-yEuyk_gMFbIGNr_SWGUi9g>`_
6. `Magnetization of ? <https://colab.research.google.com/drive/1In-gtDJ6Ud3MwHJEJdQS-c66Y6F7wpud>`_


Lattice models
~~~~~~~~~~~~~~

We use autoHF to perform
HF calculations, and afqmctools to generate Hamiltonians and write trial wavefunctions.

1. `4x4 Hubbard Model with open shell <https://colab.research.google.com/drive/1s9bdH7XgEj4qP982ZZTOGSsQkeMgAJdX>`_
2. `Stripe Ordering <https://colab.research.google.com/drive/1HrfXBp0SkiGzYWvKKJvKBQwdL-psWV-n>`_
3. `Hubbard at half-filling <https://colab.research.google.com/drive/1MjA5PnZC5V1qRjINxXdqbs1majV_FTWf>`_
4. `pair correlation functions <https://colab.research.google.com/drive/1rI5zbAB6tMAFJWRGOp1zxIpDAZIh9LTW>`_
5. `Multi-Slater determinant trial wavefunction <https://colab.research.google.com/drive/1yTDk1u8Ww-c1t6kUKhBkEAKUIsnmGm3e>`_
6. `Hubbard on a Honeycomb lattice <https://colab.research.google.com/drive/1g9_84kc92vAyDfKHTjDiuWHXeJsp12vZ>`_
7. `Hubbard with t-prime <https://colab.research.google.com/drive/1eR1RNS-BiMToOswQ4OyJWeijcpW21a4J>`_
8. `Emery Model <https://colab.research.google.com/drive/1s3nON1d7OvS7sOJBSPZZhBGj8fWm_Vsh>`_

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


