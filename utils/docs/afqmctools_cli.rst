.. _afqmctools_cli:

Command line tools
==================

Overview
--------

afqmctools provides a command line interface (CLI) to its main features for convenience.
The command line interface consists of simple wrappers of the AFQMCTools Python package.
Of course, directly using the AFQMCTools Python package is more flexible
than using the CLI.
The CLI aims to provide a simple route to setting up and analyzing the most common AFQMC use cases.

.. note::

    It is highly recommended that afqmctools be installed according to the instructions in :ref:`afqmctools`.
    This ensures that the CLI tools are easily available without needing to modify your PATH.

Command-line Usage
------------------

afqmctools includes several command line utilities to automate the generation of inputs,
and some analysis of the output data.
Each script has a help menu that can be
displayed using ``$ [script name] --help``.
The utilities are listed below in alphabetical order.

For example,


.. code-block:: bash

    $ . afqmctools/bin/active
    (afqmctools) $ write_afqmc_json --help
    usage: write_afqmc_json [-h] [--fout FOUT] [--fwfn FWFN] [--fham FHAM] [--verbose] [--steps STEPS] [--timestep TIMESTEP]
                            [--n_walkers N_WALKERS] [--mixed_est] [--time_bp TIME_BP] [--num_bp NUM_BP]
                            [--path_restoration PATH_RESTORATION]

    Writes an AFQMC json input file based on the settings provided.

    options:
    -h, --help            show this help message and exit
    --fout FOUT, -o FOUT
    --fwfn FWFN, -i FWFN
    --fham FHAM, -ih FHAM
    --verbose, -v
    --steps STEPS, -s STEPS
    --timestep TIMESTEP, -ts TIMESTEP
    --n_walkers N_WALKERS, -nw N_WALKERS
                            number of walkers per MPI rank
    --mixed_est, -me
    --time_bp TIME_BP, -tbp TIME_BP
                            maximum back propagation time
    --num_bp NUM_BP, -nbp NUM_BP
    --path_restoration PATH_RESTORATION, -pr PATH_RESTORATION
                            path restoration type "0", "1", or "e" for no, yes, and extra



Supported CLI Tools
~~~~~~~~~~~~~~~~~~~

.. list-table:: 
    :widths: 30 70
    :header-rows: 1

    * - Command
      - Description
    * - ``afqmc_to_fcidump``
      - converts from the SAFIRE format Hamiltonian format (in an HDF5 file) to FCIDUMP. Converts only the k-point factorized Hamiltonian or the generic dense Hamiltonian format.
    * - ``check_one_rdm``
      - Analyze the one-body reduced density matrix (1-rdm) output by SAFIRE.
    * - ``coqui_to_2nd_quant``
      - Convert CoQuí Hamiltonians saved in an HDF5 checkpoint file to either an FCIDUMP file or a SAFIRE file.
    * - ``dice_to_hdf5``
      - Convert from DICE wavefunction to a SAFIRE readable wavefunction format.
    * - ``fcidump_spat2spin``
      - Convert FCIDUMP expressed in a spatial orbital basis to a FCIDUMP expressed in a spinor basis.
    * - ``fcidump_to_afqmc``
      - generates an SAFIRE Hamiltonian from a plain text FCIDUMP. The integrals are assumed to be real and 8-fold symmetric.
    * - ``make_model_ham``
      - Builds a lattice model Hamiltonian in SAFIRE format based on the settings in a .toml input file and saves to HDF5.
    * - ``pyscf_to_afqmc``
      - generates Hamiltonian and Wavefunction for SAFIRE input from a pyscf scf calculation.
    * - ``scalar_stats``
      - Analyze stochastic samples of scalar data output by SAFIRE. One major use case is the analysis of the AFQMC energy, but other scalar data can be analyzed as well.
    * - ``write_afqmc_json``
      - Writes an AFQMC json input file based on the settings provided.

.. only:: developer

    Deprecated CLI Tools
    ~~~~~~~~~~~~~~~~~~~~

    .. list-table:: 
        :widths: 30 70
        :header-rows: 1

        * - Command
          - Description
        * - ``aimbes_to_afqmc``
          - (deprecated) use ``aimbes_to_2nd_quant`` instead. Convert AIMBES Hamiltonians saved in an HDF5 checkpoint file to an SAFIRE file.
        * - ``rdm_stats``
          - Analyze stochastic samples of observables output by SAFIRE
        * - ``kp_to_sparse``
          - (deprecated) sparse format is not officially supported. Converts SAFIRE K-point factorize Hamiltonian to SAFIRE generic-sparse Hamiltonian format.
        * - ``sparse_to_dense``
          - (deprecated) sparse format is not officially supported
        * - ``test_afqmc_input``
          - Sanity checks SAFIRE Hamiltonian saved in HDF5.
