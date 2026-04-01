.. _estimators:

Estimators
==========

This page is organized as follows.
We begin by explaining the different types of estimators available in SAFIRE.
Estimators are used to compute observables.
An estimator requires at least one observable to compute.
Below, we explain each :ref:`type of observable <observables>` that can be computed using these estimators.
All observables can be used with any of the estimators.

.. important::

    Unlike all other types of input blocks, the "name" parameter is used to determine the type of estimator - i.e. 
    NOT to define an identifier that can be used elsewhere in the input file.

Mixed Estimators
----------------

SAFIRE implements mixed estimators, which are used to compute observables :math:`\hat{O}` such that :math:`[\hat{O},\hat{H}] = 0`.
In this case, a mixed estimator is an unbiased estimator.
Formally, mixed estimators evaluate an observable :math:`\hat{O}` as

.. math::
  \langle \hat{O} \rangle_{Mixed} = \frac{1}{\sum_k W_{n,k}} \sum_k W_{n,k} \frac{\langle \Psi_T | \hat{O} | \Phi_{n,k} \rangle }{\langle \Psi_T | \Phi_{n,k} \rangle}


Settings
~~~~~~~~

.. code-block:: json
  :caption: Sample input block for Mixed Estimator.  
  
  "estimator": {
    "name": "mixed",
    "equil_multiplier" : 0,
    "measure_interval_multiplier" : 1,
    "onerdm": {
      "name": "one_rdm",
    }
  }


.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **equil_multiplier**
     - 0
     - Used to determine the number of projection steps in the equilibration phase using the formula below. Measurement is the most expensive operation in AFQMC.
   * - **measure_interval_multiplier**
     - Inherited from execute block (default: 1)
     - Used to determine the number of projection steps between measurements using the formula below. Measurement is the most expensive operation in AFQMC. A larger "measure_interval_multiplier" will reduce the CPU time necessary to perform AFQMC calculations.

.. important::

  The measure_interval and equilibration lengths are specified indirectly using the measure_interval_multiplier and equil_multiplier parameters, respectively. 
  They are computed using the population_control_interval according to the formula

  .. math::

    measure\_interval = measure\_interval\_multiplier \times population\_control\_interval

  and similarly for the equilibration time.

Energy Estimator
----------------

SAFIRE implements a specialized mixed estimator for the energy, which is always added 
to an execute block by default.
The only reason to explicitly define it is to customize its settings.
Formally, it evaluates

.. math::
  :name: energyEstimator

  E = \frac{\langle \Psi_T | \hat{H} | \Psi^s \rangle}{\langle \Psi_T | \Psi^s \rangle} \approx 
  \frac{1}{\sum_n W^s_n} \sum_n W^s_n \frac{\langle \Psi_T | \hat{H} | \Phi^s_n \rangle}{\langle \Psi_T | \Phi^s_n \rangle},

Settings
~~~~~~~~

.. code-block:: json
  :caption: Sample input block for Energy Estimator.  
  
  "estimator": {
    "name": "energy",
    "measure_interval_multiplier": 1,
    "print_components": true
  }


.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **measure_interval_multiplier**
     - Inherited from execute block (default: 1)
     - Used to determine the number of projection steps between measurements using the formula below. Measurement is the most expensive operation in AFQMC. A larger "measure_interval_multiplier" will reduce the CPU time necessary to perform AFQMC calculations.
   * - **print_components**
     - false
     - if true, print the one-body and two-body direct, and two-body exchange components of the energy separately (in addition to the total energy). Note: the one-body energy also includes any constant energy contributions.
 
.. only:: developer

    Advanced / developer setting
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. list-table::
      :header-rows: 1
      :widths: 25 20 55

      * - **Parameter**
        - **Default**
        - **Description**
      * - print_sign
        - false
        - if true, print detailed information about the phase and constraint in the scalar.dat output file.
      * - truncate
        - false
        -

Back-Propagation (BP) Estimators
--------------------------------

Mixed estimators are biased for observables :math:`\hat{O}` such that :math:`[\hat{O},\hat{H}] \neq 0`, 
which requires the use of pure estimators. The Back-Propagation (BP) algorithm is used to compute pure estimators in SAFIRE.
The back-propagated estimator has the form,

.. math::

  \langle \hat{O} \rangle_{BP} = \frac{1}{\sum_k W_{s+m,k}} \sum_k W_{s+m,k} \frac{\langle \tilde{\Phi}_{m,k} | \hat{O} | \Phi_{s,k} \rangle }{\langle \tilde{\Phi}_{m,k} |\Phi_{s,k}\rangle}


where :math:`| \Phi_{s,k} \rangle` are the usual forward-projected Slater determinant random walkers,
and :math:`| \tilde{\Phi}_{m,k} \rangle` are the back-propagated walkers given by,

.. math::

  | \tilde{\Phi}_{m,k} \rangle = \hat{B}^\dagger( (x - \bar{x})_{s,k} ) ... \hat{B}^\dagger( (x - \bar{x})_{s+m-1,k} ) | \Psi_T \rangle.

The index :math:`s` corresponds to the current forward projection step,
and :math:`m` is the back-propagated step index.
We note that each random walker has a corresponding back-propagated partner
which share the same path in auxiliary-field space.

Sample Input File
~~~~~~~~~~~~~~~~~

.. code-block:: json
    :caption: Sample input block for Back-Propagation (BP) Estimator.

    "estimator": {
        "name": "back_propagation",
        "path_restoration": true,
        "bp_walker_ortho_interval": 10,
        "measure_interval_multiplier": 80,
        "equil_multiplier": 80,
        "onerdm": {
            "name": "one_rdm"
        }
    }


Configuration
~~~~~~~~~~~~~

BP is invoked in SAFIRE by including a "back_propagation" estimator in the input file. 
Observables are added to the BP block in the same way as for mixed estimators.

Measurement Intervals
~~~~~~~~~~~~~~~~~~~~~

Since BP involves an additional projection, the measurement interval controls the number of back-propagated steps. 
Just as in the execute block or a mixed estimator block, the input file includes a measurement interval "multiplier",
and the actual measurement interval, :math:`m`, is determined as:

.. math::

    m = (\text{measure_interval_multiplier}) \times (\text{population_control_interval})

Multiple BP Lengths
~~~~~~~~~~~~~~~~~~~

SAFIRE implements the capability of running BP with multiple BP measurement lengths within the same calculation.
We call each measurement with a different BP length an "average". 
This allows the forward projection to be reused while checking for convergence in the BP length.
To define multiple BP lengths/averages, simply provide a JSON array for measure_interval_multiplier corresponding to all of the BP lengths
that you would like to use:

.. code-block:: json
    :caption: Sample input block for Back-Propagation (BP) Estimator with multiple BP lengths.
    
    "estimator": {
        "name": "back_propagation",
        "path_restoration": true,
        "bp_walker_ortho_interval": 10,
        "measure_interval_multiplier": [60, 70, 80],
        "equil_multiplier": 80,
        "onerdm": {
            "name": "one_rdm"
        }
    }


Equilibration
~~~~~~~~~~~~~

The BP estimator implements an equilibration phase at the beginning of the AFQMC calculation where no BP is performed. 
This is recommended since computing observables is typically expensive and since AFQMC needs to equilibrate before 
samples can meaningfully contribute to the average. 
Similarly to the measure_interval_multiplier, the equilibration time is specified as an equilibration multiplier 
(``"equil_multiplier"``) and the actual equilibration time is given by:

.. math::

    \text{equil_time} = (\text{equil_multiplier}) \times (\text{population_control_interval})


Settings
~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - **Parameter**
     - **Default**
     - **Description**
   * - **path_restoration**
     - false
     - if true, use path restoration in the back-propagation algorithm.
   * - **bp_walker_ortho_interval**
     - 10
     - Interval for walker orthogonalization during back-propagation.
   * - **measure_interval_multiplier**
     - Inherited from execute block (default: 1)
     - Either a single integer or an array of integers. Used to determine the number of back-propagation steps between measurements using the formula below. A larger value will use longer back-propagation lengths. If an array is provided, multiple back-propagation lengths will be used within the same calculation.
   * - **equil_multiplier**
     - 0
     - Used to determine the number of projection steps in the equilibration phase before back-propagation measurements begin.

.. important::

  The measure_interval and equilibration lengths are specified indirectly using the measure_interval_multiplier and equil_multiplier parameters, respectively. 
  They are computed using the population_control_interval according to the formula

  .. math::

    measure\_interval = measure\_interval\_multiplier \times population\_control\_interval

  and similarly for the equilibration time.

.. only:: developer

    Advanced / developer settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. list-table::
      :header-rows: 1
      :widths: 25 20 55

      * - **Parameter**
        - **Default**
        - **Description**
      * - **extra_path_restoration**
        - false
        - if true, perform an extra path restoration in the back-propagation algorithm.
