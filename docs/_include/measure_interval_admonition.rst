.. important::

  The measurement interval and equilibration lengths are specified indirectly using the ``measure_interval_multiplier`` and ``equil_multiplier`` parameters, respectively. 
  They are computed using the ``population_control_interval`` according to the formula

  .. math::

    \text{measure\_interval} = \text{measure\_interval\_multiplier} \times \text{population\_control\_interval}

  and similarly for the equilibration time.
