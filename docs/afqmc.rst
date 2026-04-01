.. _afqmc:

Auxiliary-Field Quantum Monte Carlo
===================================

Overview
--------

The AFQMC method is an orbitally-based many-body method which is formulated in terms of a generic, interacting second-quantized Hamiltonian.
We refer the reader to one of the review articles on the method :cite:`MaloneGPU2020,AFQMC_review,PhysRevLett.90.136401,PhysRevE.70.056702`
for a detailed description of the algorithm.
Here, we provide an overview of AFQMC with a focus on how the formalism and the inputs to the SAFIRE code relate to each other.

AFQMC uses imaginary-time projection to generate a many-body wavefunction, :math:`| \Psi \rangle`, starting from an initial wavefunction, 
:math:`| \Psi_I \rangle`, which has non-zero overlap with the target wavefunction - i.e. :math:`\langle \Psi | \Psi_I \rangle \neq 0`

.. math::
  :name: eq-imag-time-prop
  :label: eq-imag-time-prop

  \lim_{\beta \to \infty} e^{-\beta \hat{H}} | \Psi_I \rangle \rightarrow | \Psi \rangle

where :math:`\beta` is the total projection time, and :math:`\hat{H}` is the interacting Hamiltonian which, 
in its most general form, is given by 

.. math::
  :label: eq-h012-af

  H =& H_0 + \hat{H}_1 + \hat{H}_2 \\
    =& E_0 + \sum\limits_{il\sigma\sigma'} h^{\sigma \sigma'}_{il} \hat{c}^\dagger_{i\sigma}\hat{c}_{l\sigma'}
    + \sum\limits_{ijkl\sigma\sigma'\sigma'\sigma} v_{ijkl}^{\sigma\sigma'\sigma'\sigma} \hat{c}^\dagger_{i\sigma}
      \hat{c}^\dagger_{j\sigma'}\hat{c}_{k\sigma'}\hat{c}_{l\sigma}

and :math:`\hat{c}^\dagger_{i\sigma}`/ :math:`\hat{c}_{i\sigma}` are fermionic creation/annihilation operators which
create/annihilate a particle in orbital :math:`i` with spin :math:`\sigma`,
:math:`H_0` contains all constant contributions to the Hamiltonian,  
:math:`\hat{H}_1` contains all one-body Hamiltonian terms (for example, the kinetic energy in *ab initio*
calculations, or the hoping matrix in lattice models ), and :math:`\hat{H}_2` contains all two-body interactions
such as the Coulomb interaction in *ab initio* calculations or the Hubbard U in lattice models.
The matrix elements of :math:`\hat{H}_1` are given generically as :math:`h_{ij}^{\sigma\sigma'}` and the 
details of which terms are included depends on the specific system of interest.
Similarly for the matrix elements of :math:`\hat{H}_2` which are generally :math:`v_{ijkl}^{\sigma\sigma'\sigma'\sigma}`.
In specific calculations, the general form of the Hamiltonian simplifies considerably especially for :math:`\hat{H}_2`.
The different forms of Hamiltonian implement in SAFIRE are described in 
the :ref:`The Hamiltonian file formats <Hamiltonian-classes>` section.

Imaginary-time projection
-------------------------

Thoulesses' theorem :cite:`Thouless` states that the operation of the exponential of a one-body operator on a Slater 
determinant is simply another Slater determinant.
However, the application of the :ref:`imaginary-time projection operator <eq-imag-time-prop>` is non-trivial due to the interaction term in
the hamiltonian.
To handle the interaction term, the projection operator is mapped onto a high-dimensional integral where the integrand consists of exponentials
of one-body operators only.
This achieved as follows.
First, the imaginary-time projection is divided into :math:`N_{steps}` imaginary-time steps of size :math:`\tau`
which allows a Trotter–Suzuki (TS) :cite:`Trotter1959,Suzuki1976` decomposition to be applied such that the propagator now has the form,

.. math::
  :name: TS-decomp

  e^{-\tau \hat{H}} \approx  e^{-\tau \hat{H}_1/2} e^{-\tau \hat{H}_2} e^{-\tau \hat{H}_1 / 2} + \mathcal(O)(\tau^3)

where the accuracy of the TS decomposition can be improved by using a smaller time step.
Next, the interaction term is rewritten as a quadratic form of one-body operators,

.. math::
  :name: quadratic-H2

  \hat{H}_2 = \sum_\lambda (\hat{v}^\gamma)^2,

where :math:`\hat{v}^\gamma` are one-body operators. 
Next, :math:`exp[-\tau \hat{H}_2]` can be rewritten via a Hubbard-Stratonovich (HS) transformation as 

.. math::
  :name: HSInt

  e^{-\tau \sum_\lambda (\hat{v}^\gamma)^2} = \int d\mathbf{\sigma} P(\mathbf{\sigma}) e^{-\tau \mathbf{\sigma} \cdot \hat{\mathbf{v}}},

where :math:`\mathbf{\sigma}` is a vector of auxiliary-fields,
:math:`\hat{\mathbf{v}}` is the vector of one-body operators :math:`\hat{v}^\gamma`, and :math:`P(\mathbf{\sigma})` is a normal distribution function.
Combining the TS decomposition and the HS transformation, we arrive at

.. math:: 
  :name: HSProp

  e^{-\tau \hat{H} } \approx& \int d\mathbf{\sigma} P(\mathbf{\sigma}) e^{-\tau \hat{H}_1/2}  e^{-\tau \mathbf{\sigma} \cdot \hat{\mathbf{v}}} e^{-\tau \hat{H}_1/2} , \\
                     \approx& \int d\mathbf{\sigma} P(\mathbf{\sigma}) B(\mathbf{\sigma})

where :math:`B(\sigma) =  exp^{-\tau \hat{H}_1/2}  exp^{-\tau \mathbf{\sigma} \cdot \hat{\mathbf{v}}} exp^{-\tau \hat{H}_1/2}`.
Additionally, the many-body wavefunction is represented, at time step :math:`s`, in an over-complete basis of non-orthogonal Slater determinants, :math:`| \Phi^s_n \rangle`, as,

.. math:: 
  :name: MBWfnWalkers

  | \Psi^s \rangle \doteq \sum_n W^s_n | \Phi^s_n \rangle,

where :math:`W^s_n` is a weight which is accumulated during the projection.

Finally, the many-body wavefunction can be sampled as 

.. math::
  :name: MBProjection

  e^{-\tau \hat{H} } | \Psi^s \rangle \approx \int d\mathbf{\sigma} P(\mathbf{\sigma}) B(\mathbf{\sigma}) | \Psi^{s-1} \rangle


by sampling auxiliary-field configurations from :math:`P(\mathbf{\sigma})`, 
constructing :math:`B(\mathbf{\sigma})` with the sampled set of :math:`\mathbf{\sigma}`,
and evaluating :math:`B(\mathbf{\sigma}) | \Phi^s_n \rangle` for each walker.
Observables are computed on the fly to achieve low-order polynomial scaling with system size.
For example, the energy is evaluated as

.. math::
  E = \frac{\langle \Psi_T | \hat{H} | \Psi^s \rangle}{\langle \Psi_T | \Psi^s \rangle} \approx 
  \frac{1}{\sum_n W^s_n} \sum_n W^s_n \frac{\langle \Psi_T | \hat{H} | \Phi^s_n \rangle}{\langle \Psi_T | \Phi^s_n \rangle},


Importance sampling
-------------------

In practice, importance sampling is used to reduce variance.
The auxiliary-fields from the Hubbard-Stratonovich 
transformation are shifted by the so-called "force bias",
:math:`\bar{\mathbf{\sigma}}` and the walkers 
are re-weighted according to an importance function, :math:`I(\mathbf{\sigma},\bar{\mathbf{\sigma}},\Phi^{s-1}_n)`, 
at each projection step.
Formally, the force bias is given by 

.. math::
  :name: forceBias

  \bar{\mathbf{\sigma}}=-\sqrt{\tau}\frac{\langle\Psi_T|\hat{\mathbf{v}}|\Phi^s_n \rangle}{\langle\Psi_T|\Phi^s_n \rangle},

where :math:`|\Psi_T \rangle` is the trial wavefunction which is essentially a guess of the true many-body wavefunction.
The walkers weights are updated at each projection step based on the importance function as 

.. math::
  :name: importanceTransform

  W^s_n = I(\mathbf{\sigma},\bar{\mathbf{\sigma}},\Phi^{s-1}_n) W^{s-1}_n,

where :math:`I(\mathbf{\sigma},\bar{\mathbf{\sigma}},\Phi^{s-1}_n)` is given by

.. math::
  :name: importanceFunction

  I(\mathbf{\sigma},\bar{\mathbf{\sigma}},\Phi) \equiv \frac{\langle\Psi_T|\hat{B}(\mathbf{\sigma} - \bar{\mathbf{\sigma}})|\Phi \rangle}
  {\langle\Psi_T|\Phi \rangle} \textrm{Exp}\left(\bar{\mathbf{\sigma}} \cdot \mathbf{\sigma} - \frac{\bar{\mathbf{\sigma}} 
  \cdot \bar{\mathbf{\sigma}}}{2}\right)


For lattice models, some of the details are slightly different since the form of the interactions is simpler.

The Sign/Phase Problem
----------------------

As formulated above, the imaginary-time projection suffers from an exponential decrease in the signal-to-noise ratio
which arises from the invariance of any observable to an overall phase of the many-body wavefunction, :math:`e^{-i \theta} | \Psi \rangle`.
This is a manifestation of the generic fermionic sign problem that arises in all quantum Monte Carlo approaches.
The projector, :math:`B(\mathbf{\sigma})`, is typically complex-valued and introduces an arbitrary phase to the Slater determinant random walkers.
As the projection proceeds, the random walkers accumulate random phases, leading to a finite density of walkers
in the complex plane defined by :math:`\langle \Psi_T | \Phi^s_n \rangle`.
This leads to large fluctuations in the weights that ultimately diverge.
This is known as the phase problem in the AFQMC literature :cite:`PhysRevLett.90.136401`.
For lattice models, it is possible to choose a HS transformation such that :math:`B(\mathbf{\sigma})` is real-valued.
In this case, the random walkers accumulate random signs — i.e., :math:`\theta \in \{0, \pi\}` — leading to the fermionic sign problem.

The sign/phase problem is controlled using importance sampling, as shown above, and by imposing
a constraint based on the trial wavefunction, :math:`| \Psi_T \rangle`.
For the phase problem, the so-called "phaseless" approximation :cite:`PhysRevLett.90.136401` is used.
At each step, walkers are individually projected onto an
evolving line in the complex plane defined by :math:`\langle \Psi_T | \Phi^s_n \rangle` by multiplying
the walker weights by :math:`\max\{0, \cos(\Delta \theta)\}`, where 
:math:`\Delta \theta = \arg\left[\frac{\langle\Psi_T|\hat{B}(\mathbf{\sigma} - \bar{\mathbf{\sigma}})|\Phi^s_n \rangle}{\langle\Psi_T|\Phi^s_n \rangle}\right]`.
For the sign problem, the "constrained path" approximation :cite:`ZhangConstrained1997` is used,
where walkers are eliminated if :math:`\langle \Psi_T | \Phi^s_n \rangle < 0`.
This can be viewed as a special case of the phaseless approximation.
In both cases, the constraint introduces a bias that can be reduced based on the quality of the trial wavefunction.


Observables
-----------

Observables are evaluated stochastically as 

.. math::
  :name: mixedEstimators-af

  \langle \hat{O} \rangle_{Mixed} = \frac{1}{\sum_k W_{n,k}} \sum_k W_{n,k} \frac{\langle \Psi_T | \hat{O} | \Phi_{n,k} \rangle }{\langle \Psi_T | \Phi_{n,k} \rangle}


where :math:`n` is the projection step index,
:math:`| \Phi_{n,k} \rangle` are Slater determinant random walkers with weight :math:`W_{n,k}` (from importance sampling),
and :math:`| \Psi_T \rangle` is the trial wavefunction.
The mixed estimator provides an unbiased estimate for any observable which commutes with the Hamiltonian.
For the AFQMC energy, :math:`\hat{O} = \hat{H}`, and trivially :math:`[ \hat{O}, \hat{H} ] = 0`;
therefore, the mixed estimator is an unbiased choice for the total ground state energy.

On the other hand, it is often interesting to compute observables which do not commute with the Hamiltonian.
For example, the one-body reduced density matrix, from which many properties can be computed, does not commute
with the Hamiltonian.
In these cases, a back-propagated estimator is necessary to mitigate the bias that would be incurred
by using a mixed-estimator.
The back-propagated estimator has the form,

.. math::
  :label: bpEstimators-af

  \langle \hat{O} \rangle_{BP} = \frac{1}{\sum_k W_{s+m,k}} \sum_k W_{s+m,k} \frac{\langle \tilde{\Phi}_{m,k} | \hat{O} | \Phi_{s,k} \rangle }{\langle \tilde{\Phi}_{m,k} |\Phi_{s,k}\rangle}


where :math:`| \Phi_{s,k} \rangle` are the usual forward-projected Slater determinant random walkers,
and :math:`| \tilde{\Phi}_{m,k} \rangle` are the back-propagated walkers given by,

.. math::

  | \tilde{\Phi}_{m,k} \rangle = \hat{B}^\dagger( (x - \bar{x})_{s,k} ) ... \hat{B}^\dagger( (x - \bar{x})_{s+m-1,k} ) | \Psi_T \rangle.

The index :math:`s` corresponds to the current forward projection step,
and :math:`m` is the back-propagated step index.
We note that each random walker has a corresponding back-propagated partner
which share the same path in auxiliary-field space.
It is important to note that the two Slater determinants are
distinct from each other since the propagator :math:`B(x - \bar{x})` is applied differently to each.
i.e. the propagator is applied to different Slater determinants when moving in the forward-, and backward-directions.

Glossary of Symbols and Notation
--------------------------------

Indices
~~~~~~~

- :math:`i,j,k,l` are orbital basis indices. In quantum chemistry or in solids, these typically refer to 
  spatial orbitals :math:`\phi_i (\vec{r})`. In lattice models, these refer to a combined lattice site and band index.
- :math:`\sigma, \sigma', ...` are spin indices. We use additional "primes" to introduce distinct spin
  indices.
- :math:`n` is the walker index for sums over random walkers.
- :math:`s` is the AFQMC projection time step index.


Hamiltonian
~~~~~~~~~~~

- :math:`H_0`, the constant energy contributions to the Hamiltonian. For example, in *ab initio* calculations \
  performed in the Born-Oppenheimer approximation, :math:`H_0` includes the nuclear repulsion energy.

- :math:`\hat{H}_1`, a generic "one-body" Hamiltonian consisting of all one-body terms in the Hamiltonian \
  with the form :math:`\hat{H}_1 = \sum_{ij} h_{ij} \hat{c}^\dagger_i \hat{c}_j`

- :math:`\hat{H}_2`, a generic "two-body" interaction consisting of all two-body terms in the Hamiltonian \
  with the form :math:`\hat{H}_2 = \sum_{ijkl} v_{ijkl} \hat{c}^\dagger_i \hat{c}^\dagger_j \hat{c}_k \hat{c}_l`

Wavefunctions
~~~~~~~~~~~~~

- :math:`|\Psi\rangle` , a generic many-body wavefunction, typically (but not necessarily!) \
  consisting of a linear combination of Slater determinants.

- :math:`|\Psi_I\rangle` , the initial wavefunction

- :math:`| \Phi \rangle` , a single Slater determinant. \
  Subscript and/or superscript indices may also apply.

AFQMC parameters
~~~~~~~~~~~~~~~~

- :math:`\beta`, total projection time of the AFQMC imaginary-time projeciton
- :math:`\tau`, imaginary-time projection step size

bibliography
------------

.. bibliography:: /bibs/afqmc.bib

