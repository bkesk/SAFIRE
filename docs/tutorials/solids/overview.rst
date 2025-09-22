.. _solids_overview:

SAFIRE for ab initio solids
============================

.. image:: https://users.flatironinstitute.org/~beskridge/tutorial_figs/6784ee4ea455921958ac327234b91ab07702736ab22fa2df804e8dccbc36a404/00_top_level/solid.png
   :width: 500


Some preliminaries
==================

The Hamiltonian
---------------

|

Trial wavefunctions
-------------------

A trial wavefunction is used in AFQMC which is, in general, some linear
combination of Slater determinants,

.. math::

   | \Psi_T \rangle = \sum_n C_n |\Phi_m\rangle,

where :math:`|\Phi_m\rangle` are Slater determinants,
and :math:`C_n` is a coefficient.

...

Common methods for computing a trial wavefunction include:

- Hartree-Fock
- Kohn-Sham density functional theory

among others.

|

Typical Workflow
================

|

Software prerequisites
======================

Many mature electronic structure codes exist and are widely used.
For this reason, AuxiliaryFields does not implement effective one-body methods
such as density functional theory (DFT).
Instead, AuxiliaryFields is designed to use externally generated Hamiltonians and trial wavefunctions
for easy integration into existing workflows.

AuxiliaryFields can directly run AFQMC claculations using Hamiltonians and trial wavefunctions generated using Coquí.

...

|

The Tutorials
=============

The following tutorials will guide you through AFQMC calculations
using AuxiliaryFields in order to teach you the typical workflow,
and some of the main features of AuxiliaryFields.
We assume that you are familiar with typical electronic structure calculations using pseudopotentials and a plane-wave basis.
We also assume that you
have access to Coquí
and that you are familiar with its basic use.
Each "basic" tutorial builds on the previous one.
We recommend going through them in order.

|

1. Hello AuxiliaryFields `🛠️ Under construction 🛠️`
2. `Understanding the input file <https://colab.research.google.com/drive/1rWeqD-DVQNMN8ILqelEZ56OURmCYpozp>`_ `🧑‍💻 ready for edits 🧑‍💻` (note: this is the same as for molecules, lattice models)
3. Writing a Hamiltonian file `🛠️ Under construction 🛠️`
4. Writing a Trial wavefunction `🛠️ Under construction 🛠️`
5. Computing Observables `🛠️ Under construction 🛠️`

|

Worked Examples
===============

We provide the following worked examples in which we
go through the entire workflow. We use Quantum Espresso to perform
DFT calculations, and Coquí to generate a Hamiltonian and write trial wavefunctions.

1. `Charge density in Si - CoQuí interface <https://colab.research.google.com/drive/1V2qDFA5PJkM2xeliDLsKuTrWR1FpqTal>`_ `🧑‍💻 ready for edits 🧑‍💻`
2. `Carbon diamond PySCF interface <https://colab.research.google.com/drive/172tZtwLFCHI-bW6eIVXuLRgnJqPun8z6>`_ `🛠️ Under construction 🛠️`
3. `Solid Na momentum distribution <https://colab.research.google.com/drive/1qNBPcToh58UJPeg9Qx6YxcIS6RPL5a2o>`_ `🛠️ Under construction 🛠️` : Paul is working on this
4. `Bulk modulous of MnO <https://colab.research.google.com/drive/1oJRRBTptrW_66NRWs7CdQix68Qgz9WuX>`_ `🛠️ Under construction 🛠️` 
5. `Band gap of LiH <https://colab.research.google.com/drive/17x0SnUEhi-yEuyk_gMFbIGNr_SWGUi9g>`_ `🛠️ Under construction 🛠️` 
6. `Magnetization of ? <https://colab.research.google.com/drive/1In-gtDJ6Ud3MwHJEJdQS-c66Y6F7wpud>`_ `🛠️ Under construction 🛠️` 

