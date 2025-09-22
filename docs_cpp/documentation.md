@page documentation API Documentation

[TOC]

The AFQMC code uses a hierarchy of templated classes in order to achieve run-time polymorphism.
Within the hierarchy, objects can interrogate the objects which are "lower" or "deeper", but 
can't interrogate objects which are "higher" or "shallower".

TODO: Double check the order, I'm a little rusty.

The hierarchy is as follows, from top to bottom.

* Drivers
* Propagators
* HamiltonianOperations
* Hamiltonian
* Wavefunction
* Walkers

## API 

This is just a test

@ref sfqmc::afqmc::AFQMCFactory


