#! /usr/bin/env python3

# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import argparse
import sys
import numpy
from afqmctools.hamiltonian.mol import (
        write_sparse,write_dense
        )
from afqmctools.wavefunction.mol import (
        write_wfn
        )
from afqmctools.hamiltonian.converter import read_fcidump
from afqmctools.utils.linalg import modified_cholesky_direct


def parse_args():
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
        Command line arguments.
    """

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('-i', '--input', dest='input_file', type=str,
                        default=None, help='Input FCIDUMP file.')
    parser.add_argument('-o', '--output', dest='output_file',
                        type=str, default='fcidump.h5',
                        help='Output file name for PAUXY data.')
    parser.add_argument('--write-complex', dest='write_complex',
                        action='store_true', default=False,
                        help='Output integrals in complex format.')
    parser.add_argument('-t', '--cholesky-threshold', dest='thresh',
                        type=float, default=1e-5,
                        help='Cholesky convergence threshold.')
    parser.add_argument('-s', '--symmetry', dest='symm',
                        type=int, default=8,
                        help='Symmetry of integral file (1,4,8).')
    parser.add_argument('--sparse', dest='sparse',
                        action='store_true', default=False,
                        help='Write sparse Hamiltonian.')
    parser.add_argument('--add_wfn', dest='add_wfn',
                        action='store_true', default=False,
                        help='Add single determinant wavefunction to file.')
    parser.add_argument('--rohf', dest='rohf',
                        action='store_true', default=False,
                        help='ROHF wave function?.')
    parser.add_argument('--occ_up', dest='occa',nargs='+',type=int, 
                        default=None, help='List of occupied up orbitals.')
    parser.add_argument('--occ_down', dest='occb',nargs='+',type=int,
                        default=None, help='List of occupied down orbitals.')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='Verbose output.')

    options = parser.parse_args()

    if not options.input_file:
        parser.print_help()
        sys.exit(1)

    return options

def main():
    """Convert FCIDUMP to SAFIRE readable Hamiltonian format.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    options = parse_args()
    (hcore, eri, ecore, nelec) = read_fcidump(options.input_file,
                                              symmetry=options.symm,
                                              verbose=options.verbose)
    norb = hcore.shape[-1]

    # If the ERIs are complex then we need to form M_{(ik),(lj}} which is
    # Hermitian. For real integrals we will have 8-fold symmetry so trasposing
    # will have no effect.
    eri = numpy.transpose(eri,(0,1,3,2))

    chol = modified_cholesky_direct(eri.reshape(norb**2,norb**2),
                                    options.thresh, options.verbose,
                                    cmax=20).T.copy()
    cplx_chol = options.write_complex or numpy.any(abs(eri.imag)>1e-14)
    if options.sparse:
      write_sparse(hcore, chol, nelec, norb, e0=ecore,
                           real_chol=(not cplx_chol),
                           filename=options.output_file,verbose=options.verbose)
    else:
      write_dense(hcore, chol, nelec, norb, enuc=ecore,
                          filename=options.output_file,
                          real_chol=(not cplx_chol),verbose=options.verbose)

    if options.add_wfn:
      nalpha, nbeta = nelec
      if nalpha != nbeta:
        rohf=True
      else:
        rohf = options.rohf
      # For RHF only nalpha entries will be filled.
      wfn = numpy.zeros((1,norb,nalpha+nbeta), dtype=numpy.complex128)
      wfn_type = 'NOMSD'
      coeffs = numpy.array([1.0+0j])
      # Assuming we are working in MO basis, only works for RHF, ROHF trials.
      I = numpy.identity(norb, dtype=numpy.float64)
      occs = numpy.zeros(nalpha+nbeta,dtype=int)
      occs[:nalpha] = numpy.arange(nalpha) 
      occs[nalpha:] = numpy.arange(nbeta) 
      if options.occa is not None:
        occa = numpy.asarray(options.occa)
        assert(len(occa) == nalpha)
        occs[:nalpha] = occa
      if options.occb is not None:
        occb = numpy.asarray(options.occb)
        assert(len(occb) == nbeta)
        occs[nalpha:] = occb
      wfn[0,:,:nalpha] = I[:,occs[:nalpha]]
      if rohf:
        wfn[0,:,nalpha:] = I[:,occs[nalpha:]]
      write_wfn(options.output_file, (numpy.array([1.0+0j]),wfn),
                ('rhf' if not rohf else 'rohf'),
                nelec, norb, verbose=options.verbose)

if __name__ == '__main__':
    main()
