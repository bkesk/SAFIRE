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
import numpy as np
from afqmctools.hamiltonian.converter import (
    read_fcidump,
    write_fcidump,
    h1_spat2spin,
    h2_spat2spin
    )

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
        parser.add_argument('-s', '--symmetry', dest='symm',
                                                type=int, default=1,
                                                help='Symmetry of integral file (1,4,8).')
        parser.add_argument('-t', '--tol', dest='tol',
                        type=float, default=1e-12,
                        help='Cutoff for printing integrals in FCIDUMP file.')
        parser.add_argument('--occ_up', dest='occa',nargs='+',type=int, 
                                                default=None, help='List of occupied up orbitals.')
        parser.add_argument('--occ_down', dest='occb',nargs='+',type=int,
                                                default=None, help='List of occupied down orbitals.')
        parser.add_argument('-v', '--verbose', dest='verbose',
                                                action='store_true', default=False,
                                                help='Verbose output.')
        parser.add_argument('--complex-paren', dest='cplx_paren',
                        action='store_true', default=True,
                        help='Whether to write FORTRAN format complex numbers.')

        options = parser.parse_args()

        if not options.input_file:
            parser.print_help()
            sys.exit(1)

        return options

def main():
    """Convert FCIDUMP expressed in a spatial orbital basis
            to a FCIDUMP expressed in a spinor basis.

    Parameters
    ----------
    args : list of strings
            command-line arguments.
    """
    options = parse_args()
    (hcore, eri, ecore, nelec) = read_fcidump(
            options.input_file,
            symmetry=options.symm,
            verbose=options.verbose)
    norb = hcore.shape[-1]

    # If the ERIs are complex then we need to form M_{(ik),(lj}} which is
    # Hermitian. For real integrals we will have 8-fold symmetry so trasposing
    # will have no effect.
    eri = np.transpose(eri,(0,1,3,2))

    hcore = h1_spat2spin(
        h1e_new=np.zeros((2*norb, 2*norb), dtype=np.complex128),
        h1e_old=hcore,
        Mspatial=norb
    )
    eri = h2_spat2spin(
        h2e_new=np.zeros((2*norb, 2*norb, 2*norb, 2*norb),dtype=np.complex128),
        h2e_old=eri,
        Mspatial=norb
    )
    
    write_fcidump(
        filename=options.output_file,
        hcore=hcore,
        chol=eri,
        enuc=ecore,
        nmo=2*norb,
        nelec=nelec,
        tol=options.tol,
        sym=options.symm,
        cplx=True,
        paren=options.cplx_paren,
        chol_is_eri=True
    )


if __name__ == '__main__':
    main()
