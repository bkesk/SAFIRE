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
from afqmctools.hamiltonian.converter import (
        read_hamiltonian,
        write_fcidump,
        write_fcidump_kpoint
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
                        default=None, help='Input AFQMC hamiltonian file.')
    parser.add_argument('-o', '--output', dest='output_file',
                        type=str, default='FCIDUMP',
                        help='Output file for FCIDUMP.')
    parser.add_argument('-s', '--symmetry', dest='symm',
                        type=int, default=1,
                        help='Symmetry of integral file (1,4,8).')
    parser.add_argument('-t', '--tol', dest='tol',
                        type=float, default=1e-12,
                        help='Cutoff for integrals.')
    parser.add_argument('-c', '--complex', dest='cplx',
                        action='store_true', default=False,
                        help='Whether to write integrals as complex numbers.')
    parser.add_argument('--complex-paren', dest='cplx_paren',
                        action='store_true', default=False,
                        help='Whether to write FORTRAN format complex numbers.')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='Verbose output.')
    parser.add_argument('-u','--use_spinors',dest='use_spinors',
                        action='store_true', default=False,
                        help='Whether to convert to spinor basis')

    options = parser.parse_args()

    if not options.input_file:
        parser.print_help()
        sys.exit(1)

    return options

def main():
    """Convert SAFIRE format Hamiltonian to FCIDUMP Hamiltonian format.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    options = parse_args()
    hamil = read_hamiltonian(options.input_file)
    if hamil is None:
        sys.exit()

    if hamil.get('qk_k2') is None:
        write_fcidump(
            options.output_file,
            hamil['hcore'],
            hamil['chol'],
            hamil['enuc'],
            hamil['nmo'],
            hamil['nelec'],
            tol=options.tol,
            sym=options.symm,
            cplx=options.cplx,
            paren=options.cplx_paren,
            use_spinor=options.use_spinors
        )
    else:
        write_fcidump_kpoint(
            options.output_file,
            hamil['hcore'],
            hamil['chol'],
            hamil['enuc'],
            hamil['nmo'],
            hamil['nelec'],
            hamil['nmo_pk'],
            hamil['nchol_pk'],
            hamil['qk_k2'],
            tol=options.tol,
            sym=options.symm,
            cplx=options.cplx,
            paren=options.cplx_paren,
            use_spinor=options.use_spinors
        )

if __name__ == '__main__':
    main()
