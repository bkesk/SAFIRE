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
from afqmctools.hamiltonian.converter import sparse_to_dense


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
    parser.add_argument('-r', '--real-chol', dest='real_chol',
                        action='store_true', default=False,
                        help='Dump real integrals.')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='Verbose output.')

    options = parser.parse_args()

    if not options.input_file:
        parser.print_help()
        sys.exit(1)

    return options

def main():
    """Convert sparse Hamiltonian format to dense Hamiltonian format.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    options = parse_args()
    sparse_to_dense(options.input_file, options.output_file,
                    real_chol=options.real_chol)


if __name__ == '__main__':
    main()
