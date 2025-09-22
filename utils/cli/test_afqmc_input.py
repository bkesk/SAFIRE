#!/usr/bin/env python3

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

from afqmctools.hamiltonian.converter import read_hamiltonian

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
    parser.add_argument('-i', '--input', dest='input', type=str,
                        default=None, help='Input afqmc file.')

    options = parser.parse_args()

    if not options.input:
        parser.print_help()
        sys.exit()

    return options

def main():
    """Sanity check for afqmc Hamiltonian.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    options = parse_args()
    hamil = read_hamiltonian(options.input)
    if hamil is None:
        sys.exit(1)
    nerror = 0
    for k, v in hamil.items():
        if v is None:
            nerror += 1
    if nerror > 0:
        print("Found {:} non fatal error reading Hamiltonian file.".format(nerror))
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
