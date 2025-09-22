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
import os
import sys
import numpy as np
from afqmctools.wavefunction.mol import write_wfn
from afqmctools.wavefunction.converter import (
        read_dice_ascii_wavefunction,
        read_dice_h5_wavefunction
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
                        default=None, help='Input wavefunction file.',
                        required=True)
    parser.add_argument('-o', '--output', dest='output_file',
                        type=str, default='wfn.h5',
                        help='Output file name for wavefunction.')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='Verbose output.')
    parser.add_argument('-f', '--file_type', dest='file_type',
                        type=str, default='detect',
                        help='File type: {h5,ascii}. ascii: Dice standard output.')
    parser.add_argument('-s','--state', dest='state', type=int, default=0,
                        help='State (root) to use.')
    parser.add_argument('-n','--ndets', dest='ndets', type=int,
                        default=None, help='Number of determinants to write.',
                        required=True)
    parser.add_argument('-c','--core', dest='num_core', type=int,
                        default=0, 
                        help='Number of core orbitals / inactive orbitals to INCLUDE in wavefunction.'
                        ' Usefule when Dice is used as a CAS solver'
                    )
    parser.add_argument('-a','--augment-viruals',dest="aug_virts",
                        default=0,
                        help="numer of virtual orbitals to augment the basis with.")
    parser.add_argument(
        '-m','--full_basis',dest="num_full_basis",
        default=None,
        help="Full number of orbitals in the Hilbert space. Usefule when Dice is used as a CAS solver"
        )

    options = parser.parse_args()

    return options

def main():
    """Convert From DICE wavefunction to a SAFIRE readable wavefunction format.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    options = parse_args()
    input_file = options.input_file
    output_file = options.output_file
    state = options.state
    ndets = options.ndets
    if options.file_type == 'detect':
        fname, fext = os.path.splitext(input_file) 
        if fext == '.h5':
            file_type = 'h5'
        elif (fext == '.out') or (fext == '.dat') or (fext == '.txt') or (fext == '.ascii'):
            file_type = 'ascii'
        else:
            print("Unknown file extension, please specify file type with -f/--file_type.")
            assert(0)
    elif (options.file_type == 'h5') or (options.file_type == 'ascii'):
        file_type = options.file_type
    else:
        print("Unknown file type: ",options.file_type)
        assert(0)
    assert(state >= 0)
    if file_type == 'ascii':
        wfn, nmo, nup, ndn, walker_type = read_dice_ascii_wavefunction(input_file, ndets, state)
    elif file_type == 'h5':
        wfn, nmo, nup, ndn, walker_type = read_dice_h5_wavefunction(input_file, ndets, state)
    nelec = (nup,ndn)
    assert(len(wfn) == 3)

    # For calculations where DICE was run in a CAS-like active space, augment occ strings with "inactive" occs
    if options.num_core > 0:
        # add core occ string AND offset the existing one with ncore
        core_occ = np.arange(options.num_core)
        occa = wfn[1]
        occb = wfn[2]
        num_det = occa.shape[0]
        na = occa.shape[1]
        nb = occb.shape[1]
        new_occa = np.zeros((num_det,na+options.num_core),dtype=np.int64)
        new_occb = np.zeros((num_det,nb+options.num_core),dtype=np.int64)
        for n in range(ndets):
            new_occa[n,:] = np.append(core_occ, (occa[n] + options.num_core))
            new_occb[n,:] = np.append(core_occ, (occb[n] + options.num_core))
        occa,occb = new_occa,new_occb
        wfn = (wfn[0],occa,occb)
    
        nelec = (nelec[0] + options.num_core,nelec[1] + options.num_core)
        if options.num_full_basis is not None:
            nmo = int(options.num_full_basis)
        else:
            nmo = nmo + options.num_core

    if options.num_full_basis is not None:
        nmo = int(options.num_full_basis)
    elif options.aug_virts > 0:
        nmo = nmo + options.aug_virts

    write_wfn(output_file, wfn, walker_type, nelec, nmo)

if __name__ == '__main__':

    main()
