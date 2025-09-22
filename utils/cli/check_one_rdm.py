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

import h5py as h5

from afqmctools.analysis.rdm import (
    average_afqmc_rdm,
    check_1rdm_convergence,
    plot_1rdm_convergence
)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze the one-body reduced density matrix (1-rdm) output by SAFIRE.'
    )
    parser.add_argument('fname', type=str, help='SAFIRE stats file name ( [id].s[series].stat.h5 )')
    parser.add_argument('-o', '--output', type=str, default='observables.h5', help='Output file name. Will overwrite the one_rdm dataset if it exists.')
    parser.add_argument('--check_1rdm', action='store_true', help='Check 1-RDM convergence')
    parser.add_argument('--plot_1rdm', action='store_true', help='Plot 1-RDM convergence')
    parser.add_argument('--spin_trace', action='store_true', help='Trace over spin')
    parser.add_argument('--force_hermitian', action='store_true', help='Force the 1-RDM to be hermitian')

    args = parser.parse_args()
    return args

def main():
    r"""
    Analyze the one-body reduced density matrix (1-rdm) output by SAFIRE.
      to check for convergence and plot the convergence of the 1-rdm.
    """
    args = parse_args()

    # get the 1-rdm for each average
    rdm_nij, delta_rdm_nij = average_afqmc_rdm(
        rdm_file=args.fname
    )

    # check rdm convergence
    if args.check_1rdm:
        check_1rdm_convergence(rdm_nij, delta_rdm_nij)
    
    if args.plot_1rdm:
        plot_1rdm_convergence(
            rdm_nij, delta_rdm_nij, 
            spin_trace=args.spin_trace,
            force_hermitian=args.force_hermitian
        )

    with h5.File(args.output, 'a') as f:
        if 'one_rdm' in f:
            del f['one_rdm']
        f.create_dataset('one_rdm', data=rdm_nij)
        if 'delta_one_rdm' in f:
            del f['delta_one_rdm']
        f.create_dataset('delta_one_rdm', data=delta_rdm_nij)

if __name__ == '__main__':
    main()
