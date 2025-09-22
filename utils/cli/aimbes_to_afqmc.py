# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import sys
import argparse

import numpy as np

from afqmctools.hamiltonian.io import write_dense
from afqmctools.utils.aimbes_utils import aimb_1body_2_afqmc,aimb_2body_2_afqmc


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
    parser.add_argument('-i', '--input', dest='aimbes_file', type=str,
                                            default=None, help='Input AIMBES h5 file.')
    parser.add_argument('-o', '--output', dest='output_file',
                                            type=str, default='crpa_afqmc.h5',
                                            help='Output AFQMC h5 file.')
    parser.add_argument('-cd','--cholesky', dest='use_cholesky',
                            action='store_true', default=True,
                            help='use a Cholesky decomposition on two-body interaction')
    parser.add_argument('-t', '--tol', dest='tol',
                    type=float, default=1e-6,
                    help='Cholesky tolerance for two-body interactions')
    parser.add_argument('-v', '--verbose', dest='verbose',
                                            action='store_true', default=False,
                                            help='Verbose output.')
    parser.add_argument('-na',dest="na",type=int,
                            help='Number of spin-up electrons',
                            required=True)
    parser.add_argument('-nb',dest="nb",type=int,
                            help='Number of spin-down electrons',
                            required=True)
    parser.add_argument('-dt','--type',dest="downfold_type",
                            type=str, default='gw',
                            help='type of 1-body downfolding. options: dft, gw')
    parser.add_argument('-d', '--double-counting', dest='use_double_counting',
                                            action='store_true', default=True,
                                            help='Include double counting correction?')
    parser.add_argument('-r','--real',dest='real_all',
                                            action='store_true', default=False,
                                            help='force all integrals to be real valued'
                                            )
    parser.add_argument('-r1','--real1',dest='real_1body',
                                            action='store_true', default=False,
                                            help='force 1-body integrals to be real valued'
                                            )
    parser.add_argument('-r2','--real2',dest='real_2body',
                                            action='store_true', default=False,
                                            help='force 2-body integrals to be real valued'
                                            )
    parser.add_argument('--bands',dest="bands",type=str,
                            default=None,
                            help="range of input bands from Wannier90 format is `lower:upper` "
                        )

    options = parser.parse_args()

    if options.real_all:
        options.real_1body = True
        options.real_2body = True

    if not options.aimbes_file:
        parser.print_help()
        sys.exit(1)

    return options

def main():
    """Convert AIMBES Hamiltonians saved in an hdf5 checkpoint file
        to the AFQMC format.

    Parameters
    ----------
    args : list of strings
            command-line arguments.
    """
    options = parse_args()

    print("\n ====== Getting Effective 1-body Hamiltonian ====== ")
    H1 = aimb_1body_2_afqmc(
        options.aimbes_file,
        type=options.downfold_type,
        gw_iteration=1,
        input_band_range=options.bands,
        force_real=options.real_1body
        )

    if options.real_1body:
        print(f"  [!] forcing real-value one-body Hamiltonian")
        print(f"     - maximum imaginary component: {np.max(H1.imag)}")
        H1 = H1.real

    print("\n ====== Getting cRPA Screened Coulomb Interaction ====== ")

    use_crpa = True
    if options.downfold_type == 'fc' or options.downfold_type == 'gw_fc':
        use_crpa = False

    L_cRPA = aimb_2body_2_afqmc(
        options.aimbes_file,
        use_chol=True,
        tol=options.tol,
        use_crpa=use_crpa,
        force_real=options.real_2body
        )

    # TODO: read nelec from AIMBES checkpoint if possible
    write_dense(
        filename=options.output_file,
        hcore=H1,
        chol=L_cRPA,
        nelec=(options.na,options.nb),
        nmo=H1.shape[-1],
        real_chol = not np.iscomplexobj(L_cRPA),
        )


if __name__ == '__main__':
    main()
