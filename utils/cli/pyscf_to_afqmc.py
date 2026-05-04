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
import h5py
import json
from mpi4py import MPI
import os
import sys
import time
from afqmctools.inputs.from_pyscf import pyscf_to_afqmc
from afqmctools.utils.misc import get_git_hash

def parse_args(comm):
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

    if comm.rank == 0:
        parser = argparse.ArgumentParser(description = __doc__)
        parser.add_argument('-i', '--input', dest='chk_file', type=str,
                            default=None, help='Input pyscf .chk file.')
        parser.add_argument('-o', '--output', dest='hamil_file',
                            type=str, default='hamiltonian.h5',
                            help='Output file name for hamiltonian (*.h5).')
        parser.add_argument('-w', '--wavefunction', dest='wfn_file',
                            type=str, default=None,
                            help='Output file name for wavefunction (*.h5). '
                                 'By default will write to hamil_file.')
        parser.add_argument('-t', '--cholesky-threshold', dest='thresh',
                            type=float, default=1e-5,
                            help='Cholesky convergence threshold.')
        parser.add_argument('-k', '--kpoint', dest='kpoint_sym',
                            action='store_true', default=False,
                            help='Generate explicit kpoint dependent integrals.')
        parser.add_argument('--density-fit', dest='df',
                            action='store_true', default=False,
                            help='Use density fitting integrals stored in '
                            'input pyscf chkpoint file.')
        parser.add_argument('-a', '--ao', '--ortho-ao', dest='ortho_ao',
                            action='store_true', default=False,
                            help='Transform to ortho AO basis. Default assumes '
                            'we work in MO basis')
        parser.add_argument('-c', '--cas',
                            help='Specify a CAS in the form of N,M.',
                            type=lambda s: [int(item) for item in s.split(',')],
                            default=None)
        parser.add_argument('-d', '--disable-ham', dest='disable_ham',
                            action='store_true', default=False,
                            help='Disable hamiltonian generation.')
        parser.add_argument('-n', '--num-dets', dest='ndet_max',
                            type=int, default=1,
                            help='Set upper limit on number of determinants to '
                            'generate.')
        parser.add_argument('-r', '--real-ham', dest='real_chol',
                            action='store_true', default=False,
                            help='Write integrals as real numbers.')
        parser.add_argument('-p', '--phdf', dest='phdf',
                            action='store_true', default=False,
                            help='Use parallel hdf5.')
        parser.add_argument('--low', dest='low_thresh',
                            type=float, default=0.1,
                            help='Lower threshold for non-integer occupancies'
                            'to include in multi-determinant exansion.')
        parser.add_argument('--high', dest='high_thresh',
                            type=float, default=0.95,
                            help='Upper threshold for non-integer occupancies'
                            'to include in multi-determinant exansion.')
        parser.add_argument('--sfx2c', dest='sfx2c',
                            action='store_true', default=False,
                            help='Use sfx2c 1-body hamiltonian.')
        parser.add_argument('--x2c', dest='x2c',
                            action='store_true', default=False,
                            help='Use 2-component x2c hamiltonian with spin-orbit coupling.')
        parser.add_argument('-v', '--verbose', action='count', default=0,
                            help='Verbose output.')

        options = parser.parse_args()
    else:
        options = None
    options = comm.bcast(options, root=0)

    if not options.chk_file:
        if comm.rank == 0:
            parser.print_help()
        sys.exit()

    return options

def write_metadata(options, sha1, cwd, date_time):
    op_dict = vars(options)
    op_dict['git_hash'] = sha1
    op_dict['working_directory'] = os.getcwd()
    op_dict['date_time'] = date_time
    if options.wfn_file != options.hamil_file:
        with h5py.File(options.wfn_file, 'a') as fh5:
            try:
                fh5['metadata'] = json.dumps(op_dict)
            except RuntimeError:
                del fh5['metadata']
                fh5['metadata'] = json.dumps(op_dict)
    if not options.disable_ham:
        with h5py.File(options.hamil_file, 'a') as fh5:
            fh5['metadata'] = json.dumps(op_dict)

def main():
    """Generate HDF5 input from pyscf checkpoint file.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    comm = MPI.COMM_WORLD
    options = parse_args(comm)
    if comm.rank == 0:
        cwd = os.getcwd()
        sha1 = get_git_hash()
        date_time = time.asctime()
        print(" # Generating HDF5 input from PYSCF checkpoint file.")
        print(" # git sha1: {}".format(sha1))
        print(" # Date/Time: {}".format(date_time))
        print(" # Working directory: {}".format(cwd))

    if options.x2c:
        raise NotImplementedError('Converter for 2-component X2C hamiltonian not implemented.')
        print(" Using 2-component X2C hamiltonian with spin-orbit coupling.")
    if options.sfx2c:
        print(" Using spin-free X2C hamiltonian.")
    if options.x2c and options.sfx2c:
        print(" --x2c and --sfx2c are mutually exclusive. Choose only 1.")
        assert(0)

    if options.wfn_file is None:
        if options.disable_ham:
            options.wfn_file = 'wfn.dat'
        else:
            options.wfn_file = options.hamil_file
    pyscf_to_afqmc(options.chk_file, options.hamil_file, options.thresh,
                  comm=comm,
                  ortho_ao=options.ortho_ao,
                  kpoint=options.kpoint_sym, df=options.df,
                  verbose=options.verbose, cas=options.cas,
                  wfn_file=options.wfn_file,
                  write_hamil=(not options.disable_ham),
                  ndet_max=options.ndet_max,
                  real_chol=options.real_chol,
                  phdf=options.phdf,
                  low=options.low_thresh,
                  high=options.high_thresh,
                  with_sfx2c=options.sfx2c,
                  with_x2c=options.x2c)
    if comm.rank == 0:
        write_metadata(options, sha1, cwd, date_time)

    if comm.rank == 0:
        print("\n # Finished.")

if __name__ == '__main__':
    main()
