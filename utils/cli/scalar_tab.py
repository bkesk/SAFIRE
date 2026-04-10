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
from stats.scalar_dat import analyze_scalar_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', type=str, help='Scalar TABle (stab) file name')
    parser.add_argument('--mark-header', '-m', type=str, default='#',
        help='marker for start of header line, default is "#"')
    parser.add_argument('--series-column', '-s', type=str, default="time",
        help='Column used to parameterize the series. Options are "time" (default) or "index". '
            'Effects units of `--nequil` and is xaxis in plot')
    parser.add_argument('--nequil', '-e', type=float, default=0,
        help='number of equilibration blocks to throw out')
    parser.add_argument('--estimate_equil', '-ee', action='store_true',
        help='estimate the number of equilibration blocks to throw out')
    parser.add_argument('--column', '-c', type=str, default='LocalEnergy',
        help='name of column to analyze, to list all columns use the -l flag')
    parser.add_argument('--reblock', '-rb', type=int, default=1,
        help='reblock data to remove auto-correlation, default is no reblock')
    parser.add_argument('--ndiscard', '-nd', type=int, default=None,
                        help='Number of blocks to discard when plotting data trace')
    parser.add_argument('--list', '-l', action='store_true',
        help='list all columns available in the scalar file')
    parser.add_argument('--trace', '-t', action='store_true',
        help='plot the trace of column')
    parser.add_argument('--append', '-a', action='append',
        help='additional Scalar TABle (stab) file to be appended')
    parser.add_argument('--dump', action='store_true', help='dump column')
    parser.add_argument('--dump_fname', type=str, default='trace.dat')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='disable verbose output')
    parser.set_defaults(verbose=True)
    parser.add_argument('--autocorr',action='store',type=int,default=None, help='autocorrelation length. If set, then this value of the autocorrelation length will be used and it will not be automatically computed.')
    parser.add_argument('--savefig')
    parser.add_argument('--dump-avail-columns', action='store_true', help='dump available column names and exit')
    parser.add_argument('--denominator', '-d', type=str, default=None,
        help='denominator column for normalization (e.g., "deno" for energy/denominator)')
    parser.add_argument('--return-complex', action='store_true',
        help='return complex values instead of just real part (only for complex columns)')
    args = parser.parse_args()
    return args

def main():
    """
    Analyze stochastic samples of scalar data output by SAFIRE.
    One major use case is the analysis of the AFQMC energy, but other scalar
    data can be analyzed as well.
    """
    analyze_scalar_data(args=parse_args())

if __name__ == '__main__':
    main()
