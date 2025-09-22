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

import h5py

def get_metadata(fstat, path='Metadata'):
  meta = dict()
  with h5py.File(fstat, 'r') as fp:
    for k, v in fp[path].items():
      meta[k] = v[()]
  return meta

def get_bp_taus(fstat):
  sym_md = get_metadata(fstat)
  bp_md = get_metadata(fstat, 'Observables/BackPropagated/Metadata')
  dt = sym_md['Timestep']
  nsteps = bp_md['BackPropSteps']
  taus = nsteps*dt
  return taus

def calc_nequil(teq, taus):
  tbp = taus.max()
  nequil = int(round(teq/tbp))
  if (nequil < 2):
    msg = 'check nequil=%d for teq=%d' % (nequil, teq)
    raise RuntimeError(msg)
  return nequil

def collect_dm(fstat, taus, nequil):
  from stats import stat_h5
  fp = stat_h5.read(fstat)
  dm_map = {'taus': taus}
  for iav, tau in enumerate(taus):
    name = 'a%d' % iav
    dm, de = stat_h5.afobs(fp, 'FullOneRDM', nequil, numer='one_rdm', iav=iav)
    dm_map[name] = {'dm_mean': dm, 'dm_error': de}
  fp.close()
  return dm_map

def mixed_dm(fstat, nequil):
  from stats import stat_h5
  fp = stat_h5.read(fstat)
  dm, de = stat_h5.afobs(fp, 'FullOneRDM', nequil, numer='one_rdm', group='Mixed')
  fp.close()
  return {'mixed': {'dm_mean': dm, 'dm_error': de}}

def get_dms(fstat, teq, neq=None):
  dm_map = dict()
  with h5py.File(fstat, 'r') as f:
    has_mixed = 'Mixed' in f['Observables']
    has_bp = 'BackPropagated' in f['Observables']
  if has_bp:
    taus = get_bp_taus(fstat)
    nequil = calc_nequil(teq, taus)
    dm_map.update(collect_dm(fstat, taus, nequil))
  elif neq is None:
    msg = 'need to calculate nequil w/o BP metadata'
    raise NotImplementedError(msg)
  else:
    nequil = neq
  if has_mixed:
    m_dm_m = mixed_dm(fstat, nequil)
    dm_map.update(m_dm_m)
  return dm_map

def cache_dm(fout, fstat, tequil, neq=None, verbose=False):
  from time import perf_counter
  from stats import config_h5
  # calculate DM
  tick = perf_counter()
  dm_map = get_dms(fstat, tequil, neq=neq)
  tock = perf_counter()
  if verbose:
    print('averaged DMs in ', tock-tick, ' s')
  dm_map['tequil'] = tequil

  fp = config_h5.open_write(fout)
  config_h5.save_dict(dm_map, fp)
  fp.close()

def main():
  """
  Analyze stochastic samples of observables output by SAFIRE
  """
  import os
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--finp', '-i', default='afqmc.json')
  parser.add_argument('--fout', '-o')
  parser.add_argument('--tequil', '-e', type=float, default=100)
  parser.add_argument('--nequil', '-ne', type=int)
  parser.add_argument('--verbose', '-v', action='store_true')
  args = parser.parse_args()
  path = os.path.dirname(args.finp)
  if len(path) < 1:
    path = '.'

  # !!!! hard-code prefix, series
  prefix = 'qmc'
  iser = 0
  fstat = '%s/%s.s%03d.stat.h5' % (path, prefix, iser)
  fout = args.fout
  if fout is None:
    fout = '%s/%s.s%03d.dm.h5' % (path, prefix, iser)

  if os.path.isfile(fout):
    if args.verbose:
      msg = '%s exists' % fout
      print(msg)
  else:
    cache_dm(fout, fstat, args.tequil, neq=args.nequil, verbose=args.verbose)

if __name__ == '__main__':
  main()  # set no global variable
