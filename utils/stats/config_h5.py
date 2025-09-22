# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import tables
import numpy as np


def open_write(fname):
  filters = tables.Filters(complevel=5, complib='zlib')
  fp = tables.open_file(fname, mode='w', filters=filters)
  return fp


def save_vec(vec, h5file, slab, name):
  """ save numpy array into an h5 slab under name

  Args:
    vec (np.array): numpy ndarray of arbitrary dimension and type
    h5file (tables.file.File): pytables File
    slab (tables.Group): HDF5 slab
    name (str): name of CArray to create
  """
  try:
    vec.dtype
  except AttributeError as err:
    vec = np.array([vec])
  atom = tables.Atom.from_dtype(vec.dtype)
  ca = h5file.create_carray(slab, name, atom, vec.shape)
  ca[:] = vec


def save_dict(arr_dict, h5file, slab=None):
  """ save a dictionary of numpy arrays into h5file
   each entry will create its own sub-slab using key as name

  Args:
    arr_dict (dict): dictionary of numpy arrays
    h5file (tables.file.File): pytables File
    slab (tables.Group, optional): HDF5 slab, if None, then use root
  """
  if slab is None:
    slab = h5file.root
  for key, arr in arr_dict.items():
    if type(arr) is dict:
      slab1 = h5file.create_group(slab, key)
      save_dict(arr, h5file, slab=slab1)
    else:
      save_vec(arr, h5file, slab, key)
  h5file.flush()
