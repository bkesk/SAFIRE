////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#pragma

#include <vector>

#include "config.0.h"
#include "nda/h5.hpp"


namespace sfqmc
{
namespace afqmc
{
// Helper functions for reading integral data from HDF5 files.
// Currently only read one-dimensional integrals into std::vector<T> and two-dimensional
// integrals into boost::multi::array<T,2>, so only handle these overloads.

/** read data from filespace (name) to buffer (out).
 * @return true if successful. false indiciates name not found in dump.
 */
// MAM: check the complex attribute already present in h5 file
template<typename T>
bool readComplexOrReal(h5::group& grp, std::string name, std::vector<T>& out)
{
  auto l = h5::array_interface::get_dataset_info(grp,name);
  int ndim = 1; // vector
  if (shape.size() == ndim + 1)
  {
    if constexpr (not boost::is_complex<T>::value)
      APP_ABORT(" Error in readComplexOrReal: Complex data with real container. ");
    dump.readEntry(out, name);
    return true;
  }
  else if (shape.size() == ndim)
  {
    if constexpr (boost::is_complex<T>::value) {
      std::vector<RealType> out_real(out.size());
      dump.readEntry(out_real, name);
      std::copy_n(out_real.begin(), out_real.size(), out.begin());
    } else {
      dump.readEntry(out, name);
    }
    return true;
  }
  else
  {
    APP_ABORT(" Error reading " + name + " dataspace. Shape mismatch.");
    return false;
  }
}

template<typename T>
bool readComplexOrReal(hdf_archive& dump, std::string name, boost::multi::array<T, 2>& out)
{
  std::vector<int> shape;
  int ndim = 2; // matrix
  if (!dump.getShape<boost::multi::array<T, 2>>(name, shape))
  {
    // name not found in dump.
    return false;
  }
  if (shape.size() == ndim + 1)
  {
    if constexpr (not boost::is_complex<T>::value)
      APP_ABORT(" Error in readComplexOrReal: Complex data with real container. ");
    dump.readEntry(out, name);
    return true;
  }
  else if (shape.size() == ndim)
  {
    if constexpr (boost::is_complex<T>::value) {
      // Real integrals.
      boost::multi::array<RealType, 2> out_real({shape[0], shape[1]});
      dump.readEntry(out_real, name);
      std::copy_n(out_real.origin(), out_real.num_elements(), out.origin());
    } else {
      dump.readEntry(out, name);
    }
    return true;
  }
  else
  {
    APP_ABORT(" Error reading " + name + " dataspace. Shape mismatch.");
    return false;
  }
}

} // namespace afqmc
} // namespace sfqmc

