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

#ifndef SFQMC_HDF_H5_DATAPROXY_H
#define SFQMC_HDF_H5_DATAPROXY_H

#include "hdf_wrapper_functions.h"
#include "hdf_dataspace.h"

namespace sfqmc
{

/** generic h5data_proxy<T> for scalar basic datatypes defined in hdf_dataspace.h
 */
template<typename T>
struct h5data_proxy : public h5_space_type<T, 0>
{
  using data_type = T;
  using FileSpace = h5_space_type<T, 0>;
  using FileSpace::dims;
  using FileSpace::get_address;
  data_type& ref_;

  inline h5data_proxy(data_type& a) : ref_(a) { }

  inline bool read(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    return h5d_read(grp, aname, get_address(&ref_), xfer_plist);
  }

  inline bool read_attribute(hid_t grp, const std::string& aname)
  {
    return h5a_read(grp, aname, get_address(&ref_));
  }

  inline bool write(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    return h5d_write(grp, aname.c_str(), FileSpace::rank, dims, get_address(&ref_), xfer_plist);
  }

};

/** specialization for bool, convert to int
 */
template<>
struct h5data_proxy<bool> : public h5_space_type<int, 0>
{
  using data_type = bool;
  using FileSpace = h5_space_type<int, 0>;
  using FileSpace::dims;
  using FileSpace::get_address;
  data_type& ref_;

  inline h5data_proxy(data_type& a) : ref_(a) { }

  inline bool read(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    int copy = static_cast<int>(ref_);
    bool okay = h5d_read(grp, aname, get_address(&copy), xfer_plist);
    ref_ = static_cast<bool>(copy);
    return okay;
  }

  inline bool read_attribute(hid_t grp, const std::string& aname)
  {
    int copy = static_cast<int>(ref_);
    bool okay = h5a_read(grp, aname, get_address(&copy));
    ref_ = static_cast<bool>(copy);
    return okay;
  }

  inline bool write(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    int copy = static_cast<int>(ref_);
    return h5d_write(grp, aname.c_str(), FileSpace::rank, dims, get_address(&copy), xfer_plist);
  }

};

} // namespace sfqmc
#endif
