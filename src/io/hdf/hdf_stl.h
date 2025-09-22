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

#ifndef SFQMC_HDF_STL_INTERFACE_H
#define SFQMC_HDF_STL_INTERFACE_H

#include <vector>
#include <sstream>
#include <bitset>

namespace sfqmc
{
/** specialization for std::vector<T>
 *
 * Used with any T with a proper h5_space_type, e.g., intrinsic, TinyVector<T,D>, Tensor<T,D>
 */
template<typename T>
struct h5data_proxy<std::vector<T>> : public h5_space_type<T, 1>
{
  using FileSpace = h5_space_type<T, 1>;
  using FileSpace::dims;
  using FileSpace::get_address;
  typedef std::vector<T> data_type;
  data_type& ref_;

  inline h5data_proxy(data_type& a) : ref_(a) { dims[0] = ref_.size(); }

  inline bool read(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    if (!checkShapeConsistency<T>(grp, aname, FileSpace::rank, dims))
      ref_.resize(dims[0]);
    return h5d_read(grp, aname, get_address(&ref_[0]), xfer_plist);
  }

  inline bool read_attribute(hid_t grp, const std::string& aname)
  {
    if (!checkShapeConsistency<T>(grp, aname, FileSpace::rank, dims))
      ref_.resize(dims[0]);
    return h5a_read(grp, aname, get_address(&ref_[0]));
  }

  inline bool write(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    return h5d_write(grp, aname.c_str(), FileSpace::rank, dims, get_address(&ref_[0]), xfer_plist);
  }

  inline bool write(hid_t grp, const std::string& aname, const std::vector<hsize_t>& dvec, hid_t xfer_plist)
  {
    return h5d_write(grp, aname.c_str(), dvec.size(), dvec.data(), get_address(&ref_[0]), xfer_plist);
  }

};

/** specialization for std::bitset<N>
 */
template<std::size_t N>
struct h5data_proxy<std::bitset<N>>
{
  typedef std::bitset<N> ArrayType_t;
  ArrayType_t& ref;

  h5data_proxy<ArrayType_t>(ArrayType_t& a) : ref(a) {}

  inline bool write(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    unsigned long c = ref.to_ulong();
    h5data_proxy<unsigned long> hc(c);
    return hc.write(grp, aname, xfer_plist);
  }

  inline bool read(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    unsigned long c = ref.to_ulong();
    h5data_proxy<unsigned long> hc(c);
    if (hc.read(grp, aname, xfer_plist))
    {
      ref = c;
      return true;
    }
    else
      return false;
  }

  inline bool read_attribute(hid_t grp, const std::string& aname)
  {
    unsigned long c = ref.to_ulong();
    h5data_proxy<unsigned long> hc(c);
    if (hc.read_attribute(grp, aname))
    {
      ref = c;
      return true;
    }
    else
      return false;
  }
};


/** Specialization for std::string */
template<>
struct h5data_proxy<std::string>
{
  typedef std::string ArrayType_t;
  ArrayType_t& ref;

  h5data_proxy<ArrayType_t>(ArrayType_t& a) : ref(a) {}

  inline bool write(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    hid_t str80 = H5Tcopy(H5T_C_S1);
    H5Tset_size(str80, ref.size());
    hsize_t dim     = 1;
    hid_t dataspace = H5Screate_simple(1, &dim, NULL);
    hid_t dataset   = H5Dcreate(grp, aname.c_str(), str80, dataspace, H5P_DEFAULT);
    herr_t ret      = H5Dwrite(dataset, str80, H5S_ALL, H5S_ALL, xfer_plist, ref.data());
    H5Sclose(dataspace);
    H5Dclose(dataset);
    return ret != -1;
  }

  inline bool read(hid_t grp, const std::string& aname, hid_t xfer_plist = H5P_DEFAULT)
  {
    hid_t dataset = H5Dopen(grp, aname.c_str());
    if (dataset > -1)
    {
      hid_t datatype = H5Dget_type(dataset);
      auto cset = H5Tget_cset(datatype);;
      if(H5Tis_variable_str(datatype) > 0) {

        hid_t dataspace = H5Dget_space (dataset);
        int ndims  = H5Sget_simple_extent_dims(dataspace, NULL, NULL);
        if( ndims != 0 ) 
          return false;

        char* rdata[1];
        hid_t memtype = H5Tcopy (H5T_C_S1);
        herr_t ret = H5Tset_size (memtype, H5T_VARIABLE);
	if(ret < 0) return ret;
        ret = H5Tset_cset (memtype, cset);
	if(ret < 0) return ret;

        ret =  H5Dread (dataset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
	if(ret < 0) return ret;
        ref = std::string(rdata[0]);

        ret = H5Dvlen_reclaim (memtype, dataspace, H5P_DEFAULT, rdata);
        H5Tclose(memtype);
        H5Tclose(dataspace);
        H5Tclose(datatype);
        H5Dclose(dataset);
        return ret >= 0;

      } else {

        hid_t dataspace = H5Dget_space (dataset);
        int ndims  = H5Sget_simple_extent_dims(dataspace, NULL, NULL);
        if( ndims > 1 ) 
          return false;

        hsize_t dim_out;
        if (datatype == H5T_NATIVE_CHAR)
        {
          ndims  = H5Sget_simple_extent_dims(dataspace, &dim_out, NULL);
        }
        else
        {
          // MAM: Not sure what this is?
          dim_out = H5Tget_size(datatype);
        }
        ref.resize(dim_out);

        herr_t ret = H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, xfer_plist, &(ref[0]));
        H5Sclose(dataspace);
        H5Tclose(datatype);
        H5Dclose(dataset);
        return ret != -1;
      }
    }
    return false;
  }

  // no attribute strings for now
  inline bool read_attribute([[maybe_unused]] hid_t grp, [[maybe_unused]] const char* name) { return false; }
};

template<>
struct h5data_proxy<std::ostringstream>
{
  typedef std::ostringstream Data_t;
  Data_t& ref;

  h5data_proxy<Data_t>(Data_t& a) : ref(a) {}

  inline bool write(hid_t grp, const std::string& aname, [[maybe_unused]] hid_t xfer_plist = H5P_DEFAULT)
  {
    std::string clone(ref.str());
    h5data_proxy<std::string> proxy(clone);
    return proxy.write(grp, aname);
  }

  inline bool read([[maybe_unused]] hid_t grp, [[maybe_unused]] const char* name, [[maybe_unused]] hid_t xfer_plist = H5P_DEFAULT) { return false; }
  inline bool read_attribute([[maybe_unused]] hid_t grp, [[maybe_unused]] const char* name) { return false; }
};

} // namespace sfqmc
#endif
