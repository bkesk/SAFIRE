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

#ifndef SFQMC_HDF_WRAPPER_FUNCTIONS_H
#define SFQMC_HDF_WRAPPER_FUNCTIONS_H

/**@file hdf_wrapper_functions.h
 * @brief free template functions wrapping HDF5 calls.
 */

#include <vector>
#include "hdf_datatype.h"
#include "hdf_dataspace.h"
#include "Utilities/check.hpp"

namespace sfqmc
{
/** free template function to read the (user) dimension and shape of the dataset.
 * The dimensions contributed by T is excluded.
 * @tparam T data type supported by h5_space_type
 * @param grp group id
 * @param aname name of the dataspace
 * @param sizes_out sizes of each direction. For a scalar, sizes_out.size() == 0
 * @return true if sizes_out is extracted successfully
 *
 * For example, if the data on the file is Matrix<TinyVector<std::complex<double>, 3>>
 * The dataset on the file has a rank of 2 (matrix) + 1 (TinyVector) + 1 (std::complex) + 0 (double) = 4
 * getDataShape<TinyVector<std::complex<double>, 3>> only returns the first 2 dimension
 * getDataShape<std::complex<double>> only returns the first 3 dimension
 * getDataShape<double> returns all the 4 dimension
 */
template<typename T, typename IT>
inline bool getDataShape(hid_t grp, const std::string& aname, std::vector<IT>& sizes_out)
{
  using TSpaceType = h5_space_type<T, 0>;
  TSpaceType TSpace;

  hid_t h1        = H5Dopen(grp, aname.c_str());
  hid_t dataspace = H5Dget_space(h1);
  int rank        = H5Sget_simple_extent_ndims(dataspace);

  bool success = false;
  if (h1 >= 0 && dataspace >= 0 && rank >= 0)
  {
    // check if the rank is sufficient to hold the data type
    if (rank < TSpaceType::rank)
      throw std::runtime_error(aname + " dataset is too small for the requested data type");
    else
    {
      std::vector<hsize_t> sizes_in(rank);
      int status_n = H5Sget_simple_extent_dims(dataspace, sizes_in.data(), NULL);
      utils::check(status_n >= 0, "H5Sget_simple_extent_dims failed for " + aname + "\n");


      // check if the lowest dimensions match the data type
      int user_rank   = rank - TSpaceType::added_rank();
      bool size_match = true;
      for (int dim = user_rank, dim_type = 0; dim < rank; dim++, dim_type++)
        if (sizes_in[dim] != TSpace.dims[dim_type])
          size_match = false;
      if (!size_match)
        throw std::runtime_error("The lower dimensions (container element type) of " + aname + " dataset do not match the requested data type");
      else
      {
        // save the sizes of each directions excluding dimensions contributed by the data type
        sizes_out.resize(user_rank);
        for (int dim = 0; dim < user_rank; dim++)
          sizes_out[dim] = static_cast<IT>(sizes_in[dim]);
        success = true;
      }
    }
  }

  H5Sclose(dataspace);
  H5Dclose(h1);
  return success;
}

/** free function to check dimension
 * @param grp group id
 * @param aname name of the dataspace
 * @param rank rank of the multi-dimensional array
 * @param dims[rank] size for each direction, return the actual size on file
 * @return true if the dims is the same as the dataspace
 */
template<typename T>
inline bool checkShapeConsistency(hid_t grp, const std::string& aname, int rank, hsize_t* dims)
{
  using TSpaceType = h5_space_type<T, 0>;

  std::vector<hsize_t> dims_in;
  if(getDataShape<T>(grp, aname, dims_in))
  {
    const int user_rank = rank - TSpaceType::added_rank();
    if (dims_in.size() != user_rank)
      throw std::runtime_error(aname + " dataspace rank does not match\n");

    bool is_same = true;
    for (int i = 0; i < user_rank; ++i)
    {
      is_same &= (dims_in[i] == dims[i]);
      dims[i] = dims_in[i];
    }
    return is_same;
  }
  else
    return false;
}

/** return true, if successful */
template<typename T>
inline bool h5d_read(hid_t grp, const std::string& aname, T* first, hid_t xfer_plist)
{
  if (grp < 0)
    return false;
  hid_t h1 = H5Dopen(grp, aname.c_str());
  if (h1 < 0)
    return false;
  hid_t h5d_type_id = get_h5_datatype(*first);
  herr_t ret        = H5Dread(h1, h5d_type_id, H5S_ALL, H5S_ALL, xfer_plist, first);
  H5Dclose(h1);
  return ret != -1;
}

/** return true, if successful */
template<typename T>
inline bool h5a_read(hid_t grp, const std::string& aname, T* first)
{
  if (grp < 0)
    return false;
  hid_t h1 = H5Aopen(grp, aname.c_str(), H5P_DEFAULT);
  if (h1 < 0)
    return false;
  hid_t h5d_type_id = get_h5_datatype(*first);
  herr_t ret        = H5Aread(h1, h5d_type_id, first);
  H5Aclose(h1);
  return ret != -1;
}

template<typename T>
inline bool h5d_write(hid_t grp,
                      const std::string& aname,
                      hsize_t ndims,
                      const hsize_t* dims,
                      const T* first,
                      hid_t xfer_plist)
{
  if (grp < 0)
    return true;
  hid_t h5d_type_id = get_h5_datatype(*first);
  hid_t h1          = H5Dopen(grp, aname.c_str());
  herr_t ret        = -1;
  if (h1 < 0) //missing create one
  {
    hid_t dataspace = H5Screate_simple(ndims, dims, NULL);
    hid_t dataset   = H5Dcreate(grp, aname.c_str(), h5d_type_id, dataspace, H5P_DEFAULT);
    ret             = H5Dwrite(dataset, h5d_type_id, H5S_ALL, H5S_ALL, xfer_plist, first);
    H5Sclose(dataspace);
    H5Dclose(dataset);
  }
  else
  {
    ret = H5Dwrite(h1, h5d_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, first);
  }
  H5Dclose(h1);
  return ret != -1;
}

/** return true, if successful */
template<typename T>
bool h5d_read(hid_t grp,
              const std::string& aname,
              hsize_t ndims,
              [[maybe_unused]] const hsize_t* gcounts,
              const hsize_t* counts,
              const hsize_t* offsets,
              T* first,
              hid_t xfer_plist)
{
  if (grp < 0)
    return true;
  hid_t h1 = H5Dopen(grp, aname.c_str());
  if (h1 < 0)
    return false;
  //herr_t ret = H5Dread(h1, h5d_type_id, H5S_ALL, H5S_ALL, xfer_plist, first);

  hid_t dataspace = H5Dget_space(h1);
  hid_t memspace  = H5Screate_simple(ndims, counts, NULL);
  herr_t ret      = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

  hid_t h5d_type_id = get_h5_datatype(*first);
  ret               = H5Dread(h1, h5d_type_id, memspace, dataspace, xfer_plist, first);

  H5Sclose(dataspace);
  H5Sclose(memspace);

  H5Dclose(h1);
  return ret != -1;
}


template<typename T>
inline bool h5d_write(hid_t grp,
                      const std::string& aname,
                      hsize_t ndims,
                      const hsize_t* gcounts,
                      const hsize_t* counts,
                      const hsize_t* offsets,
                      const T* first,
                      hid_t xfer_plist)
{
  if (grp < 0)
    return true;
  hid_t h5d_type_id = get_h5_datatype(*first);
  hid_t h1          = H5Dopen(grp, aname.c_str());
  hid_t filespace, memspace;
  herr_t ret = -1;
  if (h1 < 0) //missing create one
  {
    hid_t dataspace = H5Screate_simple(ndims, gcounts, NULL);
    hid_t dataset   = H5Dcreate(grp, aname.c_str(), h5d_type_id, dataspace, H5P_DEFAULT);

    filespace = H5Dget_space(dataset);
    ret             = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

    memspace = H5Screate_simple(ndims, counts, NULL);
    ret            = H5Dwrite(dataset, h5d_type_id, memspace, filespace, xfer_plist, first);

    H5Dclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dataset);
    H5Sclose(dataspace);
  }
  else
  {
    filespace = H5Dget_space(h1);
    ret       = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

    memspace = H5Screate_simple(ndims, counts, NULL);
    ret      = H5Dwrite(h1, h5d_type_id, memspace, filespace, xfer_plist, first);

    H5Sclose(filespace);
    H5Dclose(memspace);
  }
  H5Dclose(h1);
  return ret != -1;
}

/** return true, if successful */
template<typename T>
bool h5d_read(hid_t grp,
              const std::string& aname,
              hsize_t ndims,
              [[maybe_unused]] const hsize_t* gcounts,
              const hsize_t* counts,
              const hsize_t* offsets,
              hsize_t mem_ndims,
              const hsize_t* mem_gcounts,
              const hsize_t* mem_counts,
              const hsize_t* mem_offsets,
              T* first,
              hid_t xfer_plist)
{
  if (grp < 0)
    return true;
  hid_t h1 = H5Dopen(grp, aname.c_str());
  if (h1 < 0)
    return false;

  hid_t dataspace = H5Dget_space(h1);
  if (ndims != H5Sget_simple_extent_ndims(dataspace))
    throw std::runtime_error(aname + " dataspace does not match ");
  // check gcounts???
  herr_t ret = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

  hid_t memspace = H5Screate_simple(mem_ndims, mem_gcounts, NULL);
  herr_t mem_ret = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offsets, NULL, mem_counts, NULL);
  utils::check(mem_ret >= 0, "H5Sselect_hyperslab failed for " + aname + "\n");

  hid_t h5d_type_id = get_h5_datatype(*first);
  ret               = H5Dread(h1, h5d_type_id, memspace, dataspace, xfer_plist, first);

  H5Sclose(dataspace);
  H5Sclose(memspace);

  H5Dclose(h1);
  return ret != -1;
}

template<typename T>
inline bool h5d_write(hid_t grp,
                      const std::string& aname,
                      hsize_t ndims,
                      const hsize_t* gcounts,
                      const hsize_t* counts,
                      const hsize_t* offsets,
                      hsize_t mem_ndims,
                      const hsize_t* mem_gcounts,
                      const hsize_t* mem_counts,
                      const hsize_t* mem_offsets,
                      const T* first,
                      hid_t xfer_plist)
{
  if (grp < 0)
    return true;
  hid_t h5d_type_id = get_h5_datatype(*first);
  hid_t h1          = H5Dopen(grp, aname.c_str());
  herr_t ret        = -1;
  if (h1 < 0) //missing create one
  {
    hid_t dataspace = H5Screate_simple(ndims, gcounts, NULL);
    hid_t dataset   = H5Dcreate(grp, aname.c_str(), h5d_type_id, dataspace, H5P_DEFAULT);

    hid_t filespace = H5Dget_space(dataset);
    ret             = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

    hid_t memspace = H5Screate_simple(mem_ndims, mem_gcounts, NULL);
    ret            = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offsets, NULL, mem_counts, NULL);
    ret            = H5Dwrite(dataset, h5d_type_id, memspace, filespace, xfer_plist, first);

    H5Dclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dataset);
    H5Sclose(dataspace);
  }
  else
  {
    hid_t filespace = H5Dget_space(h1);
    ret             = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

    hid_t memspace = H5Screate_simple(mem_ndims, mem_gcounts, NULL);
    ret            = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offsets, NULL, mem_counts, NULL);
    ret            = H5Dwrite(h1, h5d_type_id, memspace, filespace, xfer_plist, first);

    H5Sclose(filespace);
    H5Dclose(memspace);
  }
  H5Dclose(h1);
  return ret != -1;
}

template<typename T>
inline bool h5d_append(hid_t grp,
                       const std::string& aname,
                       hsize_t& current,
                       hsize_t ndims,
                       const hsize_t* dims,
                       const T* first,
                       hsize_t chunk_size = 1,
                       hid_t xfer_plist   = H5P_DEFAULT)
{
  if (grp < 0)
    return true;
  hid_t h5d_type_id = get_h5_datatype(*first);
  hid_t dataspace;
  hid_t memspace;
  hid_t dataset = H5Dopen(grp, aname.c_str());
  std::vector<hsize_t> max_dims(ndims);
  max_dims[0] = H5S_UNLIMITED;
  for (int d = 1; d < ndims; ++d)
    max_dims[d] = dims[d];
  herr_t ret = -1;
  if (dataset < 0) //missing create one
  {
    //set file pointer
    current = 0;
    // set max and chunk dims
    std::vector<hsize_t> chunk_dims(ndims);
    chunk_dims[0] = chunk_size;
    for (int d = 1; d < ndims; ++d)
      chunk_dims[d] = dims[d];
    // create a dataspace sized to the current buffer
    dataspace = H5Screate_simple(ndims, dims, max_dims.data());
    // create dataset property list
    hid_t p = H5Pcreate(H5P_DATASET_CREATE);
    // set layout (chunked, contiguous)
    hid_t sl = H5Pset_layout(p, H5D_CHUNKED);
    utils::check(sl > -1, "H5Pset_layout failed in h5d_append");
    // set chunk size
    hid_t cs = H5Pset_chunk(p, ndims, chunk_dims.data());
    utils::check(cs > -1, "H5Pset_chunk failed in h5d_append");
    // create the dataset
    dataset = H5Dcreate2(grp, aname.c_str(), h5d_type_id, dataspace, H5P_DEFAULT, p, H5P_DEFAULT);
    // create memory dataspace, size of current buffer
    memspace = H5Screate_simple(ndims, dims, NULL);
    // write the data for the first time
    ret = H5Dwrite(dataset, h5d_type_id, memspace, dataspace, xfer_plist, first);
    // update the "file pointer"
    current = dims[0];

    // close the property list
    H5Pclose(p);
  }
  else
  {
    // new end of file
    std::vector<hsize_t> start(ndims);
    std::vector<hsize_t> end(ndims);
    for (int d = 1; d < ndims; ++d)
    {
      start[d] = 0;
      end[d]   = dims[d];
    }
    start[0] = current;
    end[0]   = start[0] + dims[0];
    //extend the dataset (file)
    herr_t he = H5Dextend(dataset, end.data());
    utils::check(he > -1, "H5Dextend failed in h5d_append");
    //get the corresponding dataspace (filespace)
    dataspace = H5Dget_space(dataset);
    //set the extent
    herr_t hse = H5Sset_extent_simple(dataspace, ndims, end.data(), max_dims.data());
    utils::check(hse > -1, "H5Sset_extent_simple failed in h5d_append");
    //select hyperslab/slice of multidimensional data for appended write
    herr_t hsh = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start.data(), NULL, dims, NULL);
    utils::check(hsh > -1, "H5Sselect_hyperslab failed in h5d_append");
    //create memory space describing current data block
    memspace = H5Screate_simple(ndims, dims, NULL);
    //append the datablock to the dataset
    ret = H5Dwrite(dataset, h5d_type_id, memspace, dataspace, H5P_DEFAULT, first);
    // update the "file pointer"
    current = end[0];
  }
  // cleanup
  H5Sclose(memspace);
  H5Sclose(dataspace);
  H5Dclose(dataset);
  return ret != -1;
}

} // namespace sfqmc
#endif
