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

#ifndef SFQMC_HDF_DATASPACE_TRAITS_H
#define SFQMC_HDF_DATASPACE_TRAITS_H

/**@file hdf_dataspace.h
 * @brief define h5_space_type to handle basic datatype for hdf5
 *
 * h5_space_type is a helper class for h5data_proxy and used internally
 * - h5_space_type<T,RANK>
 * - h5_space_type<std::complex<T>,RANK>
 * - h5_space_type<TinyVector<T,D>,RANK>
 * - h5_space_type<TinyVector<std::complex<T>,D>,RANK> // removed, picked up by template recursion
 * - h5_space_type<Tensor<T,D>,RANK>
 * - h5_space_type<Tensor<std::complex<T>,D>,RANK> // removed, picked up by template recursion
 */

#include "hdf_datatype.h"
#include <complex>

namespace sfqmc
{
/** default struct to define a h5 dataspace, any intrinsic type T
 *
 * \tparm T intrinsic datatype
 * \tparm RANK rank of the multidimensional h5dataspace
 */
template<typename T, hsize_t RANK>
struct h5_space_type
{
  ///shape of the dataspace, protected for zero size array, hdf5 support scalar as rank = 0
  hsize_t dims[RANK > 0 ? RANK : 1];
  ///rank of the multidimensional dataspace
  static constexpr hsize_t rank = RANK;
  ///new rank added due to T
  static constexpr int added_rank() { return 0; }
  ///return the address
  inline static auto get_address(T* a) { return a; }
};

/** specialization of h5_space_type for std::complex<T>
 *
 * Raize the dimension of the space by 1 and set the last dimension=2
 */
template<typename T, hsize_t RANK>
struct h5_space_type<std::complex<T>, RANK> : public h5_space_type<T, RANK + 1>
{
  using Base = h5_space_type<T, RANK + 1>;
  using Base::dims;
  using Base::rank;
  static constexpr int added_rank() { return Base::added_rank() + 1; }
  inline h5_space_type() { dims[RANK] = 2; }
  inline static auto get_address(std::complex<T>* a) { return Base::get_address(reinterpret_cast<T*>(a)); }
};

} // namespace sfqmc
#endif
