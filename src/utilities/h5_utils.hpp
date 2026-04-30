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
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "configuration.hpp"
#include <typeinfo>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/nda.hpp"

#include "utilities/check.hpp"

namespace sfqmc::utils {

namespace detail {

template<typename T, bool is_complex = false, nda::MemoryArray A_t>
void read_cast(h5::group& g, std::string name, A_t && A)
{
  using value_t = nda::get_value_t<A_t>;
  if constexpr (is_complex) {
    nda::array<std::complex<T>,nda::get_rank<A_t>> B(A.shape());
    nda::h5_read(g,name,B);
    if constexpr (std::is_assignable_v<value_t&,std::complex<T>>) 
      A() = B();
    else
      check(false,"Error in utils::h5_read_with_cast: Failed is_assignable_v");
  } else {
    nda::array<T,nda::get_rank<A_t>> B(A.shape());
    nda::h5_read(g,name,B);
    if constexpr (std::is_assignable_v<value_t&,T>) 
      A() = B();
    else
      check(false,"Error in utils::h5_read_with_cast: Failed is_assignable_v");
  }
}

}

// reads and casts if types don't match
auto h5_read_with_cast(h5::group& g, std::string name, nda::MemoryArray auto && A)
{
  using A_t = std::decay_t<decltype(A)>; 
  using T = nda::get_value_t<A_t>;
  T x = {};
  if constexpr (nda::mem::on_host<A_t>) {
    auto l = h5::array_interface::get_dataset_info(g,name); 
    if (l.has_complex_attribute) {
      using T_real = nda::remove_complex_t<T>; 
      if (H5Tequal(h5::detail::hid_t_of<T_real>(),l.ty)>0) {
        nda::h5_read(g,name,A);
      } else {
        // find a better way. Tedious, so limiting to a few types
        if (H5Tequal(h5::detail::hid_t_of<double>(),l.ty)>0) {
          detail::read_cast<double,true>(g,name,A);
        } else if (H5Tequal(h5::detail::hid_t_of<float>(),l.ty)>0) {
          detail::read_cast<float,true>(g,name,A);
        } else {
          utils::check(false, "Problems with sfqmc::utils::h5_read_with_cast: Missing specialized complex type read:{}",typeid(x).name());
        }
      }  
    } else { 
      if (H5Tequal(h5::detail::hid_t_of<T>(),l.ty)) {
        nda::h5_read(g,name,A);
      } else {
        // find a better way. Tedious, so limiting to a few types
        if (H5Tequal(h5::detail::hid_t_of<int>(),l.ty)>0) {
          detail::read_cast<int>(g,name,A);
        } else if (H5Tequal(h5::detail::hid_t_of<long>(),l.ty)>0) {
          detail::read_cast<long>(g,name,A);
        } else if (H5Tequal(h5::detail::hid_t_of<double>(),l.ty)>0) {
          detail::read_cast<double>(g,name,A);
        } else if (H5Tequal(h5::detail::hid_t_of<float>(),l.ty)>0) {
          detail::read_cast<float>(g,name,A);
        } else {
          utils::check(false, "Problems with sfqmc::utils::h5_read_with_cast: Missing specialized type read:{}",typeid(x).name());
        }  
      } // l.has_complex_attribute
    }
  } else {
    // need to stage copy on host
    auto B = nda::to_host(A);
    read(g,name,B);
    A() = B();
  }
}

/// @brief Read an HDF5 dataset into an nda MemoryArray, handling all common
///        storage conventions for complex data.
///
/// Three storage conventions are supported when @p A holds complex values:
///   1. **TRIQS complex attribute** (`has_complex_attribute == true`): the
///      dataset is a plain numeric array and a companion HDF5 attribute marks
///      it as complex storage.  Delegated directly to `nda::h5_read`.
///   2. **Interleaved real/imag dimension** (`rank(dataset) == rank(A) + 1`):
///      the dataset has an extra trailing dimension of size 2 that stores
///      `[real, imag]` pairs.  A real-valued view of @p A is constructed with
///      `memory::to_real_view` and passed to `nda::h5_read`.
///   3. **Plain real dataset** (`rank(dataset) == rank(A)`): the dataset
///      contains only real values which are promoted to complex by setting the
///      imaginary parts to zero.
///
/// For non-complex @p A the call is forwarded unchanged to `nda::h5_read`.
/// Device arrays are staged through a host copy.
///
/// @param g    HDF5 group containing the dataset.
/// @param name Name of the dataset within @p g.
/// @param A    Destination array or array-view; must satisfy `nda::MemoryArray`.
auto h5_read(h5::group& g, std::string name, nda::MemoryArray auto && A)
{
  using A_t = std::decay_t<decltype(A)>;
  using T = nda::get_value_t<A_t>;
  if constexpr (nda::mem::on_host<A_t>) {
    if constexpr (nda::is_complex_v<T>) {
      using T_real = nda::remove_complex_t<T>; 
      auto l = h5::array_interface::get_dataset_info(g,name);
      if (l.has_complex_attribute) {
        nda::h5_read(g,name,A);
      } else if (nda::get_rank<A_t>+1 == l.rank()) {
        auto Ar = memory::to_real_view(A);
        nda::h5_read(g,name,Ar);
      } else {
        // Real-valued dataset read into complex array (imaginary parts = 0).
        utils::check(nda::get_rank<A_t> == (size_t)l.rank(), "Rank mismatch");
        nda::array<T_real, nda::get_rank<A_t>> Ar(A.shape());
        nda::h5_read(g,name,Ar);
        A() = Ar();
      }
    } else { 
      nda::h5_read(g,name,A);
    }
  } else {
    // need to stage copy on host
    auto B = nda::to_host(A);
    read(g,name,B);
    A() = B();
  }
}

} // namespace sfqmc::utils
