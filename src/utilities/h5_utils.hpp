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

#include <typeinfo>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/nda.hpp"

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
      check(false,"Error in utils::read_h5: Failed is_assignable_v");
  } else {
    nda::array<T,nda::get_rank<A_t>> B(A.shape());
    nda::h5_read(g,name,B);
    if constexpr (std::is_assignable_v<value_t&,T>) 
      A() = B();
    else
      check(false,"Error in utils::read_h5: Failed is_assignable_v");
  }
}

}

auto read_h5(h5::group& g, std::string name, nda::MemoryArray auto && A)
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
          utils::check(false, "Problems with sfqmc::utils::read_h5: Missing specialized complex type read:{}",typeid(x).name());
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
          utils::check(false, "Problems with sfqmc::utils::read_h5: Missing specialized type read:{}",typeid(x).name());
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

} // namespace sfqmc::utils
