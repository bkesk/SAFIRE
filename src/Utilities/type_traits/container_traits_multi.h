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

#ifndef UTILITIES_CONTAINER_TRAITS_MULTI_H
#define UTILITIES_CONTAINER_TRAITS_MULTI_H

#include <multi/array.hpp>
#include <multi/array_ref.hpp>
#include "Utilities/type_traits/container_traits.h"
//#include "Utilities/check.hpp"
//#include <cassert>

namespace sfqmc
{
// template specialization for functions in container_traits
template<typename T, boost::multi::dimensionality_type D, class Alloc>
struct container_traits<boost::multi::array<T, D, Alloc>>
{
  using element_type = T;
  using CT           = boost::multi::array<T, D, Alloc>;

  template<typename I>
  inline static void resize([[maybe_unused]] CT& ref, I* n, int d)
  {
    if (d != D)
    {
      std::ostringstream err_msg;
      err_msg << "boost::multi::array<T, " << D << ", Alloc> cannot be resized. Requested dimension = " << d
              << std::endl;
      throw std::runtime_error(err_msg.str());
    }
    std::array<I, D> shape;
    for (int i = 0; i < d; ++i)
      shape[i] = n[i];
    //RUNTIME_CHECK(0, "");
    //ref = CT(boost::multi::extensions_t<D>{shape}, ref.get_allocator());
  }

  inline static size_t getSize(const CT& ref) { return ref.num_elements(); }

  inline static auto getElementPtr(CT& ref) { return std::addressof(*ref.origin()); }
};

template<typename T, boost::multi::dimensionality_type D>
struct container_traits<boost::multi::array_ref<T, D>>
{
  using element_type = T;
  using CT           = boost::multi::array_ref<T, D>;

  template<typename I>
  inline static void resize([[maybe_unused]] CT& ref, [[maybe_unused]] I* n, [[maybe_unused]] int d)
  {
    throw std::runtime_error("Can not resize container_proxy<boost::multi::array_ref<T,D>>!\n");
  }

  inline static size_t getSize(const CT& ref) { return ref.num_elements(); }

  inline static auto getElementPtr(CT& ref) { return std::addressof(*ref.origin()); }
};

} // namespace sfqmc

#endif
