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

#ifndef SFQMC_CONTAINER_TRAITS_H
#define SFQMC_CONTAINER_TRAITS_H

#include <stdexcept>
#include <vector>

namespace sfqmc
{
template<typename CT>
struct container_traits
{
  /// the data type of elements
  using element_type = typename CT::value_type;

  /** resize container
   * @param n the size of all the dimensions
   * @param d the number of dimensions
   */
  template<typename I>
  inline static void resize([[maybe_unused]] CT& ref, [[maybe_unused]] I* n, [[maybe_unused]] int d)
  {
    throw std::runtime_error("Unknown container, resizing is not available!");
  }

  /// get the current linear storage size of a container
  inline static size_t getSize(const CT& ref) { return ref.size(); }

  /// get the linear storage pointer of a container
  inline static auto getElementPtr(CT& ref) { return ref.data(); }
};

// template specialization for std::vector
template<typename T, class ALLOC>
struct container_traits<std::vector<T, ALLOC>>
{
  using element_type = T;
  using CT           = std::vector<T, ALLOC>;

  template<typename I>
  inline static void resize(CT& ref, I* n, int d)
  {
    size_t nt = d > 0 ? 1 : 0;
    for (int i = 0; i < d; ++i)
      nt *= n[i];
    ref.resize(nt);
  }

  inline static size_t getSize(const CT& ref) { return ref.size(); }

  inline static auto getElementPtr(CT& ref) { return ref.data(); }
};

} // namespace sfqmc
#endif
