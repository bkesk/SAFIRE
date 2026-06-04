/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

#include <iostream>
#include<string>
#include <array>
#include <string_view>
#include <source_location>
#include <type_traits>
#include "IO/AppAbort.hpp"
#include "nda/nda.hpp"

namespace sfqmc {
namespace utils
{

/**
 * Checks whether the cond is true, and abort otherwise with provided message (args)
 * @tparam Args
 * @param cond - condition to be verified
 * @param args - messages
 */
template<class... Args>
struct check
{
  check(bool cond, const std::string_view format_string, Args&&... args, const std::source_location& loc = std::source_location::current())
  {
    if(not cond) {
      if constexpr (sizeof...(Args) > 0)
        APP_ABORT_with_source(loc, format_string, std::forward<Args>(args)...);
      else
        APP_ABORT_with_source(loc, format_string);
    }
  }
};

template <typename... Args>
check(bool, const std::string_view, Args&&...) -> check<Args...>;

/**
 * Checks whether `a` has the shape given by `shape...`, and aborts otherwise
 * with the message "array <name> has unexpected shape <actual> != <expected>".
 *
 * Modeled on check: a struct template plus a deduction guide, so that the
 * trailing defaulted std::source_location follows the variadic extents and is
 * still captured at the call site.
 *
 * @tparam A      - nda array type (rank must equal the number of extents)
 * @tparam Longs  - integral extents
 * @param a       - array to check
 * @param name    - name of the array, used in the abort message
 * @param shape   - expected extents (one per dimension)
 */
template<class A, class... Longs>
struct check_shape
{
  check_shape(A const& a, std::string_view name, Longs... shape,
              const std::source_location& loc = std::source_location::current())
  {
    static_assert(nda::ArrayOfRank<A, sizeof...(Longs)>,
                  "check_shape: argument must be an nda array whose rank equals the number of extents");
    static_assert((std::is_integral_v<Longs> && ...),
                  "check_shape: extents must be integral");
    std::array<long, sizeof...(Longs)> expected{ static_cast<long>(shape)... };
    if(a.shape() != expected)
      APP_ABORT_with_source(loc, "array {} has unexpected shape {} != {}",
                            name, a.shape(), expected);
  }
};

template<class A, class... Longs>
check_shape(A const&, std::string_view, Longs...) -> check_shape<A, Longs...>;

} // namespace utils
} // namespace sfqmc

