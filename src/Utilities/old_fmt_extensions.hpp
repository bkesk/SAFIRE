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

#ifndef UTILITIES_FMT_EXTENSIONS_HPP
#define UTILITIES_FMT_EXTENSIONS_HPP

#include <spdlog/fmt/bundled/format.h>
#include <complex>

template <> struct fmt::formatter<std::complex<double>> {
  char presentation = 'f';
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && (*it == 'f' || *it == 'e')) presentation = *it++;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  } 

  template <typename FormatContext>
  auto format(const std::complex<double>& p, FormatContext& ctx) -> decltype(ctx.out()) {
    return format_to(
        ctx.out(),
        presentation == 'f' ? "({:f}, {:f})" : "({:e}, {:e})",
        std::real(p), std::imag(p));
  }
};

template <> struct fmt::formatter<std::complex<float>> {
  char presentation = 'f';
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && (*it == 'f' || *it == 'e')) presentation = *it++;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }
  
  template <typename FormatContext>
  auto format(const std::complex<float>& p, FormatContext& ctx) -> decltype(ctx.out()) {
    return format_to(
        ctx.out(),
        presentation == 'f' ? "({:f}, {:f})" : "({:e}, {:e})",
        std::real(p), std::imag(p));
  }
};

#endif
