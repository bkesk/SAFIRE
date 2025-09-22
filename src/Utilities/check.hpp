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

#ifndef UTILITIES_CHECK_HPP
#define UTILITIES_CHECK_HPP

#include <iostream>
#include<string>
#include <boost/version.hpp>
#if (BOOST_VERSION >= 106500) 
#include <boost/stacktrace.hpp>
#endif
#include "Utilities/AppAbort.hpp"

namespace utils
{

/**
 * Checks whether the cond is true, and abort otherwise with provided message (args)
 * @tparam Args
 * @param cond - condition to be verified
 * @param args - messages
 */
template<class... Args>
inline void check(bool cond, Args&&... args)
{
  if(not cond) {
#if (BOOST_VERSION >= 106500) 
    std::cerr << boost::stacktrace::stacktrace() <<std::endl;
#else
    std::cerr << "stacktrace not enabled.\n" <<std::endl;
#endif
    if constexpr (sizeof...(Args) > 0)
      APP_ABORT(std::forward<Args>(args)...);
    else
      APP_ABORT("");
  }
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

// avoiding name collision with catch CHECK macro
#define RUNTIME_CHECK(cond, ...) utils::check_impl(cond, __FILE__, __PRETTY_FUNCTION__, __LINE__, #cond, ##__VA_ARGS__)

#ifdef __clang__
#pragma clang diagnostic pop
#endif

/**
 * Helper function for check_impl to handle variadic template expansion
 */
template<class Tuple, std::size_t... Ints>
inline void check_impl_helper(const char* file, const char* func, int line, const char* cond_str, Tuple&& tup, std::index_sequence<Ints...>)
{
  APP_ABORT(std::string("In file: {} function: {} at line {}: assertion failed <{}> : ").append(std::get<0>(tup)), file, func, line, cond_str, std::get<Ints + 1>(tup)...);
}

/**
 * Checks whether the cond is true, and abort otherwise with provided message (args)
 * @tparam Args
 * @param cond - condition to be verified
 * @param file - file name
 * @param line - line number
 * @param args - messages
 */
template<class... Args>
inline void check_impl(bool cond, const char* file, const char* func, int line, const char* cond_str, Args&&... args)
{
  if(not cond) {
    #if (BOOST_VERSION >= 106500) 
      std::cerr << boost::stacktrace::stacktrace() <<std::endl;
    #else
      std::cerr << "stacktrace not enabled.\n" <<std::endl;
    #endif
    if constexpr (sizeof...(Args) > 0) {
      // auto first_arg = std::get<0>(std::forward_as_tuple(args...));
      auto tup = std::forward_as_tuple(args...);
      check_impl_helper(file, func, line, cond_str, tup, std::make_index_sequence<sizeof...(Args) - 1>());
    } else if constexpr (sizeof...(Args) == 0) {
      APP_ABORT(std::string("In file: {} function: {} at line {}: assertion failed <{}>"), file, func, line, cond_str);
    }
  }
}

} // namespace utils

#endif
