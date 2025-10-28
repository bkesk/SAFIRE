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

#ifndef RAW_POINTERS_DETAIL_HPP
#define RAW_POINTERS_DETAIL_HPP

#include <iostream>
#include <cassert>
#include <type_traits>
#include <complex>
#include "AFQMC/Utilities/type_conversion.hpp"
#include "Utilities/check.hpp"

namespace sfqmc
{
namespace afqmc
{

template<class T, typename = typename std::enable_if_t<std::is_fundamental<T>::value>>
inline static T* to_address(T* p)
{
  return p;
}

template<class T>
inline static std::complex<T>* to_address(std::complex<T>* p)
{
  return p;
}

template<class T>
inline static std::complex<T> const* to_address(std::complex<T> const* p)
{
  return p;
}


template<class T, typename = typename std::enable_if_t<std::is_fundamental<T>::value>>
inline static T* raw_pointer_cast(T* p)
{
  return p;
}

template<class T>
inline static std::complex<T>* raw_pointer_cast(std::complex<T>* p)
{
  return p;
}

template<class T>
inline static std::complex<T> const* raw_pointer_cast(std::complex<T> const* p)
{
  return p;
}

template<class Q, class T>
inline static Q* reinterpret_pointer_cast(T* p)
{
  return reinterpret_cast<Q*>(p);
}

/************* copy_n_cast ****************/
template<class T, class Q, class Size>
Q* copy_n_cast(T const* A, Size n, Q* B)
{
  for (Size i = 0; i < n; i++, ++A, ++B)
    *B = static_cast<Q>(*A);
  return B;
}

/************* inplace_cast ****************/
template<class T, class Q, class Size>
void inplace_cast(T* A, Size n)
{
  Q* B(reinterpret_cast<Q*>(A));
  if (sizeof(T) >= sizeof(Q))
  {
    for (Size i = 0; i < n; i++, ++A, ++B)
      *B = static_cast<Q>(*A);
  }
  else if (sizeof(T) < sizeof(Q))
  {
    RUNTIME_CHECK(sizeof(T) * 2 <= sizeof(Q), "");
    A += (n - 1);
    B += (n - 1);
    for (; n > 0; n--, --A, --B)
      *B = static_cast<Q>(*A);
  }
}

/************* fill2D ****************/
template<typename T, typename Size>
void fill2D(Size N, Size M, T* y, Size lda, T a)
{
  for (long ip = 0; ip < long(N); ++ip)
    for (long jp = 0; jp < long(M); ++jp)
    {
      y[ip * long(lda) + jp] = a;
    }
}

/************* print ****************/
template<typename T>
void print(std::string str, T const* p, int n)
{
  std::cout << str << " ";
  for (int i = 0; i < n; i++)
    std::cout << *(p + i) << " ";
  std::cout << std::endl;
}

} // namespace afqmc
} // namespace sfqmc

#endif
