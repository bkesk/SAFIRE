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

#ifndef MULTI_CUDA_REF_ADDONS_HPP
#define MULTI_CUDA_REF_ADDONS_HPP

#include "multi/memory/adaptors/cuda/ptr.hpp"

namespace boost::multi::memory::cuda {

/*
template<class Other, typename T, std::enable_if_t<not std::is_same<Other, ref<T>>{}, int> = 0>
auto operator+=(ref<T>&& r, Other&& o)
{
  T v = T(r) + std::forward<Other>(o);
  std::move(r) = v;
  return std::move(r);
} 
template<class Other, typename T, std::enable_if_t<not std::is_same<Other, ref<T>>{}, int> = 0>
auto operator-=(ref<T>&& r, Other&& o)
{
  T v = T(r) - std::forward<Other>(o);
  std::move(r) = v;
  return std::move(r);
}   
*/
template<class Other, typename T, std::enable_if_t<not std::is_same<Other, ref<T>>{}, int> = 0>
auto operator*=(ref<T>&& r, Other&& o)
{
  T v = T(r) * std::forward<Other>(o);
  std::move(r) = v;
  return std::move(r);
}   
template<class Other, typename T, std::enable_if_t<not std::is_same<Other, ref<T>>{}, int> = 0>
auto operator/=(ref<T>&& r, Other&& o)
{
  T v = T(r) / std::forward<Other>(o);
  std::move(r) = v;
  return std::move(r);
}   

/*
template<typename T, typename TT, typename = decltype(std::declval<T&>()+=std::declval<T&&>())>
auto operator+=(ref<T>&& r, ref<TT>&& o)
{ 
  T v = T(r);
  v += TT(o); 
  std::move(r) = v;
  return std::move(r);
}
template<typename T, typename TT, typename = decltype(std::declval<T&>()-=std::declval<T&&>())>
auto operator-=(ref<T>&& r, ref<TT>&& o)
{
  T v = T(r); 
  v -= TT(o); 
  std::move(r) = v;
  return std::move(r);
}
*/
template<typename T, typename TT, typename = decltype(std::declval<T&>()*=std::declval<T&&>())>
auto operator*=(ref<T>&& r, ref<TT>&& o)
{
  T v = T(r); 
  v *= TT(o); 
  std::move(r) = v;
  return std::move(r);
}
template<typename T, typename TT, typename = decltype(std::declval<T&>()/=std::declval<T&&>())>
auto operator/=(ref<T>&& r, ref<TT>&& o)
{
  T v = T(r); 
  v /= TT(o); 
  std::move(r) = v;
  return std::move(r);
}

template<class Other, typename T>
auto operator+=(Other&& o, ref<T> const& t)
{
  return std::forward<Other>(o) += T(t);
}
template<class Other, typename T> 
auto operator-=(Other&& o, ref<T> const& t)
{
  return std::forward<Other>(o) -= T(t);
}
template<class Other, typename T> 
auto operator*=(Other&& o, ref<T> const& t)
{
  return std::forward<Other>(o) *= T(t);
}
template<class Other, typename T> 
auto operator/=(Other&& o, ref<T> const& t)
{
  return std::forward<Other>(o) /= T(t); 
}

template<class Other, typename T>
auto operator+(ref<T> const& t, Other&& o)
{ return T(t) + std::forward<Other>(o); }
template<class Other, typename T>
auto operator-(ref<T> const& t, Other&& o)
{ return T(t) - std::forward<Other>(o); }
template<class Other, typename T>
auto operator*(ref<T> const& t, Other&& o)
{ return T(t) * std::forward<Other>(o); }
template<class Other, typename T>
auto operator/(ref<T> const& t, Other&& o)
{ return T(t) / std::forward<Other>(o); }

template<class Other, typename T, std::enable_if_t<not std::is_same<Other, ref<T>>{}, int> = 0>
auto operator+(Other&& o, ref<T> const& t)
{ return std::forward<Other>(o) + T(t); }
template<class Other, typename T, std::enable_if_t<not std::is_same<Other, ref<T>>{}, int> = 0>
auto operator-(Other&& o, ref<T> const& t)
{ return std::forward<Other>(o) - T(t); }
template<class Other, typename T, std::enable_if_t<not std::is_same<Other, ref<T>>{}, int> = 0>
auto operator*(Other&& o, ref<T> const& t)
{ return std::forward<Other>(o) * T(t); }
template<class Other, typename T, std::enable_if_t<not std::is_same<Other, ref<T>>{}, int> = 0>
auto operator/(Other&& o, ref<T> const& t)
{ return std::forward<Other>(o) / T(t); }


}

#endif
