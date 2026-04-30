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

#ifndef AFQMC_TTI_HPP
#define AFQMC_TTI_HPP

#include "config.0.h"

namespace sfqmc
{
namespace afqmc
{

namespace detail
{

// checks if class has a member function called reserve that accepts a vector of size_t
template<class T, typename = decltype(std::declval<T>().reserve(std::vector<std::size_t>{}))>
std::true_type has_reserve_with_vector_aux(T);
std::false_type has_reserve_with_vector_aux(...);
template<class V>
struct has_reserve_with_vector : decltype(has_reserve_with_vector_aux(std::declval<V>()))
{};

}

// reserve with either vector or size_t
template<class Container, typename integer, typename = std::enable_if_t<detail::has_reserve_with_vector<Container>{}>>
void reserve_to_fit(Container& C, std::vector<integer> const& v)
{
  C.reserve(v);
}

template<class Container, typename integer, typename = std::enable_if_t<not detail::has_reserve_with_vector<Container>{}>>
void reserve_to_fit(Container& C, std::vector<integer> const& v, double = 0)
{
  C.reserve(std::accumulate(v.begin(), v.end(), std::size_t(0)));
}


namespace detail
{
template<class T, class tp, typename = decltype(std::declval<T>().emplace_back(std::declval<tp>()))>
std::true_type has_emplace_back_tp_aux(T,tp);
std::false_type has_emplace_back_tp_aux(...);
template<class V, class tp>
struct has_emplace_back_tp : decltype(has_emplace_back_tp_aux(std::declval<V>(),std::declval<tp>()))
{};

template<class T, class tp, typename = decltype(std::declval<T>().emplace(std::declval<tp>()))>
std::true_type has_emplace_tp_aux(T,tp);
std::false_type has_emplace_tp_aux(...);
template<class V, class tp>
struct has_emplace_tp : decltype(has_emplace_tp_aux(std::declval<V>(),std::declval<tp>()))
{};
}

// dispatch to emplace_back preferentially
template<class Container, class tp_, typename = std::enable_if_t<detail::has_emplace_back_tp<Container,tp_>{}>>
void emplace(Container& C, tp_ const& a)
{
  C.emplace_back(a);
}

// dispatch to emplace if exists (and emplace_back doesn't)
template<class Container,
         class tp_,
         typename = std::enable_if_t<not detail::has_emplace_back_tp<Container,tp_>{}>,
         typename = std::enable_if_t<detail::has_emplace_tp<Container,tp_>{}>>
void emplace(Container& C, tp_ const& a)
{
  C.emplace(a);
}

} // namespace afqmc

} // namespace sfqmc

#endif
