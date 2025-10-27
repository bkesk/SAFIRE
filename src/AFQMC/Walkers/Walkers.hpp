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

#pragma once

#include <random>
#include <type_traits>
#include <memory>

#include "config.h"
#include "IO/AppAbort.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE _MEM_, typename _value_t_>
struct walker
{
public:
  using value_type = _value_t_;
  static const MEMORY_SPACE MEM    = _MEM_;
  template<MEMORY_SPACE _M_>
  using SMType  = memory::array_view<_M_,value_type,2,nda::C_layout>;

  walker() = default;

  walker(nda::MemoryArrayOfRank<1> auto&& a, const wlk_indices& i_, const wlk_descriptor& d_)
      : w_({a.size()}, a.data()), indx(i_), desc(d_)
  {
    utils::check(a.strides()[0] == 1, "Stride mismatch."); 
  }

  ~walker() {}

  // no copy/move assignment
  walker(walker&& other)      = default;
  walker(walker const& other) = default;
  walker& operator=(walker&& other) = delete;
  walker& operator=(walker const& other) = delete;

  auto base() { return w_.data(); }
  auto size() const { return w_.size(); }
  template<MEMORY_SPACE _M_>
  auto SlaterMatrix(SpinTypes s)
  {
    utils::check(s==Alpha or desc[2] > 0, "error:walker spin out of range in SlaterMatrix(SpinType).");
    return (s == Alpha) ? (SMType<_M_>({desc[0], desc[1]}, getw_(SM)))
                        : (SMType<_M_>({desc[0], desc[2]}, getw_(SM) + desc[0] * desc[1]));
  }
  template<MEMORY_SPACE _M_>
  auto SlaterMatrixN(SpinTypes s)
  {
    utils::check(indx[SMN] >= 0, "error: access to uninitialized BP sector. ");
    utils::check(s==Alpha or desc[2] > 0, "error:walker spin out of range in SlaterMatrixN(SpinType).");
    return (s == Alpha) ? (SMType<_M_>({desc[0], desc[1]}, getw_(SMN)))
                        : (SMType<_M_>({desc[0], desc[2]}, getw_(SMN) + desc[0] * desc[1]));
  }
  template<MEMORY_SPACE _M_>
  auto SlaterMatrixAux(SpinTypes s)
  {
    utils::check(indx[SM_AUX]>=0, "error: access to uninitialized BP sector. ");
    utils::check(s==Alpha or desc[2] > 0, "error:walker spin out of range in SlaterMatrixAux(SpinType).");
    return (s == Alpha) ? (SMType<_M_>({desc[0], desc[1]},getw_(SM_AUX)))
                        : (SMType<_M_>({desc[0], desc[2]},getw_(SM_AUX) + desc[0] * desc[1]));
  }
  auto weight() { return getw_(WEIGHT); }
  auto phase() { return getw_(PHASE); }
  auto phase1() { return getw_(PHASE1); }
  auto phase2() { return getw_(PHASE2); }
  auto phase3() { return getw_(PHASE3); }
  auto theta() { return getw_(THETA); }
  auto pseudo_energy() { return getw_(PSEUDO_ELOC_); }
  auto onebody_energy() { return getw_(E1_); }
  auto exchange_energy() { return getw_(EXX_); }
  auto coulomb_energy() { return getw_(EJ_); }
  auto E1() { return getw_(E1_); }
  auto EXX() { return getw_(EXX_); }
  auto EJ() { return getw_(EJ_); }
  // MAM: problem on GPU, need memory transfer!
  auto energy() const { return *getw_(E1_) + *getw_(EXX_) + *getw_(EJ_); }
  auto overlap() { return getw_(OVLP); }
  // replaces Slater Matrix at timestep M+N to timestep N for back propagation.
  void setSlaterMatrixN()
  {
    SlaterMatrixN(Alpha) = SlaterMatrix(Alpha);
    if (desc[2] > 0)
      SlaterMatrixN(Beta) = SlaterMatrix(Beta);
  }

private:
  memory::array_view<MEM, value_type, 1, nda::C_layout> w_;
  const wlk_indices& indx;
  const wlk_descriptor& desc;

  auto getw_(int P) { return w_.data() + indx[P]; }
  auto getw_(int P) const { return w_.data() + indx[P]; }
};

template<MEMORY_SPACE _MEM_, typename _value_t_>
struct walker_iterator
    : public boost::
          iterator_facade<walker_iterator<_MEM_,_value_t_>, void, std::random_access_iterator_tag, walker<_MEM_,_value_t_>, std::ptrdiff_t>
{
public:
  walker_iterator(int k, nda::MemoryArrayOfRank<2> auto&& w_, const wlk_indices& i_, const wlk_descriptor& d_)
      : pos(k), W(w_.indexmap(),w_.data()), indx(&i_), desc(&d_)
  {}

  using element         = _value_t_; 
  using pointer         = element*;
  using Wlk_Buff        = memory::array_view<_MEM_, element, 2>;
  using difference_type = std::ptrdiff_t;
  using reference       = walker<_MEM_, element>;

private:
  int pos;
  mutable Wlk_Buff W;
  wlk_indices const* indx;
  wlk_descriptor const* desc;

  friend class boost::iterator_core_access;

  void increment() { ++pos; }
  void decrement() { --pos; }
  bool equal(walker_iterator const& other) const { return pos == other.pos; }
  reference dereference() const { return reference(W(pos,nda::range::all), *indx, *desc); }
  void advance(difference_type n) { pos += n; }
  difference_type distance_to(walker_iterator other) const { return other.pos - pos; }
};

} // namespace afqmc

} // namespace sfqmc

