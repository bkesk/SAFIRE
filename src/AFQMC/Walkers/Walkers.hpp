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
  using element = _value_t_; 
  MEMORY_SPACE MEM = _MEM_;
  using SMType  = memory::array_view<_MEM_,element,2>;

  walker() = default;

/*
  template<class ma>
  walker(ma&& a, const wlk_indices& i_, const wlk_descriptor& d_)
      : w_(a.origin(), iextensions<1u>{a.size()}), indx(i_), desc(d_)
  {
//    static_assert(::nda::get_rank<> == 1, "Wrong dimensionality");
  }
*/

  ~walker() {}

/*
  // no copy/move assignment
  walker(walker&& other)      = default;
  walker(walker const& other) = default;
  walker& operator=(walker&& other) = delete;
  walker& operator=(walker const& other) = delete;

  pointer base() { return (*w_).origin(); }
  int size() const { return (*w_).size(0); }
  SMType SlaterMatrix(SpinTypes s)
  {
    if (desc[2] <= 0 && s != Alpha)
      APP_ABORT("error:walker spin out of range in SlaterMatrix(SpinType).");
    return (s == Alpha) ? (SMType(getw_(SM), {desc[0], desc[1]}))
                        : (SMType(getw_(SM) + desc[0] * desc[1], {desc[0], desc[2]}));
  }
  SMType SlaterMatrixN(SpinTypes s)
  {
    if (indx[SMN] < 0)
      APP_ABORT("error: access to uninitialized BP sector. ");
    if (desc[2] <= 0 && s != Alpha)
      APP_ABORT("error:walker spin out of range in SlaterMatrixN(SpinType).");
    return (s == Alpha) ? (SMType(getw_(SMN), {desc[0], desc[1]}))
                        : (SMType(getw_(SMN) + desc[0] * desc[1], {desc[0], desc[2]}));
  }
  SMType SlaterMatrixAux(SpinTypes s)
  {
    if (indx[SM_AUX] < 0)
      APP_ABORT("error: access to uninitialized BP sector. ");
    if (desc[2] <= 0 && s != Alpha)
      APP_ABORT("error:walker spin out of range in SlaterMatrixAux(SpinType).");
    return (s == Alpha) ? (SMType(getw_(SM_AUX), {desc[0], desc[1]}))
                        : (SMType(getw_(SM_AUX) + desc[0] * desc[1], {desc[0], desc[2]}));
  }
  pointer weight() { return getw_(WEIGHT); }
  pointer phase() { return getw_(PHASE); }
  pointer phase1() { return getw_(PHASE1); }
  pointer phase2() { return getw_(PHASE2); }
  pointer phase3() { return getw_(PHASE3); }
  pointer theta() { return getw_(THETA); }
  pointer pseudo_energy() { return getw_(PSEUDO_ELOC_); }
  pointer onebody_energy() { return getw_(E1_); }
  pointer exchange_energy() { return getw_(EXX_); }
  pointer coulomb_energy() { return getw_(EJ_); }
  pointer E1() { return getw_(E1_); }
  pointer EXX() { return getw_(EXX_); }
  pointer EJ() { return getw_(EJ_); }
  element energy() { return *getw_(E1_) + *getw_(EXX_) + *getw_(EJ_); }
  pointer overlap() { return getw_(OVLP); }
  // replaces Slater Matrix at timestep M+N to timestep N for back propagation.
  void setSlaterMatrixN()
  {
    *SlaterMatrixN(Alpha) = *SlaterMatrix(Alpha);
    if (desc[2] > 0)
      *SlaterMatrixN(Beta) = *SlaterMatrix(Beta);
  }

private:
  boost::multi::array_ptr<element, 1, pointer> w_;
  const wlk_indices& indx;
  const wlk_descriptor& desc;

  pointer getw_(int P) const { return (*w_).origin() + indx[P]; }
*/
};

template<MEMORY_SPACE _MEM_, typename _value_t_>
struct walker_iterator
    : public boost::
          iterator_facade<walker_iterator<_MEM_,_value_t_>, void, std::random_access_iterator_tag, walker<_MEM_,_value_t_>, std::ptrdiff_t>
{
/*
public:
  template<class WBuff>
  walker_iterator(int k, WBuff&& w_, const wlk_indices& i_, const wlk_descriptor& d_)
      : pos(k), W(w_.origin(), w_.extensions()), indx(&i_), desc(&d_)
  {}

  using pointer         = Ptr;
  using element         = typename std::pointer_traits<pointer>::element_type;
  using Wlk_Buff        = boost::multi::array_ptr<element, 2, Ptr>;
  using difference_type = std::ptrdiff_t;
  using reference       = walker<Ptr>;

private:
  int pos;
  Wlk_Buff W;
  wlk_indices const* indx;
  wlk_descriptor const* desc;

  friend class boost::iterator_core_access;

  void increment() { ++pos; }
  void decrement() { --pos; }
  bool equal(walker_iterator const& other) const { return pos == other.pos; }
  reference dereference() const { return reference((*W)[pos], *indx, *desc); }
  void advance(difference_type n) { pos += n; }
  difference_type distance_to(walker_iterator other) const { return other.pos - pos; }
*/
};

} // namespace afqmc

} // namespace sfqmc

