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

#ifndef SFQMC_AFQMC_EXCITATIONS_HPP
#define SFQMC_AFQMC_EXCITATIONS_HPP

#include <boost/iterator/iterator_facade.hpp>
#include <map>
#include "AFQMC/config.h"
#include "SparseMatrix/array_of_sequences.hpp"

namespace sfqmc
{
namespace afqmc
{
// CHEAT!!!
template<class TP, class integer>
size_t get_index(TP const& tp_, integer loc)
{
  if (loc == 0)
    return size_t(std::get<0>(tp_));
  else if (loc == 1)
    return size_t(std::get<1>(tp_));
  else
    throw std::runtime_error(" Error in sfqmc::afqmc::get_index<TP,integer>(). ");
}

template<class Vector>
void push_excitation(Vector const& abij, Vector& v)
{
  if (abij.size() == 0)
    return;
  RUNTIME_CHECK(v.size() % abij.size() == 0, "");
  size_t n = abij.size();
  for (typename Vector::iterator it = v.begin(); it < v.end(); it += n)
    if (std::equal(abij.begin(), abij.end(), it))
      return;
  v.insert(v.end(), abij.begin(), abij.end());
}

template<class Vector>
size_t find_excitation(Vector const& abij, Vector& v)
{
  if (abij.size() == 0)
    return 0; // this assumes that the reference is 0
  RUNTIME_CHECK(v.size() % abij.size() == 0, "");
  size_t n   = abij.size();
  size_t loc = 0;
  for (typename Vector::iterator it = v.begin(); it < v.end(); it += n, loc++)
    if (std::equal(abij.begin(), abij.end(), it))
      return loc;
  APP_ABORT("Error: Sequence not found in find_excitation.");
  return 0;
}

template<class excitations>
std::map<int, int> find_active_space(bool single_list, WALKER_TYPES walker_type, excitations const& abij, int NMO, int NAEA, int NAEB)
{
  int npol = ( (walker_type == NONCOLLINEAR) ? 2 : 1 );
  if( walker_type == NONCOLLINEAR )
    RUNTIME_CHECK(NAEB == 0, "");
  std::map<int, int> mo2active;
  for (int i = 0; i < 2 * NMO; i++)
    mo2active[i] = -1;
  std::vector<size_t> count(2 * NMO);
  // reference first
  auto refc = abij.reference_configuration();
  for (int i = 0; i < NAEA + NAEB; i++, ++refc)
    ++count[*refc];
  auto nex = abij.maximum_excitation_number();
  for (int n = 1; n < nex[0]; ++n)
  {
    auto it    = abij.alpha_begin(n);
    auto itend = abij.alpha_end(n);
    for (; it < itend; ++it)
    {
      auto exct = *it + n; // skip locations
      for (int ak = 0; ak < n; ++ak, ++exct)
        ++count[*exct];
    }
  }
  if( walker_type == COLLINEAR) {
    for (int n = 1; n < nex[1]; ++n)
    {
      auto it    = abij.beta_begin(n);
      auto itend = abij.beta_end(n);
      for (; it < itend; ++it)
      {
        auto exct = *it + n; // skip locations
        for (int ak = 0; ak < n; ++ak, ++exct)
          ++count[*exct];
      }
    }
  }
  if (not single_list)
  {
    RUNTIME_CHECK(walker_type == COLLINEAR, "");
    int ik = 0;
    for (int i = 0; i < NMO; ++i)
      if (count[i] > 0 || count[i + NMO] > 0)
      {
        if (count[i] > 0)
          mo2active[i] = ik;
        if (count[i + NMO] > 0)
          mo2active[i + NMO] = ik;
        ++ik;
      }
  }
  else
  {
    int ik = 0;
    for (int i = 0; i < npol*NMO; ++i)
      if (count[i] > 0)
        mo2active[i] = ik++;
    if(walker_type == COLLINEAR) {
      ik = 0;
      for (int i = NMO; i < 2 * NMO; ++i)
        if (count[i] > 0)
          mo2active[i] = ik++;
    }
  }
  return mo2active;
}

/*
 * - exct stores, for each electron excitaion, the location of the orbital in the reference 
 *    being excited and the index of the excited orbital.
 */
template<class Vector, class T>
int get_excitation_number(bool getIndx, Vector& refc, Vector& confg, Vector& exct, T& ci, Vector& Iwork)
{
  int NE = refc.size();
  exct.clear();
  int cnt = 0;
  if (getIndx)
    std::copy(refc.begin(), refc.end(), Iwork.begin());
  auto it = refc.data();
  RUNTIME_CHECK(Iwork.size() >= refc.size(), "");
  for (int i = 0; i < NE; i++, it++)
    if (!std::binary_search(confg.begin(), confg.end(), *it))
    {
      if (getIndx)
      {
        // store the location, NOT the index!!!
        //exct.emplace_back(*it);
        exct.emplace_back(i);
      }
      cnt++;
    }
  if (!getIndx)
    return cnt;
  it       = confg.data();
  int cnt2 = 0;
  for (int i = 0; i < NE; i++, it++)
    if (!std::binary_search(refc.begin(), refc.end(), *it))
    {
      exct.emplace_back(*it);
      Iwork[exct[cnt2]] = *it;
      cnt2++;
    }
  RUNTIME_CHECK(cnt == cnt2, "");
  // sort Iwork and count number of exchanges to determine permutation sign
  // sooo slow but sooo simple too
  for (int i = 0; i < NE; i++)
    for (int j = i + 1; j < NE; j++)
    {
      if (Iwork[j] < Iwork[i])
      {
        ci *= T(-1.0);
        std::swap(Iwork[i], Iwork[j]);
      }
    }
  return cnt;
}

template<typename intT, class csr>
inline std::vector<size_t> get_nnz(csr const& PsiT_MO, intT* refc, size_t N, size_t shift)
{
  std::vector<size_t> res(N);
  for (size_t i = 0; i < N; i++)
    res[i] = PsiT_MO.num_non_zero_elements(*(refc + i) - shift);
  return res;
}


// Holds information necessary to reconstruct particle-hole multi-determinant expansions
// device interface should only be used after the object is properly assembled.
template<class I       = int,
         class VType   = std::complex<double>,
         class Alloc   = shared_allocator<I>,
         class is_root = ma::sparse::is_root,
         class devAlloc = std::allocator<I>    // device allocator can not be shared
        > 
struct ph_excitations
{
public:
  using integer_type       = I;
  using configuration_type = std::tuple<int, int, VType>;

private:
  using IAllocator = Alloc;
  using CAllocator = typename std::allocator_traits<Alloc>::template rebind_alloc<configuration_type>;
  using confg_aos  = ma::sparse::array_of_sequences<configuration_type, int, CAllocator, is_root>;
  using index_aos  = ma::sparse::array_of_sequences<integer_type, int, IAllocator, is_root>;
  using devIAllocator = devAlloc;
// integer allocator stays on CPU, since it is not needed/used on GPU   
  using dev_index_aos  = ma::sparse::array_of_sequences<integer_type, int, 
            devIAllocator, ma::sparse::null_is_root_<devIAllocator>,std::allocator<int>>;

  IAllocator i_allocator_;
  CAllocator c_allocator_;

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  devIAllocator dev_i_allocator_;
#endif

  template<typename Integer>
  class Iterator
      : public boost::
            iterator_facade<Iterator<Integer>, Integer*, std::random_access_iterator_tag, Integer*, std::ptrdiff_t>
  {
  public:
    using difference_type = std::ptrdiff_t;
    using reference       = Integer*;
    using const_reference = Integer const*;
    using value_tupe      = Integer*;

    Iterator(Integer* index, size_t d_) : p_index(index), D(d_) {}

    // What we implement is determined by the boost::forward_traversal_tag
  private:
    friend class boost::iterator_core_access;

    void increment() { p_index += 2 * D; }

    bool equal(Iterator const& other) const { return this->p_index == other.p_index; }

    reference dereference() const { return reference(p_index); }

    void decrement() { p_index -= 2 * D; }

    void advance(int n) { p_index += 2 * D * n; }

    difference_type distance_to(Iterator const& z) const { return ((z.p_index - p_index) / 2 / D); }

  private:
    Integer* p_index;
    size_t D;
  };

  template<typename Integer>
  class Iterator_const : public boost::iterator_facade<Iterator_const<Integer>,
                                                       Integer const*,
                                                       std::random_access_iterator_tag,
                                                       Integer const*,
                                                       std::ptrdiff_t>
  {
  public:
    using difference_type = std::ptrdiff_t;
    using reference       = Integer const*;
    using const_reference = Integer const*;
    using value_tupe      = Integer*;

    Iterator_const(Integer* index, size_t d_) : p_index(index), D(d_) {}
    Iterator_const(Integer const* index, size_t d_) : p_index(index), D(d_) {}

    // What we implement is determined by the boost::forward_traversal_tag
  private:
    friend class boost::iterator_core_access;

    void increment() { p_index += 2 * D; }

    bool equal(Iterator_const const& other) const { return this->p_index == other.p_index; }

    reference dereference() const { return reference(p_index); }

    void decrement() { p_index -= 2 * D; }

    void advance(int n) { p_index += 2 * D * n; }

    difference_type distance_to(Iterator_const const& z) const { return ((z.p_index - p_index) / 2 / D); }

  private:
    Integer* p_index;
    size_t D;
  };

public:
  using Excitation_Iterator       = Iterator<integer_type>;
  using Excitation_const_Iterator = Iterator_const<integer_type>;

  ph_excitations() = delete;

  // Note: terms_per_excitation[0] has special meaning, the number of electrons in the calculation.
  // coefficients[0] will store the reference configuration itself.
  ph_excitations(size_t number_of_configurations,
                 int na_,
                 int nb_,
                 std::vector<size_t>& unique_alpha_counts,
                 std::vector<size_t>& unique_beta_counts,
                 Alloc alloc_ = Alloc{},
                 [[maybe_unused]] devAlloc devalloc_ = devAlloc{})
      : i_allocator_(alloc_),
        c_allocator_(alloc_),
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
        dev_i_allocator_(devalloc_),
#endif
        NAEA(na_),
        NAEB(nb_),
        configurations(1, number_of_configurations, c_allocator_),
        reference(1, NAEA + NAEB, i_allocator_),
        unique_alpha(unique_alpha_counts.size(), unique_alpha_counts, i_allocator_),
        unique_beta(unique_beta_counts.size(), unique_beta_counts, i_allocator_)
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
        ,reference_dev(1, NAEA + NAEB, dev_i_allocator_, std::allocator<int>{}),
        unique_alpha_dev(unique_alpha_counts.size(), unique_alpha_counts, 
                        dev_i_allocator_,std::allocator<int>{}),
        unique_beta_dev(unique_beta_counts.size(), unique_beta_counts, 
                        dev_i_allocator_,std::allocator<int>{})
#endif
  {
    size_t emax = std::max(unique_alpha_counts.size(), unique_beta_counts.size());
    sum_of_exct.resize(emax + 1);
    sum_of_exct[0] = {0, 0};
    sum_of_exct[1] = {1, 1};
    for (size_t n = 1; n < unique_alpha.size(); ++n)
      sum_of_exct[n + 1][0] = sum_of_exct[n][0] + unique_alpha_counts[n] / size_t(2) / n;
    for (size_t n = unique_alpha.size() + 1; n <= emax; n++)
      sum_of_exct[n][0] = sum_of_exct[n - 1][0];
    for (size_t n = 1; n < unique_beta.size(); ++n)
      sum_of_exct[n + 1][1] = sum_of_exct[n][1] + unique_beta_counts[n] / size_t(2) / n;
    for (size_t n = unique_beta.size() + 1; n <= emax; n++)
      sum_of_exct[n][1] = sum_of_exct[n - 1][1];
  }

  ph_excitations(ph_excitations const& other) = delete;
  //ph_excitations(ph_excitations && other) = default;
  ph_excitations(ph_excitations&& other)
      : i_allocator_(other.i_allocator_),
        c_allocator_(other.c_allocator_),
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
        dev_i_allocator_(other.dev_i_allocator_),
#endif
        NAEA(other.NAEA),
        NAEB(other.NAEB),
        configurations(std::move(other.configurations)),
        reference(std::move(other.reference)),
        unique_alpha(std::move(other.unique_alpha)),
        unique_beta(std::move(other.unique_beta)),
        sum_of_exct(std::move(other.sum_of_exct))
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
        ,reference_dev(reference,dev_i_allocator_,std::allocator<int>{}),
        unique_alpha_dev(unique_alpha,dev_i_allocator_,std::allocator<int>{}),
        unique_beta_dev(unique_beta,dev_i_allocator_,std::allocator<int>{})
#endif
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    dev_init = true;
#endif
  }

  template<class,class,class,class,class>
  friend struct ph_excitations; 

  // limited choice for simplicity, generalize if needed
  template<class devAlloc_,
            typename = std::enable_if_t< not std::is_same<devAlloc,devAlloc_>::value >
          >
  ph_excitations(ph_excitations<I, VType, Alloc, is_root, devAlloc_> const& other, 
                 [[maybe_unused]] devAlloc dev_alloc_ = {}) 
      : i_allocator_(other.i_allocator_),
        c_allocator_(other.c_allocator_),
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
        dev_i_allocator_(dev_alloc_),
#endif
        NAEA(other.NAEA),
        NAEB(other.NAEB),
        configurations(other.configurations),
        reference(other.reference),
        unique_alpha(other.unique_alpha),
        unique_beta(other.unique_beta),
        sum_of_exct(other.sum_of_exct)
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
        ,reference_dev(reference,dev_i_allocator_,std::allocator<int>{}),
        unique_alpha_dev(unique_alpha,dev_i_allocator_,std::allocator<int>{}),
        unique_beta_dev(unique_beta,dev_i_allocator_,std::allocator<int>{})
#endif
  {
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    dev_init = true;
#endif
  }

  ph_excitations& operator=(ph_excitations const& other) = delete;
  ph_excitations& operator=(ph_excitations&& other) = default;

  std::array<size_t, 2> maximum_excitation_number() const { return {unique_alpha.size(), unique_beta.size()}; }
  size_t number_of_unique_alpha_excitations(int n) const
  {
    if (n == 0)
      return 1;
    return unique_alpha.num_elements(n) / 2 / n;
  }
  size_t number_of_unique_beta_excitations(int n) const
  {
    if (n == 0)
      return 1;
    return unique_beta.num_elements(n) / 2 / n;
  }
  std::array<size_t, 2> number_of_unique_excitations(int n) const
  {
    if (n == 0)
      return {1, 1};
    std::array<size_t, 2> res{0, 0};
    if (n < unique_alpha.size())
      res[0] = unique_alpha.num_elements(n) / 2 / n;
    if (n < unique_beta.size())
      res[1] = unique_beta.num_elements(n) / 2 / n;
    return res;
  }
  std::array<size_t, 2> number_of_unique_excitations() const { return sum_of_exct.back(); }

  size_t number_of_configurations() const { return configurations.num_elements(0); }

  // returns the number of unique excitations with particle number less than n
  std::array<size_t, 2> number_of_unique_smaller_than(int n) const { return sum_of_exct[n]; }

  template<class intIt>
  void add_alpha(size_t n, intIt indx)
  {
    RUNTIME_CHECK(n < unique_alpha.size(), "");
    RUNTIME_CHECK(n > 0, "");
    for (int i = 0; i < 2 * n; i++, ++indx)
      unique_alpha.emplace_back(n, static_cast<integer_type>(*indx));
  }

  template<class intIt>
  void add_beta(size_t n, intIt indx)
  {
    RUNTIME_CHECK(n < unique_beta.size(), "");
    RUNTIME_CHECK(n > 0, "");
    for (int i = 0; i < 2 * n; i++, ++indx)
      unique_beta.emplace_back(n, static_cast<integer_type>(*indx));
  }

  template<class Vector>
  void add_reference(Vector& refa, Vector& refb)
  {
    for (auto k : refa)
      reference.emplace_back(0, static_cast<integer_type>(k));
    for (auto k : refb)
      reference.emplace_back(0, static_cast<integer_type>(k));
  }

  // index=0 is reserved for the reference!!!
  template<typename integer, typename value>
  void add_configuration(integer alpha_indx, integer beta_index, value ci)
  {
    configurations.emplace_back(0, configuration_type{alpha_indx, beta_index, ci});
  }

  typename Excitation_Iterator::const_reference reference_configuration(int spin = 0) const
  {
    return raw_pointer_cast(reference.values(0)) + (spin == 0 ? 0 : NAEA);
  }

  typename Excitation_Iterator::reference reference_configuration(int spin = 0)
  {
    return raw_pointer_cast(reference.values(0)) + (spin == 0 ? 0 : NAEA);
  }

  configuration_type const* configurations_begin() const { return raw_pointer_cast(configurations.values(0)); }

  configuration_type const* configurations_end() const
  {
    return raw_pointer_cast(configurations.values()) + (*configurations.pointers_end(0));
  }

  configuration_type const* configuration(int i) const { return raw_pointer_cast(configurations.values()) + i; }

  Excitation_Iterator alpha_begin(int n)
  {
    RUNTIME_CHECK(n > 0, "");
    if (n < unique_alpha.size())
    {
      return Excitation_Iterator(raw_pointer_cast(unique_alpha.values(n)), n);
    }
    else
      return alpha_end(n);
  }

  Excitation_Iterator alpha_end(int n)
  {
    RUNTIME_CHECK(n > 0, "");
    if (n < unique_alpha.size())
      return Excitation_Iterator(raw_pointer_cast(unique_alpha.values()) + (*unique_alpha.pointers_end(n)), n);
    else
      return Excitation_Iterator(raw_pointer_cast(unique_alpha.values()) +
                                     (*unique_alpha.pointers_end(unique_alpha.size() - 1)),
                                 1);
  }

  Excitation_const_Iterator alpha_begin(int n) const
  {
    RUNTIME_CHECK(n > 0, "");
    if (n < unique_alpha.size())
    {
      return Excitation_const_Iterator(raw_pointer_cast(unique_alpha.values(n)), n);
    }
    else
      return alpha_end(n);
  }

  Excitation_const_Iterator alpha_end(int n) const
  {
    RUNTIME_CHECK(n > 0, "");
    if (n < unique_alpha.size())
      return Excitation_const_Iterator(raw_pointer_cast(unique_alpha.values()) + (*unique_alpha.pointers_end(n)), n);
    else
      return Excitation_const_Iterator(raw_pointer_cast(unique_alpha.values()) +
                                           (*unique_alpha.pointers_end(unique_alpha.size() - 1)),
                                       1);
  }

  Excitation_Iterator beta_begin(int n)
  {
    RUNTIME_CHECK(n > 0, "");
    if (n < unique_beta.size())
      return Excitation_Iterator(raw_pointer_cast(unique_beta.values(n)), n);
    else
      return beta_end(n);
  }

  Excitation_Iterator beta_end(int n)
  {
    RUNTIME_CHECK(n > 0, "");
    if (n < unique_beta.size())
      return Excitation_Iterator(raw_pointer_cast(unique_beta.values()) + (*unique_beta.pointers_end(n)), n);
    else
      return Excitation_Iterator(raw_pointer_cast(unique_beta.values()) + (*unique_beta.pointers_end(unique_beta.size() - 1)),
                                 1);
  }

  Excitation_const_Iterator beta_begin(int n) const
  {
    RUNTIME_CHECK(n > 0, "");
    if (n < unique_beta.size())
      return Excitation_const_Iterator(raw_pointer_cast(unique_beta.values(n)), n);
    else
      return beta_end(n);
  }

  Excitation_const_Iterator beta_end(int n) const
  {
    RUNTIME_CHECK(n > 0, "");
    if (n < unique_beta.size())
      return Excitation_const_Iterator(raw_pointer_cast(unique_beta.values()) + (*unique_beta.pointers_end(n)), n);
    else
      return Excitation_const_Iterator(raw_pointer_cast(unique_beta.values()) +
                                           (*unique_beta.pointers_end(unique_beta.size() - 1)),
                                       1);
  }

  // for generic access
  std::array<Excitation_Iterator, 2> unique_begin(int n)
  {
    return std::array<Excitation_Iterator, 2>{alpha_begin(n), beta_begin(n)};
  }
  std::array<Excitation_Iterator, 2> unique_end(int n)
  {
    return std::array<Excitation_Iterator, 2>{alpha_end(n), beta_end(n)};
  }
  std::array<Excitation_const_Iterator, 2> unique_begin(int n) const
  {
    return std::array<Excitation_const_Iterator, 2>{alpha_begin(n), beta_begin(n)};
  }
  std::array<Excitation_const_Iterator, 2> unique_end(int n) const
  {
    return std::array<Excitation_const_Iterator, 2>{alpha_end(n), beta_end(n)};
  }

  template<class Vector>
  void get_alpha_configuration(size_t index, Vector& confg) const
  {
    RUNTIME_CHECK(confg.size() >= NAEA, "");
    std::copy_n(raw_pointer_cast(reference.values(0)), NAEA, confg.data());
    if (index == 0)
      return;
    // could use lower bound
    for (int i = 1; i < unique_alpha.size(); i++)
    {
      if (index >= sum_of_exct[i][0] && index < sum_of_exct[i + 1][0])
      {
        size_t dn = index - sum_of_exct[i][0];
        auto exct = unique_alpha.values(i) + 2 * i * dn;
        for (int n = 0; n < i; n++)
          confg[exct[n]] = exct[n + i];
        return;
      }
    }
    APP_ABORT(" Error in ph_excitations::get_alpha_configuration() ");
  }

  template<class Vector>
  void get_beta_configuration(size_t index, Vector& confg) const
  {
    RUNTIME_CHECK(confg.size() >= NAEB, "");
    std::copy_n(raw_pointer_cast(reference.values(0)) + NAEA, NAEB, confg.data());
    if (index == 0)
      return;
    // could use lower bound
    for (int i = 1; i < unique_beta.size(); i++)
    {
      if (index >= sum_of_exct[i][1] && index < sum_of_exct[i + 1][1])
      {
        size_t dn = index - sum_of_exct[i][1];
        auto exct = unique_beta.values(i) + 2 * i * dn;
        for (int n = 0; n < i; n++)
          confg[exct[n]] = exct[n + i];
        return;
      }
    }
    APP_ABORT(" Error in ph_excitations::get_beta_configuration() ");
  }

  template<class Vector>
  void get_configuration(int spin, size_t index, Vector& confg) const
  {
    if (spin == 0)
      get_alpha_configuration(index, confg);
    else
      get_beta_configuration(index, confg);
  }

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  auto get_excitation_list_device(int ispin, int iex) 
  {
    if(not dev_init) {
      dev_init = true;
      reference_dev = reference;
      unique_alpha_dev = unique_alpha;
      unique_beta_dev = unique_beta;  
    }
    if(ispin == 0) {
      RUNTIME_CHECK(iex >= 0 and iex < unique_alpha_dev.size(), "");
      return unique_alpha_dev.values(iex);
    } else {
      RUNTIME_CHECK(iex >= 0 and iex < unique_beta_dev.size(), "");
      return unique_beta_dev.values(iex);
    }
  }

  auto get_reference_configuration_device(int ispin) 
  {
    if(not dev_init) {
      dev_init = true;
      reference_dev = reference;
      unique_alpha_dev = unique_alpha;   
      unique_beta_dev = unique_beta;    
    }
    return reference_dev.values(0) + (ispin == 0 ? 0 : NAEA);
  }
#else
  auto get_excitation_list_device(int ispin, int iex)
  {
    if(ispin == 0) {
      RUNTIME_CHECK(iex >= 0 and iex < unique_alpha.size(), "");
      return raw_pointer_cast(unique_alpha.values(iex));
    } else {
      RUNTIME_CHECK(iex >= 0 and iex < unique_beta.size(), "");
      return raw_pointer_cast(unique_beta.values(iex));
    }
  }

  auto get_reference_configuration_device(int ispin)
  {
    return raw_pointer_cast(reference.values(0)) + (ispin == 0 ? 0 : NAEA);
  }
#endif
  

private:
  // using array_of_seq until I switch to Boost.Multi to be able to use shared_allocator
  int NAEA, NAEB;
  confg_aos configurations;
  index_aos reference;
  index_aos unique_alpha;
  index_aos unique_beta;
  std::vector<std::array<size_t, 2>> sum_of_exct;
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  bool dev_init = false;
  dev_index_aos reference_dev;
  dev_index_aos unique_alpha_dev;
  dev_index_aos unique_beta_dev;
#endif
};

} // namespace afqmc

} // namespace sfqmc

#endif
