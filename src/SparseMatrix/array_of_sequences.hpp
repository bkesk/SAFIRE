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

/*
 * Implements a vector of sequences of diferent sizes.
 * Designed derived from ucsr_matrix. Essentually similar to ucsr_matrix, but
 * without a column index.
 */
#ifndef ARRAY_OF_SEQUENCES_HPP
#define ARRAY_OF_SEQUENCES_HPP

#include <array>
#include <cassert>
#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
#include <type_traits> // enable_if
#include <algorithm>
#include <utility>
#include <tuple>

namespace ma
{
namespace sparse
{
// also in csr_matrix, move somewhere else
// doing this to avoid needing to include csr_matrix just for this
template<class Allocator>
struct null_is_root_
{
  null_is_root_(Allocator) {}
  bool root() { return true; }
  int size() { return 1; }
  int rank() { return 0; }
  void barrier(){};
};

using size_type = std::size_t;

template<class ValTypePtr, class IntTypePtr = size_type*>
class array_of_sequences_ref
{
public:
  using value_type = decltype(*std::declval<ValTypePtr>());
  using int_type   = decltype(*std::declval<IntTypePtr>());

protected:
  using this_t = array_of_sequences_ref<ValTypePtr, IntTypePtr>;
  size_type size1_;
  size_type capacity_;
  ValTypePtr data_;
  IntTypePtr pointers_begin_;
  IntTypePtr pointers_end_;
  // set object to null state
  void reset()
  {
    size1_          = 0;
    capacity_       = 0;
    data_           = ValTypePtr(nullptr);
    pointers_begin_ = IntTypePtr(nullptr);
    pointers_end_   = IntTypePtr(nullptr);
  }

public:
  array_of_sequences_ref(size_type arr,
                         size_type cap_,
                         ValTypePtr __data_,
                         IntTypePtr __pointers_begin_,
                         IntTypePtr __pointers_end_)
      : size1_(arr), capacity_(cap_), data_(__data_), pointers_begin_(__pointers_begin_), pointers_end_(__pointers_end_)
  {}
  array_of_sequences_ref(this_t const& other) = delete;
  array_of_sequences_ref& operator=(this_t const& other) = delete;
  // pointer movement is handled by derived classes
  array_of_sequences_ref(this_t&& other) = default; //:array_of_sequences_ref() { }
  array_of_sequences_ref& operator=(this_t&& other) = default;
  ~array_of_sequences_ref() {}
  auto pointers_begin(size_type i = 0) const { return pointers_begin_ + i; }
  auto pointers_end(size_type i = 0) const { return pointers_end_ + i; }
  auto size() const { return size1_; }
  template<typename integer_type = size_type>
  auto capacity(integer_type i) const
  {
    if (not pointers_begin_)
      return size_type(0);
    return static_cast<size_type>(pointers_begin_[i + 1] - pointers_begin_[i]);
  }
  auto capacity() const
  {
    if (not pointers_begin_)
      return size_type(0);
    return capacity_;
  }
  auto num_elements() const
  {
    size_type ret = 0;
    for (size_type i = 0; i != size(); ++i)
      ret += static_cast<size_type>(pointers_end_[i] - pointers_begin_[i]);
    return ret;
  }
  template<typename integer_type = size_type>
  auto num_elements(integer_type i) const
  {
    RUNTIME_CHECK(i >= 0 && i < size1_, "");
    return static_cast<size_type>(pointers_end_[i] - pointers_begin_[i]);
  }
  auto values(size_type i = 0) const { return data_ + pointers_begin_[i]; }
  auto release_capacity()
  {
    auto t    = capacity_;
    capacity_ = size_type(0);
    return t;
  }
  auto release_values()
  {
    auto t = data_;
    data_  = ValTypePtr(nullptr);
    return t;
  }
  auto release_pointer_begin()
  {
    auto t          = pointers_begin_;
    pointers_begin_ = IntTypePtr(nullptr);
    return t;
  }
  auto release_pointer_end()
  {
    auto t        = pointers_end_;
    pointers_end_ = IntTypePtr(nullptr);
    return t;
  }
  // ??? not sure how to do this better
  auto release_size()
  {
    auto t = size1_;
    size1_ = size_type(0);
    return t;
  }
  friend decltype(auto) size(this_t const& s) { return s.size(); }
  friend auto capacity(this_t const& s) { return s.capacity(); }
  friend auto num_elements(this_t const& s) { return s.num_elements(); }
  friend auto values(this_t const& s) { return s.values(); }
  friend auto pointers_begin(this_t const& s) { return s.pointers_begin(); }
  friend auto pointers_end(this_t const& s) { return s.pointers_end(); }
};

template<class ValType,
         class IntType       = size_type,
         class ValType_alloc = std::allocator<ValType>,
         class IsRoot        = null_is_root_<ValType_alloc>,
         class IntType_alloc = typename std::allocator_traits<ValType_alloc>::template rebind_alloc<IntType>>
class array_of_sequences
    : public array_of_sequences_ref<typename std::allocator_traits<ValType_alloc>::pointer, typename std::allocator_traits<IntType_alloc>::pointer>
{
public:
  using value_type = ValType;
  using int_type   = IntType;
  using alloc_type = ValType_alloc;

protected:
  using this_t     = array_of_sequences<ValType, IntType, ValType_alloc, IsRoot, IntType_alloc>;
  using ValTypePtr = typename std::allocator_traits<ValType_alloc>::pointer;
  using IntTypePtr = typename std::allocator_traits<IntType_alloc>::pointer;
  using Valloc_ts  = std::allocator_traits<ValType_alloc>;
  using Palloc_ts  = std::allocator_traits<IntType_alloc>;
  using base       = array_of_sequences_ref<ValTypePtr, IntTypePtr>;
  ValType_alloc Valloc_;
  IntType_alloc Palloc_;
  void reset()
  {
    IsRoot r(Valloc_);
    // calling barrier for safety right now
    r.barrier();
    size_type tot_sz = base::capacity();
    r.barrier();
    /**************************************************/
    // if value_type is not a POD, you need to destroy!
    /**************************************************/
    if (base::data_)
      Valloc_.deallocate(base::data_, tot_sz);
    if (base::pointers_begin_)
      Palloc_.deallocate(base::pointers_begin_, base::size1_ + 1);
    if (base::pointers_end_)
      Palloc_.deallocate(base::pointers_end_, base::size1_);
    base::reset();
    r.barrier();
  }

public:
  template<typename integer_type = size_type>
  array_of_sequences(size_type arr, integer_type nnzpr_unique, 
                     ValType_alloc alloc, IntType_alloc ialloc)
      : array_of_sequences_ref<ValTypePtr, IntTypePtr>(arr,
                                                       arr * nnzpr_unique,
                                                       ValTypePtr(nullptr),
                                                       IntTypePtr(nullptr),
                                                       IntTypePtr(nullptr)),
        Valloc_(alloc),
        Palloc_(ialloc)
  {
    base::data_           = Valloc_.allocate(arr * nnzpr_unique);
    base::pointers_begin_ = Palloc_.allocate(arr + 1);
    base::pointers_end_   = Palloc_.allocate(arr);

    IsRoot r(Valloc_);
    if (nnzpr_unique > 0)
    {
      if (r.root())
      {
        auto pb{raw_pointer_cast(base::pointers_begin_)};
        auto pe{raw_pointer_cast(base::pointers_end_)};
        for (size_type i = 0; i != base::size1_; ++i)
        {
          *(pb + i) = i * nnzpr_unique;
          *(pe + i) = i * nnzpr_unique;
        }
        *(pb + base::size1_) = base::size1_ * nnzpr_unique;
      }
    }
    else
    {
      using std::fill_n;
      fill_n(base::pointers_begin_, base::size1_ + 1, int_type(0));
      fill_n(base::pointers_end_, base::size1_, int_type(0));
    }
    r.barrier();
  }
  template<typename integer_type = size_type>
  array_of_sequences(size_type arr, integer_type nnzpr_unique = 0,
                     ValType_alloc alloc = ValType_alloc{}) : 
                               array_of_sequences(arr, nnzpr_unique, alloc, IntType_alloc(alloc))
  {}  
  template<typename integer_type = size_type>
  array_of_sequences(size_type arr,
                     std::vector<integer_type> const& nnzpr,
                     ValType_alloc alloc, IntType_alloc ialloc)
      : array_of_sequences_ref<ValTypePtr, IntTypePtr>(arr,
                                                       0,
                                                       ValTypePtr(nullptr),
                                                       IntTypePtr(nullptr),
                                                       IntTypePtr(nullptr)),
        Valloc_(alloc),
        Palloc_(ialloc)
  {
    size_type sz = size_type(std::accumulate(nnzpr.begin(), nnzpr.end(), integer_type(0)));
    if (arr == 0)
      sz = 0; // no rows, no capacity
    base::capacity_       = sz;
    base::data_           = Valloc_.allocate(sz);
    base::pointers_begin_ = Palloc_.allocate(arr + 1);
    base::pointers_end_   = Palloc_.allocate(arr);

    RUNTIME_CHECK(nnzpr.size() >= base::size1_, "");
    IsRoot r(Valloc_);
    if(sz > 0)
    {
      if (r.root())
      {   
        IntType cnter(0);
        auto pb{raw_pointer_cast(base::pointers_begin_)};
        auto pe{raw_pointer_cast(base::pointers_end_)};
        for (size_type i = 0; i != base::size1_; ++i)
        {
          *(pb + i) = cnter;
          *(pe + i) = cnter;
          cnter += static_cast<IntType>(nnzpr[i]);
        }
        *(pb + base::size1_) = cnter;
      }
    }
    else
    {
      using std::fill_n;
      fill_n(base::pointers_begin_, base::size1_ + 1, int_type(0));
      fill_n(base::pointers_end_, base::size1_, int_type(0));
    }
    r.barrier();
  }
  template<typename integer_type = size_type>
  array_of_sequences(size_type arr,
                     std::vector<integer_type> const& nnzpr = std::vector<integer_type>(0),
                     ValType_alloc alloc                    = ValType_alloc{}) :
                            array_of_sequences(arr, nnzpr, alloc, IntType_alloc(alloc))
  {}  
  ~array_of_sequences() { reset(); }
  array_of_sequences(const this_t& other) : array_of_sequences(0, 0, other.Valloc_) 
        { *this = other; }  
  array_of_sequences(this_t&& other) : array_of_sequences(0, 0, other.Valloc_) 
        { *this = std::move(other); }
  template<class ValType_alloc_,
           class IsRoot_,
           class IntType_alloc_,
           typename = std::enable_if_t< (not std::is_same<ValType_alloc_, ValType_alloc>::value) or
                                        (not std::is_same<IntType_alloc_, IntType_alloc>::value) >
          >
  array_of_sequences(array_of_sequences<ValType,IntType,ValType_alloc_,IsRoot_,IntType_alloc_> const& other,    
            ValType_alloc alloc_ = ValType_alloc{}, 
            IntType_alloc ialloc_ = IntType_alloc{}) : array_of_sequences(0, 0, alloc_, ialloc_)
        { *this = other; }
  array_of_sequences& operator=(const this_t& other) 
  { 
    if (this != std::addressof(other))
    { 
      reset();
      base::size1_          = other.size1_;
      base::capacity_       = other.capacity_;
      base::data_           = Valloc_.allocate(base::capacity_);
      base::pointers_begin_ = Palloc_.allocate(base::size1_ + 1);
      base::pointers_end_   = Palloc_.allocate(base::size1_);
      using std::copy_n;
      copy_n(other.data_, base::capacity_, base::data_);
      copy_n(other.pointers_begin_, base::size1_ + 1, base::pointers_begin_);
      copy_n(other.pointers_end_, base::size1_, base::pointers_end_);
    }
    return *this;
  }
  template<class ValType_alloc_,
           class IsRoot_, 
           class IntType_alloc_,
           typename = std::enable_if_t< (not std::is_same<ValType_alloc_, ValType_alloc>::value) or
                                        (not std::is_same<IntType_alloc_, IntType_alloc>::value) >
          >
  array_of_sequences& operator=(array_of_sequences<ValType,IntType,ValType_alloc_,IsRoot_,IntType_alloc_> const& other)
  {
    reset();
    base::size1_          = other.size();
    base::capacity_       = other.capacity();
    base::data_           = Valloc_.allocate(base::capacity_);
    base::pointers_begin_ = Palloc_.allocate(base::size1_ + 1);
    base::pointers_end_   = Palloc_.allocate(base::size1_);
    using std::copy_n;
    // can't copy from GPU to CPU for now, FIX
    copy_n(raw_pointer_cast(other.values()), base::capacity_, base::data_);
    copy_n(raw_pointer_cast(other.pointers_begin()), base::size1_ + 1, base::pointers_begin_);
    copy_n(raw_pointer_cast(other.pointers_end()), base::size1_, base::pointers_end_);
    return *this;
  }
  // Instead of moving allocators, require they are the same right now
  array_of_sequences& operator=(this_t&& other)
  {
    if (this != std::addressof(other))
    {
      if (Valloc_ != other.Valloc_ || Palloc_ != other.Palloc_)
        APP_ABORT(" Error: Can only move assign between ucsr_matrices with equivalent allocators. ");
      reset();
      base::size1_          = std::exchange(other.size1_, size_type(0));
      base::capacity_       = std::exchange(other.capacity_, size_type(0));
      base::data_           = std::exchange(other.data_, ValTypePtr(nullptr));
      base::pointers_begin_ = std::exchange(other.pointers_begin_, IntTypePtr(nullptr));
      base::pointers_end_   = std::exchange(other.pointers_end_, IntTypePtr(nullptr));
    }
    return *this;
  }
  template<typename integer_type = size_type>
  void reserve(integer_type nnzpr_unique)
  {
    if (base::size1_ == 0)
      return;
    IntType minN = IntType(base::pointers_begin_[1] - base::pointers_begin_[0]);
    for (size_type i = 0; i != base::size1_; ++i)
      minN = std::min(minN, base::pointers_begin_[i + 1] - base::pointers_begin_[i]);
    if (static_cast<IntType>(nnzpr_unique) <= minN)
      return;
    this_t other(base::size1_, nnzpr_unique, Valloc_);
    if (base::capacity_ > 0)
    {
      IsRoot r(Valloc_);
      if (r.root())
      {
        for (size_type i = 0; i < base::size1_; i++)
        {
          size_type disp = static_cast<size_type>(base::pointers_end_[i] - base::pointers_begin_[i]);
          using std::copy_n;
          copy_n(std::addressof(base::data_[base::pointers_begin_[i]]), disp,
                      std::addressof(other.data_[other.pointers_begin_[i]]));
          other.pointers_end_[i] = other.pointers_begin_[i] + disp;
        }
      }
      r.barrier();
    }
    *this = std::move(other);
  }
  template<typename integer_type>
  void reserve(std::vector<integer_type> const& nnzpr)
  {
    if (base::size1_ == 0)
      return;
    bool resz = false;
    RUNTIME_CHECK(nnzpr.size() >= base::size1_, "");
    for (size_type i = 0; i < base::size1_; i++)
      if (static_cast<IntType>(nnzpr[i]) > base::pointers_begin_[i + 1] - base::pointers_begin_[i])
      {
        resz = true;
        break;
      }
    if (not resz)
      return;
    this_t other(base::size1_, nnzpr, Valloc_);
    if (base::capacity_ > 0)
    {
      IsRoot r(Valloc_);
      if (r.root())
      {
        for (size_type i = 0; i < base::size1_; i++)
        {
          size_type disp = static_cast<size_type>(base::pointers_end_[i] - base::pointers_begin_[i]);
          using std::copy_n;
          copy_n(std::addressof(base::data_[base::pointers_begin_[i]]), disp,
                      std::addressof(other.data_[other.pointers_begin_[i]]));
          other.pointers_end_[i] = other.pointers_begin_[i] + disp;
        }
      }
      r.barrier();
    }
    *this = std::move(other);
  }
  template<typename integer_type = IntType, class... Args>
  void emplace_back(integer_type index, Args&&... args)
  {
    using std::get;
    RUNTIME_CHECK(index >= 0, "");
    RUNTIME_CHECK(index < base::size1_, "");
    if (base::pointers_end_[index] < base::pointers_begin_[index + 1])
    {
      ::new (static_cast<void*>(std::addressof(base::data_[base::pointers_end_[index]])))
          value_type(std::forward<Args>(args)...);
      ++base::pointers_end_[index];
    }
    else
      throw std::out_of_range("row size exceeded the maximum");
  }
  template<typename integer_type = IntType, typename value_type = ValType>
  void emplace_back(std::tuple<integer_type, value_type> const& val)
  {
    using std::get;
    emplace_back(get<0>(val), get<1>(val));
  }

public:
  ValType_alloc get_allocator() const { return Valloc_; }
  ValType_alloc getValloc() const { return Valloc_; }
  IntType_alloc getPalloc() const { return Palloc_; }
};

} // namespace sparse
} // namespace ma

#endif
