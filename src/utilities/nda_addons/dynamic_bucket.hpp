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
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "nda/nda.hpp"
#include "utilities/nda_addons/align_up.hpp"

#include <algorithm>
//#include <cstddef>
//#include <cstdint>
//#include <cstdlib>
#include <vector>
#include <cstddef>

namespace nda::mem {

  /**
   * @brief Allocates memory from a pool of memory. 
   *        The size of the pool is provided at runtime and can be dynamically adjusted.
   *        If the allocator does not have enough contiguous memory to allocate a requested amount,
   *        the call throws an exception. 
   * @tparam AdrSp nda::mem::AddressSpace in which the memory is allocated.
   */
  template <AddressSpace AdrSp = Host>
  class dynamic_bucket {

    private:

    // auxiliary allocator
    using Auxiliary = mallocator<AdrSp>;

    // size of the pool
    size_t _size = 0; 

    /// alignment
    size_t _align = alignof(std::max_align_t);

    /// maximum amount of memory needed 
    size_t _maximum_needed = 0;

    /// total amount of memory requested
    size_t _total_requested = 0;

    /// total amount of memory released
    size_t _total_released = 0;

    // pool of memory
    blk_t _pool;

    // aligned start of _pool
    char* p0 = nullptr;

    // list of available memory segments
    std::vector<blk_t> _avail;

    // list of allocated memory segments
    std::vector<blk_t> _segments;

    public:
    /// Default constructor.
    dynamic_bucket(size_t s = 0, size_t align = alignof(std::max_align_t)) : 
      _size(s), _align{align}, _pool{Auxiliary::allocate(_size)} {
      if(_size==0) return;
      if(_size < 2*_align) {
        Auxiliary::deallocate(_pool);
        _pool = Auxiliary::allocate(2*_size);
      }
      // first alligned memory location
      p0 = detail::align_up(_pool.ptr, _align);
      _avail.reserve(10);   
      _segments.reserve(10);
      // initialize _avail
      _avail.emplace_back(blk_t{p0,size_t(std::distance(p0,_pool.ptr+_pool.s))});
    } 

    /// Destructor
    ~dynamic_bucket() { Auxiliary::deallocate(_pool); }

    /// Deleted copy constructor.
    dynamic_bucket(dynamic_bucket const &) = delete; 

    /// Default move constructor.
    dynamic_bucket(dynamic_bucket &&) = default;

    /// Deleted copy assignment operator.
    dynamic_bucket &operator=(dynamic_bucket const &) = delete;

    /// Default move assignment operator.
    dynamic_bucket &operator=(dynamic_bucket &&) = default;

    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = AdrSp;

    /**
     * @ brief Changes the size of the memory pool in the allocator.
     *         Only allowed if no memory is currently allocated. 
     */
    void resize(size_t s) {
      if(_segments.size() > 0) throw std::bad_alloc{};
      _size = (s >= 2*_align? s : 2*_align);
      Auxiliary::deallocate(_pool);
      _pool = Auxiliary::allocate(_size);
      p0 = detail::align_up(_pool.ptr, _align);      
      _avail.clear();
      _avail.emplace_back(blk_t{p0,size_t(std::distance(p0,_pool.ptr+_pool.s))});
    }

    /**
     *  Returns the capacity of the allocator.
     */
    auto size() const { return _size; }

    /**
     * @ brief Returns the maximum amount of memory requested
     */
    auto maximum_memory() const { return _maximum_needed; }

    /**
     * @brief Allocate memory from the pool. Returns nullptr if allocation fails. 
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate(size_t s) noexcept { 
      // round up to closest multiple of align 
      size_t aligned_s = (( s + (_align - 1) ) / _align ) * _align;
      _total_requested += aligned_s;
      _maximum_needed = std::max(_maximum_needed, _total_requested-_total_released);
      // find available block with enough space 
      auto b = std::find_if(_avail.begin(), _avail.end(), [&](auto const& a) { return a.s >= aligned_s; });
      if( b != _avail.end()) {
        auto p_s = b->ptr;
        // add segment to list, keeping list unsorted 
        _segments.push_back( {p_s,aligned_s} );
        // remove segment from _avail
        if( aligned_s == b->s ) 
          _avail.erase(b);
        else {
          b->ptr += aligned_s;
          b->s -= aligned_s; 
        }
        return {p_s, s};
      } 
      return {nullptr,0};
    }

    /**
     * @brief Allocate memory and set it to zero.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate_zero(size_t s) noexcept {
      blk_t b = allocate(s);
      if(b.ptr and b.s > 0) memset<address_space>(b.ptr, 0, b.s);
      return b;
    }

    /**
     * @ brief Deallocate memory.
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    void deallocate(blk_t b, bool &owns) noexcept { 
      if(_segments.size() == 0 or _size==0) {
        _total_released += b.s;
        owns = false;
        return;
      }
      // 1. find blk in _segments 
      auto it = std::find_if(_segments.begin(), _segments.end(), 
                             [&](auto const& a) { return std::distance(a.ptr,b.ptr)==0; });
      if( it != _segments.end() ) { 
        _total_released += it->s;
        move_blk_to_avail(*it);
        _segments.erase(it);
        owns = true;
        return;
      }
      // 2. Deallocates owned memory.
      EXPECTS_WITH_MESSAGE( (std::distance(b.ptr,p0) > 0) or 
                            (std::distance(_pool.ptr+_pool.s,b.ptr) > 0),
                            "Error in nda::mem::dynamic_bucket::deallocate: Deallocating memory within the pool of the allocator, yet not registered as a segment.")  
      // signal that memory is not owned by this allocator
      _total_released += b.s; 
      owns = false;
    }

    /**
     * @ brief Deallocate memory.
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    void deallocate(blk_t b) noexcept { 
      bool owns;
      deallocate(b,owns);
    }

    private:

    /**
     *  Adds a block of memory to _avail.
     *  _avail is kep sorted based on memory location.
     */
    void move_blk_to_avail(blk_t b) {
      if(_avail.size() == 0) {
        _avail.emplace_back(b);
        return;
      }
      // 0. find location of b.ptr in list
      auto it = std::lower_bound(_avail.begin(),_avail.end(),b,
                                [&](auto&& i, auto&& j) {return std::distance(i.ptr,j.ptr)>0;} ); 
      // add in front
      if( it == _avail.begin() ) {
        auto it_beg = _avail.begin();
        if( std::distance( b.ptr+b.s, it_beg->ptr ) == 0 ) {
          //merge
          it_beg->ptr = b.ptr;
          it_beg->s += b.s;
        } else {
          // add
          _avail.insert(it_beg,b);
        } 
        return;
      }
      // add in back
      if( it == _avail.end() ) {
        auto it_last = _avail.end() - 1;
        if( std::distance( it_last->ptr+it_last->s, b.ptr ) == 0 ) {
          //merge
          it_last->s += b.s;
        } else {
          // add
          _avail.emplace_back(b);
        }
        return;
      }
      // General scenario: 4 cases to consider
      auto prev = it-1;
      auto d_prev = std::distance(prev->ptr+prev->s, b.ptr);
      auto d_next = std::distance(b.ptr+b.s,it->ptr);
      if( (d_prev==0) and (d_next==0) ) {
        // a. b is contiguous with previous and next element in _avail: merge into a single segment
        prev->s += b.s + it->s;
        _avail.erase(it);
      } else if(d_prev==0) {
        // b. b is contiguous with previous element, not with next: merge with previous
        prev->s += b.s;
      } else if(d_next==0) {
        // c. b is contiguous with next element, not with previous: merge with next 
        it->ptr = b.ptr;
        it->s += b.s;
      } else {
        // d. b is completely isolated: add new segment
        _avail.insert(it,b);
      }
    }

  };

}

