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
#include "utilities/nda_addons/global_bucket.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
//#include <cstdlib>
//#include <memory>
//#include <utility>

namespace nda::mem {

  /**
   * @brief Attempts to allocate from a primary allocator. Falls back to mallocator 
   *        if the primary one fails. 
   * @tparam AdrSp nda::mem::AddressSpace in which the memory is allocated.
   */
  template <Allocator Primary> 
  requires requires(Primary Alloc)
           { { Alloc.deallocate(std::declval<blk_t>(),std::declval<bool&>()) } noexcept -> std::same_as<void>; }
  class fallback {

    private:

    Primary alloc;

    /// "fallback" allocator
    using Secondary = mallocator<Primary::address_space>;

    public:
    /// Default constructor.
    fallback() = default; 

    template<typename... Args>
    fallback(Args&&... args) : alloc(std::forward<Args>(args)...) {} 

    /// Deleted copy constructor.
    fallback(fallback const &) = delete; 

    /// Default move constructor.
    fallback(fallback &&) = default;

    /// Deleted copy assignment operator.
    fallback &operator=(fallback const &) = delete;

    /// Default move assignment operator.
    fallback &operator=(fallback &&) = default;

    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = Primary::address_space;

    /// Return pointer to primary allocator
    auto get_primary() { return std::addressof(alloc); }
    auto get_primary() const { return std::addressof(alloc); }

    /**
     * @brief Allocate memory from the pool. Throws exception is available memory is not sufficient. 
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate(size_t s) noexcept { 
      blk_t b = alloc.allocate(s);  
      if (b.ptr) return b;
      return Secondary::allocate(s); 
    }

    /**
     * @brief Allocate memory and set it to zero.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate_zero(size_t s) noexcept {
      blk_t b = this->allocate(s); 
      if(b.ptr and b.s > 0) memset<address_space>(b.ptr, 0, b.s);
      return b;
    }

    /**
     * @brief Deallocate memory using nda::mem::free.
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    void deallocate(blk_t b) noexcept {
      bool owns = false;
      alloc.deallocate(b,owns);
      if( not owns)
        return Secondary::deallocate(b);
    } 
  };

  /**
   * @brief Attempts to allocate from a primary allocator. Falls back to mallocator 
   *        if the primary one fails. 
   * @tparam AdrSp nda::mem::AddressSpace in which the memory is allocated.
   */
  template <AddressSpace AdrSp> 
  class global_fallback {

    private:

    /// "fallback" allocator
    using Secondary = mallocator<AdrSp>;

    public:
    /// Default constructor.
    global_fallback() = default; 

    /// Deleted copy constructor.
    global_fallback(global_fallback const &) = delete; 

    /// Default move constructor.
    global_fallback(global_fallback &&) = default;

    /// Deleted copy assignment operator.
    global_fallback &operator=(global_fallback const &) = delete;

    /// Default move assignment operator.
    global_fallback &operator=(global_fallback &&) = default;

    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = AdrSp; 

    /**
     * @brief Allocate memory from the pool. Throws exception is available memory is not sufficient. 
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate(size_t s) noexcept { 
#if defined(ENABLE_DEVICE)
      auto& alloc = std::conditional_t<AdrSp==Host,   detail::get_global_host_bucket(),
                    std::conditional_t<AdrSp==Device, detail::get_global_device_bucket(), 
                                                      detail::get_global_device_bucket()>>; 
#else
      auto& alloc = detail::get_global_host_bucket();
#endif
      blk_t b = alloc.allocate(s);  
      if (b.ptr) return b;
      return Secondary::allocate(s); 
    }

    /**
     * @brief Allocate memory and set it to zero.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate_zero(size_t s) noexcept {
      blk_t b = this->allocate(s); 
      if(b.ptr and b.s > 0) memset<address_space>(b.ptr, 0, b.s);
      return b;
    }

    /**
     * @brief Deallocate memory using nda::mem::free.
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    void deallocate(blk_t b) noexcept {
#if defined(ENABLE_DEVICE)
      auto& alloc = std::conditional_t<AdrSp==Host,   detail::get_global_host_bucket(),
                    std::conditional_t<AdrSp==Device, detail::get_global_device_bucket(),
                                                      detail::get_global_device_bucket()>>;
#else
      auto& alloc = detail::get_global_host_bucket();
#endif
      bool owns = false;
      alloc.deallocate(b,owns);
      if( not owns)
        return Secondary::deallocate(b);
    } 
  };

} // namespace nda::mem

