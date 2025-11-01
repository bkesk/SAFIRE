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
#include <cstdint>

namespace nda::mem {

  namespace detail 
  {
    inline char* align_up(char* ptr, std::size_t align = alignof(std::max_align_t)) {
      static_assert(sizeof(std::uint64_t) == sizeof(char*) or sizeof(std::uint32_t) == sizeof(char*),
                    "Routine only works in 32bit and 64bit systems.");
      if constexpr (sizeof(std::uint32_t) == sizeof(char*)) {
          std::uint32_t align_32(align);
          return reinterpret_cast<char*>( align_32 * ((reinterpret_cast<std::uint32_t&>(ptr) + (align_32 - 1)) / align_32) );
      } else if constexpr (sizeof(std::uint64_t) == sizeof(char*)) {
          return reinterpret_cast<char*>( align * ((reinterpret_cast<std::uint64_t&>(ptr) + (align - 1)) / align) );
      }
      return ptr;
    };
  }

}

