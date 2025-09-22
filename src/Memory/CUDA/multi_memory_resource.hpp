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

#ifndef MULTI_CUDA_MEMORY_RESOURCE_HPP
#define MULTI_CUDA_MEMORY_RESOURCE_HPP

#include "multi/memory/adaptors/cuda/allocator.hpp"
#include "multi/memory/adaptors/cuda/ptr.hpp"

namespace qmc_cuda 
{

class resource {
public:
        using pointer = boost::multi::memory::cuda::ptr<void>;
        using size_type = ::size_t;
        pointer allocate(std::size_t n, std::size_t alignment = alignof(std::max_align_t)) {
                if(n == 0) return pointer{nullptr};
                auto ret = static_cast<pointer>(boost::multi::memory::cuda::malloc(n));
                if(not ret) throw std::bad_alloc{};
                return ret;
        }
        void deallocate(pointer p, std::size_t n = alignof(std::max_align_t)){
                boost::multi::memory::cuda::free(p);
        }
        std::true_type operator==(resource const&) const{return {};}
        std::false_type operator!=(resource const&) const{return {};}
};

}

#endif
