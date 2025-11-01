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
        
#include "nda/nda.hpp"
#include "utilities/nda_addons/dynamic_bucket.hpp"

#define DEFAULT_BUCKET_SIZE 100000000
        
namespace nda::mem::detail 
{

  nda::mem::dynamic_bucket<nda::mem::Host>& get_global_host_bucket() {
    struct host_bucket_t {
      host_bucket_t() : alloc(DEFAULT_BUCKET_SIZE) {}
      ~host_bucket_t() = default;
      nda::mem::dynamic_bucket<nda::mem::Host> alloc;
    };
    static host_bucket_t b = {};
    return b.alloc;
  };

#if defined(ENABLE_DEVICE)
  nda::mem::dynamic_bucket<nda::mem::Device>& get_global_device_bucket() {
    struct device_bucket_t {
      device_bucket_t() : alloc(DEFAULT_BUCKET_SIZE) {}
      ~device_bucket_t() = default;
      nda::mem::dynamic_bucket<nda::mem::Device> alloc;
    };
    static device_bucket_t b = {};
    return b.alloc;
  };

  nda::mem::dynamic_bucket<nda::mem::Unified>& get_global_unified_bucket() {
    struct unified_bucket_t {
      unified_bucket_t() : alloc(DEFAULT_BUCKET_SIZE) {}
      ~unified_bucket_t() = default;
      nda::mem::dynamic_bucket<nda::mem::Unified> alloc;
    };
    static unified_bucket_t b = {};
    return b.alloc;
  };
#endif

}  // nda::mem::detail

