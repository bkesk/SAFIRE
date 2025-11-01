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

namespace nda::mem::detail 
{
  nda::mem::dynamic_bucket<nda::mem::Host>& get_global_host_bucket();
#if defined(ENABLE_DEVICE)
  nda::mem::dynamic_bucket<nda::mem::Device>& get_global_device_bucket();
  nda::mem::dynamic_bucket<nda::mem::Unified>& get_global_unified_bucket();
#endif
} 

