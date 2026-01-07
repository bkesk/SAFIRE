/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(ENABLE_CUDA)
#include "CUDA/cuda_init.h"
#include "CUDA/cuda_sync.h"
#endif

#include "config.h"
#include "nda/nda.hpp"

namespace sfqmc {
namespace arch
{
  void init(bool active_log, int output_level, int debug_level);
  bool get_device_synchronization();
  void set_device_synchronization(bool);
  void synchronize_if_set();
  void synchronize();
  void check_device_configuration();

  // device streams
  extern std::vector<nda::devStream_t> device_streams;
}
}

