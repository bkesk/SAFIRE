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

#ifndef KERNELS_SETTINGS_HPP
#define KERNELS_SETTINGS_HPP

#define BOOST_NO_AUTO_PTR

static const size_t DEFAULT_BLOCK_SIZE  = 32;
static const size_t DOT_BLOCK_SIZE      = 32;
static const size_t REDUCE_BLOCK_SIZE   = 32;
static const size_t MAX_THREADS_PER_DIM = 1024;

#endif
