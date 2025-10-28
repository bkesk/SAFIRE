/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef MEMORY_OPENMP_HPP
#define MEMORY_OPENMP_HPP

#include "config.h"

#if defined(ENABLE_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0; }
inline omp_int_t omp_get_max_threads() { return 1; }
inline omp_int_t omp_get_num_threads() { return 1; }
inline omp_int_t omp_get_level() { return 0; }
inline omp_int_t omp_get_ancestor_thread_num([[maybe_unused]] int level) { return 0; }
inline bool omp_get_nested() { return false; }
#endif

#endif 
