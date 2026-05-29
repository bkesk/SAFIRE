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

namespace sfqmc::utils {

  /**
   * @brief Initialize the threading environment for the current process.
   *
   * @details
   * Sets `OMP_NUM_THREADS` and `TBLIS_NUM_THREADS` to 1 unless the user has
   * already placed those variables in the environment, and calls
   * `omp_set_num_threads(1)` when OpenMP is compiled in.  Emits a warning if
   * either variable is user-set to a value greater than 1.
   *
   * Must be called once during application startup, after loggers are
   * initialized and before any threading-sensitive library calls.  See
   * `check_thread_oversubscription` for the complementary MPI-aware check that
   * should be called once MPI context is available.
   */
  void init_threading();

  /**
   * @internal
   * @brief Warn if the current threading configuration would oversubscribe the node.
   *
   * @param tasks_per_node  Number of MPI ranks sharing this node, used to compute
   *                        total threads = threads_per_rank × tasks_per_node.
   *
   * @details
   * Because SAFIRE does not use shared-memory threading internally, each rank
   * is normally configured with a single thread (see `init_threading`).
   * However, a user may legitimately enable threading in a BLAS or
   * tensor-contraction backend (e.g. to exploit multi-threaded MKL on a node
   * with few MPI ranks).  This function provides a runtime sanity check in
   * that scenario: it reads `OMP_NUM_THREADS` and `TBLIS_NUM_THREADS` from the
   * environment and emits a warning whenever
   *
   *   threads_per_rank × tasks_per_node > hardware logical CPUs
   *
   * for either variable.  The check is a best-effort safeguard — it relies on
   * `std::thread::hardware_concurrency()` which may return 0 on unusual
   * platforms, in which case no warning is issued.
   *
   * If threading is ever adopted natively by SAFIRE, this function should be
   * revisited alongside `init_threading`.
   */
  void check_thread_oversubscription(int tasks_per_node);

  /// Returns true if `TBLIS_NUM_THREADS` was present in the environment at startup.
  bool tblis_threads_was_user_set();

  /// Returns true if `OMP_NUM_THREADS` was present in the environment at startup.
  bool omp_threads_was_user_set();

} // namespace sfqmc::utils
