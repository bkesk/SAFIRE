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

#include "utilities/threading.h"

#include <cstdlib>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "IO/app_loggers.h"

namespace sfqmc::utils {

// File-scope flags recording whether each threading variable was already present
// in the environment when SAFIRE started (i.e. the user provided it explicitly).
static bool s_tblis_user_set = false;
static bool s_omp_user_set   = false;

bool tblis_threads_was_user_set() { return s_tblis_user_set; }
bool omp_threads_was_user_set()   { return s_omp_user_set; }

/// @copydoc sfqmc::utils::init_threading
void init_threading()
{
  auto parse_env_int = [](const char* name) -> int {
    const char* val = std::getenv(name);
    if (!val) return 1;
    try { return std::max(1, std::stoi(val)); } catch (...) { return 1; }
  };

  // Set OMP_NUM_THREADS and TBLIS_NUM_THREADS to 1 unless the user has already
  // set them. This prevents accidental CPU oversubscription in MPI parallel runs.
  s_omp_user_set = (std::getenv("OMP_NUM_THREADS") != nullptr);
  if (!s_omp_user_set) {
    ::setenv("OMP_NUM_THREADS", "1", 1);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
  }
  s_tblis_user_set = (std::getenv("TBLIS_NUM_THREADS") != nullptr);
  if (!s_tblis_user_set) ::setenv("TBLIS_NUM_THREADS", "1", 1);

  // Warn if user-set threading variables are > 1 (oversubscription awareness).
  if (s_omp_user_set && parse_env_int("OMP_NUM_THREADS") > 1)
    app_warning("OMP_NUM_THREADS={} is set in the environment. Assuming intentional; "
                "be aware of potential CPU oversubscription in MPI runs.",
                parse_env_int("OMP_NUM_THREADS"));
  if (s_tblis_user_set && parse_env_int("TBLIS_NUM_THREADS") > 1)
    app_warning("TBLIS_NUM_THREADS={} is set in the environment. Assuming intentional; "
                "be aware of potential CPU oversubscription in MPI runs.",
                parse_env_int("TBLIS_NUM_THREADS"));
}

/// @copydoc sfqmc::utils::check_thread_oversubscription
void check_thread_oversubscription(int tasks_per_node)
{
  auto parse_env_int = [](const char* name) -> int {
    const char* val = std::getenv(name);
    if (!val) return 1;
    try { return std::max(1, std::stoi(val)); } catch (...) { return 1; }
  };
  const int omp_threads   = parse_env_int("OMP_NUM_THREADS");
  const int tblis_threads = parse_env_int("TBLIS_NUM_THREADS");
  const int hw_cpus       = static_cast<int>(std::thread::hardware_concurrency());
  if (hw_cpus > 0) {
    if (omp_threads * tasks_per_node > hw_cpus)
      app_warning("OMP_NUM_THREADS={} x MPI tasks/node={} = {} threads > {} logical CPUs. "
                  "CPU oversubscription detected. Consider reducing OMP_NUM_THREADS or "
                  "the number of MPI tasks per node.",
                  omp_threads, tasks_per_node, omp_threads * tasks_per_node, hw_cpus);
    if (tblis_threads * tasks_per_node > hw_cpus)
      app_warning("TBLIS_NUM_THREADS={} x MPI tasks/node={} = {} threads > {} logical CPUs. "
                  "CPU oversubscription detected. Consider reducing TBLIS_NUM_THREADS or "
                  "the number of MPI tasks per node.",
                  tblis_threads, tasks_per_node, tblis_threads * tasks_per_node, hw_cpus);
  }
}

} // namespace sfqmc::utils
