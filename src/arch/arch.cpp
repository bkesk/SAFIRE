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


#include "arch.h"

#include <cstdlib>
#include <thread>
#include <nda/nda.hpp>

#include "config.h"


#if defined(ENABLE_CUDA)

#include "CUDA/cuda_init.h"
#include "CUDA/cuda_sync.h"

namespace sfqmc {
namespace arch
{
  bool get_device_synchronization() {return cuda::get_device_synchronization();};
  void set_device_synchronization(bool s) { cuda::set_device_synchronization(s); };
  void synchronize_if_set() { cuda::synchronize_if_set(); };
  void synchronize() { cuda::synchronize(); };
  void check_device_configuration() { cuda::check_device_configuration(); }
}
}

#else

namespace sfqmc {
namespace arch
{
  bool get_device_synchronization() {return true;};
  void set_device_synchronization(bool) {};
  void synchronize_if_set() {};
  void synchronize() {};
  void check_device_configuration() {};
}
}

#endif

#include "IO/app_loggers.h"

namespace sfqmc {
namespace arch
{

// File-scope flags recording whether each threading variable was already present
// in the environment when SAFIRE started (i.e. the user provided it explicitly).
static bool s_tblis_user_set = false;
static bool s_omp_user_set   = false;

bool tblis_threads_was_user_set() { return s_tblis_user_set; }
bool omp_threads_was_user_set()   { return s_omp_user_set; }

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

void init(bool active_log, int output_level=2, int debug_level=2)
{
  // setup loggers, can always be changed later
  setup_loggers(active_log, output_level, debug_level);

  // Set TBLIS_NUM_THREADS and OMP_NUM_THREADS to 1 unless the user has already
  // set them. This prevents accidental CPU oversubscription in MPI parallel runs.
  s_tblis_user_set = (std::getenv("TBLIS_NUM_THREADS") != nullptr);
  if (!s_tblis_user_set) ::setenv("TBLIS_NUM_THREADS", "1", 1);
  s_omp_user_set = (std::getenv("OMP_NUM_THREADS") != nullptr);
  if (!s_omp_user_set) ::setenv("OMP_NUM_THREADS", "1", 1);

#if defined(ENABLE_CUDA)
  cuda::init();
#endif

  // setup shared memory, memory buffers, etc, etc
}

// this is a problem if the cuda system is disabled before this is destroyed
std::vector<nda::devStream_t> device_streams;

}
}

