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


#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/test_common.hpp"

namespace bdft_tests
{

namespace mpi3 = boost::mpi3;
template <int Rank> using shape_t = std::array<long, Rank>;

TEST_CASE("shared_nda", "[math]") {
  auto& mpi = sfqmc::utils::make_unit_test_mpi_context();
  using Array_view_base_t = nda::array_view<double, 3>;

  auto array = math::shm::make_shared_array<Array_view_base_t>(mpi,shape_t<3>{39,17,10});
  mpi->node_comm.barrier();
  if(mpi->node_comm.root()) 
    array.local()() = 1.0;
  mpi->node_comm.barrier();
  for( auto& v : array.local() ) 
    sfqmc::utils::VALUE_EQUAL(v,1.0);
}

} // bdft_tests
