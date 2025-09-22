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

#ifndef SFQMC_AFQMC_MA_HDF5_READERS_HPP
#define SFQMC_AFQMC_MA_HDF5_READERS_HPP


#include <cassert>
#include <complex>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <vector>
#include <array>
#include <numeric>
#if defined(USE_MPI)
#include <mpi.h>
#endif

#include "Utilities/type_traits/container_traits_multi.h"
#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "config.0.h"
#include "AFQMC/Utilities/afqmc_TTI.hpp"

#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "mpi3/shared_window.hpp"

namespace sfqmc
{
namespace afqmc
{
namespace ma_hdf5
{
template<class MultiArray, class task_group_>
inline void write_distributed_MA(MultiArray& A,
                                 std::array<size_t, 2> offset,
                                 std::array<size_t, 2> gdim,
                                 hdf_archive& dump,
                                 std::string name,
                                 task_group_& TG)
{
  using value_type = typename MultiArray::element;
  // serial hdf for now
  if (TG.getNGroupsPerTG() == 1)
  {
    if (TG.Global().root())
      dump.write(A, name);
  }
  else
  {
    size_t nnodes_per_TG = TG.getNGroupsPerTG();
    // data distribution depends on whether we have devices or not!
    int TG_number(TG.getTGNumber());
    int TG_local_rank(TG.TG_local().rank());
    // assumes that Global.root() lives in TG_number
    if (TG.Global().root())
    {
      RUNTIME_CHECK(TG_number == 0, "");
      std::vector<size_t> ndim(4 * nnodes_per_TG);
      ndim[0] = offset[0];
      ndim[1] = offset[1];
      ndim[2] = A.size(0);
      ndim[3] = A.size(1);
      TG.TG_Cores().all_reduce_in_place_n(ndim.begin(), ndim.size(), std::plus<>());

      // write local piece
      {
        using Mat_ref = boost::multi::array_ref<value_type, 2>;
        Mat_ref A_(raw_pointer_cast(A.origin()), A.extensions());
        hyperslab_proxy<Mat_ref, 2> slab(A_, gdim, std::array<size_t, 2>{size_t(A.size(0)), size_t(A.size(1))}, offset);
        dump.write(slab, name);
      }

      std::vector<size_t>::iterator it = ndim.begin() + 4;
      for (size_t i = 1; i < nnodes_per_TG; i++, it += 4)
      {
        using Mat = boost::multi::array<value_type, 2>;
        Mat T({*(it + 2), *(it + 3)});
        TG.TG_Cores().receive_n(T.origin(), T.num_elements(), i, i);
        hyperslab_proxy<Mat, 2> slab(T, gdim, std::array<size_t, 2>{*(it + 2), *(it + 3)},
                                     std::array<size_t, 2>{*(it), *(it + 1)});
        dump.write(slab, name);
      }
    }
    else if ((TG_number == 0) && (TG_local_rank == 0))
    {
      std::vector<size_t> ndim(4 * nnodes_per_TG);
      RUNTIME_CHECK(TG.TG_Cores().size() == nnodes_per_TG, "");
      // all tasks on the TG have a section of the matrix
      ndim[4 * TG.TG_Cores().rank()]     = offset[0];
      ndim[4 * TG.TG_Cores().rank() + 1] = offset[1];
      ndim[4 * TG.TG_Cores().rank() + 2] = A.size(0);
      ndim[4 * TG.TG_Cores().rank() + 3] = A.size(1);
      TG.TG_Cores().all_reduce_in_place_n(ndim.begin(), ndim.size(), std::plus<>());
      TG.TG_Cores().send_n(raw_pointer_cast(A.origin()), A.num_elements(), 0, TG.TG_Cores().rank());
    }
  }
  TG.Global().barrier();
}

} // namespace ma_hdf5

} // namespace afqmc

} // namespace sfqmc

#endif
