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

#ifndef SFQMC_AFQMC_UPGRADEONEBODYINTEGRALS_HPP
#define SFQMC_AFQMC_UPGRADEONEBODYINTEGRALS_HPP

#include <vector>
#include <tuple>
#include <mpi.h>
#include <algorithm>
#include <numeric>
#include <boost/type_traits/is_complex.hpp>
#include <string>

#include "AFQMC/config.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "SparseMatrix/matrix_emplace_wrapper.hpp"
#include "Numerics/ma_operations.hpp"
#include "AFQMC/SlaterDeterminantOperations/rotate.hpp"
#include "AFQMC/Utilities/afqmc_TTI.hpp"

#include "AFQMC/Hamiltonians/rotateHamiltonian_Helper2.hpp"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

namespace sfqmc
{
namespace afqmc
{
/**
 * @brief Upgrade one-body integrals from the input spin-symmetry
 *     to the spin symmetry of the walkers.
 *
 * @details For Noncollinear walkers, will upgrade input H1 
 *          from Closed or Collinear to Noncollinear. For Collinear,
 *          will upgrade input H1 from Closed to Collinear.
 *          For Closed walkers, will copy input H1 to output H1.
 *          no upgrade is possible.
 *
  * @param H1_input multi::array<ValueType, 2>& The input one-body integrals.
  * @param H1_input multi::array_ref<ValueType, 2>& The output one-body integrals.
  * @param H1_input_shape const std::vector<int>& the shape of the input one-body integrals.
  * @param NMO int The number of basis set orbitals.
  * @param npol int The number of polarizations.
  * @param nspins int The number of spins.
  * @param type WALKER_TYPES The type of the walkers. This determines the target spin symmetry of H1_output. 
  * @param base_error const std::string& The base error message to use in case of an error.
  *
  * @note Assumes the caller is handling parallelization.
  * @note Makes no allocations.
 */
template<class ValueType>
void upgradeOneBodyIntegrals(
    multi::array<ValueType, 2>& H1_input,
    multi::array_ref<ValueType, 2>& H1_ouptut,
    const std::vector<int>& H1_input_shape,
    int NMO,
    int npol,
    int nspins,
    WALKER_TYPES type,
    const std::string& base_error = std::string()) // KE: I get an error if I set a default value here? Even if using ""
                                  //     I guess casting from const char to string is not allowed?
{
    //using shmValueMatrix = multi::array<ValueType,2 ,shared_allocator<ValueType>>;
    using ValueMatrix_ref = multi::array_ref<ValueType,2>;

    ValueMatrix_ref _H1(raw_pointer_cast(H1_ouptut.origin()), H1_ouptut.extensions());
    std::fill_n(_H1.origin(), _H1.num_elements(), ValueType(0.0));
    if (type == NONCOLLINEAR)
    {
      if (H1_input_shape == std::vector<int>({2*NMO, 2*NMO}))
      {
        copy_n_cast(raw_pointer_cast(H1_input.origin()), (npol * NMO) * (npol * NMO), _H1.origin());
      }
      else if (H1_input_shape == std::vector<int>({2*NMO, NMO}))
      {
        app_log(1, "Upgrading Collinear 1-body integrals to Noncollinear 1-body integrals.");
        ValueMatrix_ref H1up(raw_pointer_cast(H1_input.origin()), {NMO, NMO});
        ValueMatrix_ref H1dn(raw_pointer_cast(H1_input.origin()) + (NMO * NMO), {NMO, NMO});
        ma::add(ValueType(1.0), H1up, ValueType(0.0),
          _H1({0,NMO},{0,NMO}),
          _H1({0,NMO},{0,NMO})
        );
        ma::add(ValueType(1.0), H1dn, ValueType(0.0),
          _H1({NMO,npol*NMO},{NMO,npol*NMO}),
          _H1({NMO,npol*NMO},{NMO,npol*NMO})
        );
      }
      else if (H1_input_shape == std::vector<int>({NMO, NMO}))
      {
        app_log(1, "Upgrading Closed 1-body integrals to Noncollinear 1-body integrals.");
        // assign up-up
        ma::add(ValueType(1.0), H1_input, ValueType(0.0), 
            _H1({0,NMO},{0,NMO}),
            _H1({0,NMO},{0,NMO}));
        ma::add(ValueType(1.0), H1_input, ValueType(0.0), 
            _H1({NMO,npol*NMO},{NMO,npol*NMO}),
            _H1({NMO,npol*NMO},{NMO,npol*NMO}));
      }
    }   
    else if (type == COLLINEAR)
    {
      if (H1_input_shape == std::vector<int>({2*NMO, NMO}))
      {
        copy_n_cast(raw_pointer_cast(H1_input.origin()), ( nspins * NMO) * ( NMO), _H1.origin());
      }
      else if (H1_input_shape == std::vector<int>({NMO, NMO}))
      {
        app_log(1, "Upgrading Closed 1-body integrals to Collinear 1-body integrals.");
        ma::add(ValueType(1.0), H1_input, ValueType(0.0), _H1.sliced(0, NMO), _H1.sliced(0, NMO));
        ma::add(ValueType(1.0), H1_input, ValueType(0.0), _H1.sliced(NMO, 2*NMO), _H1.sliced(NMO, 2*NMO));
      }
    }
    else if (type == CLOSED || type == FULLYPOLARIZED)
    {
      copy_n_cast(raw_pointer_cast(H1_input.origin()), NMO * NMO, _H1.origin());
    }
    else
    {
      APP_ABORT(base_error + "Unknown walker type in RealDenseHamiltonian::getHamiltonianOperations().");
    }
}

/**
 * @brief Overload of upgradeOneBodyIntegrals that accepts array_ref as the first parameter
 *     to avoid unnecessary copies when the input is already an array_ref.
 */
template<class ValueType>
void upgradeOneBodyIntegrals(
    multi::array_ref<ValueType, 2>& H1_input,
    multi::array_ref<ValueType, 2>& H1_output,
    const std::vector<int>& H1_input_shape,
    int NMO,
    int npol,
    int nspins,
    WALKER_TYPES type,
    const std::string& base_error = std::string())
{
    using ValueMatrix_ref = multi::array_ref<ValueType,2>;

    ValueMatrix_ref _H1(raw_pointer_cast(H1_output.origin()), H1_output.extensions());
    std::fill_n(_H1.origin(), _H1.num_elements(), ValueType(0.0));
    if (type == NONCOLLINEAR)
    {
      if (H1_input_shape == std::vector<int>({2*NMO, 2*NMO}))
      {
        app_log(1, "Using Noncollinear 1-body integrals.");
        copy_n_cast(raw_pointer_cast(H1_input.origin()), (npol * NMO) * (npol * NMO), _H1.origin());
      }
      else if (H1_input_shape == std::vector<int>({2*NMO, NMO}))
      {
        app_log(1, "Upgrading Collinear 1-body integrals to Noncollinear 1-body integrals.");
        ValueMatrix_ref H1up(raw_pointer_cast(H1_input.origin()), {NMO, NMO});
        ValueMatrix_ref H1dn(raw_pointer_cast(H1_input.origin()) + (NMO * NMO), {NMO, NMO});
        ma::add(ValueType(1.0), H1up, ValueType(0.0),
          _H1({0,NMO},{0,NMO}),
          _H1({0,NMO},{0,NMO})
        );
        ma::add(ValueType(1.0), H1dn, ValueType(0.0),
          _H1({NMO,npol*NMO},{NMO,npol*NMO}),
          _H1({NMO,npol*NMO},{NMO,npol*NMO})
        );
      }
      else if (H1_input_shape == std::vector<int>({NMO, NMO}))
      {
        app_log(1, "Upgrading Closed 1-body integrals to Noncollinear 1-body integrals.");
        // assign up-up
        ma::add(ValueType(1.0), H1_input, ValueType(0.0), 
            _H1({0,NMO},{0,NMO}),
            _H1({0,NMO},{0,NMO}));
        ma::add(ValueType(1.0), H1_input, ValueType(0.0), 
            _H1({NMO,npol*NMO},{NMO,npol*NMO}),
            _H1({NMO,npol*NMO},{NMO,npol*NMO}));
      } else {
        APP_ABORT(base_error + "Input H1 has unexpected shape for Noncollinear upgrade.");
      }
    }   
    else if (type == COLLINEAR)
    {
      if (H1_input_shape == std::vector<int>({2*NMO, NMO}))
      {
        app_log(1, "Using Collinear 1-body integrals.");
        copy_n_cast(raw_pointer_cast(H1_input.origin()), ( nspins * NMO) * ( NMO), _H1.origin());
      }
      else if (H1_input_shape == std::vector<int>({NMO, NMO}))
      {
        app_log(1, "Upgrading Closed 1-body integrals to Collinear 1-body integrals.");
        ma::add(ValueType(1.0), H1_input, ValueType(0.0), _H1.sliced(0, NMO), _H1.sliced(0, NMO));
        ma::add(ValueType(1.0), H1_input, ValueType(0.0), _H1.sliced(NMO, 2*NMO), _H1.sliced(NMO, 2*NMO));
      }  else {
        APP_ABORT(base_error + "Input H1 has unexpected shape for Noncollinear upgrade.");
      }
    }
    else if (type == CLOSED || type == FULLYPOLARIZED)
    {
      app_log(1, "Using closed 1-body integrals.");
      copy_n_cast(raw_pointer_cast(H1_input.origin()), NMO * NMO, _H1.origin());
    }
    else
    {
      APP_ABORT(base_error + "Unknown walker type in upgradeOneBodyIntegrals with array_ref input.");
    }
}

// Explicit template instantiations for the multi::array overload
template void upgradeOneBodyIntegrals<ComplexType>(
    multi::array<ComplexType, 2>& H1_input,
    multi::array_ref<ComplexType, 2>& H1_output,
    const std::vector<int>& H1_input_shape,
    int NMO,
    int npol,
    int nspins,
    WALKER_TYPES type,
    const std::string& base_error);

template void upgradeOneBodyIntegrals<RealType>(
    multi::array<RealType, 2>& H1_input,
    multi::array_ref<RealType, 2>& H1_output,
    const std::vector<int>& H1_input_shape,
    int NMO,
    int npol,
    int nspins,
    WALKER_TYPES type,
    const std::string& base_error);

// Explicit template instantiations for the array_ref overload
template void upgradeOneBodyIntegrals<ComplexType>(
    multi::array_ref<ComplexType, 2>& H1_input,
    multi::array_ref<ComplexType, 2>& H1_output,
    const std::vector<int>& H1_input_shape,
    int NMO,
    int npol,
    int nspins,
    WALKER_TYPES type,
    const std::string& base_error);

template void upgradeOneBodyIntegrals<RealType>(
    multi::array_ref<RealType, 2>& H1_input,
    multi::array_ref<RealType, 2>& H1_output,
    const std::vector<int>& H1_input_shape,
    int NMO,
    int npol,
    int nspins,
    WALKER_TYPES type,
    const std::string& base_error);


} // namespace afqmc

} // namespace sfqmc

#endif
