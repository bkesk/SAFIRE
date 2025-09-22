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

#ifndef SFQMC_AFQMC_GENERATEPROPAGATOR_HPP
#define SFQMC_AFQMC_GENERATEPROPAGATOR_HPP

#include <cstdlib>
#include <algorithm>
#include <complex>
#include <iostream>
#include <vector>
#include <numeric>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "Numerics/ma_operations.hpp"
#include "Numerics/csr_blas.hpp"
#include "SparseMatrix/csr_matrix_construct.hpp"

namespace sfqmc
{
namespace afqmc
{
// Input H1(i,j) = h(i,j) + sum_n vMF(n)*CholMat(i,j,n) + vn0(i,j)
// Output: sparse 1 body propagator = exp(-0.5 * H1) (factor of dt included in H1)
template<class P_Type, class MultiArray2D>
P_Type generate1BodyPropagator(TaskGroup_& TG,
                               RealType cut,
                               MultiArray2D const& H1,
                               bool printP1eV = false)
{
  RUNTIME_CHECK(H1.dimensionality == 2, "");
  RUNTIME_CHECK(H1.size(0) == H1.size(1), "");
  RUNTIME_CHECK(H1.stride(1) == 1, "");
  int NMO = H1.size(0);
  if (TG.TG_local().root())
  {
    boost::multi::array<ComplexType, 2> v({NMO, NMO});
    fill_n(v.origin(), v.num_elements(), ComplexType(0));

    // running on host regardless
    boost::multi::array<ComplexType, 2> h1_(H1.extensions());
    std::copy_n(raw_pointer_cast(H1.origin()), H1.num_elements(), h1_.origin());

    for (int i = 0; i < NMO; i++)
      ma::axpy(-0.5, h1_[i], v[i]);

    boost::multi::array<ComplexType, 2> P = ma::exp(v, printP1eV);

    // need a version of this that works with gpu_ptr!!!
    return csr::shm::construct_csr_matrix_single_input<P_Type>(P, cut, 'N', TG.TG_local());
  }
  else
  {
    boost::multi::array<ComplexType, 2> P({1, 1});
    return csr::shm::construct_csr_matrix_single_input<P_Type>(P, cut, 'N', TG.TG_local());
  }
}

// Input H1(i,j) = h(i,j) + sum_n vMF(n)*CholMat(i,j,n) + vn0(i,j)
// Output: sparse 1 body propagator = exp(-0.5*H1)
template<class P_Type, class MultiArray2D, class MultiArray2DB>
P_Type generate1BodyPropagator(TaskGroup_& TG,
                               RealType cut,
                               MultiArray2D const& H1,
                               MultiArray2DB const& H1ext,
                               bool printP1eV = false)
{
  RUNTIME_CHECK(H1.dimensionality == 2, "");
  RUNTIME_CHECK(H1.size(0) == H1.size(1), "");
  RUNTIME_CHECK(H1.stride(1) == 1, "");
  RUNTIME_CHECK(H1ext.dimensionality == 2, "");
  RUNTIME_CHECK(H1ext.size(0) == H1ext.size(1), "");
  RUNTIME_CHECK(H1ext.stride(1) == 1, "");
  RUNTIME_CHECK(H1.size(0) == H1ext.size(1), "");
  int NMO = H1.size(0);
  if (TG.TG_local().root())
  {
    //      boost::multi::array<ComplexType,2> v({NMO,NMO});
    //      fill_n(v.origin(),v.num_elements(),ComplexType(0));

    // running on host regardless
    boost::multi::array<ComplexType, 2> h1_(H1);
    //boost::multi::array<ComplexType,2> h1ext_(H1ext);
    boost::multi::array<ComplexType, 2> h1ext_({NMO, NMO});
    //copy_n(H1ext.origin(),NMO*NMO,h1ext_.origin());
    h1ext_ = H1ext;


    ma::add(ComplexType(-0.5), h1_, ComplexType(-0.5), h1ext_, h1_);

    //      for(int i=0; i<NMO; i++)
    //        ma::axpy(-0.5,h1_[i],v[i]);

    boost::multi::array<ComplexType, 2> P = ma::exp(h1_, printP1eV);

    // need a version of this that works with gpu_ptr!!!
    return csr::shm::construct_csr_matrix_single_input<P_Type>(P, cut, 'N', TG.TG_local());
  }
  else
  {
    boost::multi::array<ComplexType, 2> P({1, 1});
    return csr::shm::construct_csr_matrix_single_input<P_Type>(P, cut, 'N', TG.TG_local());
  }
}

} // namespace afqmc
} // namespace sfqmc
#endif
