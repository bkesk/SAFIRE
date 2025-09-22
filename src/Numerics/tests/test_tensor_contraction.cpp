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

#include "catch_amalgamated.hpp"
#include "config.h"
#include "Utilities/AppAbort.hpp"

#include <vector>
#include <iostream>


#include "SparseMatrix/tests/matrix_helpers.h"
#include "Numerics/ma_blas.hpp"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

using boost::multi::array;
using boost::multi::array_ref;
using std::vector;
template<std::ptrdiff_t D>
using iextensions = typename boost::multi::iextensions<D>;

namespace sfqmc
{
void ma_tensor_tests()
{
  vector<double> v = {1., 2., 3.};
  {
    array_ref<double, 1> V(v.data(), iextensions<1u>{v.size()});
    ma::scal(2., V);
    {
      vector<double> v2 = {2., 4., 6.};
      array_ref<double, 1> V2(v2.data(), iextensions<1u>{v2.size()});
      verify_approx(V, V2);
    }
  }
}

TEST_CASE("test_tensor", "[tensor_operations]") { ma_tensor_tests(); }

} // namespace sfqmc
