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

#ifndef SPARSE_MATRIX_HELPER_H
#define SPARSE_MATRIX_HELPER_H

#include "SparseMatrix/SparseMatrix.h"

#include <stdio.h>
#include <string>
#include <complex>

using std::complex;
using std::string;

namespace sfqmc
{
template<typename T>
void output_data(T* data, int size)
{
  std::cout << "[ ";
  for (int i = 0; i < size; i++)
  {
    std::cout << data[i] << " ";
  }
  std::cout << "]" << std::endl;
}

template<typename T>
void output_matrix(SparseMatrix<T>& M)
{
  std::cout << "Colms = ";
  output_data(M.column_data(), M.size());
  std::cout << "Row index = ";
  output_data(M.row_index(), M.rows() + 1);
  std::cout << "Values = ";
  output_data(M.values(), M.size());
}

template<typename T>
double realPart(T& a)
{
  REQUIRE(false);
  return 0.0;
}

template<>
double realPart(const double& a)
{
  return a;
}

template<>
double realPart(double& a)
{
  return a;
}

template<>
double realPart(const complex<double>& a)
{
  return a.real();
}

template<>
double realPart(complex<double>& a)
{
  return a.real();
}

} // namespace sfqmc

#endif
