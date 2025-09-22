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

#ifndef MA_BLAS_EXTENSIONS_HPP
#define MA_BLAS_EXTENSIONS_HPP

#include "Numerics/detail/blas.hpp"
#include "Numerics/tensor_operations.hpp"
#include <utility> //std::enable_if
#include <cassert>
#include <iostream>

namespace ma
{

// res = beta*y + alpha * dot(x,y)
template<class T,
         class Q,
         class MultiArray1Dx,
         class MultiArray1Dy,
         class ptr,
         typename = std::enable_if_t<std::decay<MultiArray1Dx>::type::dimensionality == 1>,
         typename = std::enable_if_t<std::decay<MultiArray1Dy>::type::dimensionality == 1>>
void adotpby(T const alpha, MultiArray1Dx const& x, MultiArray1Dy const& y, Q const beta, ptr res)
{
  RUNTIME_CHECK(x.size() == y.size(), "");
  ma::adotpby(x.size(), alpha, pointer_dispatch(x.origin()), x.stride(0), pointer_dispatch(y.origin()), y.stride(0), beta,
          pointer_dispatch(res), select_backend<MultiArray1Dx>());
}

// res[n*inc] = beta*res[n*inc] + alpha * dot(x[n*lda],y[n*lda])
template<class T,
         class Q,
         class MultiArray2Dx,
         class MultiArray2Dy,
         class MultiArray1D,
         typename = std::enable_if_t<std::decay<MultiArray2Dx>::type::dimensionality == 2>,
         typename = std::enable_if_t<std::decay<MultiArray2Dy>::type::dimensionality == 2>,
         typename = std::enable_if_t<std::decay<MultiArray1D>::type::dimensionality == 1>>
void adotpby(T const alpha, MultiArray2Dx const& x, MultiArray2Dy const& y, Q const beta, MultiArray1D res)
{
  if (x.size(0) != y.size(0) || x.size(0) != res.size(0) || x.size(1) != y.size(1) || x.stride(1) != 1 ||
      y.stride(1) != 1)
    throw std::runtime_error(" Error: Inconsistent matrix dimensions in adotpby(2D).");
  ma::strided_adotpby(x.size(0), x.size(1), alpha, pointer_dispatch(x.origin()), x.stride(0), pointer_dispatch(y.origin()),
                  y.stride(0), beta, pointer_dispatch(res.origin()), res.stride(0), select_backend<MultiArray2Dx>());
}

template<class T,
         class MultiArray1Dx,
         class MultiArray1Dy,
         typename = std::enable_if_t<std::decay<MultiArray1Dx>::type::dimensionality == 1>,
         typename = std::enable_if_t<std::decay<MultiArray1Dy>::type::dimensionality == 1>>
MultiArray1Dy&& axty(T const alpha, MultiArray1Dx const& x, MultiArray1Dy&& y)
{
  RUNTIME_CHECK(x.size() == y.size(), "");
  ma::axty(x.size(), alpha, pointer_dispatch(x.origin()), x.stride(0), pointer_dispatch(y.origin()), y.stride(0), select_backend<MultiArray1Dx>());
  return y;
}

template<class T,
         class MultiArray2DA,
         class MultiArray2DB,
         typename = std::enable_if_t<std::decay<MultiArray2DA>::type::dimensionality == 2>,
         typename = std::enable_if_t<std::decay<MultiArray2DB>::type::dimensionality == 2>,
         typename = void>
MultiArray2DB&& axty(T const alpha, MultiArray2DA const& A, MultiArray2DB&& B)
{
  RUNTIME_CHECK(A.num_elements() == B.num_elements(), "");
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(A.stride(0) == A.size(1), "");
  RUNTIME_CHECK(B.stride(1) == 1, "");
  RUNTIME_CHECK(B.stride(0) == B.size(1), "");
  ma::axty(A.num_elements(), alpha, pointer_dispatch(A.origin()), 1, pointer_dispatch(B.origin()), 1, select_backend<MultiArray2DA>());
  return B;
}

// on fortran ordering
// implements z[i][j] = beta * z[i][j] + alpha * conj(y[i][j]) * x[i]
template<class T,
         class MultiArray2DA,
         class MultiArray1D,
         class MultiArray2DB,
         typename =
                      std::enable_if_t<(MultiArray2DA::dimensionality == 2) and (MultiArray1D::dimensionality == 1) and
                                       (std::decay<MultiArray2DB>::type::dimensionality == 2)>>
MultiArray2DB&& acAxpbB(T const alpha, MultiArray2DA const& A, MultiArray1D const& x, T const beta, MultiArray2DB&& B)
{
  RUNTIME_CHECK(A.num_elements() == B.num_elements(), "");
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == B.size(1), "");
  RUNTIME_CHECK(A.size(1) == x.size(0), "");
  ma::acAxpbB(A.size(1), A.size(0), alpha, pointer_dispatch(A.origin()), A.stride(0), pointer_dispatch(x.origin()),
          x.stride(0), beta, pointer_dispatch(B.origin()), B.stride(0), select_backend<MultiArray2DA>());
  return B;
}

template<class T,
         class MultiArray2DA,
         class MultiArray1Dy,
         typename = std::enable_if_t<std::decay<MultiArray2DA>::type::dimensionality == 2>,
         typename = std::enable_if_t<std::decay<MultiArray1Dy>::type::dimensionality == 1>>
MultiArray1Dy&& adiagApy(T const alpha, MultiArray2DA const& A, MultiArray1Dy&& y)
{
  RUNTIME_CHECK(A.size(0) == A.size(1), "");
  RUNTIME_CHECK(A.size(0) == y.size(), "");
  ma::adiagApy(y.size(), alpha, pointer_dispatch(A.origin()), A.stride(0), pointer_dispatch(y.origin()), y.stride(0), select_backend<MultiArray2DA>());
  return y;
}

template<class MultiArray1D,
         typename = std::enable_if_t<std::decay<MultiArray1D>::type::dimensionality == 1>>
auto sum(MultiArray1D const& y)
{
  return ma::sum(y.size(), pointer_dispatch(y.origin()), y.stride(0), select_backend<MultiArray1D>());
}

template<class MultiArray2D,
         typename = std::enable_if_t<std::decay<MultiArray2D>::type::dimensionality == 2>,
         typename = void>
auto sum(MultiArray2D const& A)
{
  RUNTIME_CHECK(A.stride(1) == 1, "");
  // blas call assumes fortran ordering
  return ma::sum(A.size(1), A.size(0), pointer_dispatch(A.origin()), A.stride(0), select_backend<MultiArray2D>());
}

template<class MultiArray3D,
         typename = std::enable_if_t<std::decay<MultiArray3D>::type::dimensionality == 3>,
         typename = void,
         typename = void>
auto sum(MultiArray3D const& A)
{
  // only arrays and array_refs for now
  RUNTIME_CHECK(A.stride(0) == A.size(1) * A.size(2), "");
  RUNTIME_CHECK(A.stride(1) == A.size(2), "");
  RUNTIME_CHECK(A.stride(2) == 1, "");
  return ma::sum(A.num_elements(), pointer_dispatch(A.origin()), 1, select_backend<MultiArray3D>());
}

template<class MultiArray4D,
         typename = std::enable_if_t<std::decay<MultiArray4D>::type::dimensionality == 4>,
         typename = void,
         typename = void,
         typename = void>
auto sum(MultiArray4D const& A)
{
  // only arrays and array_refs for now
  RUNTIME_CHECK(A.stride(0) == A.size(1) * A.size(2) * A.size(3), "");
  RUNTIME_CHECK(A.stride(1) == A.size(2) * A.size(3), "");
  RUNTIME_CHECK(A.stride(2) == A.size(3), "");
  RUNTIME_CHECK(A.stride(3) == 1, "");
  return ma::sum(A.num_elements(), pointer_dispatch(A.origin()), 1, select_backend<MultiArray4D>());
}

// expand = true,
//      B[ index[n] ][:] = beta * B[ index[n] ][:] + alpha * A[n][:] 
// expand = false,
//      B[n][:] = beta * B[n][:] + alpha * A[ index[n] ][:] 
template<class Ta, class Tb, 
         class MultiArray2DA,
         class MultiArray2DB,
         class IVec,
         typename = std::enable_if_t<std::decay_t<MultiArray2DA>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MultiArray2DB>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<IVec>::dimensionality == 1>
        >
void copy_select (Ta alpha, MultiArray2DA const& A, Tb beta, MultiArray2DB&& B, 
                  IVec const& index, bool expand = true )
{
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(B.stride(1) == 1, "");
  RUNTIME_CHECK(A.size(1) == B.size(1), "");
  if( expand ) 
    RUNTIME_CHECK(A.size(0) == index.size(), "");
  else 
    RUNTIME_CHECK(B.size(0) == index.size(), "");
  using ma::copy_select_impl;
  ma::copy_select_impl(index.size(), A.size(1), alpha, pointer_dispatch(A.origin()), A.stride(0),0, 
                                    beta, pointer_dispatch(B.origin()), B.stride(0),0,
                                    pointer_dispatch(index.origin()),1,expand,select_backend<MultiArray2DA>());
}

// expand = true,
//      B[k][ index[n] ][:] = beta * B[k][ index[n] ][:] + alpha * A[k][n][:] 
// expand = false,
//      B[k][n][:] = beta * B[k][n][:] + alpha * A[k][ index[n] ][:] 
template<class Ta, class Tb,
         class MultiArray3DA,
         class MultiArray3DB,
         class IVec,
         typename = std::enable_if_t<std::decay_t<MultiArray3DA>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MultiArray3DB>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<IVec>::dimensionality == 1>,
         typename = void
        >
void copy_select (Ta alpha, MultiArray3DA const& A, Tb beta, MultiArray3DB&& B,
                  IVec const& index, bool expand = true )
{ 
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(B.stride(2) == 1, "");
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(2) == B.size(2), "");
  if( expand ) 
    RUNTIME_CHECK(A.size(1) == index.size(), "");
  else 
    RUNTIME_CHECK(B.size(1) == index.size(), "");
  using ma::copy_select_impl;
  ma::copy_select_impl(index.size(), A.size(2), 
                        alpha, pointer_dispatch(A.origin()), A.stride(1), A.stride(0),
                        beta, pointer_dispatch(B.origin()), B.stride(1), B.stride(0),
                        pointer_dispatch(index.origin()),A.size(0),expand, select_backend<MultiArray3DA>());
}

template<class MultiArrayA,
         class MultiArrayB,
         class IVec
        >
void copy_select (MultiArrayA const& A, MultiArrayB&& B, IVec const& index, bool expand = true)
{
  using Atype = typename std::decay_t<MultiArrayA>::element;  
  using Btype = typename std::decay_t<MultiArrayB>::element;  
  ma::copy_select(Atype(1.0),A,Btype(0.0),std::forward<MultiArrayB>(B),index,expand);
}

// y[i] += alpha * sum_j A[j][i]  for dim==0
//      += alpha * sum_j A[i][j]  for dim==1
template<class Ta,
         class MultiArrayA,
         class MultiArrayY,
         typename = std::enable_if_t<MultiArrayA::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MultiArrayY>::dimensionality == 1>
        >
void accumulate (int dim, Ta alpha, MultiArrayA const& A, MultiArrayY&& y)
{
  RUNTIME_CHECK(A.stride(1) == 1, "");
  if( dim==0 )
    RUNTIME_CHECK(A.size(1) == y.size(0), "");
  else
    RUNTIME_CHECK(A.size(0) == y.size(0), "");
  ma::accumulate_impl(dim, A.size(0), A.size(1), alpha, pointer_dispatch(A.origin()), A.stride(0),0,
                  pointer_dispatch(y.origin()), y.stride(0), 0, 1, select_backend<MultiArrayA>());
}

// y[n][i] += alpha * sum_j A[n][j][i]  for dim==0
//         += alpha * sum_j A[n][i][j]  for dim==1
template<class Ta, 
         class MultiArrayA,
         class MultiArrayY,
         typename = std::enable_if_t<MultiArrayA::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MultiArrayY>::dimensionality == 2>,
         typename = void
        >
void accumulate (int dim, Ta alpha, MultiArrayA const& A, MultiArrayY&& y)
{ 
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(y.stride(1) == 1, "");
  RUNTIME_CHECK(A.size(0) == y.size(0), "");
  if( dim==0 )
    RUNTIME_CHECK(A.size(2) == y.size(1), "");
  else
    RUNTIME_CHECK(A.size(1) == y.size(1), "");
  using ma::accumulate_impl;
  ma::accumulate_impl(dim, A.size(1), A.size(2), alpha, pointer_dispatch(A.origin()), 
                  A.stride(1), A.stride(0), pointer_dispatch(y.origin()), 1, 
                  y.stride(0), y.size(0), select_backend<MultiArrayA>());
}

template<class MultiArrayA, class MultiArrayY>
void accumulate (int dim, MultiArrayA const& A, MultiArrayY&& y)
{
  using Atype = typename std::decay_t<MultiArrayA>::element;  
  ma::accumulate(dim, Atype(1.0), A, y);
}

template<class MA,
         typename = std::enable_if_t<std::decay_t<MA>::dimensionality == 1>
        >
void complex_conjugate (MA&& A)
{
  using ma::complex_conjugate_impl;
  ma::complex_conjugate_impl(A.size(0), 1, pointer_dispatch(A.origin()),
                  A.stride(0), 0, 1, select_backend<MA>());
}

template<class MA,
         typename = std::enable_if_t<std::decay_t<MA>::dimensionality == 2>,
         typename = void
        >
void complex_conjugate (MA&& A)
{
  RUNTIME_CHECK(A.stride(1) == 1, "");
  using ma::complex_conjugate_impl;
  ma::complex_conjugate_impl(A.size(0), A.size(1), pointer_dispatch(A.origin()),
                  A.stride(0), 0, 1, select_backend<MA>());
}

template<class MA,
         typename = std::enable_if_t<std::decay_t<MA>::dimensionality == 3>,
         typename = void,
         typename = void
        >
void complex_conjugate (MA&& A)
{
  RUNTIME_CHECK(A.stride(2) == 1, "");
  using ma::complex_conjugate_impl;
  ma::complex_conjugate_impl(A.size(1), A.size(2), pointer_dispatch(A.origin()),
                  A.stride(1), A.stride(0), A.size(0), select_backend<MA>()); 
}

/*
 * Performs the generic operation: (limited to matrices for now)
 * A[i,j] = A[i,j] op x[...], 
 *   where op is {+,-,*,/} and x[...] depends on dim (0:i, 1:j, ...}
 */
template<class Ta,
         class MultiArrayA,
         class MultiArrayX,
         typename = std::enable_if_t<std::decay_t<MultiArrayA>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MultiArrayX>::dimensionality == 1>
        >
void elementwise (TENSOR_OPERATIONS op, int dim, Ta alpha, MultiArrayX const& x,  MultiArrayA&& A)
{ 
  RUNTIME_CHECK(A.stride(1) == 1, "");
  if( dim==0 )
    RUNTIME_CHECK(A.size(0) == x.size(0), "");
  else
    RUNTIME_CHECK(A.size(1) == x.size(0), "");
  using ma::term_by_term_matrix_vector_strided;
  ma::term_by_term_matrix_vector_strided(op, dim, A.size(0), A.size(1), pointer_dispatch(A.origin()), 
        A.stride(0),0, alpha, pointer_dispatch(x.origin()), x.stride(0), 0, 1, select_backend<MultiArrayA>());
}

template<class Ta,
         class MultiArrayA,
         class MultiArrayX,
         typename = std::enable_if_t<std::decay_t<MultiArrayA>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MultiArrayX>::dimensionality == 2>,
         typename = void
        >
void elementwise (TENSOR_OPERATIONS op, int dim, Ta alpha, MultiArrayX const& x,  MultiArrayA&& A)
{ 
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(x.stride(1) == 1, "");
  RUNTIME_CHECK(A.size(0) == x.size(0), "");
  if( dim==0 ) 
    RUNTIME_CHECK(A.size(1) == x.size(1), "");
  else
    RUNTIME_CHECK(A.size(2) == x.size(1), "");
  using ma::term_by_term_matrix_vector_strided;
  ma::term_by_term_matrix_vector_strided(op, dim, A.size(1), A.size(2), pointer_dispatch(A.origin()), 
        A.stride(1),A.stride(0), alpha, pointer_dispatch(x.origin()), 1, x.stride(0), x.size(0), select_backend<MultiArrayA>());
}

template<class MultiArrayA, class MultiArrayX>
void elementwise (TENSOR_OPERATIONS op, int dim, MultiArrayX const& x,  MultiArrayA&& A)
{ 
  using Xtype = typename std::decay_t<MultiArrayX>::element;  
  elementwise(op, dim, Xtype(1.0), x, A);
}

/*
 * B[i,j] = B[i,j] op A[i,j],   where op is {+,-,*,/} 
 */
template<class Ta,
         class MultiArrayA,
         class MultiArrayB,
         typename = std::enable_if_t<std::decay_t<MultiArrayA>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MultiArrayB>::dimensionality == 2>,
         typename = void,
         typename = void
        >
void elementwise (TENSOR_OPERATIONS op, Ta alpha, MultiArrayA const& A,  MultiArrayB&& B)
{
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(B.stride(1) == 1, "");
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == B.size(1), "");
  ma::term_by_term_matrix_matrix_strided(op, A.size(0), A.size(1), alpha, pointer_dispatch(A.origin()),
        A.stride(0),0, pointer_dispatch(B.origin()), B.stride(0), 0, 1, select_backend<MultiArrayA>());
}

template<class Ta,
         class MultiArrayA,
         class MultiArrayB,
         typename = std::enable_if_t<std::decay_t<MultiArrayA>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MultiArrayB>::dimensionality == 3>,
         typename = void,
         typename = void,
         typename = void
        >
void elementwise (TENSOR_OPERATIONS op, Ta alpha, MultiArrayA const& A,  MultiArrayB&& B)
{
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(B.stride(2) == 1, "");
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == B.size(1), "");
  RUNTIME_CHECK(A.size(2) == B.size(2), "");
  ma::term_by_term_matrix_matrix_strided(op, A.size(1), A.size(2), alpha, pointer_dispatch(A.origin()),
        A.stride(1),A.stride(0), pointer_dispatch(B.origin()), B.stride(1), B.stride(0), 
        A.size(0), select_backend<MultiArrayA>());
}

template<class Ta,
         class MultiArrayA,
         class MultiArrayB,
         typename = std::enable_if_t<std::decay_t<MultiArrayA>::dimensionality == 1>,
         typename = std::enable_if_t<std::decay_t<MultiArrayB>::dimensionality == 1>,
         typename = void,
         typename = void,
         typename = void,
         typename = void
        >
void elementwise (TENSOR_OPERATIONS op, Ta alpha, MultiArrayA const& A,  MultiArrayB&& B)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  ma::term_by_term_matrix_matrix_strided(op, A.size(0), 1, alpha, pointer_dispatch(A.origin()),
        A.stride(0),0, pointer_dispatch(B.origin()), B.stride(0), 0, 1, select_backend<MultiArrayA>());
}

template<class MultiArrayA, class MultiArrayB, typename = void>
void elementwise (TENSOR_OPERATIONS op, MultiArrayA const& A,  MultiArrayB&& B)
{
  using Atype = typename std::decay_t<MultiArrayA>::element;
  elementwise(op, Atype(1.0), A, B);
}

// performs (for all i,j) A[n,i,j]=alpha, if key[n]==0.0
template<class Ta,
         class MultiArray,
         class Vec,
         typename = std::enable_if_t<std::decay_t<MultiArray>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<Vec>::dimensionality == 1>
        >
void fill_if_zero(MultiArray&& A, Vec&& key, Ta alpha)
{
  RUNTIME_CHECK(key.size(0) == A.size(0), "");
  RUNTIME_CHECK(A.stride(2) == 1, "");
  ma::fill_if_zero_impl(A.size(1), A.size(2), pointer_dispatch(key.origin()), key.stride(0), alpha,  
                    pointer_dispatch(A.origin()), A.stride(1), A.stride(0), A.size(0), select_backend<MultiArray>());
}

template<class Ta,
         class MultiArray,
         class Vec,
         typename = std::enable_if_t<std::decay_t<MultiArray>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<Vec>::dimensionality == 1>
        >
void fill_if_non_zero(MultiArray&& A, Vec&& key, Ta alpha)
{
  RUNTIME_CHECK(key.size(0) == A.size(0), "");
  RUNTIME_CHECK(A.stride(2) == 1, "");
  ma::fill_if_non_zero_impl(A.size(1), A.size(2), pointer_dispatch(key.origin()), key.stride(0), alpha,
                    pointer_dispatch(A.origin()), A.stride(1), A.stride(0), A.size(0), select_backend<MultiArray>());
}

template<class MultiArray1D, 
         typename = std::enable_if_t<std::decay<MultiArray1D>::type::dimensionality == 1>>
void zero_complex_part(MultiArray1D&& a)
{
  ma::zero_complex_part(a.num_elements(), pointer_dispatch(a.origin()), select_backend<MultiArray1D>());
}

template<class MultiArray2D, 
         typename = std::enable_if_t<std::decay<MultiArray2D>::type::dimensionality == 2>>
MultiArray2D&& set_identity(MultiArray2D&& m)
{
  ma::set_identity(m.size(1), m.size(0), pointer_dispatch(m.origin()), m.stride(0), select_backend<MultiArray2D>());
  return std::forward<MultiArray2D>(m);
}

template<class MultiArray3D,
         typename = std::enable_if_t<std::decay<MultiArray3D>::type::dimensionality == 3>,
         typename = void>
MultiArray3D&& set_identity(MultiArray3D&& m)
{
  ma::set_identity_strided(m.size(0), m.stride(0), m.size(2), m.size(1), pointer_dispatch(m.origin()), m.stride(1), select_backend<MultiArray3D>());
  return std::forward<MultiArray3D>(m);
}

template<class T,
         class MultiArray2D,
         typename = std::enable_if_t<std::decay<MultiArray2D>::type::dimensionality == 2>>
MultiArray2D&& fill(MultiArray2D&& m, T value)
{
  using std::fill_n;
#if defined(ENABLE_DEVICE)
  using boost::multi::memory::cuda::fill_n;
#endif
  if(m.stride(1)==1 and m.stride(0)==m.size(1)) {
    fill_n(m.origin(),m.num_elements(),value);
  } else {
    using sfqmc::afqmc::fill2D;
    fill2D(m.size(0), m.size(1), pointer_dispatch(m.origin()), m.stride(0), value);
  }
  return std::forward<MultiArray2D>(m);
}

template<class T,
         class MultiArray1D,
         typename = std::enable_if_t<std::decay_t<MultiArray1D>::dimensionality == 1>,
         typename = void
        >
MultiArray1D&& fill(MultiArray1D&& m, T value)
{
  using std::fill_n;
#if defined(ENABLE_DEVICE)
  using boost::multi::memory::cuda::fill_n;
#endif
  if(m.stride(0)==1)
    fill_n(m.origin(),m.size(),value);
  else {
    using sfqmc::afqmc::fill2D;
    fill2D(m.size(0), long(1), pointer_dispatch(m.origin()), m.stride(0), value);
  }
  return std::forward<MultiArray1D>(m);
}

/* copy_n_cast */
template<class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 1>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 1>
        >
MA2&& copy_n_cast(MA1&& A, MA2&& B) 
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  ma::copy_n_cast_impl(A.size(0), 1, 
              pointer_dispatch(A.origin()), A.stride(0), 0, 
              pointer_dispatch(B.origin()), B.stride(0), 0, 
              1, select_backend<MA1,MA2>());  
  return std::forward<MA2>(B);
}

template<class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 2>,
         typename = void
        >
MA2&& copy_n_cast(MA1&& A, MA2&& B)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == B.size(1), "");
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(B.stride(1) == 1, "");
  ma::copy_n_cast_impl(A.size(0), A.size(1), 
              pointer_dispatch(A.origin()), A.stride(0), 0,
              pointer_dispatch(B.origin()), B.stride(0), 0,
              1, select_backend<MA1,MA2>());  
  return std::forward<MA2>(B);
}

template<class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 3>,
         typename = void,
         typename = void
        >
MA2&& copy_n_cast(MA1&& A, MA2&& B)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == B.size(1), "");
  RUNTIME_CHECK(A.size(2) == B.size(2), "");
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(B.stride(2) == 1, "");
  ma::copy_n_cast_impl(A.size(1), A.size(2), 
              pointer_dispatch(A.origin()), A.stride(1), A.stride(0),
              pointer_dispatch(B.origin()), B.stride(1), B.stride(0),
              A.size(0), select_backend<MA1,MA2>());        
  return std::forward<MA2>(B);
}


// A[k][i] = B[k][i][i]
template<class MultiArray3D,
         class MultiArray2D,
         typename = std::enable_if_t<std::decay<MultiArray3D>::type::dimensionality == 3>,
         typename = std::enable_if_t<std::decay<MultiArray2D>::type::dimensionality == 2>>
void get_diagonal_strided(MultiArray3D const& B, MultiArray2D&& A)
{
  if (A.size(0) != B.size(0) || A.size(1) != B.size(1) || A.size(1) != B.size(2) || A.stride(1) != 1 ||
      B.stride(2) != 1)
    throw std::runtime_error(" Error: Inconsistent matrix dimensions in get_diagonal_strided.");
  ma::get_diagonal_strided(A.size(0), A.size(1), pointer_dispatch(B.origin()), B.stride(1), B.stride(0),
                       pointer_dispatch(A.origin()), A.stride(0), select_backend<MultiArray3D>());
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
inline static void gemmBatched(char Atrans, char Btrans, int M, int N, int K, T1 alpha, T2* A, int lda,
			       T3* B, int ldb, T4 beta, T5* C, int ldc, int batchSize)
{
  ma::gemmBatched(Atrans,Btrans,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,batchSize,
	typename ma_dispatch<typename std::decay_t<T5>>::backend{});
}

template<typename T1, typename T2, typename T3>
void sumGwBatched(int n, T1 x, T2* a, int inca, T3* b, int incb, int b0, int nw, int batchSize)
{
  ma::sumGwBatched(n,x,a,inca,b,incb,b0,nw,batchSize, 
	typename ma_dispatch<typename std::decay_t<T2>>::backend{});
}

template<class CSR,
         class MultiArray2D,
         typename = std::enable_if_t<(std::decay<CSR>::type::dimensionality == -2)>,
         typename = std::enable_if_t<(MultiArray2D::dimensionality == 2)>>
void Matrix2MA(char TA, CSR const& A, MultiArray2D& M)
{
  using Type     = typename MultiArray2D::element;
  using int_type = typename CSR::int_type;
  RUNTIME_CHECK(TA == 'N' || TA == 'H' || TA == 'T' || TA == 'Z', "");
  if (TA == 'N' || TA == 'Z')
  {
    if (M.size(0) != A.size(0) or M.size(1) != A.size(1))
      M = MultiArray2D({A.size(0), A.size(1)}, M.get_allocator());
  }
  else if (TA == 'T' || TA == 'H')
  {
    if (M.size(0) != A.size(1) or M.size(1) != A.size(0))
      M = MultiArray2D({A.size(1), A.size(0)}, M.get_allocator());
  }
  else
  {
    throw std::runtime_error(" Error: Unknown operation in Matrix2MA.");
  }
  using std::fill_n;
  fill_n(pointer_dispatch(M.origin()), M.num_elements(), Type(0));
  auto pbegin = A.pointers_begin();
  auto pend   = A.pointers_end();
  int_type p0(pbegin[0]);
  auto v0 = A.non_zero_values_data();
  auto c0 = A.non_zero_indices2_data();
  if (TA == 'N')
  {
    for (int i = 0; i < A.size(0); i++)
      for (int_type ip = pbegin[i], ipend = pend[i]; ip < ipend; ip++)
        M[i][c0[ip - p0]] = Type(v0[ip - p0]);
  }
  else if (TA == 'Z')
  {
    for (int i = 0; i < A.size(0); i++)
      for (int_type ip = pbegin[i], ipend = pend[i]; ip < ipend; ip++)
        M[i][c0[ip - p0]] = ma::conj(Type(v0[ip - p0]));
  }
  else if (TA == 'T')
  {
    for (int i = 0; i < A.size(0); i++)
      for (int_type ip = pbegin[i], ipend = pend[i]; ip < ipend; ip++)
        M[c0[ip - p0]][i] = Type(v0[ip - p0]);
  }
  else if (TA == 'H')
  {
    for (int i = 0; i < A.size(0); i++)
      for (int_type ip = pbegin[i], ipend = pend[i]; ip < ipend; ip++)
        M[c0[ip - p0]][i] = ma::conj(Type(v0[ip - p0]));
  }
}

template<class CSR,
         class MultiArray2D,
         typename = std::enable_if_t<(std::decay_t<CSR>::dimensionality == -2)>,
         typename = std::enable_if_t<(std::decay_t<MultiArray2D>::dimensionality == 2)>>
void Matrix2MAREF(char TA, CSR const& A, MultiArray2D&& M)
{
  using Type     = typename std::decay_t<MultiArray2D>::element;
  using int_type = typename CSR::int_type;
  RUNTIME_CHECK(TA == 'N' || TA == 'H' || TA == 'T' || TA == 'Z', "");
  if ((TA == 'N' || TA == 'Z') && ((M.size(0) != A.size(0)) || (M.size(1) != A.size(1))))
    throw std::runtime_error(" Error: Wrong dimensions in Matrix2MAREF.");
  else if ((TA == 'T' || TA == 'H') && ((M.size(0) != A.size(1)) || (M.size(1) != A.size(0))))
    throw std::runtime_error(" Error: Wrong dimensions in Matrix2MAREF.");
  using std::fill_n;
  fill_n(pointer_dispatch(M.origin()), M.num_elements(), Type(0));
  auto pbegin = A.pointers_begin();
  auto pend   = A.pointers_end();
  int_type p0(pbegin[0]);
  auto v0 = A.non_zero_values_data();
  auto c0 = A.non_zero_indices2_data();
  if (TA == 'N')
  {
    for (int i = 0; i < A.size(0); i++)
      for (int_type ip = pbegin[i], ipend = pend[i]; ip < ipend; ip++)
        M[i][c0[ip - p0]] = Type(v0[ip - p0]);
  }
  else if (TA == 'Z')
  {
    for (int i = 0; i < A.size(0); i++)
      for (int_type ip = pbegin[i], ipend = pend[i]; ip < ipend; ip++)
        M[i][c0[ip - p0]] = ma::conj(Type(v0[ip - p0]));
  }
  else if (TA == 'T')
  {
    for (int i = 0; i < A.size(0); i++)
      for (int_type ip = pbegin[i], ipend = pend[i]; ip < ipend; ip++)
        M[c0[ip - p0]][i] = Type(v0[ip - p0]);
  }
  else if (TA == 'H')
  {
    for (int i = 0; i < A.size(0); i++)
      for (int_type ip = pbegin[i], ipend = pend[i]; ip < ipend; ip++)
        M[c0[ip - p0]][i] = ma::conj(Type(v0[ip - p0]));
  }
}

/* Chooses rows of A based on occups vector and performs CSF2MA on subset of rows */
template<class CSR,
         class MultiArray2D,
         class Vector,
         typename = std::enable_if_t<(std::decay<CSR>::type::dimensionality == -2)>,
         typename = std::enable_if_t<(MultiArray2D::dimensionality == 2)>>
void Matrix2MA(char TA, CSR const& A, MultiArray2D& M, Vector const& occups)
{
  using Type = typename MultiArray2D::element;
  if (occups.size() == 0)
    throw std::runtime_error(" Error: Empty occupation array in Matrix2MA.");
  RUNTIME_CHECK(occups.size() <= A.size(0), "");
  int nrows = occups.size();
  RUNTIME_CHECK(TA == 'N' || TA == 'H' || TA == 'T' || TA == 'Z', "");
  if (TA == 'N' || TA == 'Z')
  {
    if (M.size(0) != nrows || M.size(1) != A.size(1))
      throw std::runtime_error("Incorrect Array size.");
//      M.reextent({nrows, A.size(1)});
  }
  else if (TA == 'T' || TA == 'H')
  {
    if (M.size(1) != nrows || M.size(0) != A.size(1))
      throw std::runtime_error("Incorrect Array size.");
//      M.reextent({A.size(1), nrows});
  }
  else
    throw std::runtime_error(" Error: Unknown operation in Matrix2MA.");
  using std::fill_n;
  fill_n(pointer_dispatch(M.origin()), M.num_elements(), Type(0));
  auto pbegin = A.pointers_begin();
  auto pend   = A.pointers_end();
  auto p0     = pbegin[0];
  auto v0     = A.non_zero_values_data();
  auto c0     = A.non_zero_indices2_data();
  if (TA == 'N')
  {
    for (int i = 0; i < nrows; i++)
    {
      RUNTIME_CHECK(occups[i] >= 0 && occups[i] < A.size(0), "");
      int ik = occups[i];
      for (int ip = pbegin[ik]; ip < pend[ik]; ip++)
        M[i][c0[ip - p0]] = static_cast<Type>(v0[ip - p0]);
    }
  }
  else if (TA == 'Z')
  {
    for (int i = 0; i < nrows; i++)
    {
      RUNTIME_CHECK(occups[i] >= 0 && occups[i] < A.size(0), "");
      int ik = occups[i];
      for (int ip = pbegin[ik]; ip < pend[ik]; ip++)
        M[i][c0[ip - p0]] = static_cast<Type>(ma::conj(v0[ip - p0]));
    }
  }
  else if (TA == 'T')
  {
    for (int i = 0; i < nrows; i++)
    {
      RUNTIME_CHECK(occups[i] >= 0 && occups[i] < A.size(0), "");
      int ik = occups[i];
      for (int ip = pbegin[ik]; ip < pend[ik]; ip++)
        M[c0[ip - p0]][i] = static_cast<Type>(v0[ip - p0]);
    }
  }
  else if (TA == 'H')
  {
    for (int i = 0; i < nrows; i++)
    {
      RUNTIME_CHECK(occups[i] >= 0 && occups[i] < A.size(0), "");
      int ik = occups[i];
      for (int ip = pbegin[ik]; ip < pend[ik]; ip++)
        M[c0[ip - p0]][i] = static_cast<Type>(ma::conj(v0[ip - p0]));
    }
  }
}

template<class MA,
         class MultiArray2D,
         typename = std::enable_if_t<(std::decay<MA>::type::dimensionality == 2)>,
         typename = std::enable_if_t<(MultiArray2D::dimensionality == 2)>,
         typename = void>
void Matrix2MA(char TA, MA const& A, MultiArray2D& M)
{
  using Type2 = typename MultiArray2D::element;
  RUNTIME_CHECK(TA == 'N' || TA == 'H' || TA == 'T' || TA == 'Z', "");
  if (TA == 'N' || TA == 'Z')
  {
    if (M.size(0) != A.size(0) or M.size(1) != A.size(1))
      M = MultiArray2D({A.size(0), A.size(1)}, M.get_allocator());
  }
  else if (TA == 'T' || TA == 'H')
  {
    if (M.size(0) != A.size(1) or M.size(1) != A.size(0))
      M = MultiArray2D({A.size(1), A.size(0)}, M.get_allocator());
  }
  else
  {
    throw std::runtime_error(" Error: Unknown operation in Matrix2MA.");
  }

  using ptrA = std::remove_cv_t<typename MA::element_ptr>;
  using ptrM = std::remove_cv_t<typename MultiArray2D::element_ptr>;

  if (TA == 'H')
    TA = 'C';
  if (TA == 'Z')
  {
    for (int i = 0; i < M.size(0); i++)
      for (int j = 0; j < M.size(1); j++)
        M[i][j] = ma::conj(A[i][j]);
    return;
  }

  if constexpr (std::is_same<ptrA, ptrM>::value)
  {
    ma::geam(TA, TA, M.size(1), M.size(0), Type2(1.0), pointer_dispatch(A.origin()), A.stride(0), Type2(0.0),
         pointer_dispatch(A.origin()), A.stride(0), pointer_dispatch(M.origin()), M.stride(0), select_backend<MA>());
  } 
  else 
  {
    if (TA == 'N')
    {
      for (int i = 0; i < M.size(0); i++)
        for (int j = 0; j < M.size(1); j++)
          M[i][j] = A[i][j];
    }
    else if (TA == 'T')
    {
      for (int i = 0; i < M.size(0); i++)
        for (int j = 0; j < M.size(1); j++)
          M[i][j] = A[j][i];
    }
    else if (TA == 'C')
    {
      for (int i = 0; i < M.size(0); i++)
        for (int j = 0; j < M.size(1); j++)
          M[i][j] = ma::conj(A[j][i]);
    }
  }
}

template<class MA,
         class MultiArray2D,
         typename = std::enable_if_t<(std::decay<MA>::type::dimensionality == 2)>,
         typename = std::enable_if_t<(MultiArray2D::dimensionality == 2)>,
         typename = void>
void Matrix2MAREF(char TA, MA const& A, MultiArray2D& M)
{
  using Type2 = typename MultiArray2D::element;
  RUNTIME_CHECK(TA == 'N' || TA == 'H' || TA == 'T' || TA == 'Z', "");
  if (TA == 'N' || TA == 'Z')
  {
    if (M.size(0) != A.size(0) or M.size(1) != A.size(1))
      throw std::runtime_error(" Error: Wrong dimensions in Matrix2MAREF.");
  }
  else if (TA == 'T' || TA == 'H')
  {
    if (M.size(0) != A.size(1) or M.size(1) != A.size(0))
      throw std::runtime_error(" Error: Wrong dimensions in Matrix2MAREF.");
  }
  else
  {
    throw std::runtime_error(" Error: Unknown operation in Matrix2MA.");
  }

  using ptrA = std::remove_cv_t<typename MA::element_ptr>;
  using ptrM = std::remove_cv_t<typename MultiArray2D::element_ptr>;

  if (TA == 'H')
    TA = 'C';
  if (TA == 'Z')
  {
    // bad i gpu's
    for (int i = 0; i < M.size(0); i++)
      for (int j = 0; j < M.size(1); j++)
        M[i][j] = ma::conj(A[i][j]);
    return;
  }

  if constexpr (std::is_same<ptrA, ptrM>::value)
  {
    ma::geam(TA, TA, M.size(1), M.size(0), Type2(1.0), pointer_dispatch(A.origin()), A.stride(0), Type2(0.0),
         pointer_dispatch(A.origin()), A.stride(0), pointer_dispatch(M.origin()), M.stride(0), select_backend<MA>());
  }
  else 
  {
    if (TA == 'N')
    {
      for (int i = 0; i < M.size(0); i++)
        for (int j = 0; j < M.size(1); j++)
          M[i][j] = A[i][j];
    }
    else if (TA == 'T')
    {
      for (int i = 0; i < M.size(0); i++)
        for (int j = 0; j < M.size(1); j++)
          M[i][j] = A[j][i];
    }
    else if (TA == 'C')
    {
      for (int i = 0; i < M.size(0); i++)
        for (int j = 0; j < M.size(1); j++)
          M[i][j] = ma::conj(A[j][i]);
    }
  }
}

template<class MA,
         class MultiArray2D,
         class Vector,
         typename = std::enable_if_t<(std::decay<MA>::type::dimensionality == 2)>,
         typename = std::enable_if_t<(MultiArray2D::dimensionality == 2)>,
         typename = void>
void Matrix2MA(char TA, MA const& A, MultiArray2D& M, Vector const& occups)
{
  using Type2 = typename MultiArray2D::element;
  RUNTIME_CHECK(TA == 'N' || TA == 'H' || TA == 'T' || TA == 'Z', "");
  if (occups.size() == 0)
    throw std::runtime_error(" Error: Empty occupation array in Matrix2MA.");
  RUNTIME_CHECK(occups.size() <= A.size(0), "");
  int nrows = occups.size();
  RUNTIME_CHECK(TA == 'N' || TA == 'H' || TA == 'T' || TA == 'Z', "");
  if (TA == 'N' || TA == 'Z')
  {
    if (M.size(0) != nrows || M.size(1) != A.size(1))
      M = MultiArray2D({nrows, A.size(1)}, M.get_allocator());
  }
  else if (TA == 'T' || TA == 'H')
  {
    if (M.size(1) != nrows || M.size(0) != A.size(1))
      M = MultiArray2D({A.size(1), nrows}, M.get_allocator());
  }
  else
    throw std::runtime_error(" Error: Unknown operation in Matrix2MA.");
  if (TA == 'H')
    TA = 'C';
  // bad i gpu's
  if (TA == 'N')
  {
    for (int i = 0; i < nrows; i++)
    {
      RUNTIME_CHECK(occups[i] >= 0 && occups[i] < A.size(0), "");
      int ik = occups[i];
      for (int j = 0; j < M.size(1); j++)
        M[i][j] = static_cast<Type2>(A[ik][j]);
    }
  }
  else if (TA == 'T')
  {
    for (int i = 0; i < nrows; i++)
    {
      RUNTIME_CHECK(occups[i] >= 0 && occups[i] < A.size(0), "");
      int ik = occups[i];
      for (int j = 0; j < M.size(1); j++)
        M[j][i] = static_cast<Type2>(A[ik][j]);
    }
  }
  else if (TA == 'C')
  {
    for (int i = 0; i < nrows; i++)
    {
      RUNTIME_CHECK(occups[i] >= 0 && occups[i] < A.size(0), "");
      int ik = occups[i];
      for (int j = 0; j < M.size(1); j++)
        M[j][i] = static_cast<Type2>(ma::conj(A[ik][j]));
    }
  }
  else if (TA == 'Z')
  {
    for (int i = 0; i < nrows; i++)
    {
      RUNTIME_CHECK(occups[i] >= 0 && occups[i] < A.size(0), "");
      int ik = occups[i];
      for (int j = 0; j < M.size(1); j++)
        M[i][j] = static_cast<Type2>(ma::conj(A[ik][j]));
    }
  }
}

} // namespace ma

#endif
