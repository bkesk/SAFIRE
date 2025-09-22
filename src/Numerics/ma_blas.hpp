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

#ifndef MA_BLAS_HPP
#define MA_BLAS_HPP

#include "Numerics/detail/blas.hpp"
#include <utility> //std::enable_if
#include <cassert>
#include <iostream>

#include "Numerics/ma_blas_extensions.hpp"

namespace ma
{

// no tag dispatching for copy/copy2D. We want mixed host/device pointers!
// still needs pointer_dispatch to raw_cast shm pointers.
template<class MultiArray1DX,
         class MultiArray1DY,
         typename = typename std::enable_if_t<std::decay<MultiArray1DX>::type::dimensionality == 1>,
         typename = typename std::enable_if_t<std::decay<MultiArray1DY>::type::dimensionality == 1>>
MultiArray1DY&& copy(MultiArray1DX&& x, MultiArray1DY&& y)
{
  RUNTIME_CHECK(x.num_elements() == y.num_elements(), "");
  ma::copy(x.size(), pointer_dispatch(x.origin()), x.stride(0), pointer_dispatch(y.origin()), y.stride(0));
  return std::forward<MultiArray1DY>(y);
}

template<class MultiArray2DX,
         class MultiArray2DY,
         typename = typename std::enable_if_t<std::decay<MultiArray2DX>::type::dimensionality == 2>,
         typename = typename std::enable_if_t<std::decay<MultiArray2DY>::type::dimensionality == 2>,
         typename = void>
MultiArray2DY&& copy(MultiArray2DX&& x, MultiArray2DY&& y)
{
  RUNTIME_CHECK(x.stride(1) == 1, "");
  RUNTIME_CHECK(y.stride(1) == 1, "");
  RUNTIME_CHECK(x.size(0) == y.size(0), "");
  RUNTIME_CHECK(x.size(1) == y.size(1), "");
  if ((x.stride(0) == x.size(1)) && (y.stride(0) == y.size(1)))
    ma::copy(x.num_elements(), pointer_dispatch(x.origin()), 1, pointer_dispatch(y.origin()), 1); 
  else
    ma::copy2D(x.size(0), x.size(1), pointer_dispatch(x.origin()), x.stride(0), pointer_dispatch(y.origin()), y.stride(0));
  return std::forward<MultiArray2DY>(y);
}

template<class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 3>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 3>,
         typename = void,
         typename = void
        >
MA2&& copy(MA1&& A, MA2&& B)
{
  RUNTIME_CHECK(A.size(0) == B.size(0), "");
  RUNTIME_CHECK(A.size(1) == B.size(1), "");
  RUNTIME_CHECK(A.size(2) == B.size(2), "");
  RUNTIME_CHECK(A.stride(2) == 1, "");
  RUNTIME_CHECK(B.stride(2) == 1, "");
  ma::copy_n_cast_impl(A.size(1), A.size(2),
              pointer_dispatch(A.origin()), A.stride(1), A.stride(0),
              pointer_dispatch(B.origin()), B.stride(1), B.stride(0), A.size(0),
       	      select_backend<MA1>());	
  return std::forward<MA2>(B);
}

// copy matrix into strided tensor
// B[n,i,j] = A[i,j] for all n
template<class MA1,
         class MA2,
         typename = std::enable_if_t<std::decay_t<MA1>::dimensionality == 2>,
         typename = std::enable_if_t<std::decay_t<MA2>::dimensionality == 3>,
         typename = void,
         typename = void,
         typename = void
        >
MA2&& copy(MA1&& A, MA2&& B)
{
  RUNTIME_CHECK(A.size(0) == B.size(1), "");
  RUNTIME_CHECK(A.size(1) == B.size(2), "");
  RUNTIME_CHECK(A.stride(1) == 1, "");
  RUNTIME_CHECK(B.stride(2) == 1, "");
  ma::copy_n_cast_impl(B.size(1), B.size(2),
              pointer_dispatch(A.origin()), A.stride(0), 0,
              pointer_dispatch(B.origin()), B.stride(1), B.stride(0), B.size(0),
       	      select_backend<MA1>());	
  return std::forward<MA2>(B);
}

template<class MultiArray1Dx,
         class MultiArray1Dy,
         typename = typename std::enable_if<std::decay<MultiArray1Dx>::type::dimensionality == 1>::type,
         typename = typename std::enable_if<std::decay<MultiArray1Dy>::type::dimensionality == 1>::type>
//auto
typename std::decay<MultiArray1Dx>::type::element dot(MultiArray1Dx&& x, MultiArray1Dy&& y)
{
  RUNTIME_CHECK(x.size() == y.size(), "");
  return ma::dot(x.size(), x.origin(), x.stride(0), y.origin(), y.stride(0), 
		select_backend<MultiArray1Dx>());
}

template<class T,
         class MultiArray1D,
         typename = typename std::enable_if<std::decay<MultiArray1D>::type::dimensionality == 1>::type>
MultiArray1D&& scal(T a, MultiArray1D&& x)
{
  ma::scal(x.size(), a, x.origin(), x.stride(0),
	   select_backend<MultiArray1D>());
  return std::forward<MultiArray1D>(x);
}

template<class T,
         class MultiArrayND,
         typename = typename std::enable_if<(std::decay<MultiArrayND>::type::dimensionality > 1)>::type,
         typename = void // TODO change to use dispatch
         >
MultiArrayND&& scal(T a, MultiArrayND&& x)
{
  long sz(x.size(0));
  for (int i = 1; i < int(std::decay<MultiArrayND>::type::dimensionality); ++i)
    sz *= x.size(i);
  RUNTIME_CHECK(x.num_elements() == sz, "");
  RUNTIME_CHECK(x.stride(std::decay<MultiArrayND>::type::dimensionality - 1) == 1, ""); // only on contiguous arrays
  ma::scal(x.num_elements(), a, x.origin(), 1,
	   select_backend<MultiArrayND>());
  return std::forward<MultiArrayND>(x);
}

template<class T, class MultiArray1D>
auto operator*=(MultiArray1D&& x, T a) -> decltype(scal(a, std::forward<MultiArray1D>(x)))
{
  return scal(a, std::forward<MultiArray1D>(x));
}

template<class T, class MultiArray1D,
         typename = typename std::enable_if_t<std::decay_t<MultiArray1D>::dimensionality == 1>
        >
MultiArray1D add_scalar(T a, MultiArray1D&& A)
{
  ma::add_scalar_impl(A.size(0), 1, a, pointer_dispatch(A.origin()), A.stride(0),
	              select_backend<MultiArray1D>()); 
  return std::forward<MultiArray1D>(A);
}

template<class T, class MultiArray2D,
         typename = typename std::enable_if_t<std::decay_t<MultiArray2D>::dimensionality == 2>,
         typename = void
        >
MultiArray2D add_scalar(T a, MultiArray2D&& A)
{
  ma::add_scalar_impl(A.size(0), A.size(1), a, pointer_dispatch(A.origin()), A.stride(0),
           select_backend<MultiArray2D>());
  return std::forward<MultiArray2D>(A);
}

template<class T,
         class MultiArray1DA,
         class MultiArray1DB,
         typename = typename std::enable_if<std::decay<MultiArray1DA>::type::dimensionality == 1 and
                                            std::decay<MultiArray1DB>::type::dimensionality == 1>::type>
MultiArray1DB&& axpy(T x, MultiArray1DA const& a, MultiArray1DB&& b)
{
  RUNTIME_CHECK(a.size(0) == b.size(0), "");
  ma::axpy(a.size(0), x, a.origin(), a.stride(0), b.origin(), b.stride(0),
           select_backend<MultiArray1DA>()); 
  return std::forward<MultiArray1DB>(b);
}

template<class T,
         class MultiArray2DA,
         class MultiArray2DB,
         typename = typename std::enable_if<std::decay<MultiArray2DA>::type::dimensionality == 2 and
                                            std::decay<MultiArray2DB>::type::dimensionality == 2>::type,
         typename = void // TODO change to use dispatch
         >
MultiArray2DB&& axpy(T x, MultiArray2DA const& a, MultiArray2DB&& b)
{
  RUNTIME_CHECK(a.num_elements() == b.num_elements(), "");
  RUNTIME_CHECK(a.stride(0) == a.size(1), "a is not Contiguous"); // only on contiguous arrays
  RUNTIME_CHECK(a.stride(1) == 1, "a is not Contiguous");         // only on contiguous arrays
  RUNTIME_CHECK(b.stride(0) == b.size(1), "b is not Contiguous"); // only on contiguous arrays
  RUNTIME_CHECK(b.stride(1) == 1, "b is not Contiguous");         // only on contiguous arrays
  ma::axpy(a.num_elements(), x, a.origin(), 1, b.origin(), 1,
           select_backend<MultiArray2DA>()); 
  return std::forward<MultiArray2DB>(b);
}

template<
    char IN,
    class T,
    class MultiArray2DA,
    class MultiArray1DX,
    class MultiArray1DY,
    typename = typename std::enable_if<MultiArray2DA::dimensionality == 2 and MultiArray1DX::dimensionality == 1 and
                                       std::decay<MultiArray1DY>::type::dimensionality == 1>::type>
MultiArray1DY&& gemv(T alpha, MultiArray2DA const& A, MultiArray1DX const& x, T beta, MultiArray1DY&& y)
{
  RUNTIME_CHECK((IN == 'N') || (IN == 'T') || (IN == 'C'), "");
  if (IN == 'T' or IN == 'C')
    RUNTIME_CHECK(x.size(0) == A.size(1) and y.size(0) == A.size(0), "");
  else if (IN == 'N')
    RUNTIME_CHECK(x.size(0) == A.size(0) and y.size(0) == A.size(1), "");
  RUNTIME_CHECK(A.stride(1) == 1, "gemv is not implemented for arrays with non-leading stride != 1"); // gemv is not implemented for arrays with non-leading stride != 1
  int M = A.size(1);
  int N = A.size(0);
  ma::gemv(IN, M, N, alpha, A.origin(), A.stride(0), x.origin(), x.stride(0), beta,
       			    y.origin(), y.stride(0), 
			    select_backend<MultiArray2DA>()); 
  return std::forward<MultiArray1DY>(y);
} //y := alpha*A*x + beta*y,

template<char IN, class MultiArray2DA, class MultiArray1DX, class MultiArray1DY>
MultiArray1DY&& gemv(MultiArray2DA const& A, MultiArray1DX const& x, MultiArray1DY&& y)
{
  return gemv<IN>(1., A, x, 0., std::forward<MultiArray1DY>(y));
} //y := alpha*A*x

//	gemm<'T', 'T'>(1., A, B, 0., C); // C = T(A*B) = T(B)*T(A) or T(C) = A*B
//	gemm<'N', 'N'>(1., A, B, 0., C); // C = B*A = T(T(A)*T(B)) or T(C) = T(A)*T(B)
//	gemm<'T', 'N'>(1., A, B, 0., C); // C = T(A*T(B)) = B*T(A) or T(C) = A*T(B)
//	gemm<'N', 'T'>(1., A, B, 0., C); // C =  T(T(A)*B) = T(B)*A or T(C) = T(A)*B

template<
    char TA,
    char TB,
    class T,
    class MultiArray2DA,
    class MultiArray2DB,
    class MultiArray2DC,
    typename = typename std::enable_if<MultiArray2DA::dimensionality == 2 and MultiArray2DB::dimensionality == 2 and
                                       std::decay<MultiArray2DC>::type::dimensionality == 2>::type>
MultiArray2DC&& gemm(T alpha, MultiArray2DA const& a, MultiArray2DB const& b, T beta, MultiArray2DC&& c)
{
  RUNTIME_CHECK(a.stride(1) == 1, "");
  RUNTIME_CHECK(b.stride(1) == 1, "");
  RUNTIME_CHECK(c.stride(1) == 1, "");
  RUNTIME_CHECK((TA == 'N') || (TA == 'T') || (TA == 'C'), "");
  RUNTIME_CHECK((TB == 'N') || (TB == 'T') || (TB == 'C'), "");
  int M = -1;
  int N = -1;
  int K = -1;
  if (TA == 'N' and TB == 'N')
  {
    M = a.size(1);
    N = b.size(0);
    K = a.size(0);
    RUNTIME_CHECK(a.size(0) == b.size(1) and c.size(0) == b.size(0) and c.size(1) == a.size(1), "");
  }
  if ((TA == 'T' or TA == 'C') and (TB == 'T' or TB == 'C'))
  {
    M = a.size(0);
    N = b.size(1);
    K = a.size(1);
    RUNTIME_CHECK(a.size(1) == b.size(0) and c.size(0) == b.size(1) and c.size(1) == a.size(0), "");
  }
  if ((TA == 'T' or TA == 'C') and TB == 'N')
  {
    M = a.size(0);
    N = b.size(0);
    K = a.size(1);
    RUNTIME_CHECK(a.size(1) == b.size(1) and c.size(0) == b.size(0) and c.size(1) == a.size(0), "");
  }
  if (TA == 'N' and (TB == 'T' or TB == 'C'))
  {
    M = a.size(1);
    N = b.size(1);
    K = a.size(0);
    RUNTIME_CHECK(a.size(0) == b.size(0) and c.size(0) == b.size(1) and c.size(1) == a.size(1), "");
  }
  ma::gemm(TA, TB, M, N, K, alpha, a.origin(), a.stride(0), b.origin(), b.stride(0),
       			    beta, c.origin(), c.stride(0), 
                            select_backend<MultiArray2DA>());
  return std::forward<MultiArray2DC>(c);
}

// Expect: A[nbatch][nrow][ncol]
// Expect: B[nbatch][nrow][ncol]
// Expect: C[nbatch][nrow][ncol]
template<
    char TA,
    char TB,
    class T,
    class MultiArray3DA,
    class MultiArray3DB,
    class MultiArray3DC,
    typename = typename std::enable_if<MultiArray3DA::dimensionality == 3 and 
                                       MultiArray3DB::dimensionality == 3 and
                                       std::decay<MultiArray3DC>::type::dimensionality == 3>::type>
MultiArray3DC&& gemmStridedBatched(T alpha, MultiArray3DA const& a, MultiArray3DB const& b, T beta, MultiArray3DC&& c)
{
  RUNTIME_CHECK(a.stride(2) == 1, "");
  RUNTIME_CHECK(b.stride(2) == 1, "");
  RUNTIME_CHECK(c.stride(2) == 1, "");
  RUNTIME_CHECK(a.size(0) == b.size(0), "");
  RUNTIME_CHECK(a.size(0) == c.size(0), "");
  RUNTIME_CHECK((TA == 'N') || (TA == 'T') || (TA == 'C'), "");
  RUNTIME_CHECK((TB == 'N') || (TB == 'T') || (TB == 'C'), "");
  int M = -1;
  int N = -1;
  int K = -1;
  if (TA == 'N' and TB == 'N')
  {
    M = a.size(2);
    N = b.size(1);
    K = a.size(1);
    RUNTIME_CHECK(a.size(1) == b.size(2) and c.size(1) == b.size(1) and c.size(2) == a.size(2), "");
  }
  if ((TA == 'T' or TA == 'C') and (TB == 'T' or TB == 'C'))
  {
    M = a.size(1);
    N = b.size(2);
    K = a.size(2);
    RUNTIME_CHECK(a.size(2) == b.size(1) and c.size(1) == b.size(2) and c.size(2) == a.size(1), "");
  }
  if ((TA == 'T' or TA == 'C') and TB == 'N')
  {
    M = a.size(1);
    N = b.size(1);
    K = a.size(2);
    RUNTIME_CHECK(a.size(2) == b.size(2) and c.size(1) == b.size(1) and c.size(2) == a.size(1), "");
  }
  if (TA == 'N' and (TB == 'T' or TB == 'C'))
  {
    M = a.size(2);
    N = b.size(2);
    K = a.size(1);
    RUNTIME_CHECK(a.size(1) == b.size(1) and c.size(1) == b.size(2) and c.size(2) == a.size(2), "");
  }
  ma::gemmStridedBatched(TA, TB, M, N, K, alpha, pointer_dispatch(a.origin()), a.stride(1), a.stride(0),
                     pointer_dispatch(b.origin()), b.stride(1), b.stride(0), 
                     beta, pointer_dispatch(c.origin()),
                     c.stride(1), c.stride(0), c.size(0),
                     select_backend<MultiArray3DA>()); 
  return std::forward<MultiArray3DC>(c);
}

// Expect: A[nrow][ncol]
// Expect: B[nbatch][nrow][ncol]
// Expect: C[nbatch][nrow][ncol]
template<
    char TA,
    char TB,
    class T,
    class MultiArray2DA,
    class MultiArray3DB,
    class MultiArray3DC,
    typename = typename std::enable_if_t<MultiArray2DA::dimensionality == 2 and 
                                         MultiArray3DB::dimensionality == 3 and
                                         std::decay_t<MultiArray3DC>::dimensionality == 3>,
    typename = void
        >
MultiArray3DC&& gemmStridedBatched(T alpha, MultiArray2DA const& a, MultiArray3DB const& b, T beta, MultiArray3DC&& c)
{
  RUNTIME_CHECK(a.stride(1) == 1, "");
  RUNTIME_CHECK(b.stride(2) == 1, "");
  RUNTIME_CHECK(c.stride(2) == 1, "");
  RUNTIME_CHECK(b.size(0) == c.size(0), "");
  RUNTIME_CHECK((TA == 'N') || (TA == 'T') || (TA == 'C'), "");
  RUNTIME_CHECK((TB == 'N') || (TB == 'T') || (TB == 'C'), "");
  int M = -1;
  int N = -1;
  int K = -1;
  if (TA == 'N' and TB == 'N')
  {
    M = a.size(1);
    N = b.size(1);
    K = a.size(0);
    RUNTIME_CHECK(a.size(0) == b.size(2) and c.size(1) == b.size(1) and c.size(2) == a.size(1), "");
  }
  if ((TA == 'T' or TA == 'C') and (TB == 'T' or TB == 'C'))
  {
    M = a.size(0);
    N = b.size(2);
    K = a.size(1);
    RUNTIME_CHECK(a.size(1) == b.size(1) and c.size(1) == b.size(2) and c.size(2) == a.size(0), "");
  }
  if ((TA == 'T' or TA == 'C') and TB == 'N')
  {
    M = a.size(0);
    N = b.size(1);
    K = a.size(1);
    RUNTIME_CHECK(a.size(1) == b.size(2) and c.size(1) == b.size(1) and c.size(2) == a.size(0), "");
  }
  if (TA == 'N' and (TB == 'T' or TB == 'C'))
  {
    M = a.size(1);
    N = b.size(2);
    K = a.size(0);
    RUNTIME_CHECK(a.size(0) == b.size(1) and c.size(1) == b.size(2) and c.size(2) == a.size(1), "");
  }
  ma::gemmStridedBatched(TA, TB, M, N, K, alpha, pointer_dispatch(a.origin()), a.stride(0), 0,
                     pointer_dispatch(b.origin()), b.stride(1), b.stride(0), 
                     beta, pointer_dispatch(c.origin()),
                     c.stride(1), c.stride(0), c.size(0),
                     select_backend<MultiArray2DA>());
  return std::forward<MultiArray3DC>(c);
}

// Expect: A[nbatch][nrow][ncol]
// Expect: B[nrow][ncol]
// Expect: C[nbatch][nrow][ncol]
template<
    char TA,
    char TB,
    class T,
    class MultiArray3DA,
    class MultiArray2DB,
    class MultiArray3DC,
    typename = typename std::enable_if_t<MultiArray3DA::dimensionality == 3 and 
                                       MultiArray2DB::dimensionality == 2 and
                                       std::decay_t<MultiArray3DC>::dimensionality == 3>,
    typename = void,
    typename = void
         >
MultiArray3DC&& gemmStridedBatched(T alpha, MultiArray3DA const& a, MultiArray2DB const& b, T beta, MultiArray3DC&& c)
{
  RUNTIME_CHECK(a.stride(2) == 1, "");
  RUNTIME_CHECK(b.stride(1) == 1, "");
  RUNTIME_CHECK(c.stride(2) == 1, "");
  RUNTIME_CHECK(a.size(0) == c.size(0), "");
  RUNTIME_CHECK((TA == 'N') || (TA == 'T') || (TA == 'C'), "");
  RUNTIME_CHECK((TB == 'N') || (TB == 'T') || (TB == 'C'), "");
  int M = -1;
  int N = -1;
  int K = -1;
  if (TA == 'N' and TB == 'N')
  {
    M = a.size(2);
    N = b.size(0);
    K = a.size(1);
    RUNTIME_CHECK(a.size(1) == b.size(1) and c.size(1) == b.size(0) and c.size(2) == a.size(2), "");
  }
  if ((TA == 'T' or TA == 'C') and (TB == 'T' or TB == 'C'))
  {
    M = a.size(1);
    N = b.size(1);
    K = a.size(2);
    RUNTIME_CHECK(a.size(2) == b.size(0) and c.size(1) == b.size(1) and c.size(2) == a.size(1), "");
  }
  if ((TA == 'T' or TA == 'C') and TB == 'N')
  {
    M = a.size(1);
    N = b.size(0);
    K = a.size(2);
    RUNTIME_CHECK(a.size(2) == b.size(1) and c.size(1) == b.size(0) and c.size(2) == a.size(1), "");
  }
  if (TA == 'N' and (TB == 'T' or TB == 'C'))
  {
    M = a.size(2);
    N = b.size(1);
    K = a.size(1);
    RUNTIME_CHECK(a.size(1) == b.size(0) and c.size(1) == b.size(1) and c.size(2) == a.size(2), "");
  }
  ma::gemmStridedBatched(TA, TB, M, N, K, alpha, pointer_dispatch(a.origin()), a.stride(1), a.stride(0),
                     pointer_dispatch(b.origin()), b.stride(0), 0, 
                     beta, pointer_dispatch(c.origin()),
                     c.stride(1), c.stride(0), c.size(0),
                     select_backend<MultiArray3DA>());
  return std::forward<MultiArray3DC>(c);
}

template<char TA, char TB, class T, class MultiArray2DA, class MultiArray2DB, class MultiArray2DC>
MultiArray2DC&& gemm(MultiArray2DA const& a, MultiArray2DB const& b, MultiArray2DC&& c)
{
  return gemm(1., a, b, 0., std::forward<MultiArray2DC>(c));
}

template<
    char TA,
    char TB,
    class T,
    class MultiArray2DA,
    class MultiArray2DB,
    class MultiArray2DC,
    typename = typename std::enable_if<MultiArray2DA::dimensionality == 2 and MultiArray2DB::dimensionality == 2 and
                                       std::decay<MultiArray2DC>::type::dimensionality == 2>::type>
MultiArray2DC&& geam(T alpha, MultiArray2DA const& a, T beta, MultiArray2DB const& b, MultiArray2DC&& c)
{
  RUNTIME_CHECK(a.stride(1) == 1, "");
  RUNTIME_CHECK(b.stride(1) == 1, "");
  RUNTIME_CHECK(c.stride(1) == 1, "");
  RUNTIME_CHECK((TA == 'N') || (TA == 'T') || (TA == 'C'), "");
  RUNTIME_CHECK((TB == 'N') || (TB == 'T') || (TB == 'C'), "");
  if (TA == 'N' and TB == 'N')
  {
    RUNTIME_CHECK(a.size(0) == c.size(0) and a.size(1) == c.size(1), "");
    RUNTIME_CHECK(b.size(0) == c.size(0) and b.size(1) == c.size(1), "");
  }
  if ((TA == 'T' or TA == 'C') and (TB == 'T' or TB == 'C'))
  {
    RUNTIME_CHECK(a.size(1) == c.size(0) and a.size(0) == c.size(1), "");
    RUNTIME_CHECK(b.size(1) == c.size(0) and b.size(0) == c.size(1), "");
  }
  if ((TA == 'T' or TA == 'C') and TB == 'N')
  {
    RUNTIME_CHECK(a.size(1) == c.size(0) and a.size(0) == c.size(1), "");
    RUNTIME_CHECK(b.size(0) == c.size(0) and b.size(1) == c.size(1), "");
  }
  if (TA == 'N' and (TB == 'T' or TB == 'C'))
  {
    RUNTIME_CHECK(a.size(0) == c.size(0) and a.size(1) == c.size(1), "");
    RUNTIME_CHECK(b.size(1) == c.size(0) and b.size(0) == c.size(1), "");
  }
  ma::geam(TA, TB, c.size(1), c.size(0), alpha, pointer_dispatch(a.origin()), a.stride(0), beta,
       pointer_dispatch(b.origin()), b.stride(0), pointer_dispatch(c.origin()), c.stride(0),
       select_backend<MultiArray2DA>()); 
  return std::forward<MultiArray2DC>(c);
}

template<char TA,
         class T,
         class MultiArray2DA,
         class MultiArray2DC,
         typename = typename std::enable_if<MultiArray2DA::dimensionality == 2 and
                                            std::decay<MultiArray2DC>::type::dimensionality == 2>::type>
MultiArray2DC&& geam(T alpha, MultiArray2DA const& a, MultiArray2DC&& c)
{
  RUNTIME_CHECK(a.stride(1) == 1, "");
  RUNTIME_CHECK(c.stride(1) == 1, "");
  RUNTIME_CHECK((TA == 'N') || (TA == 'T') || (TA == 'C'), "");
  if (TA == 'N')
  {
    RUNTIME_CHECK(a.size(0) == c.size(0) and a.size(1) == c.size(1), "");
  }
  if ((TA == 'T' or TA == 'C'))
  {
    RUNTIME_CHECK(a.size(1) == c.size(0) and a.size(0) == c.size(1), "");
  }
  ma::geam(TA, TA, c.size(1), c.size(0), alpha, pointer_dispatch(a.origin()), a.stride(0), T(0),
       pointer_dispatch(a.origin()), a.stride(0), pointer_dispatch(c.origin()), c.stride(0),
       select_backend<MultiArray2DA>()); 
  return std::forward<MultiArray2DC>(c);
}

} // namespace ma

#endif
