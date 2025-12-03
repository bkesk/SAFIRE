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


#pragma once

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/sparse/csr_matrix.hpp"
#include "utilities/h5_utils.hpp"

namespace math
{
namespace sparse
{

template<MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
auto to_csr(::nda::ArrayOfRank<2> auto const& A_, double vcut = 1e-8) 
{
  using value_type = typename ::nda::get_value_t<decltype(A_)>; 
  auto A = ::nda::to_host(A_());
  long nr = A.extent(0);
  long nc = A.extent(1);
  auto nnzpr = ::nda::array<IntType, 1>::zeros({nr});

  for(int r=0; r<nr; ++r)  
    for(int c=0; c<nc; ++c) 
      if(std::abs(A(r,c)) > vcut) 
        nnzpr(r)++; 

  csr_matrix<value_type,HOST_MEMORY,IndxType,IntType> csr({nr,nc},nnzpr);

  for(int r=0; r<nr; ++r) 
    for(int c=0; c<nc; ++c)
      if(std::abs(A(r,c)) > vcut)
        csr[r][c] = A(r,c);

  if constexpr (MEM == HOST_MEMORY) {
    return csr;
  } else {
    return csr_matrix<value_type,MEM,IndxType,IntType>(csr);
  }
}

template<typename value_type, typename IndxType = int, typename IntType = long>
auto combine_csr(csr_matrix<value_type,HOST_MEMORY,IndxType,IntType> const& A, 
                 csr_matrix<value_type,HOST_MEMORY,IndxType,IntType> const& B,
                 long B_col_shoft = 0)
{
  long nr = A.extent(0) + B.extent(0);
  long nrA = A.extent(0);
  long nc = std::max(A.extent(1),B.extent(1)+B_col_shoft);
  auto nnzpr = ::nda::array<IntType, 1>::zeros({nr});

  if(A.extent(0) == 0) {
    return csr_matrix<value_type,HOST_MEMORY,IndxType,IntType>{B};
  } else if(B.extent(0) == 0) {
    return csr_matrix<value_type,HOST_MEMORY,IndxType,IntType>{A};
  } else {

    for(int r=0; r<A.extent(0); ++r)
      nnzpr(r) = A.nnz(r);
    for(int r=0; r<B.extent(0); ++r)
      nnzpr(r+nrA) = B.nnz(r);

    csr_matrix<value_type,HOST_MEMORY,IndxType,IntType> csr({nr,nc},nnzpr);

    {
      auto vals = A.values();
      auto cols = A.columns();
      auto row_begin = A.row_begin();
      auto row_end = A.row_end();
      long i0 = row_begin(0);

      for(long r=0; r<A.extent(0); r++)
        for(long i=row_begin(r); i<row_end(r); ++i)
          csr.emplace_back({r, cols(i-i0)}, vals(i-i0));
    }
    {
      auto vals = B.values();
      auto cols = B.columns();
      auto row_begin = B.row_begin();
      auto row_end = B.row_end();
      long i0 = row_begin(0);

      for(long r=0; r<B.extent(0); r++)
        for(long i=row_begin(r); i<row_end(r); ++i)
          csr.emplace_back({r+nrA, B_col_shoft+cols(i-i0)}, vals(i-i0));
    }    

    return csr;
  }
}

template<typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
auto identity(long n, long nnzpr = 1)
{ 
  csr_matrix<ValType,HOST_MEMORY,IndxType,IntType> csr({n,n},
                                                       std::min(n,std::max(nnzpr,long(1))));
  
  for(int r=0; r<n; ++r) 
    csr[r][r] = ValType(1);
  
  if constexpr (MEM == HOST_MEMORY) {
    return csr;
  } else { 
    return csr_matrix<ValType,MEM,IndxType,IntType>(csr);
  }
}

template<typename ValType, MEMORY_SPACE MEM, typename IndxType, typename IntType>
auto to_compact(csr_matrix<ValType,MEM,IndxType,IntType> const& csr)
{
  auto shape = csr.shape(); 
  
  auto nnzpr = ::nda::array<IntType, 1>::zeros({shape[0]});
  for(long r=0; r<shape[0]; ++r)  
    nnzpr(r) = csr.nnz(r);
  
  csr_matrix<ValType,HOST_MEMORY,IndxType,IntType> m({shape[0],shape[1]}, nnzpr);
  
  for(long r=0; r<shape[0]; ++r) { 
    if constexpr (MEM == HOST_MEMORY or MEM==UNIFIED_MEMORY) {
      auto vals = csr[r].values();
      auto cols = csr[r].columns();
      for(long c=0; c<nnzpr[r]; ++c) 
        m[r][cols(c)] = vals(c);
    } else {
      ::nda::array<ValType,1> vals = csr[r].values();
      ::nda::array<IndxType,1> cols = csr[r].columns();
      auto row = m[r];
      for(long c=0; c<nnzpr[r]; ++c)
        row[cols(c)] = vals(c);
    }
  }
  
  if constexpr (MEM == HOST_MEMORY) {
    return m;
  } else { 
    return csr_matrix<ValType,MEM,IndxType,IntType>(m);
  }
}


template<typename NewValType, typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
requires( std::is_assignable_v<NewValType&,ValType> )
auto to_array(csr_matrix<ValType,MEM,IndxType,IntType> const& csr, nda::range row_range, nda::range col_range)
{
  sfqmc::utils::check(row_range.first() >= 0 and row_range.first() <= csr.extent(0) and
                      row_range.last() >= row_range.first() and row_range.last() <= csr.extent(0), 
                      "to_array: row_range out of bounds.");; 
  sfqmc::utils::check(col_range.first() >= 0 and col_range.first() <= csr.extent(1) and
                      col_range.last() >= col_range.first() and col_range.last() <= csr.extent(1), 
                      "to_array: col_range out of bounds.");; 
  auto vals = ::nda::to_host(csr.values());
  auto cols = ::nda::to_host(csr.columns());
  auto row_begin = ::nda::to_host(csr.row_begin());
  auto row_end = ::nda::to_host(csr.row_end());

  long nr = row_range.size(); 
  long nc = col_range.size();
  long i0 = row_begin(0);
  long r0 = row_range.first();
  long c0 = col_range.first();
  long c1 = col_range.last();

  auto A = memory::host_array<NewValType, 2>::zeros({nr,nc});
  for(long r=row_range.first(); r<row_range.last(); r++)
    for(long i=row_begin(r); i<row_end(r); ++i)
      if( cols(i-i0) >= c0 and cols(i-i0) < c1 )
        A(r-r0,cols(i-i0)-c0) = NewValType(vals(i-i0));

  if constexpr (MEM == HOST_MEMORY) {
    return A;
  } else {
    return memory::array<MEM,NewValType,2>(A);
  }
}

template<typename NewValType, typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
requires( std::is_assignable_v<NewValType&,ValType> )
auto to_array(csr_matrix<ValType,MEM,IndxType,IntType> const& csr)
{
  return to_array<NewValType>(csr,nda::range(csr.extent(0)),nda::range(csr.extent(1)));
}

// Useful routine, does nothing
template<typename... Args>
auto to_array(nda::MemoryMatrix auto const& view, nda::range row_range, nda::range col_range)
{
  return view(row_range,col_range);
}

template<typename... Args> 
auto to_array(nda::MemoryMatrix auto const& view)
{
  return view();
}

template<typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
auto HDF2CSR(h5::group grp)
{
  using csr_host = csr_matrix<ValType,HOST_MEMORY,IndxType,IntType>;
  using csr = csr_matrix<ValType,MEM,IndxType,IntType>;

  // Need to read:
  // - dims: nrow,ncols, nnz
  // - data_
  // - jdata_
  // - pointers_begin_
  // - pointers_end_

  long nrows, ncols, nnz;
  std::vector<long> dims(3);
  h5::h5_read(grp,"dims",dims);
  sfqmc::utils::check(dims.size() == 3, "Size mismatch");
  nrows = dims[0];
  ncols = dims[1];
  nnz   = dims[2];

  nda::array<IntType,1> nnz_per_row(nrows);
  nda::array<IntType,1> ptrb(nrows), ptre(nrows);
  nda::h5_read(grp,"pointers_begin_",ptrb);
  sfqmc::utils::check(ptrb.size() == nrows, "Size mismatch");
  nda::h5_read(grp,"pointers_end_",ptre);
  sfqmc::utils::check(ptre.size() == nrows, "Size mismatch");
  for (long i = 0; i < nrows; i++)
    nnz_per_row(i) = ptre(i) - ptrb(i);

  csr_host SpM({nrows, ncols}, nnz_per_row);

  nda::array<ValType,1> data(nnz);
  nda::array<IndxType,1> jdata(nnz);
  sfqmc::utils::h5_read(grp,"data_",data);
  sfqmc::utils::check(data.size() == nnz, "Size mismatch");
  nda::h5_read(grp,"jdata_",jdata);
  sfqmc::utils::check(jdata.size() == nnz, "Size mismatch");
  long cnt = 0;
  for (long r = 0; r < nrows; r++)
  {
    for(long i=ptrb[r]; i<ptre[r]; ++i) 
      SpM.emplace_back({r, jdata(i)}, data(i));
  }

  if constexpr (MEM == HOST_MEMORY)
    return SpM;
  else
    return csr{SpM};
}

/*
 * Given a csr matrix representing an operator with CLOSED spin structure,
 * return the equivalent operator matrix with COLLINEAR spin structure. 
 */
template<typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
auto closed_to_collinear(csr_matrix<ValType,MEM,IndxType,IntType> const& A)
{
  using csr = csr_matrix<ValType,MEM,IndxType,IntType>;
  using csr_host = csr_matrix<ValType,HOST_MEMORY,IndxType,IntType>;
  sfqmc::utils::check(A.extent(0) == A.extent(1), "Shape mismatch");
  IndxType N(A.extent(0));

  auto vals = ::nda::to_host(A.values());
  auto cols = ::nda::to_host(A.columns());

  nda::array<int,1> counts(2*N);
  for(IndxType r=0; r<N; r++)
    counts[r] = counts[r+N] = A.nnz(r);

  csr_host B({2*N,N}, counts);

  for(IndxType r=0; r<N; r++) {
    for(long i=A.row_begin(r); i<A.row_end(r); ++i) {
      B.emplace_back({r, cols(i)}, vals(i));
      B.emplace_back({r+N, cols(i)}, vals(i));
    }
  }

  if constexpr (MEM == HOST_MEMORY) {
    return B;
  } else {
    return csr(B); 
  }
}

/*
 * Given a csr matrix representing an operator with CLOSED spin structure,
 * return the equivalent operator matrix with NONCOLLINEAR spin structure. 
 */
template<typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
auto closed_to_noncollinear(csr_matrix<ValType,MEM,IndxType,IntType> const& A)
{
  using csr = csr_matrix<ValType,MEM,IndxType,IntType>;
  using csr_host = csr_matrix<ValType,HOST_MEMORY,IndxType,IntType>;
  sfqmc::utils::check(A.extent(0) == A.extent(1), "Shape mismatch");
  IndxType N(A.extent(0));

  auto vals = ::nda::to_host(A.values());
  auto cols = ::nda::to_host(A.columns());

  nda::array<int,1> counts(2*N);
  for(IndxType r=0; r<N; r++)
    counts[r] = counts[r+N] = A.nnz(r);

  csr_host B({2*N,2*N}, counts);

  for(IndxType r=0; r<N; r++) {
    for(long i=A.row_begin(r); i<A.row_end(r); ++i) { 
      B.emplace_back({r, cols(i)}, vals(i));
      B.emplace_back({r+N, cols(i)+N}, vals(i));
    }
  }

  if constexpr (MEM == HOST_MEMORY) {
    return B;
  } else {
    return csr(B);
  }
}

/*
 * Given a csr matrix representing an operator with COLLINEAR spin structure,
 * return the equivalent operator matrix with NONCOLLINEAR spin structure. 
 */
template<typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
auto collinear_to_noncollinear(csr_matrix<ValType,MEM,IndxType,IntType> const& A)
{
  using csr = csr_matrix<ValType,MEM,IndxType,IntType>;
  using csr_host = csr_matrix<ValType,HOST_MEMORY,IndxType,IntType>;
  sfqmc::utils::check(A.extent(0) == 2*A.extent(1), "Shape mismatch");
  IndxType N(A.extent(1));

  auto vals = ::nda::to_host(A.values());
  auto cols = ::nda::to_host(A.columns());

  nda::array<int,1> counts(2*N);
  for(IndxType r=0; r<2*N; r++)
    counts[r] = A.nnz(r);

  csr_host B({2*N,2*N}, counts);

  for(IndxType r=0; r<N; r++) 
    for(long i=A.row_begin(r); i<A.row_end(r); ++i) 
      B.emplace_back({r, cols(i)}, vals(i));
  for(IndxType r=N; r<2*N; r++) 
    for(long i=A.row_begin(r); i<A.row_end(r); ++i) 
      B.emplace_back({r, cols(i)+N}, vals(i));

  if constexpr (MEM == HOST_MEMORY) {
    return B;
  } else {
    return csr(B);
  }
}

template<class T>
void accumulate(T a, CSRMatrix auto const& X, CSRMatrix auto&& Y)
{
  using value_type = typename std::decay_t<decltype(Y)>::value_type;
  using index_type = typename std::decay_t<decltype(Y)>::index_type;
  sfqmc::utils::check((X.extent(0) <= Y.extent(0)) and (X.extent(1) <= Y.extent(1)), "Shape mismatch"); 
  auto nr = X.extent(0);
  auto vals = X.values();
  auto cols = X.columns();
  for(long r=0; r<nr; ++r) 
    for(long i=X.row_begin(r); i<X.row_end(r); ++i) 
      Y.add( {index_type(r), index_type(cols(i))}, value_type(a*vals(i)) );
} 

/*
 * Linealizes a matrix. 
 * For a sparse matrix of size {N,M}, a new matrix is generated with size: {1,N*M}
 * where:  A(i,j) -> B(0, i*M+j) 
 * The resulting csr matrix is compact.
 */
auto linearize_matrix(CSRMatrix auto const& A)
{
  using csr = std::decay_t<decltype(A)>;
  using index_type = typename csr::index_type;
  auto nr=A.extent(0);
  auto nc=A.extent(1);

  typename csr::regular_type B({1, nr*nc}, A.nnz());

  for(index_type r=0; r<index_type(nr); r++) {
    for(long n=A.row_begin(r); n<A.row_end(r); ++n) {
      index_type c_( r*index_type(nc) + A.columns(n) );
      B.emplace_back( {0,  c_}, A.values(n) );
    }
  }

  return B;
}

} // sparse
} // math

