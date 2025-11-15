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


#ifndef SPARSE_CSR_UTILS_HPP
#define SPARSE_CSR_UTILS_HPP

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/sparse/csr_matrix.hpp"

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
auto to_array(csr_matrix<ValType,MEM,IndxType,IntType> const& csr)
{
  auto vals = ::nda::to_host(csr.values());
  auto cols = ::nda::to_host(csr.columns());
  auto row_begin = ::nda::to_host(csr.row_begin());
  auto row_end = ::nda::to_host(csr.row_end());
  long nr = csr.shape(0);
  long nc = csr.shape(1);
  long i0 = row_begin(0);

  auto A = memory::host_array<NewValType, 2>::zeros({nr,nc});
  for(long r=0; r<nr; r++)
    for(long i=row_begin(r); i<row_end(r); ++i)
      A(r,cols(i-i0)) = NewValType(vals(i-i0));

  if constexpr (MEM == HOST_MEMORY) {
    return A;
  } else {
    return memory::array<MEM,NewValType,2>(A);
  }
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

  csr_host SpM(std::tuple<std::size_t, std::size_t>{nrows, ncols}, nnz_per_row);

  nda::array<ValType,1> data;
  nda::array<IndxType,1> jdata;
  nda::h5_read(grp,"data_",data);
  sfqmc::utils::check(data.size() == nnz, "Size mismatch");
  nda::h5_read(grp,"jdata_",jdata);
  sfqmc::utils::check(jdata.size() == nnz, "Size mismatch");
  long cnt = 0;
  for (long i = 0; i < nrows; i++)
  {
    long nt = ptre[i] - ptrb[i];
    for (long nn = 0; nn < nt; ++nn, ++cnt) {
      SpM.emplace_back({i, jdata(cnt)}, data(cnt));
}
  }

  if constexpr (MEM == HOST_MEMORY)
    return SpM;
  else
    return csr{SpM};
}


} // sparse
} // math

#endif

