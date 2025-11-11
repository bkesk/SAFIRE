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
////////////////////////////////////////////////////////////////////////////////

#include "configuration.hpp"

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/sparse/sparse.hpp"

namespace math
{

namespace detail
{

template<char op_A, char op_B, 
         nda::MemoryArrayOfRank<3> A_t, 
         nda::MemoryArrayOfRank<3> B_t, 
         nda::MemoryArrayOfRank<3> C_t>
requires(
   math::is_valid_op<op_A> and math::is_valid_op<op_B> and
   nda::mem::have_compatible_addr_space<A_t,B_t,C_t> and
   nda::have_same_value_type_v<B_t,C_t> and
   ((nda::blas::is_C_layout<A_t> and nda::blas::is_C_layout<B_t> and nda::blas::is_C_layout<C_t>) or
    (nda::blas::is_F_layout<A_t> and nda::blas::is_F_layout<B_t> and nda::blas::is_F_layout<C_t>)) 
        )
auto product_impl(nda::get_value_t<A_t> alpha, A_t const& a, B_t const& b,
                  nda::get_value_t<A_t> beta, C_t && c)
{
  std::string a_str = (nda::blas::is_C_layout<A_t> ? "aik" : "ika");
  std::string b_str = (nda::blas::is_C_layout<B_t> ? "akj" : "kja");
  std::string c_str = (nda::blas::is_C_layout<C_t> ? "aij" : "ija");

  
  if constexpr (op_A == 'N') {
      if(op_B == 'N') {
      } else if constexpr (op_B == 'T') {
        nda::tensor::contract(alpha,a,"aik",b,"ajk",beta,c,"aij");
      } else {
        nda::tensor::contract(alpha,a,"aik",nda::conj(b),"ajk",beta,c,"aij");
      }
    } else {
      // A is transposed and/or conjugated
    } 
  } else
  }
  nda::tensor::contract(alpha,a,a_str,b,b_str,beta,c,c_str);
  return std::forward<C_t>(c);
}

}

/**
 * 
 * General wrapper routine for array multiplication.
 * Dispatches:
 *  C = alpha*A*B + beta*C, where:
 *    - alpha/beta: scalars (same type as A)
 *    - A: nda Array of rank 1/2/3 or csr_matrix.
 *    - B: nda Array of rank 1/2/3
 *    - C: nda Array of rank 1/2/3
 *
 * Only some combinations are allowed. This routine inspects the types at 
 * compile time and dispatches to the appropriate backend.  
 * If B and C are row-major arrays, mixed real(alpha,beta,A)/complex(B,C) is accepted.
 * All nda Arrays must have the same layout (C or F) to use this routine.
 *
 */ 
template<char op_A, char op_B, typename A_t, typename B_t, typename C_t>
requires(
   math::is_valid_op<op_A> and math::is_valid_op<op_B> and 
   (math::sparse::CSRVector<A_t> or math::sparse::CSRMatrix<A_t> or 
    nda::MemoryVector<A_t> or nda::MemoryMatrix<A_t> or nda::MemoryArrayOfRank<A_t,3>) and 
   (nda::MemoryVector<B_t> or nda::MemoryMatrix<B_t> or nda::MemoryArrayOfRank<B_t,3>) and 
   (nda::MemoryVector<C_t> or nda::MemoryMatrix<C_t> or nda::MemoryArrayOfRank<C_t,3>) and
   nda::mem::have_compatible_addr_space<A_t,B_t,C_t> and
   nda::have_same_value_type_v<B_t,C_t> 
        )
auto product(nda::get_value_t<A_t> alpha, A_t const& a, B_t const& b, 
             nda::get_value_t<A_t> beta, C_t && c) 
{
  if constexpr (nda::get_rank<B_t> == 3 ) {
    static_assert( (nda::get_rank<C_t> == 3), "Rank mismatch");

    if constexpr (nda::MemoryArrayOfRank<A_t,3>) {

      // batched gemm all nda arrays 
      auto a_v = ( op_A == 'C' or op_A == 'H' ? nda::conj(a) : a());

    } else {

      // bacthed gemm with A a matrix/csr_matrix
      if constexpr ( math::sparse::CSRMatrix<A_t> ) { 

      // deal with op_A,op_B 
//      math::sparse::csrmm(alpha,a,b,beta,c);

      } else {
      
        // can use either cutensor or blas_batched. 
        if constexpr (op_A == 'N') {
          if constexpr (op_B == 'N') {
            nda::tensor::contract(alpha,);     
          } else {
         }
         } else {
        }

      }

    }

  } else if constexpr (nda::get_rank<B_t> == 2 ) {
    static_assert( (nda::get_rank<A_t> == 2) and   
                   (nda::get_rank<C_t> == 2), "Rank mismatch");

  } else {
    static_assert( (nda::get_rank<B_t> == 1) and   
                   (nda::get_rank<C_t> == 1), "Rank mismatch");
    if constexpr ( nda::get_rank<A_t> == 2 ) {
      // matrix - vector 
    } else {
      // vector - vector 
    }
  }

  return std::forward<C_t>(c);
}

template<char op_A, char op_B, typename A_t, typename B_t, typename C_t>
requires(
   math::is_valid_op<op_A> and math::is_valid_op<op_B> and
   (math::sparse::CSRVector<A_t> or math::sparse::CSRMatrix<A_t> or
    nda::MemoryVector<A_t> or nda::MemoryMatrix<A_t> or nda::MemoryArrayOfRank<A_t,3>) and
   (nda::MemoryVector<B_t> or nda::MemoryMatrix<B_t> or nda::MemoryArrayOfRank<B_t,3>) and
   (nda::MemoryVector<C_t> or nda::MemoryMatrix<C_t> or nda::MemoryArrayOfRank<C_t,3>) and
   nda::mem::have_compatible_addr_space<A_t,B_t,C_t> and
   nda::have_same_value_type_v<B_t,C_t>
        )
auto product(A_t const& a, B_t const& b, C_t && c)
{
  return product<op_A,op_B>(nda::get_value_t<A_t>(1.0),a,b,nda::get_value_t<A_t>(0.0),std::forward<C_t>(c));
}

} // math
