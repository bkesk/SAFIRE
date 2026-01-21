// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `geqp3` routine.
 */

#pragma once

#include "nda/lapack/interface/cxx_interface.hpp"
#include "nda/basic_array.hpp"
#include "nda/concepts.hpp"
#include "nda/declarations.hpp"
#include "nda/exceptions.hpp"
#include "nda/macros.hpp"
#include "nda/mem/address_space.hpp"
#include "nda/traits.hpp"

#include "nda/basic_functions.hpp"
#include "nda/layout/policies.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>
#include <utility>

namespace nda 
{

namespace lapack 
{

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `geqp3` routine.
   *
   * @details Computes a QR factorization with column pivoting of a matrix \f$ \mathbf{A} \f$:
   * \f[
   *   \mathbf{A P} = \mathbf{Q R}
   * \f]
   * using Level 3 BLAS.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam JPVT nda::MemoryVector type.
   * @tparam TAU nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the m-by-n matrix \f$ \mathbf{A} \f$. On exit, the upper triangle of the
   * array contains the `min(m,n)`-by-n upper trapezoidal matrix \f$ \mathbf{R} \f$; the elements below the diagonal,
   * together with the array `tau`, represent the unitary matrix \f$ \mathbf{Q} \f$ as a product of `min(m,n)`
   * elementary reflectors.
   * @param jpvt Input/output vector. On entry, if `jpvt(j) != 0`, the j-th column of \f$ \mathbf{A} \f$ is permuted to
   * the front of \f$ \mathbf{A P} \f$ (a leading column); if `jpvt(j) == 0`, the j-th column of \f$ \mathbf{A} \f$ is a
   * free column. On exit, if `jpvt(j) == k`, then the j-th column of \f$ \mathbf{A P} \f$ was the the k-th column of
   * \f$ \mathbf{A} \f$.
   * @param tau Output vector. The scalar factors of the elementary reflectors.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector JPVT, MemoryVector TAU, MemoryVector W>
    requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU, W>
             and mem::have_compatible_addr_space<A, JPVT, TAU, W>)
  int geqp3(A &&a, JPVT &&jpvt, TAU &&tau, W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::geqp3: C order not supported");
    static_assert(std::is_same_v<get_value_t<JPVT>, int>, "Error in nda::lapack::geqp3: Pivoting array must have elements of type int");
    static_assert(mem::have_host_compatible_addr_space<A, JPVT, TAU>, "Error in nda::lapack::geqp3: Only CPU is supported");

    auto [m, n] = a.shape();
    EXPECTS(tau.size() >= std::min(m, n));

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(jpvt.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffersize
    using value_type = get_value_t<A>;
    value_type bufferSize_T{};
    int info = 0;
    array<remove_complex_t<value_type>, 1> rwork(2 * n);
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), &bufferSize_T, -1, rwork.data(), info);
    int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

    // allocate work buffer and perform actual library call
    if (work.size() < bufferSize) work.resize(bufferSize);
    EXPECTS(work.indexmap().min_stride() == 1);
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), work.data(), bufferSize, rwork.data(), info);
    jpvt -= 1; // Shift to 0-based indexing

    if (info) NDA_RUNTIME_ERROR << "Error in nda::lapack::geqp3: info = " << info;
    return info;
  }

  template <MemoryMatrix A, MemoryVector JPVT, MemoryVector TAU>
    requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU>
             and mem::have_compatible_addr_space<A, JPVT, TAU>)
  int geqp3(A &&a, JPVT &&jpvt, TAU &&tau) { // NOLINT (temporary views are allowed here)
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return geqp3(std::forward<A>(a), std::forward<JPVT>(jpvt), std::forward<TAU>(tau), work);
  }

  namespace detail {

    template <nda::MemoryArrayOfRank<3> A, MemoryMatrix JPVT, MemoryMatrix TAU, MemoryVector W>
    requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU, W>
             and mem::have_compatible_addr_space<A, JPVT, TAU, W>)    
    auto geqp3_impl(A &&a, JPVT &&jpvt, TAU &&tau, W &&work){
      
      EXPECTS(tau.size() >= a.extent(0));

      auto batchSize = a.extent(2);
      array<int, 1> info(batchSize, 0);

      // must be lapack compatible
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(a.extent(0) == a.extent(1));
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(jpvt.indexmap().min_stride() == 1);
      EXPECTS(tau.indexmap().min_stride() == 1);

      //get the optimal buffersize
      using value_type = get_value_t<A>;
      value_type bufferSize_T{};

      auto a0 = a(range::all, range::all, 0);
      array<remove_complex_t<value_type>, 1> rwork(2 * a.extent(1));
      lapack::f77::geqp3(a0.extent(0), a0.extent(1), a0.data(), get_ld(a0), jpvt.data(), tau.data(), &bufferSize_T, -1, rwork.data(), info(0));
      int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

      if (work.size() < bufferSize) work.resize(bufferSize);
      for (int i = 0; i < batchSize; ++i) {
        auto a_b = a(range::all, range::all, i);
        lapack::f77::geqp3(a.extent(0), a.extent(1), a_b.data(), get_ld(a_b), jpvt.data() + i * get_ld(jpvt), 
                  tau.data() + i * get_ld(tau), work.data(), bufferSize, rwork.data(), info(i));
      }
      return info;
    }

  }

  template <nda::MemoryArrayOfRank<3> A, MemoryMatrix JPVT, MemoryMatrix TAU, MemoryVector W>
  requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU, W>
             and mem::have_compatible_addr_space<A, JPVT, TAU, W>)   
  auto geqp3(A && a, JPVT && jpvt, TAU && tau, W && work){
    return detail::geqp3_impl(std::forward<A>(a),jpvt,tau,work);
  }

} // namespace lapack

namespace linalg
{
  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Get the permutation vector \f$ \mathbf{\sigma} \f$ from the pivot indices returned by nda::lapack::getrf or
   * other LAPACK routines.
   *
   * @details The function constructs the permutation vector \f$ \mathbf{\sigma} \f$ of size \f$ m \f$ from the pivot
   * index vector `ipiv` of size \f$ l \f$. Starting from an identity permutation, i.e. \f$ \mathbf{\sigma} = (0, 1,
   * \ldots, m - 1) \f$, it interchanges \f$ \sigma_i \f$ with \f$ \sigma_{\mathrm{ipiv}_i - 1} \f$ for all \f$ i = 0,
   * 1, \dots, l - 1 \f$.
   *
   * @param ipiv nda::Vector containing the pivot indices returned by nda::lapack::getrf.
   * @param m Number of elements of the permutation vector \f$ \mathbf{\sigma} \f$.
   * @return Permutation vector \f$ \mathbf{\sigma} \f$ as an nda::vector.
   */
  auto get_permutation_vector(Vector auto const &ipiv, int m) {
    static_assert(nda::mem::have_host_compatible_addr_space<decltype(ipiv)>);
    EXPECTS(m >= ipiv.size());
    auto sigma = vector<int>{arange<int>(m)};
    for (int i = 0; i < ipiv.size(); ++i) std::swap(sigma(i), sigma(ipiv(i) - 1));
    return sigma;
  }

  /**
   * @brief Get the permutation matrix \f$ \mathbf{P} \f$ from a permutation vector \f$ \mathbf{\sigma} \f$.
   *
   * @details The function constructs the permutation matrix \f$ \mathbf{P} \f$ of size \f$ m \times m \f$ from the
   * permutation vector \f$ \sigma \f$ of size \f$ m \f$. 
   * 
   * The permutation matrix only has the following non-zero elements (for all \f$ i = 0, 1, \ldots, m - 1 \f$):
   * - \f$ \mathbf{P}_{i, \sigma_i} = 1 \f$ for row permutations,
   * - \f$ \mathbf{P}_{\sigma_i, i} = 1 \f$ for column permutations.
   *
   * @tparam T nda::Scalar value type of the permutation matrix.
   * @tparam LP Policy determining the memory layout of the permutation matrix.
   * @param sigma nda::Vector containing the permutation vector with values \f$ \in \{0, 1, \ldots, m - 1 \} \f$.
   * @param column_permutations If true, constructs the permutation matrix for column permutations.
   * @return Permutation matrix \f$ \mathbf{P} \f$ as an nda::matrix.
   */
  template <Scalar T, typename LP = F_layout>
  auto get_permutation_matrix(Vector auto const &sigma, bool column_permutations = false) {
    static_assert(nda::mem::have_host_compatible_addr_space<decltype(sigma)>);
    int m  = sigma.size();
    auto P = matrix<T, LP>::zeros(m, m);
    for (int i = 0; i < m; ++i){ 
      (column_permutations ? P(sigma(i), i) : P(i, sigma(i))) = T{1};
    }
    return P;
  }

  /**
   * @brief Get the permutation matrix \f$ \mathbf{P} \f$ from the pivot indices returned by nda::lapack::getrf or other
   * LAPACK routines.
   *
   * @details It simply calls nda::linalg::get_permutation_vector to get the permutation vector from the pivot indices,
   * and then calls nda::linalg::get_permutation_matrix to get the permutation matrix.
   *
   * @tparam T nda::Scalar value type of the permutation matrix.
   * @tparam LP Policy determining the memory layout of the permutation matrix.
   * @param ipiv nda::Vector containing the pivot indices returned by nda::lapack::getrf.
   * @param m Number of rows/columns of the square permutation matrix \f$ \mathbf{P} \f$.
   * @return Permutation matrix \f$ \mathbf{P} \f$ as an nda::matrix.
   */
  template <Scalar T, typename LP = F_layout>
  auto get_permutation_matrix(Vector auto const &ipiv, int m) {
    return get_permutation_matrix<T, LP>(get_permutation_vector(ipiv, m));
  }

  template <Scalar T, typename LP = F_layout>
  auto get_permutation_array(Matrix auto const &ipiv) {
    static_assert(nda::mem::have_host_compatible_addr_space<decltype(ipiv)>);
    auto [m,nbatch]  = ipiv.shape();
    int dm = m;
    auto P = nda::array<T, 3, LP>::zeros(m, m, nbatch);
    for (int i = 0; i < nbatch; ++i) {
      P(nda::range::all,nda::range::all,i) = get_permutation_matrix<T,LP>(ipiv(nda::range::all,i),dm);
    }
    return P;
  }

    /*
      Routine to get Q, R matrices from QR decomposition
    */
    template <nda::MemoryArrayOfRank<3> A, nda::MemoryMatrix TAU>
      requires(nda::mem::have_host_compatible_addr_space<A, TAU> and nda::have_same_value_type_v<A, TAU> 
              and nda::is_blas_lapack_v<nda::get_value_t<A>>)
    auto get_qr_matrices(A const &a, TAU const &tau, bool complete = false) {

      constexpr MEMORY_SPACE MEM = memory::get_memory_space<A>();

      auto all = nda::range::all;

      auto const [m, n, nbatch] = a.shape();
      auto const min_mn = std::min(m, n);
      auto const k      = (complete ? m : min_mn);
      auto Q            = memory::buffered_array<MEM, nda::get_value_t<A>, 3, nda::F_layout>::zeros(m, k, nbatch);
      auto R            = memory::buffered_array<MEM, nda::get_value_t<A>, 3, nda::F_layout>::zeros(k, n, nbatch);

      // compute Q matrix
      for(int b = 0; b < nbatch; ++b){
        Q(all, nda::range(min_mn),b) = a(all, nda::range(min_mn), b);
        int info{};
        if constexpr (nda::is_complex_v<nda::get_value_t<A>>) {
          info = nda::lapack::ungqr(Q(all,all,b), tau(all,b));
        } else {
          info = nda::lapack::orgqr(Q(all,all,b), tau(all,b));
        }
        if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::qr_in_place: orgqr/ungqr returned a non-zero value: info = " << info;

        // extract R matrix
        for (int i = 0; i < min_mn; ++i) R(nda::range(i + 1), i, b) = a(nda::range(i + 1), i, b);
        for (int i = min_mn; i < n; ++i) R(all, i, b) = a(all, i, b);
      }
      return std::make_tuple(Q, R);

    }

} // namespace linalg

} // namespace nda
