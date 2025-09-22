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

#ifndef KERNELS_TERM_BY_TERM_OPERATIONS_H
#define KERNELS_TERM_BY_TERM_OPERATIONS_H

#include <cassert>
#include <complex>

namespace kernels
{
void term_by_term_mat_vec_plus(int dim,
                               int nrow,
                               int ncol,
                               std::complex<double>* A,
                               int lda,
                               std::complex<double>* x,
                               int incx);
void term_by_term_mat_vec_minus(int dim,
                                int nrow,
                                int ncol,
                                std::complex<double>* A,
                                int lda,
                                std::complex<double>* x,
                                int incx);
void term_by_term_mat_vec_mult(int dim,
                               int nrow,
                               int ncol,
                               std::complex<double>* A,
                               int lda,
                               std::complex<double>* x,
                               int incx);
void term_by_term_mat_vec_div(int dim,
                              int nrow,
                              int ncol,
                              std::complex<double>* A,
                              int lda,
                              std::complex<double>* x,
                              int incx);
void term_by_term_mat_vec_plus(int dim, int nrow, int ncol, std::complex<double>* A, int lda, double* x, int incx);
void term_by_term_mat_vec_minus(int dim, int nrow, int ncol, std::complex<double>* A, int lda, double* x, int incx);
void term_by_term_mat_vec_mult(int dim, int nrow, int ncol, std::complex<double>* A, int lda, double* x, int incx);
void term_by_term_mat_vec_div(int dim, int nrow, int ncol, std::complex<double>* A, int lda, double* x, int incx);

} // namespace kernels

#endif
