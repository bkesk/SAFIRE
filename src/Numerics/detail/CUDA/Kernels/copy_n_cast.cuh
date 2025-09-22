////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef COPY_N_CAST_KERNELS_HPP
#define COPY_N_CAST_KERNELS_HPP

#include <stdexcept>
#include <cassert>
#include <complex>

namespace kernels
{
void copy_n_cast(double const* A, long n, float* B);
void copy_n_cast(float const* A, long n, double* B);
void copy_n_cast(std::complex<double> const* A, long n, std::complex<float>* B);
void copy_n_cast(std::complex<float> const* A, long n, std::complex<double>* B);
inline void copy_n_cast(std::complex<float> const* A, long n, std::complex<float>* B)
{
  std::cerr << " Should not be calling copy_n_cast<T,T>. \n" << std::endl;
  throw std::runtime_error("Calling cast_n_copy(std::complex<float> const*,n,std::complex<float>*)");
}
inline void copy_n_cast(std::complex<double> const* A, long n, std::complex<double>* B)
{
  std::cerr << " Should not be calling copy_n_cast<T,T>. \n" << std::endl;
  throw std::runtime_error("Calling cast_n_copy(std::complex<double> const*,n,std::complex<double>*)");
}

template<typename T1, typename T2>
void copy_n_cast_impl(int N, int M, T1 const* A, int lda, long Astride,
                      T2* B, int ldb, long Bstride, int nbatch);
template<typename T1, typename T2>
void copy_n_cast_impl(int N, int M, std::complex<T1> const* A, int lda, long Astride,
                      std::complex<T2>* B, int ldb, long Bstride, int nbatch);

} // namespace kernels

#endif
