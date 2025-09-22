////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef AXPY_BATCHED_GPU_KERNELS_HPP
#define AXPY_BATCHED_GPU_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void axpy_batched_gpu(int n,
                      std::complex<double>* x,
                      const std::complex<double>** a,
                      int inca,
                      std::complex<double>** b,
                      int incb,
                      int batchSize);

void sumGw_batched_gpu(int n,
                       std::complex<double>* x,
                       const std::complex<double>** a,
                       int inca,
                       std::complex<double>** b,
                       int incb,
                       int b0,
                       int bw,
                       int batchSize);

} // namespace kernels

#endif
