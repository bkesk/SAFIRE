////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef CUDA_WORKAROUND_LEGACY_HARDWARE_H
#define CUDA_WORKAROUND_LEGACY_HARDWARE_H

namespace kernels
{
#if __CUDA_ARCH__ < 600
inline __device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

inline __device__ float atomicAdd(float* address, float val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __float_as_int(val + __int_as_float(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

#endif
} // namespace kernels
#endif
