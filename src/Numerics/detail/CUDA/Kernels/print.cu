////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#include <cassert>
#include <complex>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace kernels
{
template<typename T>
__global__ void kernel_print(T const* p, int n)
{
  printf("D: %d ", n);
  for (int i = 0; i < n; i++)
    printf("%g ", *(p + i));
}

template<>
__global__ void kernel_print(int const* p, int n)
{
  printf("int: %d ", n);
  for (int i = 0; i < n; i++)
    printf("%d ", *(p + i));
}

template<>
__global__ void kernel_print(long const* p, int n)
{
  printf("long: %d ", n);
  for (int i = 0; i < n; i++)
    printf("%ld ", *(p + i));
}

template<>
__global__ void kernel_print(size_t const* p, int n)
{
  printf("ulong: %d ", n);
  for (int i = 0; i < n; i++)
    printf("%lu ", *(p + i));
}

template<typename T>
__global__ void kernel_print(thrust::complex<T> const* p, int n)
{
  printf("Z: %d ", n);
  for (int i = 0; i < n; i++)
    printf("(%g, %g) ", (p + i)->real(), (p + i)->imag());
}

void print(std::string str, double const* p, int n)
{
  std::cout << str << "D n: " << n << " ";
  kernel_print<<<1, 1>>>(p, n);
  cudaDeviceSynchronize();
  std::cout << std::endl;
}

void print(std::string str, int const* p, int n)
{
  std::cout << str << "I n: " << n << " ";
  kernel_print<<<1, 1>>>(p, n);
  cudaDeviceSynchronize();
  std::cout << std::endl;
}

void print(std::string str, size_t const* p, int n)
{
  std::cout << str << "UL n: " << n << " ";
  kernel_print<<<1, 1>>>(p, n);
  cudaDeviceSynchronize();
  std::cout << std::endl;
}

void print(std::string str, long const* p, int n)
{
  std::cout << str << "L n: " << n << " ";
  kernel_print<<<1, 1>>>(p, n);
  cudaDeviceSynchronize();
  std::cout << std::endl;
}

void print(std::string str, std::complex<double> const* p, int n)
{
  std::cout << str << " Z n: " << n << " ";
  kernel_print<<<1, 1>>>(reinterpret_cast<thrust::complex<double> const*>(p), n);
  cudaDeviceSynchronize();
  std::cout << std::endl;
}

} // namespace kernels
