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
  hipLaunchKernelGGL(kernel_print, dim3(1), dim3(1), 0, 0, p, n);
  hipDeviceSynchronize();
  std::cout << std::endl;
}

void print(std::string str, int const* p, int n)
{
  std::cout << str << "I n: " << n << " ";
  hipLaunchKernelGGL(kernel_print, dim3(1), dim3(1), 0, 0, p, n);
  hipDeviceSynchronize();
  std::cout << std::endl;
}

void print(std::string str, size_t const* p, int n)
{
  std::cout << str << "UL n: " << n << " ";
  hipLaunchKernelGGL(kernel_print, dim3(1), dim3(1), 0, 0, p, n);
  hipDeviceSynchronize();
  std::cout << std::endl;
}

void print(std::string str, long const* p, int n)
{
  std::cout << str << "L n: " << n << " ";
  hipLaunchKernelGGL(kernel_print, dim3(1), dim3(1), 0, 0, p, n);
  hipDeviceSynchronize();
  std::cout << std::endl;
}

void print(std::string str, std::complex<double> const* p, int n)
{
  std::cout << str << " Z n: " << n << " ";
  hipLaunchKernelGGL(kernel_print, dim3(1), dim3(1), 0, 0, reinterpret_cast<thrust::complex<double> const*>(p), n);
  hipDeviceSynchronize();
  std::cout << std::endl;
}

} // namespace kernels
