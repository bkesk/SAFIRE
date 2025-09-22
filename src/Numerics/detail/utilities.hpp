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

#ifndef MA_UTILITIES_HPP
#define MA_UTILITIES_HPP

#include <complex>
#include "config.0.h"
#include "Numerics/detail/define.hpp"
#include "Memory/raw_pointers.hpp"
#include "Memory/SharedMemory/shm_ptr_with_raw_ptr_dispatch.hpp"

namespace ma
{
using sfqmc::afqmc::raw_pointer_cast;

/*
#if defined(HAVE_MKL)
typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
#endif
*/

//inline double const& real(double const& d) { return d; }
//inline float const& real(float const& f) { return f; }

inline double real(double const& d) { return d; }
inline float real(float const& f) { return f; }
inline double real(std::complex<double> const& d) { return std::real(d); }
inline float real(std::complex<float> const& f) { return std::real(f); }

inline double imag([[maybe_unused]] double const& d) { return 0.0; }
inline float imag([[maybe_unused]] float const& f) { return 0.0f; }
inline double imag(std::complex<double> const& d) { return std::imag(d); }
inline float imag(std::complex<float> const& f) { return std::imag(f); }

inline double conj(double const& d) { return d; }
inline float conj(float const& f) { return f; }
inline std::complex<double> conj(std::complex<double> const& d) { return std::conj(d); }
inline std::complex<float> conj(std::complex<float> const& f) { return std::conj(f); }

template<class Ptr>
auto pointer_dispatch(Ptr p)
{
  return p;
}

template<typename T>
T* pointer_dispatch(shm::shm_ptr_with_raw_ptr_dispatch<T> p)
{
  return raw_pointer_cast(p);
}


} // namespace ma

#endif
