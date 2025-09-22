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

#ifndef MA_DEFINE_HPP
#define MA_DEFINE_HPP

namespace ma
{

enum TENSOR_OPERATIONS
{
  TOp_PLUS,
  TOp_MINUS,
  TOp_MUL,
  TOp_DIV
};

static const int INCX                   = 1;
static const int INCY                   = 1;
static const char UPLO                  = 'L';
static const char TRANS                 = 'T';
static const char NOTRANS               = 'N';
static const float sone                 = 1.0;
static const float szero                = 0.0;
static const double done                = 1.0;
static const double dzero               = 0.0;
static const std::complex<float> cone   = std::complex<float>(1.0, 0.0);
static const std::complex<float> czero  = std::complex<float>(0.0, 0.0);
static const std::complex<double> zone  = std::complex<double>(1.0, 0.0);
static const std::complex<double> zzero = std::complex<double>(0.0, 0.0);

}

#endif
