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

#ifndef SFQMC_H5DATATYPE_DEFINE_H
#define SFQMC_H5DATATYPE_DEFINE_H

#include <type_traits>
#include <hdf5.h>

namespace sfqmc
{
/** map C types to hdf5 native types
 * bool is explicit removed due to the fact that it is implementation-dependant
 */
template<typename T, typename ENABLE = std::enable_if_t<!std::is_same<bool, T>::value>>
inline hid_t get_h5_datatype(const T&);

#define BOOSTSUB_H5_DATATYPE(CppType, H5DTYPE)          \
  template<>                                            \
  inline hid_t get_h5_datatype<CppType>(const CppType&) \
  {                                                     \
    return H5DTYPE;                                     \
  }

BOOSTSUB_H5_DATATYPE(char, H5T_NATIVE_CHAR);
BOOSTSUB_H5_DATATYPE(short, H5T_NATIVE_SHORT);
BOOSTSUB_H5_DATATYPE(int, H5T_NATIVE_INT);
BOOSTSUB_H5_DATATYPE(long, H5T_NATIVE_LONG);
BOOSTSUB_H5_DATATYPE(long long, H5T_NATIVE_LLONG);
BOOSTSUB_H5_DATATYPE(unsigned char, H5T_NATIVE_UCHAR);
BOOSTSUB_H5_DATATYPE(unsigned short, H5T_NATIVE_USHORT);
BOOSTSUB_H5_DATATYPE(unsigned int, H5T_NATIVE_UINT);
BOOSTSUB_H5_DATATYPE(unsigned long, H5T_NATIVE_ULONG);
BOOSTSUB_H5_DATATYPE(unsigned long long, H5T_NATIVE_ULLONG);
//BOOSTSUB_H5_DATATYPE(uint32_t, H5T_NATIVE_UINT32);
//BOOSTSUB_H5_DATATYPE(uint64_t, H5T_NATIVE_UINT64);
BOOSTSUB_H5_DATATYPE(float, H5T_NATIVE_FLOAT);
BOOSTSUB_H5_DATATYPE(double, H5T_NATIVE_DOUBLE);

} // namespace sfqmc
#endif
