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

#ifndef AFQMC_CONFIG_H
#define AFQMC_CONFIG_H

#include <string>
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <vector>
#include <map>
#include <complex>
#include <tuple>
#include <fstream>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "Utilities/check.hpp"
#include "config.0.h"

#include "io/ptree/ptree_utilities.hpp"

// platform-specific definition of pointer types, memory resources, etc...
#include "Memory/config.h"
#include "Memory/custom_pointers.hpp"

#include "SparseMatrix/csr_matrix.hpp"
#include "SparseMatrix/coo_matrix.hpp"

//#include "mpi3/shared_window.hpp"
#include "Memory/SharedMemory/shm_ptr_with_raw_ptr_dispatch.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "multi/memory/fallback.hpp"

namespace mpi3 = boost::mpi3;

namespace sfqmc
{
namespace afqmc
{

// Global Constant Definitions
const int DEFAULT_MEASURE_INTERVAL_MULTIPLIER = 1; // in units of population control interval
const int DEFAULT_POPULATION_CONTROL_INTERVAL = 10; // in units of steps
const int DEFAULT_WALKER_ORTHO_INTERVAL = 10; // in units of steps
const float DEFAULT_TIME_STEP = 0.01f; // in units of inverse energy (depending on Hamiltonian units)

namespace multi = boost::multi;

// ultil we switch to c++17, to reduce extra lines
using tp_ul_ul = std::tuple<std::size_t, std::size_t>;

enum WALKER_TYPES
{
  UNDEFINED_WALKER_TYPE,
  CLOSED,
  COLLINEAR,
  NONCOLLINEAR,
  FULLYPOLARIZED
};

inline WALKER_TYPES initWALKER_TYPES(int i)
{
  if (i == 0)
    return UNDEFINED_WALKER_TYPE;
  else if (i == 1)
    return CLOSED;
  else if (i == 2)
    return COLLINEAR;
  else if (i == 3)
    return NONCOLLINEAR;
  else if (i == 4)
    return FULLYPOLARIZED;
  return UNDEFINED_WALKER_TYPE;
}

enum INTEGRAL_TYPES
{
  UNDEFINED_INTEGRAL_TYPE,
  CLOSED_INTEGRAL,
  COLLINEAR_INTEGRAL,
  NONCOLLINEAR_INTEGRAL,
};

inline INTEGRAL_TYPES initINTEGRAL_TYPES(int i)
{
  if (i == 0)
    return UNDEFINED_INTEGRAL_TYPE;
  else if (i == 1)
    return CLOSED_INTEGRAL;
  else if (i == 2)
    return COLLINEAR_INTEGRAL;
  else if (i == 3)
    return NONCOLLINEAR_INTEGRAL;
  return UNDEFINED_INTEGRAL_TYPE;
}

template<typename T>
using s1D = std::tuple<IndexType, T>;
template<typename T>
using s2D = std::tuple<IndexType, IndexType, T>;
template<typename T>
using s3D = std::tuple<IndexType, IndexType, IndexType, T>;
template<typename T>
using s4D = std::tuple<IndexType, IndexType, IndexType, IndexType, T>;

enum SpinTypes
{
  Alpha,
  Beta
};

// new types
template<typename T>
using mpi3_csr_matrix =
    ma::sparse::csr_matrix<T, int, std::size_t, shared_allocator<T>, ma::sparse::is_root>;

template<typename T>
using dev_csr_Matrix = ma::sparse::csr_matrix<T, int, int, device_allocator<T>>;

//#ifdef PsiT_IN_SHM
template<typename T>
using PsiT_Matrix_t = ma::sparse::csr_matrix<T, int, int, shared_allocator<T>, ma::sparse::is_root>;
using PsiT_Matrix   = PsiT_Matrix_t<ComplexType>;
#if defined(ENABLE_DEVICE)
template<typename T, typename IntT = int>
using local_csr_Matrix = ma::sparse::csr_matrix<T, int, IntT, device_allocator<T>>;
#else
template<typename T, typename IntT = int>
using local_csr_Matrix = ma::sparse::csr_matrix<T, int, IntT, shared_allocator<T>, ma::sparse::is_root>;
#endif
//#else
//  using PsiT_Matrix = ma::sparse::csr_matrix<ComplexType,int,int>;
//  using devPsiT_Matrix = ma::sparse::csr_matrix<ComplexType,int,int>;
//#endif


#if defined(ENABLE_DEVICE)
using P1Type = ma::sparse::csr_matrix<ComplexType, int, int, localTG_allocator<ComplexType>>;
#else
using P1Type        = ma::sparse::csr_matrix<ComplexType, int, int, localTG_allocator<ComplexType>, ma::sparse::is_root>;
#endif

enum HamiltonianTypes
{
  THC,
  KPTHC,
  KPFactorized,
  RealDenseFactorized,
  ModelHamiltonian,  
  UNKNOWN
};

/* Remember to propagate any changes to this enum to the device Kernel
   routines for construct_X_Model */
enum PropagatorTypes
{
  ContinuousChargePropagator,
  ContinuousSpinPropagator,
  DiscreteChargePropagator,
  DiscreteSpinPropagator,
  UndefinedPropagator
};

template<std::ptrdiff_t D>
using iextensions = typename boost::multi::iextensions<D>;

// general matrix definitions
template<class Alloc = std::allocator<int>>
using IntegerVector = boost::multi::array<int, 1, Alloc>;
template<class Alloc = std::allocator<ComplexType>>
using ComplexVector = boost::multi::array<ComplexType, 1, Alloc>;
template<class Ptr = ComplexType*>
using ComplexVector_ref = boost::multi::array_ref<ComplexType, 1, Ptr>;

template<class Alloc = std::allocator<int>>
using IntegerMatrix = boost::multi::array<int, 2, Alloc>;
template<class Alloc = std::allocator<ComplexType>>
using ComplexMatrix = boost::multi::array<ComplexType, 2, Alloc>;
template<class Ptr = ComplexType*>
using ComplexMatrix_ref = boost::multi::array_ref<ComplexType, 2, Ptr>;

template<class Alloc = std::allocator<ComplexType>>
using Complex3Tensor = boost::multi::array<ComplexType, 3, Alloc>;
template<class Ptr = ComplexType*>
using Complex3Tensor_ref = boost::multi::array_ref<ComplexType, 3, Ptr>;

template<std::ptrdiff_t D, class Alloc = std::allocator<ComplexType>>
using ComplexArray = boost::multi::array<ComplexType, D, Alloc>;
template<std::ptrdiff_t D, class Ptr = ComplexType*>
using ComplexArray_ref = boost::multi::array_ref<ComplexType, D, Ptr>;

/* move to these types */
template<class T, class Alloc = std::allocator<T>> 
using Vector = boost::multi::array<T, 1, Alloc>;
template<class T, class Alloc = std::allocator<T>> 
using StaticVector = boost::multi::static_array<T, 1, Alloc>;
template<class T, class Ptr = T*> 
using Vector_ref = boost::multi::array_ref<T, 1, Ptr>;

template<class T, class Alloc = std::allocator<T>>
using Matrix = boost::multi::array<T, 2, Alloc>;
template<class T, class Alloc = std::allocator<T>>
using StaticMatrix = boost::multi::static_array<T, 2, Alloc>;
template<class T, class Ptr = T*>              
using Matrix_ref = boost::multi::array_ref<T, 2, Ptr>;

template<class T, std::ptrdiff_t D, class Alloc = std::allocator<T>> 
using Array = boost::multi::array<T, D, Alloc>;
template<class T, std::ptrdiff_t D, class Alloc = std::allocator<T>> 
using StaticArray = boost::multi::static_array<T, D, Alloc>;
template<class T, std::ptrdiff_t D, class Ptr = T*> 
using Array_ref = boost::multi::array_ref<T, D, Ptr>;

template<class Alloc> 
using Vector_ = boost::multi::array<typename std::allocator_traits<Alloc>::value_type, 1, Alloc>;
template<class Alloc> 
using StaticVector_ = boost::multi::static_array<typename std::allocator_traits<Alloc>::value_type, 1, Alloc>;
template<class ptr> 
using Vector_ref_ = boost::multi::array_ref<typename std::pointer_traits<ptr>::element_type, 1, ptr>; 

template<class Alloc> 
using Matrix_ = boost::multi::array<typename std::allocator_traits<Alloc>::value_type, 2, Alloc>;
template<class Alloc> 
using StaticMatrix_ = boost::multi::static_array<typename std::allocator_traits<Alloc>::value_type, 2, Alloc>;
template<class ptr> 
using Matrix_ref_ = boost::multi::array_ref<typename std::pointer_traits<ptr>::element_type, 2, ptr>; 

template<std::ptrdiff_t D, class Alloc>
using Array_ = boost::multi::array<typename std::allocator_traits<Alloc>::value_type, D, Alloc>;
template<std::ptrdiff_t D, class Alloc>
using StaticArray_ = boost::multi::static_array<typename std::allocator_traits<Alloc>::value_type, D, Alloc>;
template<std::ptrdiff_t D, class ptr>
using Array_ref_ = boost::multi::array_ref<typename std::pointer_traits<ptr>::element_type, D, ptr>; 

struct AFQMCInfo
{
public:
  // default constructor
  AFQMCInfo()
      : name(""),
        NMO(-1),
        NAEA(-1),
        NAEB(-1),
        MS2(-99),
        ISYM(-1)
  {}

  AFQMCInfo(std::string nm, int nmo_, int naea_, int naeb_)
      : name(nm),
        NMO(nmo_),
        NAEA(naea_),
        NAEB(naeb_),
        MS2(-99),
        ISYM(-1)
  {}

  AFQMCInfo(const AFQMCInfo& other) = default;
  AFQMCInfo& operator=(const AFQMCInfo& other) = default;

  // destructor
  ~AFQMCInfo() {}

  // identifier
  std::string name;

  // number of active orbitals
  int NMO;

  // number of active electrons alpha/beta
  int NAEA, NAEB;

  // ms2
  int MS2 = -1;

  // isym
  int ISYM = -1;

  // copies values from object
  void copyInfo(const AFQMCInfo& a)
  {
    name           = a.name;
    NMO            = a.NMO;
    NAEA           = a.NAEA;
    NAEB           = a.NAEB;
    MS2            = a.MS2;
    ISYM           = a.ISYM;
  }

  // no fully spin polarized yet, not sure what it will break
  bool checkAFQMCInfoState()
  {
    if ( NAEA < 1 || NAEB < 0 ) 
      return false;
    return true;
  }

  void printAFQMCInfoState(std::ostream& out)
  {
    out << "AFQMC info: "
        << "name: " << name << ""
        << "NMO: "      << NMO << ""
        << "NAEA: " << NAEA << ""
        << "NAEB: " << NAEB << std::endl; 
// FIX        << "MS2: " << MS2 << std::endl; 
  }

  bool parse(ptree pt)
  {
    name = pt.get<std::string>("name");
    NMO      = pt.get<int>("NMO", -1);
    NAEA     = pt.get<int>("NAEA", -1);
    NAEB     = pt.get<int>("NAEB", -1);
    MS2      = pt.get<int>("MS2", -99);
    ISYM      = pt.get<int>("ISYM", -1);
    // fix! either specify MS2 or NAEA/NAEB, but not both
    // right now MS2 is not a useful option
    return true;
  }
};

} // namespace afqmc
} // namespace sfqmc

#endif
