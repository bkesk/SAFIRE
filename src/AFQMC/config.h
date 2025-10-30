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

#pragma once

#include <string>
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <vector>
#include <map>
#include <complex>
#include <tuple>
#include <fstream>

#include "configuration.hpp"
#include "config.0.h"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"

#include "IO/ptree/ptree_utilities.hpp"

#include"numerics/sparse/sparse.hpp"

namespace sfqmc
{
namespace afqmc
{

// Global Constant Definitions
const int DEFAULT_MEASURE_INTERVAL_MULTIPLIER = 1; // in units of population control interval
const int DEFAULT_POPULATION_CONTROL_INTERVAL = 10; // in units of steps
const int DEFAULT_WALKER_ORTHO_INTERVAL = 10; // in units of steps
const float DEFAULT_TIME_STEP = 0.01f; // in units of inverse energy (depending on Hamiltonian units)

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

enum SpinTypes
{
  Alpha,
  Beta
};

// Trial wave function types
template<typename T, MEMORY_SPACE MEM>
using PsiT_Matrix_t = math::sparse::csr_matrix<T, MEM, int, int>;
template<MEMORY_SPACE MEM>
using PsiT_Matrix   = PsiT_Matrix_t<ComplexType,MEM>;

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

