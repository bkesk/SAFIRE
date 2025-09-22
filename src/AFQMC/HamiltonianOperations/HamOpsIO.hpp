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

#ifndef SFQMC_AFQMC_HAMOPSIO_HPP
#define SFQMC_AFQMC_HAMOPSIO_HPP

#include <fstream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/config.h"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.hpp"
#if !defined(ENABLE_DEVICE)
#include "AFQMC/HamiltonianOperations/SparseTensorIO.hpp"
#endif
#include "AFQMC/HamiltonianOperations/THCOpsIO.hpp"
//#if defined(ENABLE_COMPLEX)
//#include "AFQMC/HamiltonianOperations/KP3IndexFactorizationIO.hpp"
//#endif

namespace sfqmc
{
namespace afqmc
{
template<bool MP, bool REAL>
HamiltonianOperations loadHamOps(hdf_archive& dump,
                                 WALKER_TYPES type,
                                 int NMO,
                                 int NAEA,
                                 int NAEB,
                                 std::vector<PsiT_Matrix>& PsiT,
                                 TaskGroup_& TGprop,
                                 TaskGroup_& TGwfn,
                                 RealType cutvn,
                                 RealType cutv2)
{
  int hops_type = -1;
  if (TGwfn.Global().root())
  {
    if(dump.push("HamiltonianOperations", false)<0)
      APP_ABORT(" Error in loadHamOps: Group HamiltonianOperations not found. ");
    if (dump.is_group(std::string("THCOps")))
      hops_type = 1;
    else if (dump.is_group(std::string("KP3IndexFactorization")))
      hops_type = 3;
#if !defined(ENABLE_DEVICE)
    else if (dump.is_group(std::string("SparseTensor")))
    {
      dump.push("SparseTensor", false);
      std::vector<int> type;
      if (!dump.readEntry(type, "type"))
        APP_ABORT(" Error in loadHamOps: Problems reading type dataset. ");
      if (type[0] == 11)
        hops_type = 211;
      else if (type[0] == 12)
        hops_type = 212;
      else if (type[0] == 21)
        hops_type = 221;
      else if (type[0] == 22)
        hops_type = 222;
      else
      {
        app_error(" Unknown SparseTensor/type: {}", type[0]);
        APP_ABORT("");
      }
      dump.pop();
    }
#endif
    else
    {
      APP_ABORT(" Error in loadHamOps: Unknown hdf5 format. ");
    }
    dump.pop();
  }
  TGwfn.Global().broadcast_value(hops_type);

  if (hops_type == 1)
    return HamiltonianOperations(loadTHCOps<MP,REAL>(dump, type, NMO, NAEA, NAEB, PsiT, TGprop, TGwfn, cutvn, cutv2));
#if !defined(ENABLE_DEVICE)
  else if (hops_type == 211)
    return HamiltonianOperations(
        loadSparseTensor<ValueType, ValueType>(dump, type, NMO, NAEA, NAEB, PsiT, TGprop, TGwfn, cutvn, cutv2));
  else if (hops_type == 212)
    return HamiltonianOperations(
        loadSparseTensor<ValueType, ComplexType>(dump, type, NMO, NAEA, NAEB, PsiT, TGprop, TGwfn, cutvn, cutv2));
  else if (hops_type == 221)
    return HamiltonianOperations(
        loadSparseTensor<ComplexType, ValueType>(dump, type, NMO, NAEA, NAEB, PsiT, TGprop, TGwfn, cutvn, cutv2));
  else if (hops_type == 222)
    return HamiltonianOperations(
        loadSparseTensor<ComplexType, ComplexType>(dump, type, NMO, NAEA, NAEB, PsiT, TGprop, TGwfn, cutvn, cutv2));
  //  else if(hops_type == 3)
  //    return  HamiltonianOperations(loadKP3IndexFactorization(dump,type,NMO,NAEA,NAEB,PsiT,TGprop,TGwfn,cutvn,cutv2));
#endif

  app_error(" Error in loadHamOps: Unknown HOps type: {}", hops_type);
  APP_ABORT("");
  return HamiltonianOperations{};
}

} // namespace afqmc
} // namespace sfqmc

#endif
