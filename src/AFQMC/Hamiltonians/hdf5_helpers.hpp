/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

#include <iostream>

#include "config.h"
#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "nda/h5.hpp"

namespace sfqmc
{
namespace afqmc
{

inline HamiltonianTypes peekHamType(h5::group grp, std::string format = "std")
{
  if (format  == "coqui") {
    // only format available, add choices as they are implemented
    // this is not enough, it could be THC, KPTHC, etc... Look for cholesky vectors...
    utils::check(grp.has_subgroup("/Interaction"), "Missing Interaction dataset.");
    h5::group igrp = grp.open_group("/Interaction");
    std::vector<int> shape;
    if (igrp.has_key("Vq0"))
      return KPFactorized;
    if (igrp.has_key("factorized_coulomb_matrix"))
    {
      auto l = h5::array_interface::get_dataset_info(igrp,"factorized_coulomb_matrix");
      utils::check(l.lengths[0]>0,"  Error: Found Interaction/factorized_coulomb_matrix with dimension=0 "); 
      return (l.lengths[0]==1?THC:KPTHC);
    }
  } else if(format == "std") {
    h5::group hgrp = grp.open_group("/Hamiltonian");
    if (hgrp.has_subgroup("KPFactorized"))
    {
      return KPFactorized;
    }
    if (hgrp.has_subgroup("DenseFactorized"))
    {
      return RealDenseFactorized;
    }
    if (hgrp.has_subgroup("ModelHamiltonian"))
    {
      return ModelHamiltonian;
    }
  } else {
    utils::check(false, "  Error: Invalid format in peekHamType. ");
    return UNKNOWN;
  }
  utils::check(false,"  Error: Invalid hdf5 file format in peekHamType(). ");
  return UNKNOWN;
}

inline std::string get_hamiltonian_format(h5::group& grp)
{
  if(grp.has_subgroup(std::string("/Hamiltonian")))
    return "std";
  else if(grp.has_subgroup(std::string("/System")) and grp.has_subgroup(std::string("/Interaction")))
    return "coqui"; 
  else
    utils::check(false,"Error in get_hamiltonian_format: Invalid format");
  return "";
}

} // namespace afqmc

} // namespace sfqmc

