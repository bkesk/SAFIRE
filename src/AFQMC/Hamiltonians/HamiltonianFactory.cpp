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

#include <cstdlib>
#include <memory>
#include <algorithm>
#include <complex>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <vector>
#include <numeric>
#include <boost/version.hpp>

#include "AFQMC/config.h"
#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"
#include "HamiltonianFactory.h"

#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/Hamiltonians/RealDenseHamiltonian.h"
#include "AFQMC/Hamiltonians/RealDenseHamiltonian_v2.h"
#include "AFQMC/Hamiltonians/FactorizedSparseHamiltonian.h"
#include "AFQMC/Hamiltonians/THCHamiltonian.h"
#include "AFQMC/Hamiltonians/KPFactorizedHamiltonian.h"
#include "AFQMC/Hamiltonians/ModelHamOpsGenerator.h"

#include "AFQMC/Utilities/Utils.hpp"

#include "Numerics/ma_operations.hpp"
#include "SparseMatrix/csr_matrix.hpp"

#include "AFQMC/Utilities/hdf5_consistency_helper.hpp"
#include "SparseMatrix/array_partition.hpp"

namespace sfqmc
{
namespace afqmc
{
Hamiltonian HamiltonianFactory::fromHDF5(GlobalTaskGroup& gTG, ptree pt)
{
  std::string info;
  info = pt.get<std::string>("system", "");

  if (InfoMap.find(info) == InfoMap.end())
    APP_ABORT("ERROR: Undefined system in execute block. ");

  AFQMCInfo& AFinfo = InfoMap[info];

  int NMO  = AFinfo.NMO;
  int NAEA = AFinfo.NAEA;
  int NAEB = AFinfo.NAEB;

  std::string filename = pt.get<std::string>("filename");
  int number_of_TGs = pt.get<int>("number_of_TGs", 1);
  int n_reading_cores = pt.get<int>("num_io_cores", -1);

  // make or get TG
  number_of_TGs  = std::max(1, std::min(number_of_TGs, gTG.getTotalNodes()));
  TaskGroup_& TG = getTG(gTG, number_of_TGs);

  // processor info
  int ncores = TG.getTotalCores(), coreid = TG.getCoreID();
  int nread = (n_reading_cores <= 0) ? (ncores) : (std::min(n_reading_cores, ncores));
  int head  = TG.Global().rank() == 0;

  app_log(1," Initializing Hamiltonian from file: {}", filename);

  // FIX FIX FIX
  hdf_archive dump(TG.Global());
  // these cores will read from hdf file
  if (coreid < nread)
  {
    if (!dump.open(filename, H5F_ACC_RDONLY))
      APP_ABORT("Error opening integral file in HamiltonianFactory. ");
  }
  std::string format = get_hamiltonian_format(dump,TG.Global());
  app_log(1, " Found hamiltonian with format: {}", format);
  if (coreid < nread)
  {
    if(format.substr(0,6) == "coqui") {
      if (dump.push("System", false)<0)
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group System not found. ");
    } else {
      if (dump.push("Hamiltonian", false)<0)
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group Hamiltonian not found. ");
    } 
  }

  HamiltonianTypes htype = UNKNOWN;
  if (head)
    htype = peekHamType(dump,format);
  {
    int htype_ = int(htype);
    TG.Global().broadcast_n(&htype_, 1, 0);
    htype = HamiltonianTypes(htype_);
  }

  int complex_integrals;
  // Hamiltonian file may not contain flag.
  bool have_complex_flag = true;
  if (head)
  {
    if(format.substr(0,6) == "coqui") { // coqui always complex for now!
      have_complex_flag = true;
    } else if (!dump.readEntry(complex_integrals, "ComplexIntegrals"))
    {
      have_complex_flag = false;
    }
  }
  TG.Global().broadcast_n(&have_complex_flag, 1, 0);

  std::vector<int> Idata(8);
  if (head) {
    if(format.substr(0,6) == "coqui") { // coqui always complex for now!
      if (!dump.readAttributeEntry(Idata[3], "number_of_bands"))
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Problems reading attribute /System/number_of_bands. ");
      if (dump.push("BZ", false)<0)
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group /System/BZ not found. ");
      if (!dump.readAttributeEntry(Idata[2], "number_of_kpoints"))
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Problems reading attribute /System/BZ/number_of_kpoints. ");
      dump.pop();
      Idata[3] *= Idata[2];
    } else if (!dump.readEntry(Idata, "dims")) {
      APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Problems reading dims. ");
    }
  }
  TG.Global().broadcast(Idata.begin(), Idata.end());

  if (Idata[3] != NMO)
    APP_ABORT(" Error: NMO differs from value in integral file. ");
// MAM: checking this doesn't really do anything
//  if (Idata[4] != NAEA)
//    app_warning(" WARNING: NAEA differs from value in integral file. ");
//  if (Idata[5] != NAEB)
//    app_warning(" WARNING: NAEB differs from value in integral file. ");

  ComplexType NuclearCoulombEnergy(0);
  ComplexType FrozenCoreEnergy(0);
  ComplexType ElecSelfIntEnergy(0);

  if (head)
  {
    if(format == "std") {
      std::vector<RealType> Rdata(2);
      if (!dump.readEntry(Rdata, "Energies"))
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Problems reading  dataset. ");
      if (Rdata.size() > 0)
        NuclearCoulombEnergy = Rdata[0];
      if (Rdata.size() > 1)
        FrozenCoreEnergy = Rdata[1];
    } else if(format.substr(0,6) == "coqui") {
      double et(0);
      if (dump.readAttributeEntry(et, "nuclear_energy")) {
        NuclearCoulombEnergy = et;
      } else {
        app_warning("Could not find nuclear_energy in h5 file. Setting to 0."); 
      }
      if (dump.readAttributeEntry(et, "frozen_core_energy")) 
        FrozenCoreEnergy = et;
      if (dump.readAttributeEntry(et, "madelung_constant"))
        ElecSelfIntEnergy = -1.0*et*(NAEA+NAEB);
    }
  }
  app_log(2, "");
  app_log(2, " - Nuclear coulomb energy: {}",NuclearCoulombEnergy);
  app_log(2, " - Frozen Core energy: {}",FrozenCoreEnergy);
  app_log(2, " - Electron self-interaction energy: {}",ElecSelfIntEnergy);

  TG.Global().broadcast_n(&NuclearCoulombEnergy, 1, 0);
  TG.Global().broadcast_n(&FrozenCoreEnergy, 1, 0);
  TG.Global().broadcast_n(&ElecSelfIntEnergy, 1, 0);
  // MAM: FrozenCoreEnergy is not handled correctly! FIX! Add all terms into a single E0
  NuclearCoulombEnergy += ElecSelfIntEnergy;

  if (htype == KPTHC)
  {
    APP_ABORT(" Error: KPTHC hamiltonian not yet working. ");
    if (coreid < nread and dump.push("KPTHC", false)<0)
      APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group KPTHC not found. ");
    if (coreid < nread)
    {
      dump.pop();
      dump.pop();
      dump.close();
    }
    TG.Global().barrier();
    //      return Hamiltonian(KPTHCHamiltonian(AFinfo,pt,TG,
    //                                        NuclearCoulombEnergy,FrozenCoreEnergy));
  }
  else if (htype == KPFactorized)
  {
    if (format == "std" and coreid < nread and dump.push("KPFactorized", false)<0)
      APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group KPFactorized not found. ");
    if (coreid < nread)
    {
      if(format == "std") dump.pop();
      dump.pop();
      dump.close();
    }
    TG.Global().barrier();
    return Hamiltonian(KPFactorizedHamiltonian(AFinfo, pt, TG, NuclearCoulombEnergy, FrozenCoreEnergy));
  }
  else if (htype == RealDenseFactorized)
  {
    if(format != "std")
      APP_ABORT("Error: format: {} not yet implemented with this hamiltonian type.", format);
    if (coreid < nread and dump.push("DenseFactorized", false)<0)
      APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group DenseFactorized not found. ");
    if (coreid < nread)
    {
      dump.pop();
      dump.pop();
      dump.close();
    }
    TG.Global().barrier();
#if defined(ENABLE_DEVICE)
    return Hamiltonian(RealDenseHamiltonian_v2(AFinfo, pt, TG, NuclearCoulombEnergy, FrozenCoreEnergy));
#else
    return Hamiltonian(RealDenseHamiltonian(AFinfo, pt, TG, NuclearCoulombEnergy, FrozenCoreEnergy));
#endif
  }
  else if ( htype == ModelHamiltonian ) 
  {
    if(format != "std")
      APP_ABORT("Error: format: {} not yet implemented with this hamiltonian type.", format);
    if (coreid < nread and dump.push("ModelHamiltonian", false)<0)
      APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group ModelHamiltonian not found. ");
    if (coreid < nread)
    {
      dump.pop();
      dump.pop();
      dump.close();
    }
    TG.Global().barrier();
    return Hamiltonian(ModelHamOpsGenerator(AFinfo, pt, TG, NuclearCoulombEnergy, FrozenCoreEnergy));
  }
  else if ( htype == THC )
  {
    if(format != "coqui")
      APP_ABORT("Error: format: {} not yet implemented with this hamiltonian type.", format);
    if (coreid < nread)
    {
      dump.pop();
      dump.pop();
      dump.close();
    }
    TG.Global().barrier();
    return Hamiltonian(THCHamiltonian(AFinfo, pt, TG, NuclearCoulombEnergy, FrozenCoreEnergy));
  }
  else if (htype == FactorizedSparse)
  {
    if(format != "std")
      APP_ABORT("Error: format: {} not yet implemented with this hamiltonian type.", format);
    if (coreid < nread and dump.push("Factorized", false)<0)
      APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group Factorized not found. ");

    if (coreid < nread)
    {
      dump.pop();
      dump.pop();
      dump.close();
    }
    TG.Global().barrier();

    return Hamiltonian(FactorizedSparseHamiltonian(AFinfo, pt, TG, 
                                                   NuclearCoulombEnergy, FrozenCoreEnergy));
  }

  APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Unknown Hamiltonian Type. ");
  return Hamiltonian{};
}
} // namespace afqmc
} // namespace sfqmc
