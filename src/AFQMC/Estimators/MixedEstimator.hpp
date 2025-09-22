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

#ifndef SFQMC_AFQMC_MIXED_ESTIMATOR_HPP
#define SFQMC_AFQMC_MIXED_ESTIMATOR_HPP

#include "AFQMC/config.h"
#include <vector>
#include <string>
#include <iostream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Estimators/MixedObsHandler.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Top class for mixed estimators. 
 * An instance of this class will manage a set of observables evaluated at the mixed distribution.
 */
class MixedEstimator : public EstimatorBase
{

public:
  MixedEstimator(afqmc::TaskGroup_& tg_,
                          AFQMCInfo& info,
                          std::string name,
                          ptree pt,
                          WALKER_TYPES wlk,
                          [[maybe_unused]] WalkerSet& wset,
                          Wavefunction& wfn)
      : EstimatorBase(info),
        TG(tg_),
        walker_type(wlk),
        observ0(TG, info, name, pt, wlk, wfn),
        wfn0(wfn)
  {
    int _pop_control_interval, equil_multiplier;
    _pop_control_interval = pt.get<int>("_population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL);
    block_size = pt.get<int>("block_size", 1);
    equil_multiplier = pt.get<int>("equil_multiplier", 0); // units of population control interval
    measure_interval_multiplier = pt.get<int>("measure_interval_multiplier", DEFAULT_MEASURE_INTERVAL_MULTIPLIER); // units of population control interval
    measure_interval = measure_interval_multiplier * _pop_control_interval;
    if (equil_multiplier % measure_interval_multiplier != 0)
      APP_ABORT("Error in MixedEstimator user input: 'equil_multiplier' must be evenly divisible by 'measure_interval_multiplier'");
    nblocks_skip = equil_multiplier / measure_interval_multiplier;
    writer = (TG.Global().rank() == 0);
  }

  ~MixedEstimator() {}

  void accumulate_step([[maybe_unused]] double time, 
                       [[maybe_unused]] WalkerSet& wset,
                       [[maybe_unused]] std::vector<ComplexType>& curData) {}

  void accumulate_block([[maybe_unused]] double time, WalkerSet& wset)
  {
    accumulated_in_last_block = false;

    // 0. skip if requested
    // MAM: problematic on restarts!!!
    if (iblock < nblocks_skip) { 
      iblock++;
      return;
    }

    AFQMCTimer.start(mixed_estimator_timer);
    observ0.accumulate(wset);
    iblock++;
    accumulated_in_last_block = true;
    AFQMCTimer.stop(mixed_estimator_timer);
  }

  void tags([[maybe_unused]] std::ofstream& out)
  {
  }

  int get_measurement_interval() { 
    return measure_interval;
  }

  void print([[maybe_unused]] std::ofstream& out, hdf_archive& dump, [[maybe_unused]] WalkerSet& wset)
  {
    // I doubt we will ever collect a billion blocks of data.
    if (accumulated_in_last_block)
    {
      if (writer)
      {
        dump.push("Observables");
        dump.push("Mixed");
      }
      observ0.print(iblock, dump);
      if (writer)
      {
        dump.pop();
        dump.pop();
      }
    }
  }

private:
  TaskGroup_& TG;

  [[maybe_unused]] WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE;

  bool writer = false;
  bool accumulated_in_last_block = false;

  MixedObsHandler observ0;

  [[maybe_unused]] Wavefunction& wfn0;

  // Blocking info 
  int block_size   = 1;
  int iblock       = 0;
  int nblocks_skip = 0;

  int measure_interval = 1;
  int measure_interval_multiplier = 1;
};
} // namespace afqmc
} // namespace sfqmc

#endif
