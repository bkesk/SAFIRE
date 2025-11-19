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

#include <tuple>
#include <map>
#include <string>
#include <iomanip>

#include "config.h"
#include "Utilities/AppAbort.hpp"

#include "AFQMC/config.h"
#include "Utilities/app_loggers.h"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMCNewDriver.h"
#include "AFQMC/Walkers/WalkerIO.hpp"
#include "Memory/buffer_managers.h"

namespace sfqmc
{
namespace afqmc
{
bool AFQMCNewDriver::run(WalkerSet& wset)
{

  app_log(1,"****************************************************");
  app_log(1,"              Beginning AFQMC calculation           ");
  app_log(1,"****************************************************");
  setup_AFQMC_timer();

  std::vector<ComplexType> curData;

  RealType w0   = wset.GlobalWeight();
  int nwalk_ini = wset.GlobalPopulation();

  app_log(1, "Initial weight and number of walkers: {}, {}", w0 ,nwalk_ini);
  app_log(1, "Initial Eshift: {} ", Eshift);

  // problems with using step_tot to do ortho and load balance
  double total_time = step0 * dt;
  int step_tot      = step0, iBlock;
  // KE: the concept of a "block" is now implicitly defined by the measure_interval
  iBlock = 0;
  
  // KE: need to change the hard-coded 1.0 to an equilibration phase.
  
  AFQMCTimer.start(block_timer);
  for (int iStep = 0; iStep < nStep; ++iStep, ++step_tot)
  {
    prop0.Propagate(wset, Eshift, dt);
    total_time += dt;

    if ((step_tot + 1) % nStabilize == 0)
    {
      AFQMCTimer.start(ortho_timer);
      prop0.Orthogonalize(wset);
      AFQMCTimer.stop(ortho_timer);
    }

    if (total_time < weight_reset_period && !prop0.free_propagation())
      wset.resetWeights();

    if (total_time < 1.0 || (iStep + 1) % nPopulation == 0 || iStep == 0)
    {
      AFQMCTimer.start(popcont_timer);
      wset.processWalkerData(curData);
      wset.popControl(); // make this a call to actual pop control
      AFQMCTimer.stop(popcont_timer);
      estim0.accumulate_step(total_time, wset, curData);
    }
    
    if (total_time < 1.0)
    {
      Eshift = estim0.getEloc_step();
    }
    else if ((iStep + 1) % nAccumulate == 0)
      Eshift += dShift * (estim0.getEloc_step() - Eshift);
  

    // checkpoint
    if (nCheckpoint > 0 && (iStep + 1) % nCheckpoint == 0)
      if (!checkpoint(wset, iStep, step_tot))
      {
        app_error(" Error in AFQMCNewDriver::checkpoint(). ");
        app_error_flush();
        return false;
      }
    
    // write samples
    if (samplePeriod > 0 && (iStep + 1) % samplePeriod == 0)
      if (!writeSamples(wset))
      {
        app_error(" Error in AFQMCNewDriver::writeSamples(). ");
        app_error_flush();
        return false;
      }
    
    if ((iStep + 1) % _measure_interval == 0 )
    {
      // quantities that are measured once per block
      estim0.accumulate_block(total_time, wset);
      estim0.print(iBlock + 1, total_time, Eshift, wset);
      iBlock++;
    }

    // resize stack pointers to match maximum buffer use
    update_memory_managers();

    AFQMCTimer.stop(block_timer);
    
  }

  if (nCheckpoint > 0)
    checkpoint(wset, iBlock, step_tot);

  // print timers
  if(globalComm.root()) AFQMCTimer.print_all();
  
  app_log(1,"****************************************************");
  app_log(1,"               Finished AFQMC calculation           ");
  app_log(1,"****************************************************");

  return true;
}

// writes checkpoint file
bool AFQMCNewDriver::checkpoint(WalkerSet& wset, int block, int step)
{
  // hack until hdf_archive works with mpi3
  hdf_archive dump(globalComm, false);
  if (globalComm.rank() == 0)
  {
    std::string file;
    if (hdf_write_restart != std::string(""))
      file = hdf_write_restart;
    else
      file = project_title + std::string(".chk.h5");

    if (!dump.create(file))
    {
      app_error(" Error opening checkpoint file for write. ");
      app_error_flush();
      return false;
    }

    std::vector<RealType> Rdata(2);
    Rdata[0] = Eshift;
    Rdata[1] = Eshift;

    std::vector<IndexType> Idata(2);
    Idata[0] = block;
    Idata[1] = step;

    // always write driver data and walkers
    dump.push("AFQMCNewDriver");
    dump.write(Idata, "DriverInts");
    dump.write(Rdata, "DriverReals");
    dump.pop();
  }

  if (!dumpToHDF5(wset, dump))
  {
    app_error(" Problems writing checkpoint file in Driver/AFQMCNewDriver::checkpoint(). ");
    app_error_flush();
    return false;
  }

  if (globalComm.rank() == 0)
  {
    dump.close();
  }

  return true;
}


// writes samples
bool AFQMCNewDriver::writeSamples(WalkerSet& wset)
{
  // hack until hdf_archive works with mpi3
  hdf_archive dump(globalComm, false);
  if (globalComm.rank() == 0)
  {
    std::string file;
    file = project_title + std::string(".confg.h5");

    if (!dump.create(file))
    {
      app_error(" Error opening checkpoint file for write. ");
      app_error_flush();
      return false;
    }
  }

  int nwtowrite = -1;
  if (!dumpSamplesHDF5(wset, dump, nwtowrite))
  {
    app_error(" Problems writing checkpoint file in Driver/AFQMCNewDriver::writeSample(). ");
    app_error_flush();
    return false;
  }

  if (globalComm.rank() == 0)
  {
    dump.close();
  }

  return true;
}

bool AFQMCNewDriver::clear() { return true; }

} // namespace afqmc

} // namespace sfqmc
