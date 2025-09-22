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

#ifndef SFQMC_AFQMC_AFQMCDRIVER_H
#define SFQMC_AFQMC_AFQMCDRIVER_H

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"
#include "mpi3/communicator.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/EstimatorHandler.h"

namespace sfqmc
{
namespace afqmc
{
class AFQMCDriver : public AFQMCInfo
{
public:
  AFQMCDriver(boost::mpi3::communicator& comm,
              AFQMCInfo& info,
              std::string& title,
              int mser,
              int blk0,
              int stp0,
              double eshft_,
              ptree pt_in,
              Wavefunction& wfn_,
              Propagator& prpg_,
              EstimatorHandler& estim_)
      : AFQMCInfo(info),
        globalComm(comm),
        m_series(mser),
        project_title(title),
        block0(blk0),
        step0(stp0),
        wfn0(wfn_),
        prop0(prpg_),
        estim0(estim_),
        weight_reset_period(0.0),
        Eshift(eshft_)
  {
    name = "AFQMCDriver";
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2, "\nAFQMCDriver input:");
    app_log(2, "{}\n", io::to_string(pt));
    // initialize using verbose input
    hdf_write_restart = pt.get<std::string>("hdf_write_file");
    nBlock = pt.get<int>("blocks");
    nStep = pt.get<int>("steps");
    nSubstep = pt.get<int>("substeps");
    fix_bias = pt.get<int>("fix_bias");
    nStabilize = pt.get<int>("ortho");
    nCheckpoint = pt.get<int>("checkpoint");
    samplePeriod = pt.get<int>("sample_period");
    weight_reset_period = pt.get<double>("weight_reset"); // in units of time
    dt = pt.get<double>("timestep");
    dShift = pt.get<double>("dshift");  // Etrial shift scale
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string hdf_write_file;
    int blocks, steps, substeps, fix_bias, ortho, checkpoint, sample_period;
    double weight_reset, timestep, dshift;
    hdf_write_file = pt0.get<std::string>("hdf_write_file", "");
    blocks        = pt0.get<int>("blocks", 100);
    steps         = pt0.get<int>("steps", 1);
    substeps      = pt0.get<int>("substeps", 1);
    fix_bias      = pt0.get<int>("fix_bias", 1);
    ortho         = pt0.get<int>("ortho", 1);
    checkpoint    = pt0.get<int>("checkpoint", -1);
    sample_period = pt0.get<int>("sample_period", -1);
    weight_reset = pt0.get<double>("weight_reset", 0.0);
    timestep     = pt0.get<double>("timestep", 0.01);
    dshift       = pt0.get<double>("dshift", 1.0);
    // validate inputs
    fix_bias = std::min(fix_bias, substeps);
    // create verbose internal inputs
    ptree pt1;
    pt1.put("hdf_write_file", hdf_write_file);
    pt1.put("blocks", blocks);
    pt1.put("steps", steps);
    pt1.put("substeps", substeps);
    pt1.put("fix_bias", fix_bias);
    pt1.put("ortho", ortho);
    pt1.put("checkpoint", checkpoint);
    pt1.put("sample_period", sample_period);
    pt1.put("weight_reset", weight_reset);
    pt1.put("timestep", timestep);
    pt1.put("dshift", dshift);
    // check for unkown input keys
    std::unordered_set<std::string> pass_through_keys = {
      "walker_set",
      "wavefunction",
      "propagator",
      "estimator",
      "hamiltonian"
    };
    io::compare_known_keys("(legacy) AFQMC Driver",pt1, pt0, pass_through_keys);
    return pt1;
  }

  ~AFQMCDriver() {}

  bool run(WalkerSet&);

  bool checkpoint(WalkerSet&, int, int);

  bool clear();

protected:
  boost::mpi3::communicator& globalComm;

  std::string name;

  int m_series;
  std::string project_title;

  std::string hdf_write_restart;

  int nBlock;
  int nStep;
  int nSubstep;
  int fix_bias;

  int nCheckpoint;
  int nStabilize;
  RealType dt;
  int block0, step0;

  Wavefunction& wfn0;

  Propagator& prop0;

  EstimatorHandler& estim0;

  bool writeSamples(WalkerSet&);

  int samplePeriod;

  double weight_reset_period;

  RealType dShift;
  RealType Eshift;
  RealType Etav;
};

} // namespace afqmc
} // namespace sfqmc

#endif
