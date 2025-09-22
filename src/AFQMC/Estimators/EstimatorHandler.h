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

#ifndef SFQMC_AFQMC_ESTIMATORHANDLER_H
#define SFQMC_AFQMC_ESTIMATORHANDLER_H

#include "AFQMC/config.h"

#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Utilities/taskgroup.h"

#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Estimators/EnergyEstimator.h"
#include "AFQMC/Estimators/BasicEstimator.h"
#include "AFQMC/Estimators/MixedEstimator.hpp"
#include "AFQMC/Estimators/BackPropagatedEstimator.hpp"
#include "AFQMC/Estimators/BPWithTimeEvolvedOperators.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Propagators/Propagator.hpp"

#include "mpi3/communicator.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Manager class for all estimators/observables.
 * This class contains and manages a list of estimator objects.
 * An arbitrary combination of estimators can be used simultaneously
 * during a simulation, including: 
 *   1) mixed distribution estimators,  
 *   2) back propagated estimators, 
 *   3) any number of 1),2), 
 *   4) each with independent wavefunctions.
 */
class EstimatorHandler : public AFQMCInfo
{
  using EstimPtr     = std::shared_ptr<EstimatorBase>;
  using communicator = boost::mpi3::communicator;

public:
  EstimatorHandler(afqmc::TaskGroupHandler& TGgen,
                   AFQMCInfo info,
                   std::string title,
                   ptree exec_pt,
                   WalkerSet& wset,
                   WavefunctionFactory& WfnFac,
                   Wavefunction& wfn0,
                   Propagator& prop0,
                   WALKER_TYPES walker_type,
                   HamiltonianFactory& HamFac,
                   std::string ham0,
                   double dt,
                   bool defaultEnergyEstim = false,
                   bool impsamp            = true)
      : AFQMCInfo(info), project_title(title), dt(dt), hdf_output(false)
  {
    estimators.reserve(10);

    app_log(1,"\n****************************************************");
    app_log(1,"               Initializing Estimators ");
    app_log(1,"\n****************************************************");
    app_log(2,"\n EstimatorHandler input:\n{}\n", io::to_string(exec_pt));
    ptree est_pt, basic_pt;
    int est_index = 0;
    // default measure_interval to DEFAULT_MEASURE_INTERVAL if not specified in exec_pt
    int measure_interval_multiplier = exec_pt.get<int>("measure_interval_multiplier", DEFAULT_MEASURE_INTERVAL_MULTIPLIER);
    int population_control_interval = exec_pt.get<int>("population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL);

    bool overwrite_default_energy=false;
    bool remove_default_energy=false;
    for(auto& it : exec_pt)
    {
      std::string cname = it.first;
      if (cname == "estimator")
      {
        ptree child = it.second;
        std::string name = child.get<std::string>("name");
        if (name == "basic" || name == "Basic" || name == "standard")
        {
          basic_pt = child;
        }
        else if (name == "energy")
        {
          overwrite_default_energy = child.get<bool>("overwrite", false);
          remove_default_energy = child.get<bool>("remove", false);
        }
      }
    }

    basic_pt.put("measure_interval_multiplier", measure_interval_multiplier);
    if (remove_node_if_exists(basic_pt, "_population_control_interval"))
    {
      app_warning("'_population_control_interval' is set in a 'basic' estimator block. The value in the 'basic' estimator block will be ignored.");
    }
    basic_pt.put("_population_control_interval", population_control_interval); // to compute measure_interval
    est_pt.put("measure_interval_multiplier", measure_interval_multiplier);
    est_pt.put("_population_control_interval", population_control_interval); // to compute measure_interval

    estimators.emplace_back(
        static_cast<EstimPtr>(std::make_shared<BasicEstimator>(TGgen.getTG(1), info, title, basic_pt, impsamp)));
    measure_schedule[est_index] = estimators.back()->get_measurement_interval();
    est_index++;

    // add an EnergyEstimator if requested
    if (defaultEnergyEstim && 
            not(overwrite_default_energy or remove_default_energy) 
        )
      {
      estimators.emplace_back(
          static_cast<EstimPtr>(std::make_shared<EnergyEstimator>(TGgen.getTG(1), info, est_pt, wfn0, impsamp)));
        measure_schedule[est_index] = estimators.back()->get_measurement_interval();
        est_index++;
      }

    int bp_estimator(false);
    for(auto& it : exec_pt)
    {
      std::string cname = it.first;
      if (cname == "estimator")
      {
        est_pt = it.second;
        std::string name, wfn_name, ham_name;
        name = est_pt.get<std::string>("name");
        // Estimator can use different ham & wfn from Driver
        wfn_name = est_pt.get<std::string>("wfn", "");
        ham_name = est_pt.get<std::string>("ham", "");
        //  default is same ham & wfn from Driver
        
        int child_measure_interval_multiplier = est_pt.get<int>("measure_interval_multiplier", measure_interval_multiplier);
        if (remove_node_if_exists(est_pt, "_population_control_interval"))
        {
          app_warning("'_population_control_interval' is set in an estimator block. The value in the estimator block will be ignored!");
        }
        est_pt.put("_population_control_interval", population_control_interval); // to compute measure_interval


        if (name == "basic" || name == "Basic" || name == "standard")
        {
          // do nothing
          // first process estimators that do not need a wfn
        }
        else
        {
          // now do those that do

          Wavefunction* wfn = &wfn0;
          //Hamiltonian* ham = &ham0;
          // not sure how to do this right now
          if (wfn_name != "")
          { // wfn_name must produce a viable wfn object
            ptree wfn_pt = WfnFac.get_input(wfn_name);
            int nnodes = wfn_pt.get<int>("nnodes", 1);
            if (WfnFac.is_constructed(wfn_name))
            {
              wfn = std::addressof(
                  WfnFac.getWavefunction(TGgen.getTG(1), TGgen.getTG(nnodes), wfn_name, wfn0.getWalkerType(), nullptr));
            }
            else if (ham_name != "")
            {
              //APP_ABORT(" Estimator wfn must used default hamiltonian for execute block for now.");
              Hamiltonian& ham = HamFac.getHamiltonian(TGgen.gTG(), ham_name);
              wfn              = std::addressof(WfnFac.getWavefunction(TGgen.getTG(1), TGgen.getTG(nnodes), wfn_name,
                                                          wfn0.getWalkerType(), std::addressof(ham)));
            }
            else
            {
              Hamiltonian& ham = HamFac.getHamiltonian(TGgen.gTG(), ham0);
              wfn              = std::addressof(WfnFac.getWavefunction(TGgen.getTG(1), TGgen.getTG(nnodes), wfn_name,
                                                          wfn0.getWalkerType(), std::addressof(ham)));
            }
            if (wfn == nullptr)
            {
              app_error("WavefunctionFactory returned nullptr, check that given Wavefunction has been defined. ");
              APP_ABORT(" Error: Problems generating wavefunction in DriverFactory::executeAFQMCDriver(). ");
            }
          }

          if (name == "back_propagation")
          {
            if(bp_estimator) 
              APP_ABORT(" Error: Only one back propagator estimator allowed. ");
            est_pt.put("measure_interval_multiplier", child_measure_interval_multiplier);
            estimators.emplace_back(static_cast<EstimPtr>(
                std::make_shared<BackPropagatedEstimator>(TGgen.getTG(1), info, title, est_pt, walker_type, wset, *wfn,
                                                          prop0, impsamp)));
            measure_schedule[est_index] = estimators.back()->get_measurement_interval();
            est_index++;
            hdf_output = true;
            bp_estimator = true;
          }
          else if (name == "time_evolved_operators")
          {
            if(bp_estimator) 
              APP_ABORT(" Error: Only one back propagator estimator allowed. ");
            est_pt.put("measure_interval_multiplier", child_measure_interval_multiplier);
            estimators.emplace_back(static_cast<EstimPtr>(
                std::make_shared<BPWithTimeEvolvedOperators>(TGgen.getTG(1), info, title, 
                            exec_pt, est_pt, walker_type, wset, *wfn, prop0, impsamp)));
            measure_schedule[est_index] = estimators.back()->get_measurement_interval();
            est_index++;
            hdf_output = true;
            bp_estimator = true;
          }
          else if (name == "mixed")
          {
            est_pt.put("measure_interval_multiplier", child_measure_interval_multiplier);
            estimators.emplace_back(static_cast<EstimPtr>(
                std::make_shared<MixedEstimator>(TGgen.getTG(1), info, title, est_pt, walker_type, 
                                                 wset, *wfn)));
            measure_schedule[est_index] = estimators.back()->get_measurement_interval();
            est_index++;
            hdf_output = true;
          }
          else if (name == "energy")
          {
            // NOTE: do not put child_measure_interval into est_pt "measure_interval_multiplier"
            //       to ensure synchronization with other estimators that print to the
            //       scalar data file
            est_pt.put("measure_interval_multiplier", measure_interval_multiplier);
            bool remove = est_pt.get<bool>("remove", false);
            if(not remove)
              estimators.emplace_back(
                static_cast<EstimPtr>(std::make_shared<EnergyEstimator>(TGgen.getTG(1), info, est_pt, *wfn, impsamp)));
            measure_schedule[est_index] = estimators.back()->get_measurement_interval();
            est_index++;
          }
          else
          {
            app_log(1," Ignoring unknown estimator type: {}", name); 
          }
        }
      }
    }

    check_synchronized(); // for Estimators that print to the same line of the scalar.dat file

    if (TGgen.getTG(1).Global().rank() == 0)
    {
      //out.open(filename.c_str(),std::ios_base::app | std::ios_base::out);
      std::string filename = project_title + ".scalar.dat";
      if (hdf_output)
      {
        hdf_file = project_title + ".stat.h5";
        hdf_archive dump;
        if (!dump.create(hdf_file))
          APP_ABORT("Problems opening estimator hdf5 output file: " + hdf_file + ""); 
        write_hdf_metadata(dump, walker_type, !impsamp);
        dump.close();
      }
      out.open(filename.c_str());
      if (out.fail())
        APP_ABORT("Problems opening estimator output file: " + filename + ""); 
      out << "# block  time  ";
      for (std::vector<EstimPtr>::iterator it = estimators.begin(); it != estimators.end(); it++)
        (*it)->tags(out);
      out << "Eshift freeMemory ";
      estimators[0]->tags_timers(out);
      out << std::endl;
    }
  }

  ~EstimatorHandler() {}

  EstimatorHandler(EstimatorHandler const& other) = delete;
  EstimatorHandler& operator=(EstimatorHandler const& other) = delete;
  EstimatorHandler(EstimatorHandler&& other) = default;
  EstimatorHandler& operator=(EstimatorHandler&& other) = default;

  double getEloc() { return estimators[0]->getEloc(); }

  double getEloc_step() { return estimators[0]->getEloc_step(); }

  void print(int block, double time, double Es, WalkerSet& wlks)
  {
    hdf_archive dump;
    bool printed_row_prefix = false;
    if (hdf_output)
      dump.open(hdf_file);
   
    // print must follow the measure_schedule as well (otherwise the data may not be updated)
    long step = std::lround(time / dt);
    int estimator_index = 0;
    for (std::vector<EstimPtr>::iterator it = estimators.begin(); it != estimators.end(); it++, estimator_index++)
    {
      if (step % measure_schedule[estimator_index] == 0)
      { 
        if (estimator_index == 0 && not printed_row_prefix)
        {
          // print prefix
          out << block << " " << time << " ";
          printed_row_prefix = true;
        }
        (*it)->print(out, dump, wlks);
      }
    }

    if (printed_row_prefix)
    { 
      // print suffix
      out << std::setprecision(12) << Es << "  " << freemem() << " ";
      estimators[0]->print_timers(out);
      out << std::endl;
    }
    if (hdf_output)
      dump.close();
    if ((block + 1) % 10 == 0)
      out.flush();
  }

  // 1) acumulates estimators over steps, and 2) reduces and accumulates substep estimators
  void accumulate_step(double time, WalkerSet& wlks, std::vector<ComplexType>& curData)
  {
    for (std::vector<EstimPtr>::iterator it = estimators.begin(); it != estimators.end(); it++)
      (*it)->accumulate_step(time, wlks, curData);
  }

  // 1) acumulates estimators over steps, and 2) reduces and accumulates substep estimators
  /* Requests that each estimator measure. Will check the current time against the measurement schedule.*/
  void accumulate_block(double time, WalkerSet& wlks)
  {
    long step = std::lround(time / dt); // tmp for debug
    int estimator_index = 0;
    for (std::vector<EstimPtr>::iterator it = estimators.begin(); it != estimators.end(); it++, estimator_index++)
    {
      if (step % measure_schedule[estimator_index] == 0)
        (*it)->accumulate_block(time, wlks);
    }
  }

  void write_hdf_metadata(hdf_archive& dump, WALKER_TYPES wlk, bool free_projection)
  {
    dump.push("Metadata");
    dump.write(NMO, "NMO");
    dump.write(NAEA, "NAEA");
    dump.write(NAEB, "NAEB");
    int wlk_t_copy = wlk; // the actual data type of enum is implementation-defined. convert to int for file
    dump.write(wlk_t_copy, "WalkerType");
    dump.write(free_projection, "FreeProjection");
    dump.write(dt, "Timestep");
    dump.pop();
  }

  
  int get_max_common_interval()
  {
    return get_max_common_interval( std::vector<int> {});
  }

  /*
  We can compute the maximum interval which is commensurate with all estimators.
  It is simply the greatest common devisor among all measurement intervals.
  Any additional intervals to consider (for example the accumulate interval)
  can be passed in the intervals vector.

  The exception is if no gcd exists; in this case, we assume that the maximum interval is 1.

  Notes:
  - All EnergyEstimators and BasicEstimators need to have the same measurement interval since
    they print inline scalar data to the same outout file.
  */
  int get_max_common_interval(std::vector<int>&& intervals)
  { 
    int max = 0;
    int max_common_interval = 1;
    bool all_equal = true;

    for (auto& it : measure_schedule)
      intervals.push_back(it.second);

    for (auto& it : intervals)
      if (it > max)
        max = it;

    max_common_interval = max;
    for (auto& it : intervals)
      {
        max_common_interval = std::gcd(max_common_interval, it);
        if (it != max)
          all_equal = false;
      }
    
    if (max_common_interval == max && !all_equal)
      max_common_interval = 1;
    
    return max_common_interval;
  }


  void display_measurement_intervals()
  {
    //TODO: get a descriptive name for each estimator
    app_log(1, "\n\n======= Measurement Schedule: =========\n");
    for (auto& it : measure_schedule)
      app_log(1, "Estimator {} has measurement interval {}", it.first, it.second);
    app_log(1, "\n======================================\n\n");
  }

private:
  std::string project_title;

  std::vector<EstimPtr> estimators;
  std::vector<std::string> tags;

  std::map<int,int> measure_schedule;
  double dt; // timestep

  std::ofstream out;
  std::string hdf_file;
  bool hdf_output;

  /*
  Check that all Estimators that need to be synchronized are synchronized.
  This is important because some Estimators print to the same output file.

  The following set of Estimators need to be synchronized:
  - EnergyEstimator
  - BasicEstimator

  The following set of Estimators do not need to be synchronized:
  - BackPropagatedEstimator
  - BPWithTimeEvolvedOperators
  - MixedEstimator
  */
  void check_synchronized()
  {
    int synchronized_interval = -1;
    for (auto& it : measure_schedule)
    { 
      if (it.first < estimators.size())
      {
        auto estimator = estimators[it.first];
        if (std::dynamic_pointer_cast<BasicEstimator>(estimator) || std::dynamic_pointer_cast<EnergyEstimator>(estimator))
        {
          if (synchronized_interval == -1)
            synchronized_interval = it.second;
          else if (it.second != synchronized_interval)
            APP_ABORT(" Error: EnergyEstimator and BasicEstimator must have the same measurement interval.");
        }
      }
    }
  }

};
} // namespace afqmc
} // namespace sfqmc

#endif
