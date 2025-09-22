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

#ifndef SFQMC_AFQMC_ENERGYESTIMATOR_H
#define SFQMC_AFQMC_ENERGYESTIMATOR_H

#include "AFQMC/config.h"
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/Utilities/AFQMCTimer.h"

#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Walkers/WalkerConfig.hpp"


namespace sfqmc
{
namespace afqmc
{

// use atan2 to enforce [-pi,pi] range of values
inline ComplexType mod2pi(ComplexType x){
  RealType x_r = std::real(x);
  RealType x_i = std::imag(x);
  return ComplexType(atan2(sin(x_r),cos(x_r)),
                     atan2(sin(x_i),cos(x_i))
                     );
}

class EnergyEstimator : public EstimatorBase
{
public:
  EnergyEstimator(afqmc::TaskGroup_& tg_,
                  AFQMCInfo info,
                  ptree pt_in,
                  Wavefunction& wfn,
                  bool impsamp_ = true,
                  [[maybe_unused]] bool timer    = true)
      : EstimatorBase(info), 
        TG(tg_), 
        wfn0(wfn), 
        importanceSampling(impsamp_), 
        energy_components(false)
  {
    // convert user input to verbose input
    int population_control_interval;
    ptree pt = interpret_inputs(pt_in);
    app_log(1,"EnergyEstimator input:\n{}\n",io::to_string(pt));
    // initialize using verbose input
    print_sign = pt.get<bool>("print_sign");
    truncate = pt.get<bool>("truncate");
    energy_components = pt.get<bool>("print_components");
    nblocks_equil = pt.get<int>("equil");
    nblocks_skip = pt.get<int>("skip");
    population_control_interval = pt.get<int>("_population_control_interval");
    measure_interval = pt.get<int>("measure_interval_multiplier") * population_control_interval;

    data.resize(11); // hardcoded to the current number of working fields
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    bool print_components = pt0.get<bool>("print_components", false);
    bool print_sign       = pt0.get<bool>("print_sign", false);
    bool truncate         = pt0.get<bool>("truncate", false);
    int equil = pt0.get<int>("equil", 0);
    int skip  = pt0.get<int>("skip", 0);
    int measure_interval_multiplier = pt0.get<int>("measure_interval_multiplier", DEFAULT_MEASURE_INTERVAL_MULTIPLIER);
    int population_control_interval = pt0.get<int>("_population_control_interval", DEFAULT_POPULATION_CONTROL_INTERVAL);
    // validate inputs
    if (equil < 0 || skip < 0)
      APP_ABORT("EnergyEstimator: equil, skip must both be > 0");
    // create verbose internal inputs
    ptree pt1;
    pt1.put("print_components", print_components);
    pt1.put("print_sign", print_sign);
    pt1.put("truncate", truncate);
    pt1.put("equil", equil);
    pt1.put("skip", skip);
    pt1.put("measure_interval_multiplier", measure_interval_multiplier);
    pt1.put("_population_control_interval", population_control_interval);
    std::unordered_set<std::string> pass_through_keys = {
      "name",
      "overwrite",
      "remove"
    };
    io::compare_known_keys("Energy Estimator",pt1, pt0,pass_through_keys);
    return pt1;
  }

  ~EnergyEstimator() {}

  void accumulate_step([[maybe_unused]] double total_time, [[maybe_unused]] WalkerSet& wlks,
                       [[maybe_unused]] std::vector<ComplexType>& curData) {}

  void accumulate_block([[maybe_unused]] double total_time, WalkerSet& wset)
  {
    AFQMCTimer.start(energy_timer);
    size_t nwalk = wset.size();
    if (eloc.size(0) != nwalk || eloc.size(1) != 3)
      eloc = ComplexMatrix<device_allocator<ComplexType>>({nwalk, 3});
    if (ovlp.size(0) != nwalk)
      ovlp = ComplexVector<device_allocator<ComplexType>>(iextensions<1u>{nwalk});
    if (wprop.size(0) != 7 || wprop.size(1) != nwalk)
      wprop = ComplexMatrix<std::allocator<ComplexType>>({7, nwalk});

    ComplexType dum, et;
    // MAM: if nblocks_skip > 0, this will produce data filled with zeros.
    //      Can output to hdf5 instead for better format
    if( iblock < nblocks_equil or (iblock-nblocks_equil)%(nblocks_skip+1) != 0) {
      if (TG.TG_local().root())
        std::fill_n(data.begin(), data.size(), ComplexType(0.0));
    } else {
      wfn0.Energy(wset, eloc, ovlp);
      // in case GPU
      ComplexMatrix<std::allocator<ComplexType>> eloc_(eloc);
      ComplexVector<std::allocator<ComplexType>> ovlp_(ovlp);

      if (TG.TG_local().root())
      {

        int np = TG.TG_heads().size();
        int i0 = TG.TG_heads().rank()*nwalk;
        int nx = ( truncate ? (1) : (0));
        ComplexMatrix<std::allocator<ComplexType>> wet({4*nx,np*nwalk*nx},0.0);
        if(truncate) {
          for(int i=0; i<nwalk; i++) { 
            wet[0][i0 + i] = eloc_[i][0] + eloc_[i][1] + eloc_[i][2];
            wet[1][i0 + i] = eloc_[i][0];
            wet[2][i0 + i] = eloc_[i][1];
            wet[3][i0 + i] = eloc_[i][2];
          }
          TG.Global().all_reduce_in_place_n(wet.origin(), wet.num_elements(), std::plus<>());
          variance_based_truncation(wet[0](multi::ALL),3.0);
          variance_based_truncation(wet[1](multi::ALL),3.0);
          variance_based_truncation(wet[2](multi::ALL),3.0);
          variance_based_truncation(wet[3](multi::ALL),3.0);
        }

        wset.getProperty(WEIGHT, wprop[0]);
        wset.getProperty(OVLP, wprop[1]);
        wset.getProperty(PHASE, wprop[2]);
        wset.getProperty(PHASE1, wprop[3]);
        wset.getProperty(PHASE2, wprop[4]);
        wset.getProperty(PHASE3, wprop[5]);
        wset.getProperty(THETA, wprop[6]);
        std::fill_n(data.begin(), data.size(), ComplexType(0.0));
        for (int i = 0; i < nwalk; i++)
        {
          if (std::isnan(real(wprop[0][i])))
            continue;
          if (importanceSampling)
          {
            dum = (wprop[0][i]) * ovlp_[i] / (wprop[1][i]);
          }
          else
          {
            dum = (wprop[0][i]) * ovlp_[i] * (wprop[2][i]);
          }
          if(truncate) {
            et = wet[0][ i0 + i ]; 
          } else {
            et = eloc_[i][0] + eloc_[i][1] + eloc_[i][2];
          }
          if ((!std::isfinite(real(dum))) || (!std::isfinite(real(et * dum))))
            continue;
          data[1] += dum;
          data[0] += et * dum;
          if(truncate) {
            data[2] += wet[1][i0+i] * dum;
            data[3] += wet[2][i0+i] * dum;
            data[4] += wet[3][i0+i] * dum;
          } else {
            data[2] += eloc_[i][0] * dum;
            data[3] += eloc_[i][1] * dum;
            data[4] += eloc_[i][2] * dum;
          }
          data[5] += ( dum / std::abs(dum) );  
          data[6] += ( wprop[2][i] );  
          data[7] += ( wprop[3][i] );  
          data[8] += ( wprop[4][i] );  
          data[9] += ( wprop[5][i] );
          data[10] += ( mod2pi(wprop[6][i]) );

        }
        TG.TG_heads().all_reduce_in_place_n(data.begin(), data.size(), std::plus<>());
      }
    }
    // increase counter
    iblock ++;
    AFQMCTimer.stop(energy_timer);
  }

  void tags(std::ofstream& out)
  {
    if (TG.Global().root())
    {
      out << "EnergyEstim_" << name << "_nume_real  EnergyEstim_" << name << "_nume_imag "
          << "EnergyEstim_" << name << "_deno_real  EnergyEstim_" << name << "_deno_imag "
          << "EnergyEstim_" << name << "_timer ";
      if(print_sign) 
      {
        /*
        printing the real and imaginary parts of:
        - data[5] = dum /std::abs(dum)  with dum = w_i * O_i^(n+1) / O_i^(n)   (i is a walker index)
        - data[6] = wset property PHASE  (KE: unclear exactly what phase this is)
        - data[7] = PHASE1 = prod_k exp(-(i dt)/2 * (E_k + E_k')) / scale  : where scale is the constraint factor
        - data[8] = PHASE2 = O_i^(n+1) : i.e. the "new" overlap
        - data[9] = PHASE3 = w_i * scale : i.e. the "new" weight (this is cumulative product!)

        Printing just the real part of the following
        - THETA = arg( O_i^(n+1) / O_i^(n) ) - Im[ x <V> ]  : real by construction (note: <V> is the mean-field subtraction)
        */
        out << " Energy_sign_real   Energy_sign_imag "
            << "  Phase_real   Phase_imag   "
            << "  Phase1_real   Phase1_imag   "
            << "  NewOverlap_real   NewOverlap_imag   " 
            << "  ConstraintWeight_real   ConstraintWeight_imag   "
            << "  ConstraintTheta_real    ConstraintTheta_imag  ";
      }  
      if (energy_components)
      {
        out << "OneBodyEnergyEstim__nume_real "
            << "EXXEnergyEstim__nume_real "
            << "ECoulEnergyEstim__nume_real ";
      }
    }
  }

  int get_measurement_interval()
  {
    return measure_interval;
  }

  void print(std::ofstream& out, [[maybe_unused]] hdf_archive& dump, WalkerSet& wset)
  {
    if (TG.Global().root())
    {
      int n = wset.get_global_target_population();
      out << data[0].real() / n << " " << data[0].imag() / n << " " << data[1].real() / n << " " << data[1].imag() / n
          << " " << AFQMCTimer.elapsed(energy_timer) << " ";
      if(print_sign) 
      {
        out <<data[5].real() / n <<" " <<data[5].imag() / n <<" " 
            <<data[6].real() / n <<" " <<data[6].imag() / n <<" "
            <<data[7].real() / n <<" " <<data[7].imag() / n <<" " 
            <<data[8].real() / n <<" " <<data[8].imag() / n <<" " 
            <<data[9].real() / n <<" " <<data[9].imag() / n <<" "
            <<data[10].real() / n <<" " <<data[10].imag() / n <<" ";
      }
      if (energy_components)
      {
        out << data[2].real() / n << " " << data[3].real() / n << " " << data[4].real() / n << " ";
      }
      AFQMCTimer.reset(energy_timer);
    }
  }

private:
  std::string name;

  TaskGroup_& TG;

  Wavefunction& wfn0;

  int nblocks_skip = 0;
  int nblocks_equil = 0;
  int iblock = 0;
  int measure_interval = 1;

  ComplexMatrix<device_allocator<ComplexType>> eloc;
  ComplexVector<device_allocator<ComplexType>> ovlp;
  ComplexMatrix<std::allocator<ComplexType>> wprop;

  std::vector<std::complex<double>> data;

  bool importanceSampling = true;
  bool energy_components = false;
  bool print_sign = false;
  bool truncate = false;

};
} // namespace afqmc
} // namespace sfqmc

#endif
