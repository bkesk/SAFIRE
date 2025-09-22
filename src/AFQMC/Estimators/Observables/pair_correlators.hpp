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

#ifndef SFQMC_AFQMC_PAIRCORR_HPP
#define SFQMC_AFQMC_PAIRCORR_HPP

#include "AFQMC/config.h"
#include <vector>
#include <string>
#include <iostream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Numerics/ma_operations.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged spin*spin correlation
 */
class pair_correlator : public AFQMCInfo
{
  // allocators
  using Allocator = device_allocator<ComplexType>;

  // type defs
  using pointer       = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;

  using CVector_ref    = boost::multi::array_ref<ComplexType, 1, pointer>;
  using CMatrix_ref    = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CVector        = boost::multi::array<ComplexType, 1, Allocator>;
  using CMatrix        = boost::multi::array<ComplexType, 2, Allocator>;
  using IMatrix        = boost::multi::array<int, 2, device_allocator<int>>;
  using stdCVector_ref = boost::multi::array_ref<ComplexType, 1>;
  using stdCMatrix_ref = boost::multi::array_ref<ComplexType, 2>;
  using mpi3IMatrix    = boost::multi::array<int, 2, shared_allocator<int>>;
  using mpi3CVector    = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3CMatrix    = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using mpi3CTensor    = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;
  using mpi3C4Tensor   = boost::multi::array<ComplexType, 4, shared_allocator<ComplexType>>;

  using mpi3IMatrix_ref = boost::multi::array_ref<int, 2, shared_allocator<int>>;

public:
   pair_correlator(afqmc::TaskGroup_& tg_, AFQMCInfo& info, ptree pt0, WALKER_TYPES wlk, int nave_ = 1, int bsize = 1)
      : AFQMCInfo(info),
        block_size(bsize),
        nave(nave_),
        num_correlators(0),
        TG(tg_),
        walker_type(wlk),
        writer(false),
        hdf_walker_output(""),
        pair_map({}),
        PairCorrAverage({0, 0, 0}, shared_allocator<ComplexType>{TG.TG_local()})
  {

    std::string base_error(" Error in pair_correlator::pair_correlator(): \n    ");

    ptree pt = interpret_inputs(pt0);

    app_log(1,"  --  Pair Correlator (PairCorr) estimator. -- ");
    hdf_walker_output = pt.get<std::string>("walker_output", "");
    filename = pt.get<std::string>("filename","");

    for (const auto& item : pt0.get_child("pair_type")) {
      correlator_names.push_back(item.second.get_value<std::string>());
    }

    if (walker_type == CLOSED)
    {
      APP_ABORT("pair_correlator not yet implemented for CLOSED walkers. Use COLLINEAR or NONCOLLINEAR Walkers instead.");
    }
    else if (walker_type == FULLYPOLARIZED)
    {
      APP_ABORT("pair_correlator not yet implemented for FULLYPOLARIZED Walkers.");
    }

    hdf_archive indump;
    if (TG.TG_local().root())
    {
      app_log(1, "reading pair correlators from: {}", filename);
      if (!indump.open(filename, H5F_ACC_RDONLY))
        APP_ABORT(base_error + "Problems opening pair_correlator file.");
      if (indump.push("PairCorrelator", false)<0)
        APP_ABORT(base_error + "Group PairCorrelator not found.");
      indump.readEntry(num_pair,"orbital_map/num_pair");
      // iterate over the correlator names
      for (auto& correlator_name : correlator_names)
      {
        IMatrix pair_map_current = IMatrix({num_pair,1});
        if (!indump.readEntry(pair_map_current,"orbital_map/" + correlator_name))
          app_error("Problems reading pair_map for correlator: {}", correlator_name);
        pair_map[num_correlators] = pair_map_current;
        num_correlators++;
      }
      indump.pop();
      indump.close();
    }
    TG.Node().barrier();

    if (hdf_walker_output != std::string(""))
    {
      hdf_walker_output = "G" + std::to_string(TG.TG_heads().rank()) + "_" + hdf_walker_output;
      hdf_archive dump;
      if (not dump.create(hdf_walker_output))
      {
        app_error("Problems creating walker output hdf5 file: {}", hdf_walker_output);
        APP_ABORT("Problems creating walker output hdf5 file.");
      }
      if (writer)
      {
        dump.push("PairCorrelator");
        dump.push("Metadata");
        dump.write(NMO, "NMO");
        dump.write(NAEA, "NUP");
        dump.write(NAEB, "NDOWN");
        dump.write(num_correlators, "NCORRELATORS");
        // TODO: find a better to write correlator names to HDF5
        for (int i = 0; i < num_correlators; i++)
        { // this isn't writing anything for some reason?? run through the debugger
          dump.write(correlator_names[i], "CORRELATOR_NAMES_" + std::to_string(i));
        }
        int wlk_t_copy = walker_type; // the actual data type of enum is implementation-defined. convert to int for file
        dump.write(wlk_t_copy, "WalkerType");
        dump.pop();
        dump.pop();
      }
      dump.close();
    }

    using std::fill_n;
    writer  = (TG.Global().rank() == 0);
    correlator_names_printed = false;

    dm_size = NMO*NMO;
    correlators_size = num_correlators*num_correlators;

    PairCorrAverage = mpi3CTensor({nave, correlators_size,dm_size}, shared_allocator<ComplexType>{TG.TG_local()});
    fill_n(PairCorrAverage.origin(), PairCorrAverage.num_elements(), ComplexType(0.0, 0.0));
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string hdf_walker_output, filename, name;
    std::vector<std::string> correlator_names;
  
    name = pt0.get<std::string>("name","pair_correlator");
    hdf_walker_output = pt0.get<std::string>("walker_output", "");
    filename = pt0.get<std::string>("filename","");
    
    for (const auto& item : pt0.get_child("pair_type")) {
      correlator_names.push_back(item.second.get_value<std::string>());
    }

    if (correlator_names.empty())
      APP_ABORT("pair_correlator: No pair_type specified. Please specify at least one pair_type.");

    hdf_walker_output = pt0.get<std::string>("walker_output", "");
    filename = pt0.get<std::string>("filename","");
    
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("walker_output", hdf_walker_output);
    pt1.put("filename", filename);

    ptree pair_type_node;
    for (const auto& correlator_name : correlator_names) {
      ptree child;
      child.put("", correlator_name);
      pair_type_node.push_back(std::make_pair("", child));
    }
    pt1.add_child("pair_type", pair_type_node);

    app_log(1, "pair_correlator pt1 ");
    io::to_string(pt1);
    
    io::compare_known_keys("pair_correlators observable",pt1, pt0);
    return pt1;
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  template<class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, MatG&& G, MatG_host&& G_host, HostCVec1&& Xw, [[maybe_unused]] bool impsamp)
  {
    std::string base_error(" Error in pair_correlator::pair_correlator(): \n    ");
    static_assert(std::decay<MatG>::type::dimensionality == 4, "Wrong dimensionality");
    static_assert(std::decay<MatG_host>::type::dimensionality == 4, "Wrong dimensionality");
    using std::copy_n;
    using std::fill_n;

    // assumes G[nwalk][spin][M][M]
    int nw(G.size(0));
    RUNTIME_CHECK(G.size(0) == Xw.size(0), "");

    // no parallelization over ncores for now, fix if needed 
    if(TG.TG_local().root())
    {
      for (int iw = 0; iw < nw; iw++)
      {
        for (int alpha = 0; alpha < num_correlators; alpha++)
        {
          IMatrix const pair_map_i = this->pair_map.at(alpha); 
          for (int beta = 0; beta < num_correlators; beta++)
          {
            IMatrix const pair_map_j = this->pair_map.at(beta);
            int pair_ind = alpha * num_correlators + beta;
            ComplexType* ptr(PairCorrAverage[iav][pair_ind].origin());
            
            if (walker_type == CLOSED)
            {
              APP_ABORT(base_error + "accumulate not yet implemented for CLOSED walkers.");
            }
            else if (walker_type == COLLINEAR)
            {
              auto&& Gup_ = G_host[iw][0];
              auto&& Gdn_ = G_host[iw][1];
              for (int i = 0; i < NMO; i++)
              {
                int ibar = pair_map_i[i][0];
                if (ibar < 0)  // negative index means no valid pair!
                  continue;
                for (int j = i + 1; j < NMO; j++, ptr++)
                {
                  int jbar = pair_map_j[j][0];
                  if (jbar < 0)  // negative index means no valid pair!
                    continue;
                  //term 1
                  *ptr += 0.5 * Xw[iw] * Gup_[i][j] * Gdn_[ibar][jbar];
                  //term 2
                  *ptr += 0.5 * Xw[iw] * Gup_[i][jbar] * Gdn_[ibar][j];
                  //term 3  
                  *ptr += 0.5 * Xw[iw] * Gdn_[i][jbar] * Gup_[ibar][j];
                  //term 4
                  *ptr += 0.5 * Xw[iw] * Gdn_[i][j] * Gup_[ibar][jbar];
                }
              }
            }
            else if (walker_type == FULLYPOLARIZED)
            { 
              APP_ABORT(base_error + "accumulate not yet implemented for FULLYPOLARIZED walkers.");
            }
            else if (walker_type == NONCOLLINEAR) {
              auto&& G_ = G_host[iw][0]; // contains all spins
              for (int i = 0; i < NMO; i++)
              {
                int ibar = pair_map_i[i][0];
                if (ibar < 0)  // negative index means no valid pair!
                  continue;
                for (int j = 0; j < NMO; j++, ptr++)
                {
                  int jbar = pair_map_j[j][0]; 
                  if (jbar < 0)  // negative index means no valid pair!
                    continue;
                  //term 1
                  *ptr += 0.5 * Xw[iw] * (G_[i][j] * G_[NMO + ibar][NMO + jbar] - G_[i][NMO + jbar] * G_[NMO + ibar][j]);
                  //term 2
                  *ptr += 0.5 * Xw[iw] * (G_[i][jbar] * G_[NMO + ibar][NMO + j] - G_[i][NMO + j] * G_[NMO + ibar][jbar]);
                  //term 3  
                  *ptr += 0.5 * Xw[iw] * (G_[NMO+i][NMO+jbar] * G_[ibar][j] - G_[NMO+i][j] * G_[ibar][NMO+jbar]);
                  //term 4
                  *ptr += 0.5 * Xw[iw] * (G_[NMO + i][NMO + j] * G_[ibar][jbar] - G_[NMO + i][jbar] * G_[ibar][NMO+j]);
                }
              }
            }
          } // loop over beta
        } // loop over alpha
     } // loop over walkers 
   }
    TG.TG_local().barrier();
  }

  // Second interface, including factorized G in addition to full G and G_host 
  template<class Mat1, class Mat2, class Mat3, class Mat4,
           class MatG, class MatG_host, class HostCVec1>
  void accumulate(int iav, [[maybe_unused]] Mat1&& Sa, [[maybe_unused]] Mat2&& Ga, 
                  [[maybe_unused]] Mat3&& Sb, [[maybe_unused]] Mat4&& Gb,
                  MatG&& G, MatG_host&& G_host, HostCVec1&& Xw, bool impsamp)
  {
    accumulate(iav,std::forward<MatG>(G),std::forward<MatG_host>(G_host),
                   std::forward<HostCVec1>(Xw),impsamp);
  }


/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/
  template<class... Args>
  void accumulate_reference_configuration([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_reference_configuration ");
  }

  template<class... Args>
  void accumulate_excited_configuration_first([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_first ");
  }

  template<class... Args>
  void accumulate_excited_configuration_second([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_second ");
  }

  template<class HostCVec>
  void print(int iblock, hdf_archive& dump, HostCVec&& Wsum)
  {
    using std::fill_n;
    const int n_zero = 9;

    if (TG.TG_local().root())
    {
      
      ma::scal(ComplexType(1.0 / block_size), PairCorrAverage);
      TG.TG_heads().reduce_in_place_n(raw_pointer_cast(PairCorrAverage.origin()), PairCorrAverage.num_elements(), std::plus<>(), 0);
      if (writer)
      {
        dump.push(std::string("PairCorr"));
        if (!correlator_names_printed){
          dump.push("Metadata");
          dump.write(num_correlators, "NCORRELATORS");
          for (int i = 0; i < num_correlators; i++)
            dump.write(correlator_names[i], "CORRELATOR_NAME_" + std::to_string(i));
          dump.pop();
          correlator_names_printed = true;
        }
        for (int i = 0; i < nave; ++i)
        {
          dump.push(std::string("Average_") + std::to_string(i));
          std::string padded_iblock =
              std::string(n_zero - std::to_string(iblock).length(), '0') + std::to_string(iblock);
          // TODO: write PairCorrAverage flattened over alpha and beta
          for (int alpha = 0; alpha < num_correlators; alpha++)
          {
            for (int beta = 0; beta < num_correlators; beta++)
            {
              stdCVector_ref PairCorrAverage_(raw_pointer_cast(PairCorrAverage[i][alpha * num_correlators + beta].origin()), {dm_size});
              std::string out_name = "P_" + std::to_string(alpha) + "_" + std::to_string(beta) + "_ij_" + padded_iblock;
              dump.write(PairCorrAverage_, out_name);
              dump.write(Wsum[i], "denominator_" + padded_iblock);
            }
          }
          dump.pop();
        }
        dump.pop();
      }
    }
    TG.TG_local().barrier();
    fill_n(PairCorrAverage.origin(), PairCorrAverage.num_elements(), ComplexType(0.0, 0.0));
  }

private:
  int block_size;

  int nave;

  int num_pair, num_correlators; // number of correlators which are actually used! (vs number of correlators defined in the HDF5 input)

  std::string filename;

  TaskGroup_& TG;

  WALKER_TYPES walker_type;

  int dm_size, correlators_size;

  bool writer;
  bool correlator_names_printed;

  std::string hdf_walker_output;

  std::vector<std::string> correlator_names;
  std::map<int, IMatrix> pair_map;
  mpi3CTensor PairCorrAverage;
};

} // namespace afqmc
} // namespace sfqmc

#endif