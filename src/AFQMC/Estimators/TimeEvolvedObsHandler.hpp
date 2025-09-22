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

#ifndef SFQMC_AFQMC_TIMEEVOLVEDOBSHANDLER_HPP
#define SFQMC_AFQMC_TIMEEVOLVEDOBSHANDLER_HPP

#include <vector>
#include <string>
#include <iostream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"
#include "AFQMC/Utilities/taskgroup.h"

#include "AFQMC/Estimators/Observables/Observable.hpp"
#include "AFQMC/config.h"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "Numerics/ma_operations.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "Memory/buffer_managers.h"

namespace sfqmc
{
namespace afqmc
{
/*
 * Handler class for time-evolved observables. 
 */
class TimeEvolvedObsHandler : public AFQMCInfo
{
  // allocators
  using sharedAllocator = localTG_allocator<ComplexType>;

  using shared_pointer       = typename std::allocator_traits<sharedAllocator>::pointer;
  using const_shared_pointer = typename std::allocator_traits<sharedAllocator>::const_pointer;

  using devCMatrix_ptr = boost::multi::array_ptr<ComplexType, 2, device_ptr<ComplexType>>;

  using sharedCVector      = boost::multi::array<ComplexType, 1, sharedAllocator>;
  using sharedCVector_ref  = boost::multi::array_ref<ComplexType, 1, shared_pointer>;
  using sharedCMatrix_ref  = boost::multi::array_ref<ComplexType, 2, shared_pointer>;
  using sharedC4Tensor_ref = boost::multi::array_ref<ComplexType, 4, shared_pointer>;

  using mpi3C4Tensor = boost::multi::array<ComplexType, 4, shared_allocator<ComplexType>>;

  using stdCVector     = boost::multi::array<ComplexType, 1>;
  using stdCMatrix     = boost::multi::array<ComplexType, 2>;
  using stdCVector_ref = boost::multi::array_ref<ComplexType, 1>;

  using shm_stack_alloc_type = LocalTGBufferManager::template allocator_t<ComplexType>;
  using StaticSHMVector       = boost::multi::static_array<ComplexType, 1, shm_stack_alloc_type>;
  using StaticSHMMatrix       = boost::multi::static_array<ComplexType, 2, shm_stack_alloc_type>;
  using StaticSHM3Tensor       = boost::multi::static_array<ComplexType, 3, shm_stack_alloc_type>;
  using StaticSHM4Tensor      = boost::multi::static_array<ComplexType, 4, shm_stack_alloc_type>;

public:
  TimeEvolvedObsHandler(afqmc::TaskGroup_& tg_,
                 AFQMCInfo& info,
                 std::string name_,
                 ptree pt,
                 WALKER_TYPES wlk,
                 int nave,
                 Wavefunction& wfn)
      : AFQMCInfo(info),
        TG(tg_),
        walker_type(wlk),
        wfn0(wfn),
        number_of_averages(nave),
        name(name_)
  {
    using std::fill_n;
    block_size = pt.get<int>("block_size", 1);
    if(number_of_averages <= 0)
      APP_ABORT("Error:  Empty measure_at.");

    for(const ptree::value_type &it : pt)
    {
      std::string cname = it.first;
      io::tolower(cname);
      if (cname == "onerdm")
      {
        properties_1body.emplace_back(Observable(full1rdm(TG, info, it.second, walker_type, number_of_averages, block_size)));
      }
      else if (cname == "gfock" || cname == "genfock" || cname == "ekt")
      {
        properties.emplace_back(Observable(
            generalizedFockMatrix(TG, info, it.second, walker_type, wfn0, number_of_averages, block_size)));
      }
      else if (cname == "diag2rdm")
      {
        properties.emplace_back(Observable(diagonal2rdm(TG, info, it.second, walker_type, number_of_averages, block_size)));
      }
      else if (cname == "twordm")
      {
        properties.emplace_back(Observable(full2rdm(TG, info, it.second, walker_type, number_of_averages, block_size)));
      }
      else if (cname == "n2r" || cname == "ontop2rdm")
      {
#if defined(ENABLE_DEVICE)
        ptree pt1 = it.second;
        bool use_host_memory = pt1.get<bool>("use_host_memory", false);
        if (use_host_memory)
        {
          properties.emplace_back(Observable(
              n2r<device_allocator<ComplexType>>(TG, info, it.second, walker_type, false, device_allocator<ComplexType>{},
                                                 device_allocator<ComplexType>{}, number_of_averages, block_size)));
        }
        else
#endif
        {
          properties.emplace_back(Observable(
              n2r<shared_allocator<ComplexType>>(TG, info, it.second, walker_type, true,
                                                 shared_allocator<ComplexType>{TG.TG_local()},
                                                 shared_allocator<ComplexType>{TG.Node()}, number_of_averages, block_size)));
        }
      }
      else if (cname == "realspace_correlators")
      {
        properties.emplace_back(Observable(realspace_correlators(TG, info, it.second, walker_type, number_of_averages, block_size)));
      }
      else if (cname == "correlators")
      {
        properties.emplace_back(Observable(atomcentered_correlators(TG, info, it.second, walker_type, number_of_averages, block_size)));
      }
      else if (cname == "spinspin")
      {
        properties.emplace_back(Observable(spinspinobs(TG, info, it.second, walker_type, number_of_averages, block_size)));
      }
    }

    if (properties.size() == 0 && properties_1body.size() == 0)
      APP_ABORT("empty observables list is not allowed.");

    writer = (TG.Global().rank() == 0);

    denominator = stdCVector(iextensions<1u>{number_of_averages});
    ma::fill(denominator, ComplexType(0.0, 0.0));
  }

  void print(int iblock, hdf_archive& dump)
  {
    using std::fill_n;

    if (TG.TG_local().root())
    {
      ma::scal(ComplexType(1.0 / block_size), denominator);
      TG.TG_heads().reduce_in_place_n(raw_pointer_cast(denominator.origin()), denominator.num_elements(), std::plus<>(), 0);
    }

    for (auto& v : properties_1body)
      v.print(iblock, dump, denominator);
    for (auto& v : properties)
      v.print(iblock, dump, denominator);
    ma::fill(denominator, ComplexType(0.0, 0.0));
  }

  // call for MixedEstimator
  template<class WlkSet, class TVec>
  void accumulate(int iav, WlkSet& wset, TVec& wgt, bool importanceSampling)
  {
    // accumulate denominator
    if (importanceSampling)
      denominator[iav] += std::accumulate(wgt.begin(), wgt.end(), ComplexType(0.0));
    else
    {
      APP_ABORT(" Finish implementation of free projection. \n\n");
    }

    LocalTGBufferManager shm_buffer_manager;
    StaticSHM4Tensor dummy({0,0,0,0},
            shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    wfn0.accumulate_estimators(iav, wset, wgt, properties_1body, properties, 
                               dummy, dummy, dummy, false, importanceSampling);
  }

  // call for TimeEvolvedOperators
  template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3,
            typename = typename std::enable_if_t<std::decay_t<Mat1>::dimensionality == 4>,
            typename = typename std::enable_if_t<std::decay_t<Mat2>::dimensionality == 4>,
            typename = typename std::enable_if_t<std::decay_t<Mat3>::dimensionality == 4>
            >
  void accumulate(int iav, WlkSet& wset, TVec& wgt, Mat1 const& X, Mat2 const& Y, Mat3 const& M, 
                  bool importanceSampling)
  {
    // accumulate denominator
    if (importanceSampling)
      denominator[iav] += std::accumulate(wgt.begin(), wgt.end(), ComplexType(0.0));
    else
    {
      APP_ABORT(" Finish implementation of free projection. \n\n");
    } 

    wfn0.accumulate_estimators(iav, wset, wgt, properties_1body, properties, 
                               X, Y, M, true, importanceSampling);
  }

private:
  TaskGroup_& TG;

  WALKER_TYPES walker_type;

  Wavefunction& wfn0;

  bool writer = false;

  int block_size = 1;

  int number_of_averages = 1;

  std::string name;

  std::vector<Observable> properties_1body;
  std::vector<Observable> properties;

  stdCVector denominator;

};

} // namespace afqmc

} // namespace sfqmc

#endif
