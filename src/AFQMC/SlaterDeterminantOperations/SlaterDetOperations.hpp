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

#ifndef SFQMC_AFQMC_SLATERDETOPERATIONS_HPP
#define SFQMC_AFQMC_SLATERDETOPERATIONS_HPP

#include "AFQMC/config.h"
#include "boost/variant.hpp"

#include "Utilities/app_loggers.h"
#include "Memory/buffer_managers.h"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations_shared.hpp"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations_serial.hpp"

namespace sfqmc
{
namespace afqmc
{
class SlaterDetOperations : public boost::variant<
#if !defined(ENABLE_CUDA) && !defined(ENABLE_HIP)
                                    SlaterDetOperations_shared<ComplexType>,
#endif
                                    SlaterDetOperations_serial<ComplexType, DeviceBufferManager>
                                                 >
{
public:
  SlaterDetOperations() : variant()
  {
    app_warning(" WARNING: Building SlaterDetOperations with default constructor. ");
  }

#if !defined(ENABLE_CUDA) && !defined(ENABLE_HIP)
  explicit SlaterDetOperations(SlaterDetOperations_shared<ComplexType>&& other) : variant(std::move(other)) {}

  explicit SlaterDetOperations(SlaterDetOperations_shared<ComplexType> const& other) = delete;
#endif

  explicit SlaterDetOperations(SlaterDetOperations_serial<ComplexType, DeviceBufferManager> const& other) = delete;
  explicit SlaterDetOperations(SlaterDetOperations_serial<ComplexType, DeviceBufferManager>&& other)
      : variant(std::move(other))
  {}

  SlaterDetOperations(SlaterDetOperations const& other) = delete;
  SlaterDetOperations(SlaterDetOperations&& other)      = default;

  SlaterDetOperations& operator=(SlaterDetOperations const& other) = delete;
  SlaterDetOperations& operator=(SlaterDetOperations&& other) = default;

  // member functions visible outside the variant
  template<class... Args>
  ComplexType MixedDensityMatrix(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.MixedDensityMatrix(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void BatchedOverlap(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.BatchedOverlap(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void BatchedMixedDensityMatrix(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.BatchedMixedDensityMatrix(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void BatchedDensityMatrix(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.BatchedDensityMatrix(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void BatchedPropagate(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.BatchedPropagate(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  ComplexType MixedDensityMatrix_noHerm(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.MixedDensityMatrix_noHerm(std::forward<Args>(args)...); },
                                *this);
  }

  template<class... Args>
  ComplexType MixedDensityMatrixForWoodbury(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.MixedDensityMatrixForWoodbury(std::forward<Args>(args)...); },
                                *this);
  }

  template<class... Args>
  ComplexType MixedDensityMatrixFromConfiguration(Args&&... args)
  {
    return boost::
        apply_visitor([&](auto&& a) { return a.MixedDensityMatrixFromConfiguration(std::forward<Args>(args)...); },
                      *this);
  }

  template<class... Args>
  void BatchedMixedDensityMatrixForWoodbury(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.BatchedMixedDensityMatrixForWoodbury(std::forward<Args>(args)...); },
                                *this);
  }

  template<class... Args>
  void BatchedMixedDensityMatrixFromConfiguration(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.BatchedMixedDensityMatrixFromConfiguration(std::forward<Args>(args)...); },
                      *this);
  }

  template<class... Args>
  ComplexType Overlap(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.Overlap(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  ComplexType Overlap_noHerm(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.Overlap_noHerm(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  ComplexType OverlapForWoodbury(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.OverlapForWoodbury(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void BatchedOverlapForWoodbury(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.BatchedOverlapForWoodbury(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void Propagate(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.Propagate(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  ComplexType Orthogonalize(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.Orthogonalize(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void BatchedOrthogonalize(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.BatchedOrthogonalize(std::forward<Args>(args)...); }, *this);
  }
};

} // namespace afqmc

} // namespace sfqmc

#endif
