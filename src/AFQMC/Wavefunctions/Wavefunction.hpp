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

#pragma once

#include <variant>
#include "AFQMC/config.h"

#include "numerics/shared_array/const_shared_array.hpp"
#include "AFQMC/Wavefunctions/NOMSD.hpp"
#include "AFQMC/Wavefunctions/PHMSD.hpp"
#include "AFQMC/Wavefunctions/NOMSD_FT.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class Wavefunction 
{
public:
  Wavefunction() = delete;

  explicit Wavefunction(NOMSD<MEM,PsiT_Matrix<MEM>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD<MEM,PsiT_Matrix<MEM>> const& other) : var(other) {} 

  explicit Wavefunction(NOMSD<MEM,memory::const_shared_array<MEM,ComplexType,2>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD<MEM,memory::const_shared_array<MEM,ComplexType,2>> const& other) : var(other) {}  

  explicit Wavefunction(PHMSD<MEM>&& other) : var(std::move(other)) {}
  explicit Wavefunction(PHMSD<MEM> const& other) : var(other) {} 
  
  // Add finite-T NOMSD wavefunctions
  explicit Wavefunction(NOMSD_FT<MEM,PsiT_Matrix<MEM>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD_FT<MEM,PsiT_Matrix<MEM>> const& other) = delete;

  explicit Wavefunction(NOMSD_FT<MEM,memory::const_shared_array<MEM,ComplexType,2>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD_FT<MEM,memory::const_shared_array<MEM,ComplexType,2>> const& other) = delete; 


  Wavefunction(Wavefunction const& other) = delete;
  Wavefunction(Wavefunction&& other)      = default;

  Wavefunction& operator=(Wavefunction const& other) = delete;
  Wavefunction& operator=(Wavefunction&& other) = default;

  /*
   * Returns the memory space.
   */
  auto get_memory_space() const 
  {
    return std::visit([&](auto&& a) { return a.get_memory_space(); }, var);
  }

  int number_of_cholesky_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_cholesky_vectors(); }, var);
  }

  template<class WlkSet>
  void runtime_optimization(WlkSet& wset)
  {
    std::visit([&](auto&& a) { a.runtime_optimization(wset); }, var);
  }

  WALKER_TYPES getWalkerType() const
  {
    return std::visit([&](auto&& a) { return a.getWalkerType(); }, var);
  }

  bool isFiniteTemperature() const
  {
    return std::visit([&](auto&& a) { return a.isFiniteTemperature(); }, var);
  }

  template<class... Args>
  void vMF(Args&&... args)
  {
    std::visit([&](auto&& a) { a.vMF(std::forward<Args>(args)...); }, var);
  }

  auto G_MF()
  {
    return std::visit([&](auto&& a) { return a.G_MF(); }, var);
  }

  template<class... Args>
  void vbias(Args&&... args)
  {
    std::visit([&](auto&& a) { a.vbias(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  auto vHS(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.vHS(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  auto vHS_sparse(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.vHS_sparse(std::forward<Args>(args)...); }, var);
  }

  auto vHS_dims() const
  {
    return std::visit([&](auto&& a) { return a.vHS_dims(); }, var);
  }


  template<class... Args>
  void Energy(Args&&... args)
  {
    std::visit([&](auto&& a) { a.Energy(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void DensityMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.DensityMatrix(std::forward<Args>(args)...); }, var);
  }

  
  template<class... Args>
  void MixedDensityMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.MixedDensityMatrix(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void Log_Overlap(Args&&... args)
  {
    std::visit([&](auto&& a) { a.Log_Overlap(std::forward<Args>(args)...); }, var);
  }
  

  auto total_number_of_references() const
  {
    return std::visit([&](auto&& a) { return a.total_number_of_references(); }, var);
  }

  int getNMO() const
  {
    return std::visit([&](auto&& a) { return a.getNMO(); }, var);
  }

  template<class... Args>
  ComplexType getReferenceWeight(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.getReferenceWeight(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void getReferences(Args&&... args) 
  {
    std::visit([&](auto&& a) { a.getReferences(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void accumulate_estimators(Args&&... args)
  {
    std::visit([&](auto&& a) { a.accumulate_estimators(std::forward<Args>(args)...); }, var);
  }
/*
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.generalizedFockMatrix(std::forward<Args>(args)...); }, var);
  } 
*/

  HamiltonianTypes getHamType() const
  {
    return std::visit([&](auto&& a) { return a.getHamType(); }, var);
  }

  auto getFieldTypes()
  {
    return std::visit([&](auto&& a) { return a.getFieldTypes(); }, var);
  }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    std::visit([&](auto&& a) { a.update_potentials(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  auto getOneBodyPropagatorMatrix(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.getOneBodyPropagatorMatrix(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void updateLogScale(Args&&... args)
  {
    std::visit([&](auto&& a) { a.updateLogScale(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  auto getLogScale(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.getLogScale(std::forward<Args>(args)...); }, var);
  }
  
  void resetLogScale()
  {
    std::visit([&](auto&& a) { a.resetLogScale(); }, var);
  }

  template<class... Args>
  void setLogPT0(Args&&... args)
  {
    std::visit([&](auto&& a) { a.setLogPT0(std::forward<Args>(args)...); }, var);
  }

  //auto getLogPT0()
  //{
  //  return std::visit([&](auto&& a) -> decltype(auto) { return a.getLogPT0(); }, var);
  //}

  auto getLogPT0()
  {
    using R = memory::array<MEM,ComplexType,1>;
    return std::visit([&](auto&& a) -> R { return a.getLogPT0(); }, var);
  }



  private:


  std::variant<NOMSD<MEM,PsiT_Matrix<MEM>>,
               NOMSD<MEM,memory::const_shared_array<MEM,ComplexType,2>>,
               NOMSD_FT<MEM,PsiT_Matrix<MEM>>,
               NOMSD_FT<MEM,memory::const_shared_array<MEM,ComplexType,2>>,
               PHMSD<MEM>
              > var;

};

} // namespace afqmc

} // namespace sfqmc

