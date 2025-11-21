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

#include "AFQMC/Wavefunctions/NOMSD.hpp"
//#include "AFQMC/Wavefunctions/PHMSD.hpp"

namespace sfqmc
{
namespace afqmc
{

class Wavefunction //: public boost::variant<
                   //                      NOMSD<true,local_csr_Matrix<ComplexType>>,
                   //                      NOMSD<false,local_csr_Matrix<ComplexType>>,
                   //                      NOMSD<true,ComplexMatrix<node_allocator<ComplexType>>>,
                   //                      NOMSD<false,ComplexMatrix<node_allocator<ComplexType>>>,
                   //                      PHMSD<true>,
                   //                      PHMSD<false>>
{
public:
  Wavefunction() { APP_ABORT(" Error: Reached default constructor of Wavefunction. "); }

  explicit Wavefunction(NOMSD<HOST_MEMORY,PsiT_Matrix<HOST_MEMORY>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD<HOST_MEMORY,PsiT_Matrix<HOST_MEMORY>> const& other) = delete;

  explicit Wavefunction(NOMSD<HOST_MEMORY,memory::shared_array<HOST_MEMORY,ComplexType,2>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD<HOST_MEMORY,memory::shared_array<HOST_MEMORY,ComplexType,2>> const& other) = delete; 

#if defined(ENABLE_DEVICE)
  explicit Wavefunction(NOMSD<DEVICE_MEMORY,PsiT_Matrix<DEVICE_MEMORY>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD<DEVICE_MEMORY,PsiT_Matrix<DEVICE_MEMORY>> const& other) = delete;

  explicit Wavefunction(NOMSD<DEVICE_MEMORY,memory::shared_array<DEVICE_MEMORY,ComplexType,2>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD<DEVICE_MEMORY,memory::shared_array<DEVICE_MEMORY,ComplexType,2>> const& other) = delete;
#endif

/*
  explicit Wavefunction(PHMSD<true>&& other) : variant(std::move(other)) {}
  explicit Wavefunction(PHMSD<true> const& other) = delete;

  explicit Wavefunction(PHMSD<false>&& other) : variant(std::move(other)) {}
  explicit Wavefunction(PHMSD<false> const& other) = delete;
*/
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

  int number_of_references_for_back_propagation() const
  {
    return std::visit([&](auto&& a) { return a.number_of_references_for_back_propagation(); }, var);
  }

  WALKER_TYPES getWalkerType() const
  {
    return std::visit([&](auto&& a) { return a.getWalkerType(); }, var);
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
  void Overlap(Args&&... args)
  {
    std::visit([&](auto&& a) { a.Overlap(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  ComplexType getReferenceWeight(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.getReferenceWeight(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  auto getReferences(Args&&... args) const
  {
    std::visit([&](auto&& a) { a.getReferences(std::forward<Args>(args)...); }, var);
  }
/*
  template<class... Args>
  void accumulate_estimators(Args&&... args)
  {
    std::visit([&](auto&& a) { a.accumulate_estimators(std::forward<Args>(args)...); }, var);
  }

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

  template<class... Args>
  void getFieldTypes(Args&&... args)
  {
    std::visit([&](auto&& a) { a.getFieldTypes(std::forward<Args>(args)...); }, var);
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
/*
  template<class... Args>
  std::tuple<dev_csr_Matrix<ComplexType> const*, dev_csr_Matrix<ComplexType> const*> vHS_sparse(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.vHS_sparse(std::forward<Args>(args)...); }, var);
  }
*/
  
  private:

#if defined(ENABLE_DEVICE)
  std::variant<NOMSD<HOST_MEMORY,PsiT_Matrix<HOST_MEMORY>>,
               NOMSD<HOST_MEMORY,memory::shared_array<HOST_MEMORY,ComplexType,2>>,
               NOMSD<DEVICE_MEMORY,PsiT_Matrix<DEVICE_MEMORY>>,
               NOMSD<DEVICE_MEMORY,memory::shared_array<DEVICE_MEMORY,ComplexType,2>>
              > var;
#else
  std::variant<NOMSD<HOST_MEMORY,PsiT_Matrix<HOST_MEMORY>>,
               NOMSD<HOST_MEMORY,memory::shared_array<HOST_MEMORY,ComplexType,2>>> var;
#endif

};

} // namespace afqmc

} // namespace sfqmc

