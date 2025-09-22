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

#ifndef SFQMC_AFQMC_MODELCOMPONENT_HPP
#define SFQMC_AFQMC_MODELCOMPONENT_HPP


#include "AFQMC/config.h"
#include "boost/variant.hpp"

#include "AFQMC/HamiltonianOperations/ModelComponents/Continuous_GeneralUJ.hpp"
#include "AFQMC/HamiltonianOperations/ModelComponents/Discrete_GeneralUJ.hpp"

namespace sfqmc
{
namespace afqmc
{
namespace dummy
{
/*
 * Empty class to avoid need for default constructed ModelComponent.
 * Throws is any visitor is called.
 */
class dummy_ModCom
{
public:
  dummy_ModCom(){};

  template<class... Args>
  void addOneBodyPropagatorMatrix([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_ModCom object");
  }

  template<class... Args>
  void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_ModCom object");
  }

  template<class... Args>
  void vHS([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_ModCom object");
  }

  template<class... Args>
  void vbias([[maybe_unused]]Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_ModCom object");
  }

  int number_of_ke_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_ModCom object");
    return 0;
  }

  int local_number_of_cholesky_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_ModCom object");
    return 0;
  }

  template<class TVec>
  void getFieldTypes([[maybe_unused]] TVec&& v) {
    throw std::runtime_error("calling visitor on dummy_ModCom object");
  }

  template<class... Args>
  void update([[maybe_unused]] Args&&... args) {
    throw std::runtime_error("calling visitor on dummy_ModCom object");
  }

};

} // namespace dummy


template<bool MP, bool REAL>
class ModelComponent : public boost::variant<dummy::dummy_ModCom,
                                             Continuous_GeneralUJ<MP>, 
                                             Discrete_GeneralUJ<MP,REAL> 
                                            >
{

using Base = boost::variant<dummy::dummy_ModCom,
                            Continuous_GeneralUJ<MP>,
                            Discrete_GeneralUJ<MP,REAL>
                           >;
public:
  ModelComponent() = default; 

  explicit ModelComponent(Continuous_GeneralUJ<MP> const& other) = delete; 
  explicit ModelComponent(Continuous_GeneralUJ<MP>&& other) : Base::variant(std::move(other)) {}

  explicit ModelComponent(Discrete_GeneralUJ<MP,REAL> const& other) = delete; 
  explicit ModelComponent(Discrete_GeneralUJ<MP,REAL>&& other) : Base::variant(std::move(other)) {}

  ModelComponent(ModelComponent const& other) = delete;
  ModelComponent(ModelComponent&& other)      = default;

  ModelComponent& operator=(ModelComponent const& other) = delete;
  ModelComponent& operator=(ModelComponent&& other) = default;

  template<class... Args>
  void addOneBodyPropagatorMatrix(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.addOneBodyPropagatorMatrix(std::forward<Args>(args)...); },
                                *this);
  }

  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.generalizedFockMatrix(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void vHS(Args&&... args)
  {
    boost::apply_visitor([&](auto&& s) { s.vHS(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void vbias(Args&&... args)
  {
    boost::apply_visitor([&](auto&& s) { s.vbias(std::forward<Args>(args)...); }, *this);
  }

  int local_number_of_cholesky_vectors() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.local_number_of_cholesky_vectors(); }, *this);
  }

  int number_of_ke_vectors() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.number_of_ke_vectors(); }, *this);
  }

  template<class... Args>
  void getFieldTypes(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.getFieldTypes(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void update(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.update(std::forward<Args>(args)...); }, *this);
  }

};

} // namespace afqmc

} // namespace sfqmc

#endif
