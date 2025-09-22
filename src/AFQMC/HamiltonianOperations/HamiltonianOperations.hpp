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

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_HPP


#include "AFQMC/config.h"
#include <boost/variant.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector/vector30.hpp>

#if defined(BOOST_VARIANT_NO_TYPE_SEQUENCE_SUPPORT)
#warning "Error: BOOST_VARIANT_NO_TYPE_SEQUENCE_SUPPORT defined."
#warning "       Compiler does not support make_variant_over."
#warning "       Contact developers. Provide compiler information." 
#error
#endif


#if !defined(ENABLE_DEVICE)
#include "AFQMC/HamiltonianOperations/KP3IndexFactorization.hpp"
#include "AFQMC/HamiltonianOperations/Real3IndexFactorization.hpp"
#endif
#include "AFQMC/HamiltonianOperations/SparseTensor.hpp"
#include "AFQMC/HamiltonianOperations/THCOps.hpp"
#include "AFQMC/HamiltonianOperations/KP3IndexFactorization_batched.hpp"
#include "AFQMC/HamiltonianOperations/Real3IndexFactorization_batched_v2.hpp"
#include "AFQMC/HamiltonianOperations/ModelHamOps.hpp"


namespace sfqmc
{
namespace afqmc
{

namespace dummy
{
/*
 * Empty class to avoid need for default constructed HamiltonianOperations.
 * Throws is any visitor is called.
 */
class dummy_HOps
{
public:
  dummy_HOps(){};

  template<class... Args>
  boost::multi::array<ComplexType, 2> getOneBodyPropagatorMatrix([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return boost::multi::array<ComplexType, 2>{};
  }

  template<class... Args>
  void energy([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
  }

  template<class... Args>
  void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
  }

  template<class... Args>
  void ph_reference_energy([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
  }

  template<class... Args>
  void ph_excited_energy([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
  }

  template<class... Args>
  void vHS([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
  }

  template<class... Args>
  void vbias([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
  }

  template<class... Args>
  void write2hdf([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
  }

  int number_of_ke_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return 0;
  }

  int local_number_of_cholesky_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return 0;
  }

  int global_origin_cholesky_vector() const
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return 0;
  }

  int global_number_of_cholesky_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return 0;
  }

  bool transposed_G_for_vbias() const
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return false;
  }

  bool transposed_G_for_E() const
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return false;
  }

  bool transposed_vHS() const
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return false;
  }

  bool distribution_over_cholesky_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return false;
  }

  boost::multi::array<ComplexType, 2> getHSPotentials()
  {
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return boost::multi::array<ComplexType, 2>{};
  }

  HamiltonianTypes getHamType() const { return UNKNOWN; }

  template<class TVec>
  void getFieldTypes([[maybe_unused]] TVec&& v) {
    throw std::runtime_error("calling visitor on dummy_HOps object");
  }

  bool spin_dependent_vHS() const { 
    throw std::runtime_error("calling visitor on dummy_HOps object");
    return false;
  }

};
} // namespace dummy

namespace detail 
{
  // MAM: If too many template arguments, use vectorn and add dummy_HOps to fill up to n.
  template<bool MP>
  using HOps_types = boost::mpl::vector< dummy::dummy_HOps
#if !defined(ENABLE_DEVICE) 
		,KP3IndexFactorization<MP>
		,Real3IndexFactorization<MP,true>
    ,Real3IndexFactorization<MP,false>
		,ModelHamOps<MP,true,Matrix_<shared_allocator<typename to_working_precision<MP, ComplexType>::type>>>
		,ModelHamOps<MP,false,Matrix_<shared_allocator<typename to_working_precision<MP, ComplexType>::type>>>
#endif
		,SparseTensor<MP, ComplexType, ComplexType, ComplexType>
		,SparseTensor<MP, RealType, RealType, RealType>
		,SparseTensor<MP, RealType, ComplexType, ComplexType>
		,KP3IndexFactorization_batched<MP,Matrix_<device_allocator<typename to_working_precision<MP, ComplexType>::type>>>
		,KP3IndexFactorization_batched<MP,Matrix_<shared_allocator<typename to_working_precision<MP, ComplexType>::type>>>
		,THCOps<MP,true>
		,THCOps<MP,false>
		,ModelHamOps<MP,true,Matrix_<device_allocator<typename to_working_precision<MP, ComplexType>::type>>>
		,ModelHamOps<MP,false,Matrix_<device_allocator<typename to_working_precision<MP, ComplexType>::type>>>
		,Real3IndexFactorization_batched_v2<MP,true>
    ,Real3IndexFactorization_batched_v2<MP,false>
					>;

  template<bool MP>
  using HOps_variant = typename boost::make_variant_over< HOps_types<MP> >::type;

} // namespace detail

template<bool MP>
class HamiltonianOperations : public detail::HOps_variant<MP>
{

  using Base = detail::HOps_variant<MP>;
  using SPComplexType = typename to_working_precision<MP, ComplexType>::type;

public:

  HamiltonianOperations() : Base::variant() 
  {
    print_stacktrace;
    APP_ABORT(" Error: Calling default constructor of HamiltonianOperations. ");
  } 

  // host only !
#if !defined(ENABLE_DEVICE) 
  explicit HamiltonianOperations(ModelHamOps<MP,true,Matrix_<shared_allocator<SPComplexType>>>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(ModelHamOps<MP,false,Matrix_<shared_allocator<SPComplexType>>>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(Real3IndexFactorization<MP,true>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(Real3IndexFactorization<MP,false>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(KP3IndexFactorization<MP>&& other) : Base::variant(std::move(other)) {}
#endif
  // GPU enabled
  explicit HamiltonianOperations(SparseTensor<MP, ComplexType, ComplexType, ComplexType>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(SparseTensor<MP, RealType, RealType, RealType>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(SparseTensor<MP, RealType, ComplexType, ComplexType>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(THCOps<MP,true>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(THCOps<MP,false>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(ModelHamOps<MP,true,Matrix_<device_allocator<SPComplexType>>>&& other) : Base::variant(std::move(other)) {}
 explicit HamiltonianOperations(ModelHamOps<MP,false,Matrix_<device_allocator<SPComplexType>>>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(Real3IndexFactorization_batched_v2<MP,true>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(Real3IndexFactorization_batched_v2<MP,false>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(KP3IndexFactorization_batched<MP,Matrix_<device_allocator<SPComplexType>>>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(KP3IndexFactorization_batched<MP,Matrix_<shared_allocator<SPComplexType>>>&& other) : Base::variant(std::move(other)) {}

  // host only !
#if !defined(ENABLE_DEVICE) 
  explicit HamiltonianOperations(Real3IndexFactorization<MP,true> const& other) = delete;
  explicit HamiltonianOperations(Real3IndexFactorization<MP,false> const& other) = delete;
  explicit HamiltonianOperations(KP3IndexFactorization<MP> const& other)   = delete;
  explicit HamiltonianOperations(ModelHamOps<MP,true,Matrix<shared_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(ModelHamOps<MP,false,Matrix<shared_allocator<SPComplexType>>> const& other) = delete;
#endif
// GPU enabled
  explicit HamiltonianOperations(SparseTensor<MP, ComplexType, ComplexType, ComplexType> const& other)                    = delete;
  explicit HamiltonianOperations(SparseTensor<MP, RealType, RealType, RealType> const& other)                    = delete;
  explicit HamiltonianOperations(SparseTensor<MP, RealType, ComplexType, ComplexType> const& other)                    = delete;
  explicit HamiltonianOperations(THCOps<MP,true> const& other) = delete;
  explicit HamiltonianOperations(THCOps<MP,false> const& other) = delete;
  explicit HamiltonianOperations(ModelHamOps<MP,true,Matrix<device_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(ModelHamOps<MP,false,Matrix<device_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(Real3IndexFactorization_batched_v2<MP,true> const& other) = delete;
  explicit HamiltonianOperations(Real3IndexFactorization_batched_v2<MP,false> const& other) = delete;
  explicit HamiltonianOperations(KP3IndexFactorization_batched<MP,Matrix_<device_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(KP3IndexFactorization_batched<MP,Matrix_<shared_allocator<SPComplexType>>> const& other) = delete;

  HamiltonianOperations(HamiltonianOperations const& other) = delete;
  HamiltonianOperations(HamiltonianOperations&& other)      = default;

  HamiltonianOperations& operator=(HamiltonianOperations const& other) = delete;
  HamiltonianOperations& operator=(HamiltonianOperations&& other) = default;

  template<class... Args>
  boost::multi::array<ComplexType, 2> getOneBodyPropagatorMatrix(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.getOneBodyPropagatorMatrix(std::forward<Args>(args)...); },
                                *this);
  }

  template<class... Args>
  void write2hdf(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.write2hdf(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void energy(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.energy(std::forward<Args>(args)...); }, *this);
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
  std::tuple<dev_csr_Matrix<ComplexType> const*, dev_csr_Matrix<ComplexType> const*> vHS_sparse(Args&&... args)
  {
    // ugly, but this is trully limited to ModelHamOps types only. 
    // ModelHamOps<MP,true,Matrix_<device_allocator<SPComplexType>>>
    using ModHOps1 = ModelHamOps<MP,true,Matrix_<device_allocator<SPComplexType>>>;
    using ModHOps2 = ModelHamOps<MP,false,Matrix_<device_allocator<SPComplexType>>>;
#if !defined(ENABLE_DEVICE)
    using ModHOps3 = ModelHamOps<MP,true,Matrix_<shared_allocator<SPComplexType>>>;
    using ModHOps4 = ModelHamOps<MP,false,Matrix_<shared_allocator<SPComplexType>>>;
    if( ModHOps3* ptr3 = boost::get<ModHOps3>(this) ) {
      return ptr3->vHS_sparse(std::forward<Args>(args)...);
    } else if( ModHOps4* ptr4 = boost::get<ModHOps4>(this) ) {
      return ptr4->vHS_sparse(std::forward<Args>(args)...);
    } else 
#endif
    if( ModHOps1* ptr1 = boost::get<ModHOps1>(this) ) {
      return ptr1->vHS_sparse(std::forward<Args>(args)...);
    } else if( ModHOps2* ptr2 = boost::get<ModHOps2>(this) ) {
      return ptr2->vHS_sparse(std::forward<Args>(args)...);
    } else {
      throw std::runtime_error("calling vHS_sparse with non-ModelHamOps variant. ");
      dev_csr_Matrix<ComplexType> const* t(nullptr);
      return std::make_tuple(t,t); 
    }
  }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    // ugly, but this is trully limited to ModelHamOps types only. 
    // ModelHamOps<MP,true,Matrix_<device_allocator<SPComplexType>>>
    using ModHOps1 = ModelHamOps<MP,true,Matrix_<device_allocator<SPComplexType>>>;
    using ModHOps2 = ModelHamOps<MP,false,Matrix_<device_allocator<SPComplexType>>>;
#if !defined(ENABLE_DEVICE)
    using ModHOps3 = ModelHamOps<MP,true,Matrix_<shared_allocator<SPComplexType>>>;
    using ModHOps4 = ModelHamOps<MP,false,Matrix_<shared_allocator<SPComplexType>>>;
    if( ModHOps3* ptr3 = boost::get<ModHOps3>(this) ) {
      ptr3->update_potentials(std::forward<Args>(args)...);
    } else if( ModHOps4* ptr4 = boost::get<ModHOps4>(this) ) {
      ptr4->update_potentials(std::forward<Args>(args)...);
    } else
#endif
    if( ModHOps1* ptr1 = boost::get<ModHOps1>(this) ) {
      ptr1->update_potentials(std::forward<Args>(args)...);
    } else if( ModHOps2* ptr2 = boost::get<ModHOps2>(this) ) {
      ptr2->update_potentials(std::forward<Args>(args)...);
    } else {
      throw std::runtime_error("calling update_potentials with non-ModelHamOps variant. ");
    }
  }

  template<class... Args>
  void ph_reference_energy(Args&&... args)
  { 
    boost::apply_visitor([&](auto&& s) { s.ph_reference_energy(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void ph_excited_energy(Args&&... args)
  { 
    boost::apply_visitor([&](auto&& s) { s.ph_excited_energy(std::forward<Args>(args)...); }, *this);
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

  int global_origin_cholesky_vector() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.global_origin_cholesky_vector(); }, *this);
  }

  int number_of_ke_vectors() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.number_of_ke_vectors(); }, *this);
  }

  int global_number_of_cholesky_vectors() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.global_number_of_cholesky_vectors(); }, *this);
  }

  bool distribution_over_cholesky_vectors() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.distribution_over_cholesky_vectors(); }, *this);
  }


  bool transposed_G_for_vbias() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.transposed_G_for_vbias(); }, *this);
  }

  bool transposed_G_for_E() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.transposed_G_for_E(); }, *this);
  }

  bool transposed_vHS() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.transposed_vHS(); }, *this);
  }

  HamiltonianTypes getHamType() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.getHamType(); }, *this);
  }

  template<class... Args>
  boost::multi::array<ComplexType, 2> getHSPotentials()
  {
    return boost::apply_visitor([&](auto&& a) { return a.getHSPotentials(); }, *this);
  }

  template<class... Args>
  void getFieldTypes(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.getFieldTypes(std::forward<Args>(args)...); }, *this);
  }

  bool spin_dependent_vHS() const { 
    return boost::apply_visitor([&](auto&& a) { return a.spin_dependent_vHS(); }, *this);
  }
};

} // namespace afqmc

} // namespace sfqmc

#endif
