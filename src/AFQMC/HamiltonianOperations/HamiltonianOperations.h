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

#include "AFQMC/config.h"
//#include "AFQMC/Utilities/type_conversion.hpp"

//#if !defined(ENABLE_DEVICE)
//#include "AFQMC/HamiltonianOperations/KP3IndexFactorization.hpp"
//#include "AFQMC/HamiltonianOperations/Real3IndexFactorization.hpp"
//#endif
#include "AFQMC/HamiltonianOperations/THCOps.hpp"
#include "AFQMC/HamiltonianOperations/KPTHCOps.hpp"
//#include "AFQMC/HamiltonianOperations/KP3IndexFactorization_batched.hpp"
//#include "AFQMC/HamiltonianOperations/Real3IndexFactorization_batched_v2.hpp"
//#include "AFQMC/HamiltonianOperations/ModelHamOps.hpp"


namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class HamiltonianOperations 
{

public:

  HamiltonianOperations(); 

  HamiltonianOperations(THCOps<MEM,true>&& other);
  HamiltonianOperations(THCOps<MEM,false>&& other);

  HamiltonianOperations(KPTHCOps<MEM>&& other);

/*
  // host only !
#if !defined(ENABLE_DEVICE) 
  explicit HamiltonianOperations(ModelHamOps<MP,true,Matrix_<shared_allocator<SPComplexType>>>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(ModelHamOps<MP,false,Matrix_<shared_allocator<SPComplexType>>>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(Real3IndexFactorization<MP,true>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(Real3IndexFactorization<MP,false>&& other) : Base::variant(std::move(other)) {}
  explicit HamiltonianOperations(KP3IndexFactorization<MP>&& other) : Base::variant(std::move(other)) {}
#endif
  // GPU enabled
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
*/
  explicit HamiltonianOperations(THCOps<MEM,true> const& other) = delete;
  explicit HamiltonianOperations(THCOps<MEM,false> const& other) = delete;
  explicit HamiltonianOperations(KPTHCOps<MEM> const& other) = delete;
/*
  explicit HamiltonianOperations(ModelHamOps<MP,true,Matrix<device_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(ModelHamOps<MP,false,Matrix<device_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(Real3IndexFactorization_batched_v2<MP,true> const& other) = delete;
  explicit HamiltonianOperations(Real3IndexFactorization_batched_v2<MP,false> const& other) = delete;
  explicit HamiltonianOperations(KP3IndexFactorization_batched<MP,Matrix_<device_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(KP3IndexFactorization_batched<MP,Matrix_<shared_allocator<SPComplexType>>> const& other) = delete;
*/

  HamiltonianOperations(HamiltonianOperations const& other) = default;
  HamiltonianOperations(HamiltonianOperations&& other)      = default;

  HamiltonianOperations& operator=(HamiltonianOperations const& other) = default;
  HamiltonianOperations& operator=(HamiltonianOperations&& other) = default;

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF) 
  {
    return getOneBodyPropagatorMatrix_impl(dt,vMF());
  }

/*
  template<class... Args>
  void write2hdf(Args&&... args)
  {
    std::visit([&](auto&& a) { a.write2hdf(std::forward<Args>(args)...); }, var);
  }
*/
  void energy(nda::MemoryArrayOfRank<2> auto && E, nda::MemoryArrayOfRank<2> auto const& G,
              int idet, bool addH1  = true, bool addEJ  = true,bool addEXX = true)
  {
    auto E_=E();
    energy_impl(E_,G(),idet,addH1,addEJ,addEXX);
  }

/*
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.generalizedFockMatrix(std::forward<Args>(args)...); }, var);
  }
*/

  memory::buffered_array<MEM,ComplexType,4> vHS(nda::MemoryArrayOfRank<2> auto const& X, double dt)
  {
    return vHS_impl(X(),dt);
  }

/*
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
    std::visit([&](auto&& s) { s.ph_reference_energy(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void ph_excited_energy(Args&&... args)
  { 
    std::visit([&](auto&& s) { s.ph_excited_energy(std::forward<Args>(args)...); }, var);
  }
*/

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto&& v, double dt)
  {  
    auto v_ = v();
    vbias_impl(G(),v_,dt);
  }

  int number_of_cholesky_vectors() const;

  int number_of_ke_vectors() const;

  std::array<int,2> vHS_dims() const;

  HamiltonianTypes getHamType() const;

  nda::array<int,1> getFieldTypes() const;

  private:

  std::variant<THCOps<MEM,true>, THCOps<MEM,false>, KPTHCOps<MEM>> var;

  // makes instantiations easier
  nda::array<ComplexType,3> getOneBodyPropagatorMatrix_impl(double dt,
                                                       nda::MemoryVector auto const& vMF); 

  void energy_impl(nda::MemoryArrayOfRank<2> auto& E, nda::MemoryArrayOfRank<2> auto const& G,
              int idet, bool addH1, bool addEJ,bool addEXX);

  void vbias_impl(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt);

  memory::buffered_array<MEM,ComplexType,4> vHS_impl(nda::MemoryArrayOfRank<2> auto const& X, double dt);
};

} // namespace afqmc

} // namespace sfqmc

