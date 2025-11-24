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

#include <variant>

#include "AFQMC/config.h"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"
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

  template<MEMORY_SPACE M>
  HamiltonianOperations<M>::HamiltonianOperations()  
  {
    APP_ABORT(" Error: Calling default constructor of HamiltonianOperations. ");
  } 
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations();
#if defined(ENABLE_DEVICE)
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations();
#endif

  template<MEMORY_SPACE M>
  HamiltonianOperations<M>::HamiltonianOperations(THCOps<M,true>&& other) : var(std::move(other)) {} 
  template<MEMORY_SPACE M>
  HamiltonianOperations<M>::HamiltonianOperations(THCOps<M,false>&& other) : var(std::move(other)) {} 
  template<MEMORY_SPACE M>
  HamiltonianOperations<M>::HamiltonianOperations(KPTHCOps<M>&& other) : var(std::move(other)) {} 

  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(THCOps<HOST_MEMORY,true>&&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(THCOps<HOST_MEMORY,false>&&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(KPTHCOps<HOST_MEMORY>&&);

#if defined(ENABLE_DEVICE)
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(THCOps<DEVICE_MEMORY,true>&&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(THCOps<DEVICE_MEMORY,false>&&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(KPTHCOps<DEVICE_MEMORY>&&);
#endif

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

/*
  explicit HamiltonianOperations(ModelHamOps<MP,true,Matrix<device_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(ModelHamOps<MP,false,Matrix<device_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(Real3IndexFactorization_batched_v2<MP,true> const& other) = delete;
  explicit HamiltonianOperations(Real3IndexFactorization_batched_v2<MP,false> const& other) = delete;
  explicit HamiltonianOperations(KP3IndexFactorization_batched<MP,Matrix_<device_allocator<SPComplexType>>> const& other) = delete;
  explicit HamiltonianOperations(KP3IndexFactorization_batched<MP,Matrix_<shared_allocator<SPComplexType>>> const& other) = delete;
*/
/*
  HamiltonianOperations(HamiltonianOperations const& other) = default;
  HamiltonianOperations(HamiltonianOperations&& other)      = default;

  HamiltonianOperations& operator=(HamiltonianOperations const& other) = default;
  HamiltonianOperations& operator=(HamiltonianOperations&& other) = default;
*/
  template<MEMORY_SPACE M>
  nda::array<ComplexType,3> HamiltonianOperations<M>::getOneBodyPropagatorMatrix_impl(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    return std::visit([&](auto&& a) { return a.getOneBodyPropagatorMatrix(dt,vMF); },
                                var);
  }

#define __getOneBodyPropagatorMatrix__(M)    \
  template nda::array<ComplexType,3> HamiltonianOperations<M>::getOneBodyPropagatorMatrix_impl(double, memory::array_view<M,const ComplexType,1> const&);   \
  template nda::array<ComplexType,3> HamiltonianOperations<M>::getOneBodyPropagatorMatrix_impl(double, memory::array_view<M,const ComplexType,1,nda::C_layout> const&);   \
  template nda::array<ComplexType,3> HamiltonianOperations<M>::getOneBodyPropagatorMatrix_impl(double, memory::array_view<M,const ComplexType,1,nda::basic_layout<0,nda::C_stride_order<1>,nda::layout_prop_e::strided_1d>> const&);
__getOneBodyPropagatorMatrix__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__getOneBodyPropagatorMatrix__(DEVICE_MEMORY)
#endif

/*
  template<class... Args>
  void write2hdf(Args&&... args)
  {
    std::visit([&](auto&& a) { a.write2hdf(std::forward<Args>(args)...); }, var);
  }
*/

  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::energy_impl(nda::MemoryArrayOfRank<2> auto& E, 
                                        nda::MemoryArrayOfRank<2> auto const& G,
              int idet, bool addH1, bool addEJ, bool addEXX)
  {
    std::visit([&](auto&& a) { a.energy(E,G,idet,addH1,addEJ,addEXX); }, var);
  }

#define __energy__(M) \
  template void HamiltonianOperations<M>::energy_impl(memory::array_view<M,ComplexType,2>&, memory::array_view<M,const ComplexType,2>const&,int,bool,bool,bool); \
  template void HamiltonianOperations<M>::energy_impl(memory::array_view<M,ComplexType,2>&, memory::array_view<M,const ComplexType,2,nda::C_layout>const&,int,bool,bool,bool); \
  template void HamiltonianOperations<M>::energy_impl(memory::array_view<M,ComplexType,2,nda::C_layout>&, memory::array_view<M,const ComplexType,2>const&,int,bool,bool,bool); \
  template void HamiltonianOperations<M>::energy_impl(memory::array_view<M,ComplexType,2,nda::C_layout>&, memory::array_view<M,const ComplexType,2,nda::C_layout>const&,int,bool,bool,bool);  
__energy__(HOST_MEMORY) 
#if defined(ENABLE_DEVICE)
__energy__(DEVICE_MEMORY) 
#endif

/*
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.generalizedFockMatrix(std::forward<Args>(args)...); }, var);
  }
*/

  template<MEMORY_SPACE M>
  memory::buffered_array<M,ComplexType,4> HamiltonianOperations<M>::vHS_impl(nda::MemoryArrayOfRank<2> auto const& X, double dt)
  {
    return std::visit([&](auto&& s) { return s.vHS(X,dt); }, var);
  }
#define __vHS__(M) \
  template memory::buffered_array<M,ComplexType,4> HamiltonianOperations<M>::vHS_impl(memory::array_view<M,const ComplexType,2>const&,double); \
  template memory::buffered_array<M,ComplexType,4> HamiltonianOperations<M>::vHS_impl(memory::array_view<M,const ComplexType,2,nda::C_layout>const&,double); 
__vHS__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__vHS__(DEVICE_MEMORY)
#endif

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

  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::vbias_impl(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    std::visit([&](auto&& s) { s.vbias(G,v,dt); }, var);
  }
#define __vbias__(M) \
  template void HamiltonianOperations<M>::vbias_impl(memory::array_view<M,ComplexType const,2>const&,memory::array_view<M,ComplexType,2>&,double); \
  template void HamiltonianOperations<M>::vbias_impl(memory::array_view<M,ComplexType const,2,nda::C_layout>const&,memory::array_view<M,ComplexType,2>&,double); \
  template void HamiltonianOperations<M>::vbias_impl(memory::array_view<M,ComplexType const,2>const&,memory::array_view<M,ComplexType,2,nda::C_layout>&,double); \
  template void HamiltonianOperations<M>::vbias_impl(memory::array_view<M,ComplexType const,2,nda::C_layout>const&,memory::array_view<M,ComplexType,2,nda::C_layout>&,double); 
__vbias__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__vbias__(DEVICE_MEMORY)
#endif

  template<MEMORY_SPACE M>
  int HamiltonianOperations<M>::number_of_cholesky_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_cholesky_vectors(); }, var);
  }

  template<MEMORY_SPACE M>
  int HamiltonianOperations<M>::number_of_ke_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_ke_vectors(); }, var);
  }

  template<MEMORY_SPACE M>
  std::array<int,2> HamiltonianOperations<M>::vHS_dims() const
  {
    return std::visit([&](auto&& a) { return a.vHS_dims(); }, var);
  }

  template<MEMORY_SPACE M>
  HamiltonianTypes HamiltonianOperations<M>::getHamType() const
  {
    return std::visit([&](auto&& a) { return a.getHamType(); }, var);
  }

  template<MEMORY_SPACE M>
  nda::array<int,1> HamiltonianOperations<M>::getFieldTypes() const
  {
    return std::visit([&](auto&& a) { return a.getFieldTypes(); }, var);
  }
  
  template int HamiltonianOperations<HOST_MEMORY>::number_of_cholesky_vectors() const;
  template int HamiltonianOperations<HOST_MEMORY>::number_of_ke_vectors() const;
  template std::array<int,2> HamiltonianOperations<HOST_MEMORY>::vHS_dims() const;
  template HamiltonianTypes HamiltonianOperations<HOST_MEMORY>::getHamType() const;
  template nda::array<int,1> HamiltonianOperations<HOST_MEMORY>::getFieldTypes() const;

#if defined(ENABLE_DEVICE)
  template int HamiltonianOperations<DEVICE_MEMORY>::number_of_cholesky_vectors() const;
  template int HamiltonianOperations<DEVICE_MEMORY>::number_of_ke_vectors() const;
  template std::array<int,2> HamiltonianOperations<DEVICE_MEMORY>::vHS_dims() const;
  template HamiltonianTypes HamiltonianOperations<DEVICE_MEMORY>::getHamType() const;
  template nda::array<int,1> HamiltonianOperations<DEVICE_MEMORY>::getFieldTypes() const;
#endif

} // namespace afqmc

} // namespace sfqmc

