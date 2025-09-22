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

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_REAL3INDEXFACTORIZATION_BATCHED_V2_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_REAL3INDEXFACTORIZATION_BATCHED_V2_HPP

#include <vector>
#include <type_traits>

#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "Numerics/ma_operations.hpp"

#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Utilities/Utils.hpp"
#include "Numerics/batched_operations.hpp"
#include "Numerics/tensor_operations.hpp"

#include "AFQMC/Wavefunctions/detail/phmsd_impl.hpp"

namespace sfqmc
{
namespace afqmc
{
// Custom implementation for real build
template<bool MP, bool REAL_ONEBODY>
class Real3IndexFactorization_batched_v2
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;

  using ValueType = typename std::conditional<REAL_ONEBODY, RealType, ComplexType>::type;

  // allocators
  using Allocator           = device_allocator<ComplexType>;
  using SpAllocator         = device_allocator<SPComplexType>;
  using SpRAllocator        = device_allocator<SPRealType>;
  using Allocator_shared    = node_allocator<ComplexType>;
  using SpAllocator_shared  = node_allocator<SPComplexType>;
  using SpRAllocator_shared = node_allocator<SPRealType>;

  // type defs
  using pointer            = typename std::allocator_traits<Allocator>::pointer;
  using sp_pointer         = typename std::allocator_traits<SpAllocator>::pointer;
  using const_sp_pointer   = typename std::allocator_traits<SpAllocator>::const_pointer;
  using sp_rpointer        = typename std::allocator_traits<SpRAllocator>::pointer;
  using pointer_shared     = typename std::allocator_traits<Allocator_shared>::pointer;
  using sp_pointer_shared  = typename std::allocator_traits<SpAllocator_shared>::pointer;
  using sp_rpointer_shared = typename std::allocator_traits<SpRAllocator_shared>::pointer;

  using device_alloc_type  = DeviceBufferManager::template allocator_t<SPComplexType>;
  using device_alloc_Rtype = DeviceBufferManager::template allocator_t<SPRealType>;

  using CVector        = Vector_<Allocator>;
  using SpVector       = Vector_<SpAllocator>;
  using SpCMatrix      = Matrix_<SpAllocator>;
  using CMatrix_ref    = Matrix_ref_<pointer>;
  using SpCMatrix_ref  = Matrix_ref_<sp_pointer>;
  using SpRVector_ref  = Vector_ref_<sp_rpointer>; 
  using SpRMatrix_ref  = Matrix_ref_<sp_rpointer>;
  using SpCTensor_ref  = Array_ref_<3, sp_pointer>;
  using SpC4Tensor_ref = Array_ref_<4, sp_pointer>;
  using C4Tensor_ref   = Array_ref_<4, pointer>;

  using StaticSpVector   = StaticVector_<device_alloc_type>;
  using StaticSpMatrix   = StaticMatrix_<device_alloc_type>;
  using Static3Tensor  = StaticArray_<3, device_alloc_type>;
  using Static4Tensor  = StaticArray_<4, device_alloc_type>;
  using StaticRVector  = StaticVector_<device_alloc_Rtype>;
  using StaticRMatrix  = StaticMatrix_<device_alloc_Rtype>;
  using Static3RTensor = StaticArray_<3, device_alloc_Rtype>;
  using Static4RTensor = StaticArray_<4, device_alloc_Rtype>;

  using shmCVector    = Vector_<Allocator_shared>;
  using shmCMatrix    = Matrix_<Allocator_shared>;
  using shmSpC3Tensor = Array_<3, SpAllocator_shared>;
  using shmSpCMatrix  = Matrix_<SpAllocator_shared>;
  using shmSpRMatrix  = Matrix_<SpRAllocator_shared>;

  using mpi3RMatrix = boost::multi::array<RealType, 2, shared_allocator<RealType>>;
  using mpi3CMatrix = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;

  using mpi3VMatrix = boost::multi::array<ValueType, 2, shared_allocator<ValueType>>;

public:
  static const HamiltonianTypes HamOpType = RealDenseFactorized;
  HamiltonianTypes getHamType() const { return HamOpType; }

  template<class shmCMatrix_, class shmSpRMatrix_, class shmSpC3Tensor_>
  Real3IndexFactorization_batched_v2(WALKER_TYPES type,
                                     TaskGroup_& TG,
                                     mpi3VMatrix&& hij_,
                                     shmCMatrix_&& haj_,
                                     shmSpRMatrix_&& vik,
                                     std::vector<shmSpC3Tensor_>&& vnak,
                                     mpi3CMatrix&& vn0_,
                                     ComplexType e0_,
                                     Allocator const& alloc_,
                                     int cv0,
                                     int gncv,
                                     long maxMem = 2000)
      : allocator_(alloc_),
        sp_allocator_(alloc_),
        buffer_manager(),
        walker_type(type),
        max_memory_MB(maxMem),
        global_origin(cv0),
        global_nCV(gncv),
        local_nCV(0),
        E0(e0_),
        hij(std::move(hij_)),
        hij_dev(hij.extensions(), make_node_allocator<ComplexType>(TG)),
        haj(std::move(haj_)),
        Likn(std::move(vik)),
        Lnak(std::move(move_vector<shmSpC3Tensor>(std::move(vnak)))),
        vn0(std::move(vn0_)),
        Twina_ph(iextensions<1u>{0},sp_allocator_),
        Swia_ph(iextensions<1u>{0},allocator_)
  {
    local_nCV = Likn.size(1);
    size_t lnak(0);
    for (auto& v : Lnak)
      lnak += v.num_elements();
    for (int i = 0; i < hij.size(0); i++)
    {
      for (int j = 0; j < hij.size(1); j++)
      {
        hij_dev[i][j] = ComplexType(hij[i][j]);
      }
    }
    app_log(1,"****************************************************************** ");
    app_log(1,"  Static memory usage by Real3IndexFactorization_batched_v2 (node 0 in MB) ");
    app_log(1,"  Likn: {}", double(Likn.num_elements() * sizeof(SPRealType)) / 1024.0 / 1024.0);
    app_log(1,"  Lnak: {}", double(lnak * sizeof(SPComplexType)) / 1024.0 / 1024.0);
    app_log(1,"  Buffer memory limited to (not yet allocated) : {} MB", max_memory_MB);
    memory_report();
  }

  ~Real3IndexFactorization_batched_v2() {}

  Real3IndexFactorization_batched_v2(const Real3IndexFactorization_batched_v2& other) = delete;
  Real3IndexFactorization_batched_v2& operator=(const Real3IndexFactorization_batched_v2& other) = delete;
  Real3IndexFactorization_batched_v2(Real3IndexFactorization_batched_v2&& other)                 = default;
  Real3IndexFactorization_batched_v2& operator=(Real3IndexFactorization_batched_v2&& other) = delete;

  boost::multi::array<ComplexType, 2> getOneBodyPropagatorMatrix(TaskGroup_& TG, double dt,
                                                                 boost::multi::array<ComplexType, 1> const& vMF)
  {
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO = hij.size(1) / npol;

    RUNTIME_CHECK((hij.size(0) == NMO) || (hij.size(0) == 2*NMO), "");
    if( hij.size(0) == 2*NMO )
      RUNTIME_CHECK((walker_type == COLLINEAR) || (walker_type == NONCOLLINEAR), "");
    int I0 = (hij.size(0) == 2*NMO) ? NMO : 0;
    if( hij.size(1) == 2*NMO )
      RUNTIME_CHECK(walker_type == NONCOLLINEAR, "");
    int J0 = (hij.size(1) == 2*NMO) ? NMO : 0;

    CVector vMF_(vMF);
    CVector P1D(iextensions<1u>{NMO * NMO});
    fill_n(P1D.origin(), P1D.num_elements(), ComplexType(0));
    vHS(vMF_, P1D, dt);
    if (TG.TG().size() > 1)
      TG.TG().all_reduce_in_place_n(raw_pointer_cast(P1D.origin()), P1D.num_elements(), std::plus<>());

    boost::multi::array<ComplexType, 2> P1({nspin * npol * NMO, npol *  NMO});
    std::fill_n(P1.origin(), P1.num_elements(), ComplexType(0));

    if (walker_type != NONCOLLINEAR)
      copy_n(P1D.origin(), NMO * NMO, P1.origin());

    if(walker_type == COLLINEAR)
      copy_n(P1D.origin(), NMO * NMO, P1[NMO].origin());
    else if (walker_type == NONCOLLINEAR)
    {
      boost::multi::array<ComplexType, 2> P0({NMO, NMO});
      copy_n(P1D.origin(), NMO * NMO, P0.origin());
      ma::add(ComplexType(1.0), P0, ComplexType(0.0), 
        P1({0,NMO},{0,NMO}), 
        P1({0,NMO},{0,NMO}));
      ma::add(ComplexType(1.0), P0, ComplexType(0.0), 
        P1({NMO,npol*NMO},{NMO,npol*NMO}), 
        P1({NMO,npol*NMO},{NMO,npol*NMO}));
    }

    using ma::conj;

    for (int i = 0; i < NMO; i++)
    {
      P1[i][i] += dt * (hij[i][i] + vn0[i][i]);
      if(walker_type == COLLINEAR)
        P1[NMO+i][i] += dt * (hij[I0+i][i] + vn0[i][i]);   
      else if(walker_type == NONCOLLINEAR)
        P1[NMO+i][NMO+i] += dt * (hij[I0+i][J0+i] + vn0[i][i]);
      for (int j = i + 1; j < NMO; j++)
      {
        P1[i][j] += dt * (hij[i][j] + vn0[i][j]);
        P1[j][i] += dt * (hij[j][i] + vn0[j][i]);
        // This is really cutoff dependent!!!
        if (std::abs(P1[i][j] - ma::conj(P1[j][i])) > 1e-6)
        {
          app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 is not hermitian. ");
          app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,P1[i][j],P1[j][i]);
          app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[i][j],hij[j][i]);
          app_warning("             vn0[I,J]:{}, vn0[J,I]:{} ",vn0[i][j],vn0[j][i]);
        }
        P1[i][j] = 0.5 * (P1[i][j] + ma::conj(P1[j][i]));
        if(walker_type == COLLINEAR) {
          P1[NMO+i][j] += dt * (hij[I0+i][j] + vn0[i][j]);
          P1[NMO+j][i] += dt * (hij[I0+j][i] + vn0[j][i]);
          // This is really cutoff dependent!!!
          if (std::abs(P1[NMO+i][j] - ma::conj(P1[NMO+j][i])) > 1e-6)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (beta) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,P1[NMO+i][j],P1[NMO+j][i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[I0+i][j],hij[I0+j][i]);
            app_warning("             vn0[I,J]:{}, vn0[J,I]:{} ",vn0[i][j],vn0[j][i]);
          }
          P1[NMO+i][j] = 0.5 * (P1[NMO+i][j] + ma::conj(P1[NMO+j][i]));
          P1[NMO+j][i] = ma::conj(P1[NMO+i][j]);
        } else if (walker_type == NONCOLLINEAR ) {
          // dt * (hij[I0+i][J0+j] + vn0[i][j]); is already included in P1[i][j]
          P1[I0+i][J0+j] = P1[i][j];
          P1[I0+j][J0+i] = P1[j][i];

          // a-b and b-a are hij
          P1[I0+i][j] = dt * hij[I0+i][j];
          P1[I0+j][i] = dt * hij[I0+j][i];

          P1[i][J0+j] = dt * hij[i][J0+j];
          P1[j][J0+i] = dt * hij[j][J0+i];

          // This is really cutoff dependent!!!
          if (std::abs(P1[I0+i][J0+j] - ma::conj(P1[J0+j][I0+i])) > 1e-6) // b-b
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (beta-beta) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,P1[I0+i][J0+j],P1[I0+j][J0+i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[I0+i][J0+j],hij[I0+j][J0+i]);
            app_warning("             vn0[I,J]:{}, vn0[J,I]:{} ",vn0[i][j],vn0[j][i]);
          }

          if (std::abs(P1[i][J0+j] - ma::conj(P1[J0+j][i])) > 1e-6)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (spin-flip) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,P1[i][J0+j],P1[I0+j][i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[i][J0+j],hij[I0+j][i]);
          }

          P1[I0+i][J0+j] = 0.5 * (P1[I0+i][J0+j] + ma::conj(P1[J0+j][I0+i]));
          P1[J0+j][I0+i] = ma::conj(P1[I0+i][J0+j]);

          P1[i][J0+j] = 0.5 * (P1[i][J0+j] + ma::conj(P1[J0+j][i]));
          P1[J0+j][i] = ma::conj(P1[i][J0+j]);

          P1[I0+i][j] = 0.5 * (P1[I0+i][j] + ma::conj(P1[j][I0+i]));
          P1[j][I0+i] = ma::conj(P1[I0+i][j]);

        }
        P1[j][i] = ma::conj(P1[i][j]);
      }
    }

    return P1;
  }

  template<class TVec>
  void getFieldTypes(TVec&& v) {
    int localnvc = local_number_of_cholesky_vectors();
    RUNTIME_CHECK(v.size() == localnvc, "");
    using std::fill_n;
    fill_n( v.origin(), v.size(), ContinuousChargePropagator );
  }

  template<class Mat, class MatB>
  void energy(Mat&& E,
              MatB const& Gc,
              int nd,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  { 
    using std::fill_n;
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nwalk = Gc.size(0);
    int NMO   = hij.size(1) / npol;
    int nel[2];
    nel[0] = Lnak[nspin * nd].size(1);
    nel[1] = ((nspin == 2) ? Lnak[nspin * nd + 1].size(1) : 0);
    
    RUNTIME_CHECK(Gc.num_elements() == nwalk * (nel[0]+nel[1]) * npol * NMO, "");
    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error Real3IndexFactorization_batched_v2::energy: Incorrect matrix dimensions ");
    ma::fill(E({0,nwalk},{0,3}),ComplexType(0.0));
    //if(addH1) 
    //  ma::fill(E({0,nwalk},0),ComplexType(E0));
    if(addH1){
      for(int i=0; i<nwalk; i++)
        E[i][0] += E0;
    }

    if (addEJ)
    {
      StaticSpMatrix Kl({nwalk,local_nCV}, SPComplexType(0.0),
                        buffer_manager.get_generator().template get_allocator<SPComplexType>());

      for(int is=0, is0=0; is<nspin; ++is) {
        energy_impl(is, E, Gc(Gc.extension(0), {is0, is0+nel[is]*NMO*npol}), nd, Kl,
                  addH1, addEJ, addEXX);
        is0 += nel[is]*NMO*npol;
      }

      SPRealType scl = (walker_type == CLOSED ? 2.0 : 1.0);
      ma::dot('N','N',ComplexType(0.5*scl*scl),Kl,Kl,ComplexType(0.0),E.rotated()[2].unrotated());
    } else {
      SpCMatrix Kl({0,0});  // dummy matrix, not used 
      for(int is=0, is0=0; is<nspin; ++is) {
        energy_impl(is, E, Gc(Gc.extension(0), {is0, is0+nel[is]*NMO*npol}), nd, Kl,
                  addH1, addEJ, addEXX);
        is0 += nel[is]*NMO*npol;
      }
    }
  }

  template<class Mat, class MatB, class MatC>
  void energy(SpinTypes spin_component,
              Mat&& E,
              MatB const& Gc,
              int nd,
              MatC&& EJn,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    using std::fill_n;
    int ispin = ( spin_component == Alpha ? 0 : 1);
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int nwalk = Gc.size(0);
    int NMO   = hij.size(1);
    int nel[2];
    nel[0] = Lnak[nspin * nd].size(1);
    nel[1] = ((nspin == 2) ? Lnak[nspin * nd + 1].size(1) : 0);

    RUNTIME_CHECK(Gc.num_elements() == nwalk * nel[ispin] * NMO, "");
    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error Real3IndexFactorization::energy: Incorrect matrix dimensions ");
    ma::fill(E({0,nwalk},{0,3}),ComplexType(0.));
    // by convention, add E0 only to Alpha
    if(addH1 and (spin_component == Alpha)) 
      ma::fill(E({0,nwalk},0),ComplexType(E0));

    if (addEJ)
    {
      RUNTIME_CHECK(EJn.size(0) == nwalk, "");
      RUNTIME_CHECK(EJn.size(1) == local_nCV, "");
      using EJType = typename std::decay_t<MatC>::element_type;
      if constexpr  (not std::is_same_v<EJType,SPComplexType>) {
        StaticSpMatrix Kl({nwalk,local_nCV}, SPComplexType(0.0),
                        buffer_manager.get_generator().template get_allocator<SPComplexType>());
        energy_impl(ispin, E, Gc, nd, Kl, addH1, addEJ, addEXX);
        ma::dot('N','N',ComplexType(0.5),Kl,Kl,ComplexType(0.0),E.rotated()[2].unrotated());
	ma::copy_n_cast(Kl,EJn);
      } else {
        energy_impl(ispin, E, Gc, nd, EJn, addH1, addEJ, addEXX);
        ma::dot('N','N',ComplexType(0.5),EJn,EJn,ComplexType(0.0),E.rotated()[2].unrotated());
      }
    } else {
      SpCMatrix Kl({0,0});  // dummy matrix, not used 
      energy_impl(ispin, E, Gc, nd, Kl, addH1, addEJ, addEXX);
    }
  }

  template<class... Args>
  void fast_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: fast_energy not implemented in Real3IndexFactorization_batched_v2. ");
  }

  template<class Mat, class MatB, class MatC,
           typename = typename std::enable_if_t<(std::decay_t<Mat>::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay_t<MatB>::dimensionality == 2)>
          >
  void ph_reference_energy(SpinTypes spin_component,
              Mat&& E,
              MatB const& Gc,
              MatC&& EJn,
              bool addH1 = true)
  {
    using std::fill_n;
    RUNTIME_CHECK(haj.size() == 1, "");
    if (walker_type == COLLINEAR)
      RUNTIME_CHECK(Lnak.size() == 2, "");
    else
      RUNTIME_CHECK(Lnak.size() == 1, "");

    int ispin = ( spin_component == Alpha ? 0 : 1);
    int nwalk = Gc.size(0);
    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error Real3IndexFactorization::energy: Incorrect matrix dimensions ");
    ma::fill(E({0,nwalk},{0,3}),ComplexType(0.));
    // by convention, add E0 only to Alpha
    if(addH1 and (spin_component == Alpha)) 
      ma::fill(E({0,nwalk},0),ComplexType(E0));
    
    RUNTIME_CHECK(EJn.size(0) == nwalk, "");
    RUNTIME_CHECK(EJn.size(1) == local_nCV, "");
    using EJType = typename std::decay_t<MatC>::element_type;
    if constexpr  (not std::is_same_v<EJType,SPComplexType>) {
      StaticSpMatrix Kl({nwalk,local_nCV}, SPComplexType(0.0),
                      buffer_manager.get_generator().template get_allocator<SPComplexType>());
      ph_ref_energy_impl(ispin, E, Gc, Kl, addH1);
      ma::dot('N','N',ComplexType(0.5),Kl,Kl,ComplexType(0.0),E.rotated()[2].unrotated());
      ma::copy_n_cast(Kl,EJn);
    } else {
      ph_ref_energy_impl(ispin, E, Gc, EJn, addH1);
      ma::dot('N','N',ComplexType(0.5),EJn,EJn,ComplexType(0.0),E.rotated()[2].unrotated());
    }
  }

  template<class Iptr, class Mat, class MatB, class MatC, class MatW,
           typename = typename std::enable_if_t<(std::decay<Mat>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatW>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 4)>
          >
  void ph_excited_energy(SpinTypes spin_component,
              int ndet,
              int nex,
              int nelec,
              int nact,
              Iptr const iexcit,
              Iptr const refc,
              Mat&& E,
              MatW&& wgt,
              MatB const& R,
              MatC&& EJn,
              bool addH1 = true)
  {
    // HamOps and Wfn must have the same mixed_precision setting, otherwise abort
    using RType = typename std::decay_t<typename std::decay_t<MatB>::element_type>;
    using EJType = typename std::decay_t<typename std::decay_t<MatC>::element_type>;
    if constexpr (std::is_same<SPComplexType, RType>::value and
                  std::is_same<SPComplexType, EJType>::value) {
        ph_excited_energy_impl(spin_component, ndet, nex, nelec, nact, iexcit, refc, E, wgt, R, EJn, addH1);
    } else {
        // Reaching this line at runtime is a bug! 
        APP_ABORT("Inconsistent mixed_precision definition. Submit issue to developers. ");
    }
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 1)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 1)>,
           typename = void>
  void vHS(MatA& X, MatB&& v, double dt, double a = 1., double c = 0.)
  {
    using BType = typename std::decay<MatB>::type::element;
    using AType = typename std::decay<MatA>::type::element;
    boost::multi::array_ref<BType, 2, decltype(v.origin())> v_(v.origin(), {v.size(0), 1});
    boost::multi::array_ref<AType, 2, decltype(X.origin())> X_(X.origin(), {X.size(0), 1});
    return vHS(X_, v_, dt, a, c);
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>>
  void vHS(MatA& X, MatB&& v, double dt, double a = 1., double c = 0.)
  {
    using XType = typename std::decay_t<typename MatA::element>;
    using vType = typename std::decay<MatB>::type::element;
    RUNTIME_CHECK(Likn.size(1) == X.size(0), "");
    RUNTIME_CHECK(Likn.size(0) == v.size(0), "");
    RUNTIME_CHECK(X.size(1) == v.size(1), "");
    // scale a by sqrt(dt)
    a *= std::sqrt(dt);
    // setup buffer space if changing precision in X or v
    size_t vmem(0), Xmem(0);
    if (not std::is_same<XType, SPComplexType>::value)
      Xmem = X.num_elements();
    if (not std::is_same<vType, SPComplexType>::value)
      vmem = v.num_elements();
    StaticSpVector SPBuff(iextensions<1u>{Xmem + vmem},
                        buffer_manager.get_generator().template get_allocator<SPComplexType>());
    sp_pointer vptr(nullptr);
    const_sp_pointer Xptr(nullptr);
    // setup origin of Gsp and copy_n_cast if necessary
    using sfqmc::afqmc::reinterpret_pointer_cast;
    if (std::is_same<XType, SPComplexType>::value)
    {
      Xptr = reinterpret_pointer_cast<SPComplexType const>(make_device_ptr(X.origin()));
    }
    else
    {
      copy_n_cast(make_device_ptr(X.origin()), X.num_elements(), make_device_ptr(SPBuff.origin()));
      Xptr = make_device_ptr(SPBuff.origin());
    }
    // setup origin of vsp and copy_n_cast if necessary
    if (std::is_same<vType, SPComplexType>::value)
    {
      vptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(v.origin()));
    }
    else
    {
      vptr = make_device_ptr(SPBuff.origin()) + Xmem;
      if (std::abs(c) > 1e-12)
        copy_n_cast(make_device_ptr(v.origin()), v.num_elements(), vptr);
    }
    // work
    boost::multi::array_cref<SPComplexType const, 2, const_sp_pointer> Xsp(Xptr, X.extensions());
    boost::multi::array_ref<SPComplexType, 2, sp_pointer> vsp(vptr, v.extensions());
    ma::product(SPRealType(a), Likn, Xsp, SPRealType(c), vsp);
    if (not std::is_same<vType, SPComplexType>::value)
    {
      copy_n_cast(make_device_ptr(vsp.origin()), v.num_elements(), make_device_ptr(v.origin()));
    }
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 1)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 1)>,
           typename = void>
  void vbias(const MatA& G, MatB&& v, double dt, double a = 1., double c = 0., int k = 0)
  {
    using BType = typename std::decay<MatB>::type::element;
    using AType = typename std::decay<MatA>::type::element;
    boost::multi::array_ref<BType, 2, decltype(v.origin())> v_(v.origin(), {v.size(0), 1});
    if(haj.size(0) == 1) {
      boost::multi::array_ref<AType const, 2, decltype(G.origin())> G_(G.origin(), {1, G.size(0)});
      return vbias(G_, v_, dt, a, c, k);
    } else {
      boost::multi::array_ref<AType const, 2, decltype(G.origin())> G_(G.origin(), {G.size(0), 1});
      return vbias(G_, v_, dt, a, c, k);
    }
  }

  // v(n,w) = sum_ak L(ak,n) G(w,ak)
  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>>
  void vbias(const MatA& G, MatB&& v, double dt, double a = 1., double c = 0., [[maybe_unused]] int k = 0)
  {
    // scale a by sqrt(dt)
    a *= std::sqrt(dt);
    using GType = typename std::decay_t<typename MatA::element>;
    using vType = typename std::decay_t<MatB>::element;
    if (walker_type == CLOSED)
      a *= 2.0;
    // setup buffer space if changing precision in G or v
    size_t vmem(0), Gmem(0);
    if (not std::is_same<GType, SPComplexType>::value)
      Gmem = G.num_elements();
    if (not std::is_same<vType, SPComplexType>::value)
      vmem = v.num_elements();
    StaticSpVector SPBuff(iextensions<1u>{Gmem + vmem},
                        buffer_manager.get_generator().template get_allocator<SPComplexType>());
    sp_pointer vptr(nullptr);
    const_sp_pointer Gptr(nullptr);
    // setup origin of Gsp and copy_n_cast if necessary
    using sfqmc::afqmc::reinterpret_pointer_cast;
    if (std::is_same<GType, SPComplexType>::value)
    {
      Gptr = reinterpret_pointer_cast<SPComplexType const>(make_device_ptr(G.origin()));
    }
    else
    {
      boost::multi::array_ref<SPComplexType, 2, sp_pointer> Gsp(make_device_ptr(SPBuff.origin()), 
                                                                 G.extensions());
      ma::copy_n_cast(G,Gsp);
      Gptr = make_device_ptr(SPBuff.origin());
    }
    // setup origin of vsp and copy_n_cast if necessary
    if (std::is_same<vType, SPComplexType>::value)
    {
      vptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(v.origin()));
    }
    else
    {
      vptr = make_device_ptr(SPBuff.origin()) + Gmem;
      if (std::abs(c) > 1e-12)
        copy_n_cast(make_device_ptr(v.origin()), v.num_elements(), vptr);
    }
    // setup array references
    boost::multi::array_cref<SPComplexType const, 2, const_sp_pointer> Gsp(Gptr, G.extensions());
    boost::multi::array_ref<SPComplexType, 2, sp_pointer> vsp(vptr, v.extensions());

    if (haj.size(0) == 1)
    {
      if (walker_type == COLLINEAR)
      {
        RUNTIME_CHECK(G.size(0) == v.size(1), "");
        int NMO, nel[2];
        NMO    = Lnak[0].size(2);
        nel[0] = Lnak[0].size(1);
        nel[1] = Lnak[1].size(1);
        double c_[2];
        c_[0] = c;
        c_[1] = c;
        if (std::abs(c) < 1e-8)
          c_[1] = 1.0;
        for (int ispin = 0, is0 = 0; ispin < 2; ispin++)
        {
          RUNTIME_CHECK(Lnak[ispin].size(0) == v.size(0), "");
          auto Ln = Lnak[ispin].rotated().flatted().unrotated();
          ma::product(SPComplexType(SPRealType(a)), Ln, 
                      ma::T(Gsp.rotated().sliced(is0, is0 + nel[ispin] * NMO).unrotated()), 
                      SPComplexType(SPRealType(c_[ispin])), vsp);
          is0 += nel[ispin] * NMO;
        }
      }
      else
      {
        RUNTIME_CHECK(G.size(0) == v.size(1), "");
        RUNTIME_CHECK(Lnak[0].size(1) * Lnak[0].size(2) == G.size(1), "");
        RUNTIME_CHECK(Lnak[0].size(0) == v.size(0), "");
        SpCMatrix_ref Ln(make_device_ptr(Lnak[0].origin()), {local_nCV, Lnak[0].size(1) * Lnak[0].size(2)});
        ma::product(SPComplexType(SPRealType(a)), Ln, ma::T(Gsp), SPComplexType(SPRealType(c)), vsp);
      }
    }
    else
    {
      // multideterminant is not half-rotated, so use Likn
      RUNTIME_CHECK(Likn.size(0) == G.size(0), "");
      RUNTIME_CHECK(Likn.size(1) == v.size(0), "");
      RUNTIME_CHECK(G.size(1) == v.size(1), "");

      ma::product(SPRealType(a), ma::T(Likn), Gsp, SPRealType(c), vsp);
    }
    if (not std::is_same<vType, SPComplexType>::value)
    {
      copy_n_cast(make_device_ptr(vsp.origin()), v.num_elements(), make_device_ptr(v.origin()));
    }
  }

  template<class Mat, class MatB>
  void generalizedFockMatrix(Mat&& G, MatB&& Fp, MatB&& Fm)
  {
    int nwalk = G.size(0);
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int NMO   = hij.size(1);
    RUNTIME_CHECK(Fp.size(0) == nwalk, "");
    RUNTIME_CHECK(Fm.size(0) == nwalk, "");
    RUNTIME_CHECK(G[0].num_elements() == nspin * NMO * NMO, "");
    RUNTIME_CHECK(Fp[0].num_elements() == nspin * NMO * NMO, "");
    RUNTIME_CHECK(Fm[0].num_elements() == nspin * NMO * NMO, "");

    if(hij.size(0) == 2*NMO)
      APP_ABORT(" Error: generalizedFockMatrix not implemented with spin dependent H1.");

    // Rwn[nwalk][nCV]: 1+nspin copies
    // Twpqn[nwalk][NMO][NMO][nCV]: 1+nspin copies
    // extra copies

    // can you find out how much memory is available on the buffer?
    long LBytes = max_memory_MB * 1024L * 1024L / long(sizeof(SPComplexType));
    if constexpr (MP) {
      LBytes -= long((3 * nspin + 1) * nwalk * NMO * NMO); // G, Fp, Fm and Gt
    } else {
      LBytes -= long((1 + nspin) * nwalk * NMO * NMO); //  G and Gt
    }
    LBytes *= long(sizeof(SPComplexType));
    int Bytes = int(LBytes / long(2 * (NMO * NMO + 1) * local_nCV * sizeof(SPComplexType)));
    int nwmax = std::min(std::max(1, Bytes), nwalk);
    RUNTIME_CHECK(nwmax >= 1 && nwmax <= nwalk, "");

    sp_pointer ptr_Fp(nullptr);
    sp_pointer ptr_Fm(nullptr);
    auto buffer_alloc=buffer_manager.get_generator().template get_allocator<SPComplexType>();
    if constexpr (MP) {
      buffer_alloc.deallocate(ptr_Fp, nwalk * nspin * NMO * NMO);
      buffer_alloc.deallocate(ptr_Fp, nwalk * nspin * NMO * NMO);
    } else {
      ptr_Fm = make_device_ptr(Fm.origin());
      ptr_Fp = make_device_ptr(Fp.origin());
    }
    SpCMatrix_ref Fp_(ptr_Fp, {nwalk, nspin * NMO * NMO});
    SpCMatrix_ref Fm_(ptr_Fm, {nwalk, nspin * NMO * NMO});
    fill_n(Fp_.origin(), Fp_.num_elements(), SPComplexType(0.0));
    fill_n(Fm_.origin(), Fm_.num_elements(), SPComplexType(0.0));

    SPComplexType scl = (walker_type == CLOSED ? 2.0 : 1.0);
    std::vector<sp_pointer> Aarray;
    std::vector<sp_pointer> Barray;
    std::vector<sp_pointer> Carray;
    Aarray.reserve(nwalk);
    Barray.reserve(nwalk);
    Carray.reserve(nwalk);

    long gsz(0);
    if constexpr (MP) {
      gsz = nspin * nwmax * NMO * NMO;
    } else {
      if (nspin > 1)
        gsz = nspin * nwmax * NMO * NMO;  
    }
    StaticSpVector GBuff(iextensions<1u>{gsz}, buffer_manager.get_generator().template get_allocator<SPComplexType>());

    int nw0(0);
    while (nw0 < nwalk)
    {
      int nw = std::min(nwalk - nw0, nwmax);

      // transpose/cast G
      sp_pointer ptr(nullptr);
      if constexpr (MP) {
        ptr = GBuff.origin();
        for (int ispin = 0, is0 = 0, ip = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
          for (int n = 0; n < nw; ++n, ip += NMO * NMO)
            copy_n_cast(make_device_ptr(G[nw0 + n].origin()) + is0, NMO * NMO, ptr + ip);
      } else {
        if (nspin == 1)
        {
          ptr = make_device_ptr(G[nw0].origin());
        }
        else
        {
          ptr = GBuff.origin();
          using std::copy_n;
          for (int ispin = 0, is0 = 0, ip = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
            for (int n = 0; n < nw; ++n, ip += NMO * NMO)
              copy_n(make_device_ptr(G[nw0 + n].origin()) + is0, NMO * NMO, ptr + ip);
        }
      }
      SpCTensor_ref GF(ptr, {nspin, nw * NMO, NMO}); // now contains G in the correct structure [spin][w][i][j]
      StaticSpMatrix Gt({NMO * NMO, nw}, buffer_manager.get_generator().template get_allocator<SPComplexType>());
      fill_n(Gt.origin(), Gt.num_elements(), SPComplexType(0.0));

      StaticSpMatrix Rnw({local_nCV, nw}, buffer_manager.get_generator().template get_allocator<SPComplexType>());
      // calculate Rwn
      for (int ispin = 0; ispin < nspin; ispin++)
      {
        SpCMatrix_ref G_(GF[ispin].origin(), {nw, NMO * NMO});
        ma::add(SPComplexType(1.0), Gt, SPComplexType(1.0), ma::T(G_), Gt);
      }
      // R[n,w] = \sum_ik L[n,ik] G[ik,w]
      ma::product(SPRealType(1.0), ma::T(Likn), Gt, SPRealType(0.0), Rnw);
      StaticSpMatrix Rwn({nw, local_nCV}, buffer_manager.get_generator().template get_allocator<SPComplexType>());
      ma::transpose(Rnw, Rwn);

      // add coulomb contribution of <pr||qs>Grs term to Fp, reuse Gt for temporary storage
      // Fp[p,t] = \sum_{jl} L[p,t,n] L[j,l,n] P[j,l]
      // Fp[pt,w] = \sum_n L[pt,n] R[n,w]
      ma::product(SPRealType(1.0), Likn, Rnw, SPRealType(0.0), Gt);
      for (int ispin = 0; ispin < nspin; ispin++)
      {
        ma::add(SPComplexType(1.0), Fp_({nw0, nw0 + nw}, {ispin * NMO * NMO, (ispin + 1) * NMO * NMO}),
                SPComplexType(scl), ma::T(Gt), Fp_({nw0, nw0 + nw}, {ispin * NMO * NMO, (ispin + 1) * NMO * NMO}));
      }

      // L[i,kn]
      SpRMatrix_ref Ln(make_device_ptr(Likn.origin()), {NMO, NMO * local_nCV});
      // T[w,p,t,n] = \sum_{l} L[l,t,n] P[w,l,p]
      StaticSpMatrix Twptn({nw * NMO, NMO * local_nCV},
                         buffer_manager.get_generator().template get_allocator<SPComplexType>());
      // transpose for faster contraction
      StaticSpMatrix Taux({nw * NMO, NMO * local_nCV},
                        buffer_manager.get_generator().template get_allocator<SPComplexType>());
      SpCMatrix_ref Ttnwp(Taux.origin(), {NMO * local_nCV, nw * NMO});
      // Twptn3D: {nw, NMO * NMO, local_nCV} 
      auto&& Twptn3D= Twptn.partitioned(nw).rotated().flatted().unrotated(); 	
      // Twptn4D: {nw, NMO, NMO, local_nCV} 
      auto&& Twptn4D= Twptn.rotated().partitioned(NMO).unrotated().partitioned(nw); 	
      // Taux4D: {nw, NMO, NMO, local_nCV}	
      auto&& Taux4D= Taux.rotated().partitioned(NMO).unrotated().partitioned(nw); 	
      SpCMatrix_ref Gt_(Gt.origin(), {NMO, nw * NMO});

      for (int ispin = 0, is0 = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
      {
        SpCMatrix_ref G_(GF[ispin].origin(), {nw * NMO, NMO});
        ma::transpose(G_, Gt_);

        // J = \sum_{iklr} L[i,k,n] L[q,l,n] P[s,p,l] P[r,i,k]
        // R[n] = \sum_{ik} L[i,k,n] P[r,i,k]
        // Here T[tn,wp] = \sum_{l} L[tn,l] P[l,wp]
        // T[ln,p] = T[npl] = L[nkl] P[p,l]
        ma::product(SPRealType(1.0), ma::T(Ln), Gt_, SPRealType(0.0), Ttnwp);
        // T[wp,tn]
        ma::transpose(Ttnwp, Twptn);

        // transpose Twptn -> Twtpn=Taux
        // T[wt,pn]
	ma::transpose_wabn_to_wban(Twptn4D, Taux4D);

        // add exchange component to Fm_
        Aarray.clear();
        Barray.clear();
        Carray.clear();
        for (int w = 0; w < nw; w++)
        {
          Aarray.push_back(Taux.partitioned(nw)[w].origin());
          Barray.push_back(Twptn.partitioned(nw)[w].origin());
          Carray.push_back(Fm_[w].origin() + is0);
        }
        using ma::gemmBatched;
        // careful with expected Fortran ordering here!!!
        // K[p,q] = \sum_{ln} T[n,l,p] T[n,q,l]
        //          \sum_{ln} T[nl,p] T[nl,q]
        gemmBatched('T', 'N', NMO, NMO, NMO * local_nCV, SPComplexType(1.0), Aarray.data(), NMO * local_nCV,
                    Barray.data(), NMO * local_nCV, SPComplexType(1.0), Carray.data(), NMO, nw);

        // add coulomb component to Fm_
        Aarray.clear();
        Barray.clear();
        Carray.clear();
        for (int w = 0; w < nw; w++)
        {
          Aarray.push_back(Twptn3D[w].origin());
          Barray.push_back(Rwn[w].origin());
          Carray.push_back(Fm_[w].origin() + is0);
        }
        using ma::gemmBatched;
        // careful with expected Fortran ordering here!!!
        // J[w][pq] = \sum_{n} T[w][pq,n] R[w][n]
        gemmBatched('T', 'N', NMO * NMO, 1, local_nCV, SPComplexType(-1.0) * scl, Aarray.data(), local_nCV,
                    Barray.data(), local_nCV, SPComplexType(1.0), Carray.data(), NMO * NMO, nw);

        // Fp
        // Need Gt_[i][wj]
	ma::transpose_wabn_to_wban( G_.rotated().partitioned(NMO).unrotated().partitioned(1), 
				    Gt_.rotated().partitioned(nw).unrotated().partitioned(1));
        ma::product(SPRealType(1.0), ma::T(Ln), Gt_, SPRealType(0.0), Ttnwp);
        ma::transpose(Ttnwp, Twptn);
        ma::transpose_wabn_to_wban(Twptn4D, Taux4D); 
        // add coulomb component to Fp_, same as Fm_ above
        Aarray.clear();
        Barray.clear();
        Carray.clear();
        for (int w = 0; w < nw; w++)
        {
          Aarray.push_back(Twptn3D[w].origin());
          Barray.push_back(Rwn[w].origin());
          Carray.push_back(Fp_[nw0+w].origin() + is0);
        }
        using ma::gemmBatched;
        // careful with expected Fortran ordering here!!!
        // Coulomb component
        gemmBatched('T', 'N', NMO * NMO, 1, local_nCV, SPComplexType(-1.0) * scl, Aarray.data(), local_nCV,
                    Barray.data(), local_nCV, SPComplexType(1.0), Carray.data(), NMO * NMO, nw);

        // add exchange component of Fp_
        Aarray.clear();
        Barray.clear();
        Carray.clear();
        for (int w = 0; w < nw; w++)
        {
          Aarray.push_back(Taux.partitioned(nw)[w].origin());
          Barray.push_back(Twptn.partitioned(nw)[w].origin());
          Carray.push_back(Fp_[nw0+w].origin() + is0);

          // add exchange contribution of <pr||qs>Grs term by adding Lptn to Twptn
          // dispatch directly from here to be able to add to the real part only
          // K1B[p,q] = -\sum_{jl} L[jt,n] L[pl,n] P[jl]
          ma::axpy(Likn.num_elements(), SPRealType(-1.0), ma::pointer_dispatch(Likn.origin()), 1,
               reinterpret_pointer_cast<SPRealType>(ma::pointer_dispatch(Twptn3D[w].origin())), 2,
	       ma::select_backend<shmSpRMatrix>());
        }
        using ma::gemmBatched;
        // careful with expected Fortran ordering here!!!
        gemmBatched('T', 'N', NMO, NMO, NMO * local_nCV, SPComplexType(1.0), Aarray.data(), NMO * local_nCV,
                    Barray.data(), NMO * local_nCV, SPComplexType(1.0), Carray.data(), NMO, nw);

      } // ispin

      nw0 += nw;
    }

    if constexpr (MP) {
      copy_n_cast(Fp_.origin(), Fp_.num_elements(), make_device_ptr(Fp.origin()));
      copy_n_cast(Fm_.origin(), Fm_.num_elements(), make_device_ptr(Fm.origin()));
      buffer_alloc.deallocate(ptr_Fp, nwalk * nspin * NMO * NMO);	
      buffer_alloc.deallocate(ptr_Fm, nwalk * nspin * NMO * NMO);	
    }

    //fill_n(Fp.origin(),Fp.num_elements(),SPComplexType(0.0));
    //fill_n(Fm.origin(),Fm.num_elements(),SPComplexType(0.0));
    // add one body terms now
    {
      std::vector<pointer> Aarr;
      std::vector<pointer> Barr;
      std::vector<pointer> Carr;
      Aarr.reserve(nspin * nwalk);
      Barr.reserve(nspin * nwalk);
      Carr.reserve(nspin * nwalk);
      // Fm -= G[w][p][r] * h[q][r]
      Aarr.clear();
      Barr.clear();
      Carr.clear();
      for (int ispin = 0, is0 = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
      {
        for (int w = 0; w < nwalk; w++)
        {
          Aarr.push_back(make_device_ptr(hij_dev.origin()));
          Barr.push_back(make_device_ptr(G[w].origin()) + is0);
          Carr.push_back(make_device_ptr(Fm[w].origin()) + is0);
        }
      }
      using ma::gemmBatched;
      // careful with expected Fortran ordering here!!!
      gemmBatched('T', 'N', NMO, NMO, NMO, ComplexType(-1.0), Aarr.data(), NMO, Barr.data(), NMO, ComplexType(1.0),
                  Carr.data(), NMO, Aarr.size());


      // Fp -= G[w][r][p] * h[q][r]
      Aarr.clear();
      Barr.clear();
      Carr.clear();
      C4Tensor_ref Fp4D(make_device_ptr(Fp.origin()), {nwalk, nspin, NMO, NMO});
      for (int ispin = 0, is0 = 0; ispin < nspin; ispin++, is0 += NMO * NMO)
      {
        for (int w = 0; w < nwalk; w++)
        {
          Aarr.push_back(make_device_ptr(hij_dev.origin()));
          Barr.push_back(make_device_ptr(G[w].origin()) + is0);
          Carr.push_back(make_device_ptr(Fp[w].origin()) + is0);

          // add diagonal contribution to Fp
          ma::add(ComplexType(1.0), Fp4D[w][ispin], ComplexType(1.0), ma::T(hij_dev), Fp4D[w][ispin]);
        }
      }
      using ma::gemmBatched;
      // careful with expected Fortran ordering here!!!
      gemmBatched('T', 'T', NMO, NMO, NMO, ComplexType(-1.0), Aarr.data(), NMO, Barr.data(), NMO, ComplexType(1.0),
                  Carr.data(), NMO, Aarr.size());
    }
  }

  bool distribution_over_cholesky_vectors() const { return true; }
  int number_of_ke_vectors() const { return local_nCV; }
  int local_number_of_cholesky_vectors() const { return local_nCV; }
  int global_number_of_cholesky_vectors() const { return global_nCV; }
  int global_origin_cholesky_vector() const { return global_origin; }

  // transpose=true means G[nwalk][ik], false means G[ik][nwalk]
  bool transposed_G_for_vbias() const { return (haj.size(0)==1); } 
  bool transposed_G_for_E() const { return true; }
  // transpose=true means vHS[nwalk][ik], false means vHS[ik][nwalk]
  bool transposed_vHS() const { return false; }

  bool fast_ph_energy() const { return false; }
  bool spin_dependent_vHS() const { return false; }

  boost::multi::array<ComplexType, 2> getHSPotentials() { return boost::multi::array<ComplexType, 2>{}; }

private:
  Allocator allocator_;
  SpAllocator sp_allocator_;
  DeviceBufferManager buffer_manager;

  WALKER_TYPES walker_type;

  long max_memory_MB;
  int global_origin;
  int global_nCV;
  int local_nCV;

  ComplexType E0;

  // bare one body hamiltonian
  mpi3VMatrix hij;

  // one body hamiltonian
  shmCMatrix hij_dev;

  // (potentially half rotated) one body hamiltonian
  shmCMatrix haj;

  //Cholesky Tensor Lik[i][k][n]
  shmSpRMatrix Likn;

  // permuted half-tranformed Cholesky tensor
  // Lnak[ 2*idet + ispin ]
  std::vector<shmSpC3Tensor> Lnak;

  // one-body piece of Hamiltonian factorization
  mpi3CMatrix vn0;

  // used in PH reference/excited energy evaluation
  // Twian = sum_k G_ref[w][i][k] L[a][n][k]
  SpVector Twina_ph;
  // Swia = sum_k G_ref[w][i][k] h[a][k]
  CVector Swia_ph;

  template<class Mat, class MatB, class MatC>
  void energy_impl(int ispin,
              Mat&& E,
              MatB const& Gc,
              int nd,
              MatC& Kl,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    RUNTIME_CHECK(ispin==0 or ispin==1, "");
    RUNTIME_CHECK(E.size(1) >= 3, "");
    RUNTIME_CHECK(nd >= 0, "");
    RUNTIME_CHECK(nd < haj.size(), "");
    if (walker_type == COLLINEAR)
      RUNTIME_CHECK(2 * nd + 1 < Lnak.size(), "");
    else
      RUNTIME_CHECK(nd < Lnak.size(), "");

    int nwalk = Gc.size(0);
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int NMO   = hij.size(1) / npol;
    int nel[2];
    nel[0] = Lnak[nspin * nd].size(1);
    nel[1] = ((nspin == 2) ? Lnak[nspin * nd + 1].size(1) : 0);
    RUNTIME_CHECK(Lnak[nspin * nd].size(0) == local_nCV, "");
    RUNTIME_CHECK(Lnak[nspin * nd].size(2) == npol*NMO, "");
    if (nspin == 2)
    {
      RUNTIME_CHECK(Lnak[nspin * nd + 1].size(0) == local_nCV, "");
      RUNTIME_CHECK(Lnak[nspin * nd + 1].size(2) == NMO, "");
    }
    RUNTIME_CHECK(Gc.num_elements() == nwalk * nel[ispin] * NMO * npol, "");

    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error in Real3IndexFactorization_batched_v2: Incorrect matrix dimensions "); 
    if (addEJ)
      RUNTIME_CHECK(Kl.size(0) == nwalk && Kl.size(1) == local_nCV, "");

    SPRealType scl = (walker_type == CLOSED ? 2.0 : 1.0);

    // one-body contribution
    // haj[ndet][nocc*nmo]
    // not parallelized for now, since it would require customization of Wfn
    if (addH1)
    {
      if(ispin == 0) {
        ma::product(ComplexType(scl), Gc, haj[nd].sliced(0,nel[0]*NMO*npol),
                    ComplexType(1.0), E(E.extension(0), 0));
      } else if(ispin==1) {
        ma::product(ComplexType(scl), Gc, haj[nd].sliced(nel[0]*NMO,(nel[0]+nel[1])*NMO),
                    ComplexType(scl), E(E.extension(0), 0));
      }
    }

    if (addEXX)
    {
      // MAM: if the memory available in the MR is larger than max_memory_MB, use it!
      int max_nCV = 0;
      long LBytes = max_memory_MB * 1024L * 1024L;
      int Bytes   = int(LBytes / long(nwalk * nel[ispin] * nel[ispin] * sizeof(SPComplexType)));
      max_nCV     = std::min(std::max(1, Bytes), local_nCV);
      RUNTIME_CHECK(max_nCV > 1 && max_nCV <= local_nCV, "");

      StaticSpMatrix GF({nwalk * nel[ispin], npol*NMO}, 
                            buffer_manager.get_generator().template get_allocator<SPComplexType>());
      for (int n = 0; n < nwalk; ++n)
      {
        copy_n_cast(make_device_ptr(Gc[n].origin()), nel[ispin]*NMO*npol,
                    GF.origin() + n * nel[ispin] * NMO * npol);
      }

      int nCV = 0;
      while (nCV < local_nCV)
      {
        int nvecs = std::min(local_nCV - nCV, max_nCV);
        auto Lna= Lnak[nd * nspin * npol + ispin].sliced(nCV, nCV+nvecs).flatted();
        Static4Tensor Twbna({nwalk, nel[ispin], nvecs, nel[ispin]},
                           buffer_manager.get_generator().template get_allocator<SPComplexType>());

        ma::product(GF, ma::T(Lna), Twbna.flatted().rotated().flatted().unrotated());

        // E[w] = -0.5*scl* sum_abn Twanb * Twbna
        ma::Awanb_Awbna_Bw(SPComplexType(SPRealType(-0.5 * scl)), Twbna, E.rotated()[1].unrotated());

        if (addEJ)
          ma::Awana_Bwn(Twbna, Kl);

        nCV += max_nCV;
      }
    }

    if(addEXX and not addEJ)
      APP_ABORT(" Error: addEXX and not addEJ not yet implemented. \n\n");

  }  // energy_impl


  // similar to energy_impl, but generates Twina_ph to be used subsequently in
  // ph_excited_energy.  
  // Due to the nature of the implementation, looping over blocks of CV is not possible.
  // Therr has to be enough memory to do everything at once
  // Careful here! Gc is the Green's function of the reference!
  // It is size {nwalk, #electrons, NMO} and not! {nwalk, nactive, NMO} like Lnak!!! 
  template<class Mat, class MatB, class MatC>
  void ph_ref_energy_impl(int ispin,
              Mat&& E,
              MatB const& Gc,
              MatC& Kl,
              bool addH1  = true)
  {
    RUNTIME_CHECK(ispin==0 or ispin==1, "");
    RUNTIME_CHECK(E.size(1) >= 3, "");

    int nwalk = Gc.size(0);
    int NMO   = hij.size(1);
    int nact[2], nel;
    nact[0] = Lnak[0].size(1);
    nact[1] = ((walker_type == COLLINEAR) ? Lnak[1].size(1) : 0);
    nel = Gc.size(1)/NMO;
    RUNTIME_CHECK(Lnak[0].size(0) == local_nCV, "");
    RUNTIME_CHECK(Lnak[0].size(2) == NMO, "");
    if (walker_type == COLLINEAR)
    {
      RUNTIME_CHECK(Lnak[1].size(0) == local_nCV, "");
      RUNTIME_CHECK(Lnak[1].size(2) == NMO, "");
    }
    RUNTIME_CHECK(Gc.num_elements() == nwalk * nel * NMO, "");
    RUNTIME_CHECK(Kl.size(0) == nwalk && Kl.size(1) == local_nCV, "");

    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error in Real3IndexFactorization_batched_v2: Incorrect matrix dimensions "); 

    // one-body contribution
    // haj[ndet][nocc*nmo]
// right now this only works if the reference configuration is refc[i] = i!!!!
    if (addH1)
    {
      if(Swia_ph.size(0) < nwalk * nel * std::max(nact[0],nact[1]))
        Swia_ph = CVector(iextensions<1u>{nwalk * nel * std::max(nact[0],nact[1])}, SPComplexType(0.0));
                           
      Array_ref<ComplexType, 3, pointer> Swia(Swia_ph.origin(), {nwalk, nel, nact[ispin]});
      auto Gwik= Gc.rotated().partitioned(nel).unrotated();
      auto h2D=haj[0].partitioned(nact[0]+nact[1]);
      RUNTIME_CHECK(h2D.size(1) == NMO, "");
      if(ispin == 0) {
        ma::productStridedBatched(Gwik,ma::T(h2D.sliced(0,nact[0])),Swia);
      } else if(ispin==1) {
        ma::productStridedBatched(Gwik,ma::T(h2D.sliced(nact[0],nact[0]+nact[1])),Swia);
      }
// need kernel!!!
      for(int iw=0; iw<nwalk; iw++)
        for(int i=0; i<nel; i++)
          E[iw][0] += Swia[iw][i][i];
          //E[iw][0] += Swia[iw][i][refc[i]];
    }

    SPRealType scl = (walker_type == CLOSED ? 2.0 : 1.0);

    StaticSpMatrix GF({nwalk * nel, NMO},
                            buffer_manager.get_generator().template get_allocator<SPComplexType>());
    for (int n = 0; n < nwalk; ++n)
    {
      copy_n_cast(make_device_ptr(Gc[n].origin()), nel*NMO,
                    GF.origin() + n * nel * NMO);
    }

    auto Lna= Lnak[ispin].flatted();
    if(Twina_ph.size(0) < nwalk * nel * local_nCV * nact[ispin])
      Twina_ph = SpVector(iextensions<1u>{nwalk * nel * local_nCV * nact[ispin]}, SPComplexType(0.0)); 
    SpC4Tensor_ref Twina(Twina_ph.origin(), {nwalk, nel, local_nCV, nact[ispin]});  

    ma::product(GF, ma::T(Lna), Twina.flatted().rotated().flatted().unrotated());

    // E[w] = -0.5*scl* sum_abn Twanb * Twbna
    ma::Awanb_Awbna_Bw(SPComplexType(SPRealType(-0.5 * scl)), Twina, E.rotated()[1].unrotated());

    ma::Awana_Bwn(Twina, Kl);

  }  // ph_ref_energy_impl 

  template<class Iptr, class Mat, class MatB, class MatC, class MatW,
           typename = typename std::enable_if_t<(std::decay<Mat>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatW>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 4)>
          >
  void ph_excited_energy_impl([[maybe_unused]] SpinTypes spin_component,
              int ndet,
              int nex,
              int nelec,
              int nact,
              Iptr const iexcit,
              Iptr const refc,
              Mat&& E,
              MatW&& wgt,
              MatB const& R,
              MatC& Kl,
              bool addH1)
  {
    using std::copy_n;
    // R[nwalk, ndet, nex, nact] 
    // E[nwalk, 3]
    // wgt[ndet, nwalk]
    // K[ndet, nwalk, nkev]
    int nwalk = R.size(0);

    RUNTIME_CHECK(R.size(1) == ndet, "");
    RUNTIME_CHECK(R.size(2) == nex, "");
    RUNTIME_CHECK(R.size(3) == nact, "");
    RUNTIME_CHECK(E.size(0) == nwalk, "");
    RUNTIME_CHECK(E.size(1) == 3, "");
    RUNTIME_CHECK(wgt.size(0) == ndet, "");
    RUNTIME_CHECK(wgt.size(1) == nwalk, "");

    // by convention, add E0 only to Alpha
    /* One body terms */
    if(addH1) {
      if(Swia_ph.size(0) < nwalk * nelec * nact)
        APP_ABORT("Error in ph_excited_energy: Unexpected size in Swia_ph.");
      Array_ref<ComplexType, 3, pointer> Swia(Swia_ph.origin(), {nwalk, nelec, nact});
      using ma::ph_excited_1body_energy;
      ph_excited_1body_energy(iexcit, refc, Swia, R, wgt, E.rotated()[0]);
    }

    /* Two body terms */
    // Make sure Twina_ph has appropriate dimensions
    if(Twina_ph.size(0) < nwalk * nelec * local_nCV * nact)
      APP_ABORT("Error in ph_excited_energy: Unexpected size in Twina_ph.");
    SpC4Tensor_ref Twina(Twina_ph.origin(), {nwalk, nelec, local_nCV, nact});

    using ma::ph_excited_2body_energy_dense_cholesky_Tpna;
    RUNTIME_CHECK(Kl.stride(0) == Kl.size(1)*Kl.size(2), ""); 
    RUNTIME_CHECK(Kl.stride(1) == Kl.size(2), ""); 
    RUNTIME_CHECK(Kl.stride(2) == 1, ""); 
    RUNTIME_CHECK(Kl.size(0) == ndet, "");
    RUNTIME_CHECK(Kl.size(1) == nwalk, "");
    RUNTIME_CHECK(Kl.size(2) == local_nCV, "");
    ph_excited_2body_energy_dense_cholesky_Tpna(iexcit, refc, Twina, R, wgt,
        E.rotated()[1], E.rotated()[2], Kl);
  }


};

} // namespace afqmc

} // namespace sfqmc

#endif
