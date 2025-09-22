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

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_REAL3INDEXFACTORIZATION_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_REAL3INDEXFACTORIZATION_HPP

#include <vector>
#include <type_traits>

//#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "mpi3/shared_communicator.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "Numerics/ma_operations.hpp"

#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/taskgroup.h"

#include "AFQMC/Wavefunctions/detail/phmsd_impl.hpp"

namespace sfqmc
{
namespace afqmc
{
// MAM: to make the working precision of this class a template parameter,
//      sed away SPComplexType by another type ( which is then defined through a template parameter)
//      This should be enough, since some parts are kept at ComplexType precision
//      defined through the cmake

// Custom implementation for real build
template<bool MP, bool REAL_ONEBODY>
class Real3IndexFactorization
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;

  using ValueType = typename std::conditional<REAL_ONEBODY, RealType, ComplexType>::type;
  using shmVMatrix  = multi::array<ValueType, 2, shared_allocator<ValueType>>;

  using sp_pointer       = SPComplexType*;
  using const_sp_pointer = SPComplexType const*;

  using IVector  = multi::array<int, 1>;
  using CVector  = multi::array<ComplexType, 1>;
  using SpVector = multi::array<SPComplexType, 1>;

  using CMatrix      = multi::array<ComplexType, 2>;
  using CMatrix_cref = multi::array_cref<ComplexType, 2>;
  using CMatrix_ref  = multi::array_ref<ComplexType, 2>;
  using CVector_ref  = multi::array_ref<ComplexType, 1>;

  using RMatrix      = multi::array<RealType, 2>;
  using RMatrix_cref = multi::array_cref<RealType, 2>;
  using RMatrix_ref  = multi::array_ref<RealType, 2>;
  using RVector_ref  = multi::array_ref<RealType, 1>;

  using SpCMatrix      = multi::array<SPComplexType, 2>;
  using SpCMatrix_cref = multi::array_cref<SPComplexType, 2>;
  using SpCVector_ref  = multi::array_ref<SPComplexType, 1>;
  using SpCMatrix_ref  = multi::array_ref<SPComplexType, 2>;

  using SpRMatrix      = multi::array<SPRealType, 2>;
  using SpRMatrix_cref = multi::array_cref<SPRealType, 2>;
  using SpRVector_ref  = multi::array_ref<SPRealType, 1>;
  using SpRMatrix_ref  = multi::array_ref<SPRealType, 2>;

  using C3Tensor       = multi::array<ComplexType, 3>;
  using SpC3Tensor     = multi::array<SPComplexType, 3>;
  using SpC3Tensor_ref = multi::array_ref<SPComplexType, 3>;
  using SpC4Tensor_ref = multi::array_ref<SPComplexType, 4>;

  using shmCVector  = multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using shmRMatrix  = multi::array<RealType, 2, shared_allocator<RealType>>;
  using shmCMatrix  = multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using shmC3Tensor = multi::array<ComplexType, 3, shared_allocator<ComplexType>>;

  using shmSpRVector  = multi::array<SPRealType, 1, shared_allocator<SPRealType>>;
  using shmSpRMatrix  = multi::array<SPRealType, 2, shared_allocator<SPRealType>>;
  using shmSpCVector  = multi::array<SPComplexType, 1, shared_allocator<SPComplexType>>;
  using shmSpCMatrix  = multi::array<SPComplexType, 2, shared_allocator<SPComplexType>>;
  using shmSpC3Tensor = multi::array<SPComplexType, 3, shared_allocator<SPComplexType>>;

  template<class T>
  using device_alloc_type = HostBufferManager::template allocator_t<T>;
  template<class T>
  using shm_alloc_type = LocalTGBufferManager::template allocator_t<T>;

  template<class U, int N>
  using ShmArray = multi::static_array<U, N, shm_alloc_type<U>>;
  template<class U, int N>
  using DevArray = multi::static_array<U, N, device_alloc_type<U>>;

  using this_t = Real3IndexFactorization;

public:
  static const HamiltonianTypes HamOpType = RealDenseFactorized;
  HamiltonianTypes getHamType() const { return HamOpType; }

  Real3IndexFactorization(afqmc::TaskGroup_& tg_,
                          WALKER_TYPES type,
                          shmVMatrix&& hij_,
                          shmCMatrix&& haj_,
                          shmSpRMatrix&& vik,
                          shmSpCMatrix&& vak,
                          std::vector<shmSpC3Tensor>&& vank,
                          shmCMatrix&& vn0_,
                          ComplexType e0_,
                          int cv0,
                          int gncv)
      : TG(tg_),
        walker_type(type),
        device_buffer_manager(),
        shm_buffer_manager(),
        global_origin(cv0),
        global_nCV(gncv),
        local_nCV(0),
        E0(e0_),
        hij(std::move(hij_)),
        haj(std::move(haj_)),
        Likn(std::move(vik)),
        Lank(std::move(vank)),
        Lakn(std::move(vak)),
        vn0(std::move(vn0_)),
        Twian_ph(iextensions<1u>{0},shared_allocator<SPComplexType>{TG.TG_local()}),
        Swia_ph(iextensions<1u>{0},shared_allocator<ComplexType>{TG.TG_local()})
  {
    local_nCV = Likn.size(1);
    TG.Node().barrier();
  }

  ~Real3IndexFactorization() {}

  Real3IndexFactorization(const Real3IndexFactorization& other) = delete;
  Real3IndexFactorization& operator=(const Real3IndexFactorization& other) = delete;
  Real3IndexFactorization(Real3IndexFactorization&& other)                 = default;
  Real3IndexFactorization& operator=(Real3IndexFactorization&& other) = delete;

  CMatrix getOneBodyPropagatorMatrix(TaskGroup_& TG_, double dt, 
                                     multi::array<ComplexType, 1> const& vMF)
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

    // MAM: This is wrong if hij's structure is not consistent with the walker_type
    // FIX FIX FIX!!!
    /* KE: This should be correct. The Hamiltonian class handles reading the input integrals and is (therefore) resonsible for making hij's structre consistent with the walker_type.
    There are checks above to ensure that this is true.
    */
    CMatrix H1({nspin * npol * NMO, npol * NMO});
    CMatrix P0({NMO, NMO});

    // add sum_n vMF*Spvn, vMF has local contribution only!
    multi::array_ref<ComplexType, 1> P1D(P0.origin(), {NMO * NMO});
    std::fill_n(P1D.origin(), P1D.num_elements(), ComplexType(0));
    vHS(vMF, P1D, dt);
    TG_.TG().all_reduce_in_place_n(P1D.origin(), P1D.num_elements(), std::plus<>());

    std::fill_n(H1.origin(), H1.num_elements(), ComplexType(0));

    // add hij + vn0 and symmetrize
    using ma::conj;

    for (int i = 0; i < NMO; i++)
    {
      H1[i][i] =  P0[i][i] + dt * (hij[i][i] + vn0[i][i]);
      if(walker_type == COLLINEAR)
        H1[NMO+i][i] = P0[i][i] + dt * (hij[I0+i][i] + vn0[i][i]);
      else if(walker_type == NONCOLLINEAR)
        H1[NMO+i][NMO+i] = P0[i][i] + dt * (hij[I0+i][J0+i] + vn0[i][i]);
      for (int j = i + 1; j < NMO; j++)
      {
        H1[i][j] = P0[i][j] + dt * (hij[i][j] + vn0[i][j]);
        H1[j][i] = P0[j][i] + dt * (hij[j][i] + vn0[j][i]);
        // This is really cutoff dependent!!!
        if (std::abs(H1[i][j] - ma::conj(H1[j][i])) > 1e-6)
        {
          app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 is not hermitian. ");
          app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[i][j],H1[j][i]);
          app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[i][j],hij[j][i]);
          app_warning("             vn0[I,J]:{}, vn0[J,I]:{} ",vn0[i][j],vn0[j][i]);
        }
        H1[i][j] = 0.5 * (H1[i][j] + ma::conj(H1[j][i]));
        H1[j][i] = ma::conj(H1[i][j]);
        if(walker_type == COLLINEAR) {
          H1[NMO+i][j] = P0[i][j] + dt * (hij[I0+i][j] + vn0[i][j]);
          H1[NMO+j][i] = P0[j][i] + dt * (hij[I0+j][i] + vn0[j][i]);
          // This is really cutoff dependent!!!
          if (std::abs(H1[NMO+i][j] - ma::conj(H1[NMO+j][i])) > 1e-6)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (beta) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[NMO+i][j],H1[NMO+j][i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[I0+i][j],hij[I0+j][i]);
            app_warning("             vn0[I,J]:{}, vn0[J,I]:{} ",vn0[i][j],vn0[j][i]);
          }
          H1[NMO+i][j] = 0.5 * (H1[NMO+i][j] + ma::conj(H1[NMO+j][i]));
          H1[NMO+j][i] = ma::conj(H1[NMO+i][j]);
        } else if(walker_type == NONCOLLINEAR) {
          H1[I0+i][J0+j] = P0[i][j] + dt * (hij[I0+i][J0+j] + vn0[i][j]);
          H1[I0+j][J0+i] = P0[j][i] + dt * (hij[I0+j][J0+i] + vn0[j][i]);

          // a-b and b-a are hij
          H1[I0+i][j] = dt * hij[I0+i][j];
          H1[I0+j][i] = dt * hij[I0+j][i];

          H1[i][J0+j] = dt * hij[i][J0+j];
          H1[j][J0+i] = dt * hij[j][J0+i];

          // This is really cutoff dependent!!!
          if (std::abs(H1[I0+i][J0+j] - ma::conj(H1[J0+j][I0+i])) > 1e-6) // b-b
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (beta-beta) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[I0+i][J0+j],H1[I0+j][J0+i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[I0+i][J0+j],hij[I0+j][J0+i]);
            app_warning("             vn0[I,J]:{}, vn0[J,I]:{} ",vn0[i][j],vn0[j][i]);
          }

          if (std::abs(H1[i][J0+j] - ma::conj(H1[J0+j][i])) > 1e-6)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (spin-flip) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[i][J0+j],H1[I0+j][i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[i][J0+j],hij[I0+j][i]);
          }

          H1[I0+i][J0+j] = 0.5 * (H1[I0+i][J0+j] + ma::conj(H1[J0+j][I0+i]));
          H1[J0+j][I0+i] = ma::conj(H1[I0+i][J0+j]);

          H1[i][J0+j] = 0.5 * (H1[i][J0+j] + ma::conj(H1[J0+j][i]));
          H1[J0+j][i] = ma::conj(H1[i][J0+j]);

          H1[I0+i][j] = 0.5 * (H1[I0+i][j] + ma::conj(H1[j][I0+i]));
          H1[j][I0+i] = ma::conj(H1[I0+i][j]);
        }
      }
    }

    return H1;
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
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nwalk = Gc.size(0);
    int NMO   = hij.size(1) / npol;
    int nel[2];
    nel[0] = Lank[nspin * nd].size(0);
    nel[1] = ((nspin == 2) ? Lank[nspin * nd + 1].size(0) : 0);

    RUNTIME_CHECK(Gc.num_elements() == nwalk * (nel[0]+nel[1]) * npol * NMO, "");
    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error Real3IndexFactorization::energy: Incorrect matrix dimensions ");
    for (int n = 0; n < nwalk; n++)
      std::fill_n(E[n].origin(), 3, ComplexType(0.));
    if(addH1) {
      for (int i = 0; i < nwalk; i++)
        E[i][0] += E0;
    }

    if (addEJ)
    {
      ShmArray<SPComplexType, 2> Kl({nwalk,local_nCV}, SPComplexType(0.0),
                shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());

      for(int is=0, is0=0; is<nspin; ++is) {
        energy_impl(is, E, Gc(Gc.extension(0), {is0, is0+nel[is]*NMO*npol}), nd, Kl,  
                  addH1, addEJ, addEXX); 
        is0 += nel[is]*NMO*npol;
        TG.TG_local().barrier();
      }
 
      SPRealType scl = (walker_type == CLOSED ? 2.0 : 1.0);
      for (int n = 0; n < nwalk; ++n)
      {
        if (n % TG.TG_local().size() == TG.TG_local().rank())
          E[n][2] += 0.5 * static_cast<ComplexType>(scl * scl * ma::dot(Kl[n], Kl[n]));
      }
    } else {
      SpCMatrix Kl({0,0});  // dummy matrix, not used 
      for(int is=0, is0=0; is<nspin; ++is) {
        energy_impl(is, E, Gc(Gc.extension(0), {is0, is0+nel[is]*NMO*npol}), nd, Kl,
                  addH1, addEJ, addEXX); 
        is0 += nel[is]*NMO*npol;
      }
    }
    TG.TG_local().barrier();
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
    using EJType = typename std::decay_t<MatC>::element_type;
    int ispin = ( spin_component == Alpha ? 0 : 1);
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int nwalk = Gc.size(0);
    int NMO   = hij.size(1);
    int nel[2];
    nel[0] = Lank[nspin * nd].size(0);
    nel[1] = ((nspin == 2) ? Lank[nspin * nd + 1].size(0) : 0);

    RUNTIME_CHECK(Gc.num_elements() == nwalk * nel[ispin] * NMO, "");
    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error Real3IndexFactorization::energy: Incorrect matrix dimensions ");

    if constexpr  (not std::is_same_v<EJType,SPComplexType>) {
      
      if( not addEJ ) {
	SpCMatrix Kl({0,0});
	energy(spin_component, std::forward<Mat>(E), Gc, nd, Kl, addH1, addEJ, addEXX);
      } else {	
	assert(EJn.size(0) == nwalk);
	assert(EJn.size(1) == local_nCV);
        long i0, iN;
        std::tie(i0, iN) = FairDivideBoundary(long(TG.TG_local().rank()),
                            long(EJn.size(0)), long(TG.TG_local().size()));
        ShmArray<SPComplexType, 2> Kl({nwalk,local_nCV}, SPComplexType(0.0),
                shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());
        TG.TG_local().barrier();
        energy(spin_component, std::forward<Mat>(E), Gc, nd, Kl, addH1, addEJ, addEXX);
        if(iN>i0)
          ma::copy_n_cast(Kl.sliced(i0,iN), EJn.sliced(i0,iN));
      }

    } else { 

      for (int n = 0; n < nwalk; n++) 
        std::fill_n(E[n].origin(), 3, ComplexType(0.));
      // by convention, add E0 only to Alpha
      if(addH1 and (spin_component == Alpha)) { 
        for (int i = 0; i < nwalk; i++)
          E[i][0] += E0;  
      }
    
      if (addEJ)
      { 
	assert(EJn.size(0) == nwalk);
	assert(EJn.size(1) == local_nCV);

        energy_impl(ispin, E, Gc, nd, EJn, addH1, addEJ, addEXX);
        TG.TG_local().barrier();
      
        for (int n = 0; n < nwalk; ++n)
        { 
          if (n % TG.TG_local().size() == TG.TG_local().rank()) 
            E[n][2] += 0.5 * static_cast<ComplexType>(ma::dot(EJn[n], EJn[n]));
        }

      } else {
        energy_impl(ispin, E, Gc, nd, EJn, addH1, addEJ, addEXX);
      }
    }
    TG.TG_local().barrier();
  }

  template<class Mat, class MatB, class MatC,
           typename = typename std::enable_if_t<(std::decay<Mat>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>
          >
  void ph_reference_energy(SpinTypes spin_component,
              Mat&& E,
              MatB const& Gc,
              MatC&& EJn,
              bool addH1 = true)
  { 
    using std::fill_n;
    // check that ndet == 1
    RUNTIME_CHECK(haj.size() == 1, "");
    if (walker_type == COLLINEAR)
      RUNTIME_CHECK(Lank.size() == 2, "");
    else
      RUNTIME_CHECK(Lank.size() == 1, "");
    
    int ispin = ( spin_component == Alpha ? 0 : 1);
    int nwalk = Gc.size(0);
    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT(" Error Real3IndexFactorization::energy: Incorrect matrix dimensions ");
    ma::fill(E({0,nwalk},{0,3}),ComplexType(0.));
    // by convention, add E0 only to Alpha
    if(addH1 and (spin_component == Alpha)) {
      for (int i = 0; i < nwalk; i++)
        E[i][0] += E0;
    }
    RUNTIME_CHECK(EJn.size(0) == nwalk, "");
    RUNTIME_CHECK(EJn.size(1) == local_nCV, "");

    using EJType = typename std::decay_t<MatC>::element_type;
    if constexpr  (not std::is_same_v<EJType,SPComplexType>) {
    
      ShmArray<SPComplexType, 2> Kl({nwalk,local_nCV}, SPComplexType(0.0),
                shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      ph_ref_energy_impl(ispin, E, Gc, Kl, addH1);
      TG.TG_local().barrier();

      for (int n = 0; n < nwalk; ++n)
      {
        if (n % TG.TG_local().size() == TG.TG_local().rank())
          E[n][2] += 0.5 * static_cast<ComplexType>(ma::dot(Kl[n], Kl[n]));
      }

      long i0, iN; 
      std::tie(i0, iN) = FairDivideBoundary(long(TG.TG_local().rank()), 
                          long(EJn.size(0)), long(TG.TG_local().size()));
      if(iN>i0)
        ma::copy_n_cast(Kl.sliced(i0,iN),EJn.sliced(i0,iN));
      TG.TG_local().barrier();

    } else {

      ph_ref_energy_impl(ispin, E, Gc, EJn, addH1);
      TG.TG_local().barrier();
      
      for (int n = 0; n < nwalk; ++n)
      { 
        if (n % TG.TG_local().size() == TG.TG_local().rank())
          E[n][2] += 0.5 * static_cast<ComplexType>(ma::dot(EJn[n], EJn[n]));
      }

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
    using EJType = typename std::decay_t<MatC>::element_type;
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
    multi::array_ref<BType, 2> v_(raw_pointer_cast(v.origin()), {v.size(0), 1});
    multi::array_ref<const AType, 2> X_(raw_pointer_cast(X.origin()), {X.size(0), 1});
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
    long ik0, ikN;
    std::tie(ik0, ikN) = FairDivideBoundary(long(TG.TG_local().rank()), long(Likn.size(0)), long(TG.TG_local().size()));
    // setup buffer space if changing precision in X or v
    size_t vmem(0), Xmem(0);
    if (not std::is_same<XType, SPComplexType>::value)
      Xmem = X.num_elements();
    if (not std::is_same<vType, SPComplexType>::value)
      vmem = v.num_elements();
    ShmArray<SPComplexType, 1> buff(iextensions<1u>{vmem + Xmem},
                shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    sp_pointer vptr(nullptr);
    const_sp_pointer Xptr(nullptr);
    // setup origin of Xsp and copy_n_cast if necessary
    if (std::is_same<XType, SPComplexType>::value)
    {
      Xptr = reinterpret_cast<const_sp_pointer>(raw_pointer_cast(X.origin()));
    }
    else
    {
      long i0, iN;
      std::tie(i0, iN) =
          FairDivideBoundary(long(TG.TG_local().rank()), long(X.num_elements()), long(TG.TG_local().size()));
      copy_n_cast(raw_pointer_cast(X.origin()) + i0, iN - i0, raw_pointer_cast(buff.origin()) + i0);
      Xptr = raw_pointer_cast(buff.origin());
    }
    // setup origin of vsp and copy_n_cast if necessary
    if (std::is_same<vType, SPComplexType>::value)
    {
      vptr = reinterpret_cast<sp_pointer>(raw_pointer_cast(v.origin()));
    }
    else
    {
      long i0, iN;
      std::tie(i0, iN) =
          FairDivideBoundary(long(TG.TG_local().rank()), long(v.num_elements()), long(TG.TG_local().size()));
      vptr = raw_pointer_cast(buff.origin()) + Xmem;
      if (std::abs(c) > 1e-12)
        copy_n_cast(raw_pointer_cast(v.origin()) + i0, iN - i0, vptr + i0);
    }
    // setup array references
    multi::array_cref<SPComplexType const, 2> Xsp(Xptr, X.extensions());
    multi::array_ref<SPComplexType, 2> vsp(vptr, v.extensions());
    TG.TG_local().barrier();

    ma::product(SPRealType(a), Likn.sliced(ik0, ikN), Xsp, SPRealType(c), vsp.sliced(ik0, ikN));

    if (not std::is_same<vType, SPComplexType>::value)
    {
      copy_n_cast(raw_pointer_cast(vsp[ik0].origin()), vsp.size(1) * (ikN - ik0), raw_pointer_cast(v[ik0].origin()));
    }
    TG.TG_local().barrier();
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
    multi::array_ref<BType, 2> v_(raw_pointer_cast(v.origin()), {v.size(0), 1});
    if(haj.size(0)==1) { 
      multi::array_cref<AType, 2> G_(raw_pointer_cast(G.origin()), {1, G.size(0)});
      return vbias(G_, v_, dt, a, c, k);
    } else {
      multi::array_cref<AType, 2> G_(raw_pointer_cast(G.origin()), {G.size(0), 1});
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
    using GType = typename std::decay_t<typename MatA::element>;
    using vType = typename std::decay<MatB>::type::element;
    // scale a by sqrt(dt)
    a *= std::sqrt(dt);
    long ic0, icN;
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    // setup buffer space if changing precision in G or v
    size_t vmem(0), Gmem(0);
    if (not std::is_same<GType, SPComplexType>::value)
      Gmem = G.num_elements();
    if (not std::is_same<vType, SPComplexType>::value)
      vmem = v.num_elements();
    ShmArray<SPComplexType, 1> buff(iextensions<1u>{vmem + Gmem},
                shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    const_sp_pointer Gptr(nullptr);
    sp_pointer vptr(nullptr);
    // setup origin of Gsp and copy_n_cast if necessary
    if (std::is_same<GType, SPComplexType>::value)
    {
      Gptr = reinterpret_cast<const_sp_pointer>(raw_pointer_cast(G.origin()));
    }
    else
    {
      long i0, iN;
      std::tie(i0, iN) =
          FairDivideBoundary(long(TG.TG_local().rank()), long(G.size(0)), long(TG.TG_local().size()));
      Array_ref<SPComplexType, 2> Gsp(raw_pointer_cast(buff.origin()), G.extensions());
      if(iN>i0)
        ma::copy_n_cast(G.sliced(i0,iN),Gsp.sliced(i0,iN));
      Gptr = raw_pointer_cast(buff.origin());
    }
    // setup origin of vsp and copy_n_cast if necessary
    if (std::is_same<vType, SPComplexType>::value)
    {
      vptr = reinterpret_cast<sp_pointer>(raw_pointer_cast(v.origin()));
    }
    else
    {
      long i0, iN;
      std::tie(i0, iN) =
          FairDivideBoundary(long(TG.TG_local().rank()), long(v.num_elements()), long(TG.TG_local().size()));
      vptr = raw_pointer_cast(buff.origin()) + Gmem;
      if (std::abs(c) > 1e-12)
        copy_n_cast(raw_pointer_cast(v.origin()) + i0, iN - i0, vptr + i0);
    }
    // setup array references
    multi::array_cref<SPComplexType const, 2> Gsp(Gptr, G.extensions());
    multi::array_ref<SPComplexType, 2> vsp(vptr, v.extensions());
    TG.TG_local().barrier();

    if (haj.size(0) == 1)
    {
      RUNTIME_CHECK(Lakn.size(0) == G.size(1), "");
      RUNTIME_CHECK(Lakn.size(1) == v.size(0), "");
      RUNTIME_CHECK(G.size(0) == v.size(1), "");
      std::tie(ic0, icN) =
          FairDivideBoundary(long(TG.TG_local().rank()), long(Lakn.size(1)), long(TG.TG_local().size()));

      if (walker_type == CLOSED)
        a *= 2.0;
      ma::product(SPRealType(a), ma::T(Lakn(Lakn.extension(0), {ic0, icN})), ma::T(Gsp), SPRealType(c),
                  vsp.sliced(ic0, icN));
    }
    else
    {
      // multideterminant is not half-rotated, so use Likn
      if (walker_type == NONCOLLINEAR)
        APP_ABORT("Known bug in Real3IndexFactorization::vbias for noncollinear walkers with multiple determinants. Please contact developers.");
      
      // Likn has only spatial degrees of freedom - for noncollinear, we need to handle each polarization separately
      RUNTIME_CHECK(G.size(0) / Likn.size(0) == npol*npol, "");
      RUNTIME_CHECK(G.size(0) % Likn.size(0)  == 0, "");
      RUNTIME_CHECK(Likn.size(1) == v.size(0), "");
      RUNTIME_CHECK(G.size(1) == v.size(1), "");
      std::tie(ic0, icN) =
          FairDivideBoundary(long(TG.TG_local().rank()), long(Likn.size(1)), long(TG.TG_local().size()));

      if (walker_type == CLOSED)
        a *= 2.0;
      // BUG: For noncollinear, we need to sum over the polarizations - need to make a view of Gsp for
      // for each polarization
      ma::product(SPRealType(a), ma::T(Likn(Likn.extension(0), {ic0, icN})), Gsp, SPRealType(c),
                  vsp.sliced(ic0, icN));
    }
    // copy data back if changing precision
    if (not std::is_same<vType, SPComplexType>::value)
    {
      copy_n_cast(raw_pointer_cast(vsp[ic0].origin()), vsp.size(1) * (icN - ic0), raw_pointer_cast(v[ic0].origin()));
    }
    TG.TG_local().barrier();
  }

  template<class Mat, class MatB>
  void generalizedFockMatrix([[maybe_unused]] Mat&& G, [[maybe_unused]] MatB&& Fp, [[maybe_unused]] MatB&& Fm)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
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
  bool spin_dependent_vHS() const { return false; }

  multi::array<ComplexType, 2> getHSPotentials() { return multi::array<ComplexType, 2>{}; }

private:
  afqmc::TaskGroup_& TG;

  WALKER_TYPES walker_type;

  HostBufferManager device_buffer_manager;
  LocalTGBufferManager shm_buffer_manager;

  int global_origin;
  int global_nCV;
  int local_nCV;

  ComplexType E0;

  // bare one body hamiltonian
  shmVMatrix hij;

  // (potentially half rotated) one body hamiltonian
  shmCMatrix haj;

  //Cholesky Tensor Lik[i][k][n]
  shmSpRMatrix Likn;

  // permuted half-tranformed Cholesky tensor
  // Lank[ 2*idet + ispin ]
  std::vector<shmSpC3Tensor> Lank;

  // half-tranformed Cholesky tensor
  // only used in single determinant case, haj.size(0)==1.
  shmSpCMatrix Lakn;

  // one-body piece of Hamiltonian factorization
  shmCMatrix vn0;

  // used in PH reference/excited energy evaluation
  // Twian = sum_k G_ref[w][i][k] L[a][n][k]
  shmSpCVector Twian_ph;
  // Swia = sum_k G_ref[w][i][k] h[a][k]
  shmCVector Swia_ph;

  // Accumulates 1-Body and EXX energy components associated with spin ispin.
  // Accumulates KE vectors if requested. DOES NOT accumulate EJ.  
  // Gc does not need to be contiguous in memory!
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
      RUNTIME_CHECK(2 * nd + 1 < Lank.size(), "");
    else
      RUNTIME_CHECK(nd < Lank.size(), "");

    int nwalk = Gc.size(0);
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int NMO   = hij.size(1) / npol;
    int nel[2];
    nel[0] = Lank[nspin * nd].size(0);
    nel[1] = ((nspin == 2) ? Lank[nspin * nd + 1].size(0) : 0);
    RUNTIME_CHECK(Lank[nspin * nd].size(1) == local_nCV, "");
    RUNTIME_CHECK(Lank[nspin * nd].size(2) == npol*NMO, "");
    if (nspin == 2)
    {
      RUNTIME_CHECK(Lank[nspin * nd + 1].size(1) == local_nCV, "");
      RUNTIME_CHECK(Lank[nspin * nd + 1].size(2) == NMO, "");
    }
    RUNTIME_CHECK(Gc.num_elements() == nwalk * nel[ispin] * NMO * npol, "");

    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT("Error in Real3IndexFactorization::energy(...). Incorrect matrix dimensions ");
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

      ShmArray<SPComplexType, 2> GF({nwalk * nel[ispin], NMO*npol},
                shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      for (int n = 0; n < nwalk; ++n)
      {
        if (n % TG.TG_local().size() != TG.TG_local().rank())
          continue;
        copy_n_cast(raw_pointer_cast(Gc[n].origin()), nel[ispin]*NMO*npol, 
                    raw_pointer_cast(GF.origin()) + n * nel[ispin] * NMO*npol);
      }
      TG.TG_local().barrier();

      //SpCMatrix_ref Lan(raw_pointer_cast(Lank[nd * nspin + ispin].origin()), {nel[ispin] * local_nCV, NMO});
      auto Lan = Lank[nd * nspin + ispin].flatted();
      RUNTIME_CHECK(Lan.size(0) == nel[ispin] * local_nCV, "");  
      RUNTIME_CHECK(Lan.size(1) == npol*NMO, "");
      ShmArray<SPComplexType, 2> Twban({nwalk*nel[ispin], nel[ispin]*local_nCV},
                shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      SpC4Tensor_ref T4Dwban(raw_pointer_cast(Twban.origin()), 
                {nwalk, nel[ispin], nel[ispin], local_nCV});

      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(TG.TG_local().rank()), 
                                long(nel[ispin] * local_nCV), long(TG.TG_local().size()));
      ma::product(GF, ma::T(Lan.sliced(i0, iN)), Twban(Twban.extension(0), {i0, iN}));
      TG.TG_local().barrier();

      for (int n = 0, an = 0; n < nwalk; ++n)
      {
        ComplexType E_(0.0);
        for (int a = 0; a < nel[ispin]; ++a, an++)
        {
          if (an % TG.TG_local().size() != TG.TG_local().rank())
            continue;
          for (int b = 0; b < nel[ispin]; ++b)
            E_ += static_cast<ComplexType>(ma::dot(T4Dwban[n][a][b], T4Dwban[n][b][a]));
        }
        E[n][1] -= 0.5 * scl * E_;
      }

      if (addEJ)
      {
        for (int n = 0; n < nwalk; ++n)
        {
          if (n % TG.TG_local().size() != TG.TG_local().rank())
            continue;
          for (int a = 0; a < nel[ispin]; ++a)
            ma::axpy(SPComplexType(1.0), T4Dwban[n][a][a], Kl[n]);
        }
      }

    }
    if(addEXX and not addEJ)
      APP_ABORT(" Error: addEXX and not addEJ not yet implemented. \n\n");

    TG.TG_local().barrier();
  }

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
    nact[0] = Lank[0].size(0);
    nact[1] = ((walker_type == COLLINEAR) ? Lank[1].size(0) : 0);
    nel = Gc.size(1)/NMO;
    RUNTIME_CHECK(Lank[0].size(1) == local_nCV, "");
    RUNTIME_CHECK(Lank[0].size(2) == NMO, "");
    if (walker_type == COLLINEAR)
    {
      RUNTIME_CHECK(Lank[1].size(1) == local_nCV, "");
      RUNTIME_CHECK(Lank[1].size(2) == NMO, "");
    }
    RUNTIME_CHECK(Gc.num_elements() == nwalk * nel * NMO, "");

    if (E.size(0) != nwalk || E.size(1) < 3)
      APP_ABORT("Error in Real3IndexFactorization::energy(...). Incorrect matrix dimensions ");
    RUNTIME_CHECK(Kl.size(0) == nwalk && Kl.size(1) == local_nCV, "");

    SPRealType scl = (walker_type == CLOSED ? 2.0 : 1.0);

    // one-body contribution
    // haj[ndet][nocc*nmo]
    // not parallelized for now, since it would require customization of Wfn
    if (addH1)
    {
      if(Swia_ph.size(0) < nwalk * nel * std::max(nact[0],nact[1]))
        Swia_ph = shmCVector(iextensions<1u>{nwalk * nel * std::max(nact[0],nact[1])},
                           shared_allocator<ComplexType>{TG.TG_local()}); 
      Array_ref<ComplexType, 3> Swia(raw_pointer_cast(Swia_ph.origin()),
                                        {nwalk, nel, nact[ispin]});
      auto Gwik= Gc.rotated().partitioned(nel).unrotated();
      auto h2D=haj[0].partitioned(nact[0]+nact[1]);
      RUNTIME_CHECK(h2D.size(1) == NMO, "");
      if(ispin == 0) {
        ma::productStridedBatched(Gwik,ma::T(h2D.sliced(0,nact[0])),Swia);
      } else if(ispin==1) {
        ma::productStridedBatched(Gwik,ma::T(h2D.sliced(nact[0],nact[0]+nact[1])),Swia);
      }  
      //KYLE: TODO: Scale by `scl`
      for(int iw=0; iw<nwalk; iw++)
        for(int i=0; i<nel; i++)
          E[iw][0] += Swia[iw][i][i];
          //E[iw][0] += Swia[iw][i][refc[i]];
    }


    ShmArray<SPComplexType, 2> GF({nwalk * nel, NMO}, 
              shm_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    for (int n = 0; n < nwalk; ++n)
    {
      if (n % TG.TG_local().size() != TG.TG_local().rank())
        continue;
      copy_n_cast(raw_pointer_cast(Gc[n].origin()), nel*NMO, 
                  raw_pointer_cast(GF.origin()) + n * nel * NMO);
    }
    TG.TG_local().barrier();

    auto Lan= Lank[ispin].flatted();
    RUNTIME_CHECK(Lan.size(0) == nact[ispin] * local_nCV, "");  
    RUNTIME_CHECK(Lan.size(1) == NMO, "");  
    if(Twian_ph.size(0) < nwalk * nel * local_nCV * std::max(nact[0],nact[1]))
      Twian_ph = shmSpCVector(iextensions<1u>{nwalk * nel * local_nCV * std::max(nact[0],nact[1])},
                           shared_allocator<SPComplexType>{TG.TG_local()}); 
    Matrix_ref<SPComplexType> Twban(raw_pointer_cast(Twian_ph.origin()), 
                                        {nwalk*nel, nact[ispin]*local_nCV});
    Array_ref<SPComplexType, 4> T4Dwban(raw_pointer_cast(Twban.origin()), 
              {nwalk, nel, nact[ispin], local_nCV});

    long i0, iN;
    std::tie(i0, iN) = FairDivideBoundary(long(TG.TG_local().rank()), 
                                long(nact[ispin] * local_nCV), long(TG.TG_local().size()));
    ma::product(GF, ma::T(Lan.sliced(i0, iN)), Twban(Twban.extension(0), {i0, iN}));
    TG.TG_local().barrier();

    for (int n = 0, an = 0; n < nwalk; ++n)
    {
      ComplexType E_(0.0);
      for (int a = 0; a < nel; ++a, an++)
      {
        if (an % TG.TG_local().size() != TG.TG_local().rank())
          continue;
        for (int b = 0; b < nel; ++b)
          E_ += static_cast<ComplexType>(ma::dot(T4Dwban[n][a][b], T4Dwban[n][b][a]));
      }
      E[n][1] -= 0.5 * scl * E_;
    }

    for (int n = 0; n < nwalk; ++n)
    {
      if (n % TG.TG_local().size() != TG.TG_local().rank())
        continue;
      for (int a = 0; a < nel; ++a)
        ma::axpy(SPComplexType(1.0), T4Dwban[n][a][a], Kl[n]);
    }

    TG.TG_local().barrier();
  }

  template<class Iptr, class Mat, class MatB, class MatC, class MatW,
           typename = typename std::enable_if_t<(std::decay<Mat>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatW>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 4)>
          >
  void ph_excited_energy_impl([[maybe_unused]] SpinTypes spin_component, // KE: why is this not used?? 
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
      Array_ref<ComplexType, 3> Swia(raw_pointer_cast(Swia_ph.origin()), {nwalk, nelec, nact});
      Array_ref<int const, 3> iex(raw_pointer_cast(iexcit), {ndet, 2, nex});
      for (int iw = 0; iw < nwalk; iw++) {
        for (int d = 0; d < ndet; d++) { 
          ComplexType e_(0.0);
          for (int p = 0; p < nex; p++) { 
            int ip = iex[d][0][p];
            int ap = iex[d][1][p];
            auto S_=Swia[iw][ip];
            auto R_=R[iw][d][p];
            for(int a=0; a<nact; a++)
              e_ += S_[a]*static_cast<ComplexType>(R_[a]);
            e_ += Swia[iw][ip][ap] - Swia[iw][ip][refc[ip]];
          }
          E[iw][0] += wgt[d][iw] * e_;
        }
      }
    }

    /* Two body terms */
    // Make sure Twian_ph has appropriate dimensions
    if(Twian_ph.size(0) < nwalk * nelec * local_nCV * nact)
      APP_ABORT("Error in ph_excited_energy: Unexpected size in Twian_ph.");
    Array_ref<SPComplexType, 4> Twian(raw_pointer_cast(Twian_ph.origin()),
              {nwalk, nelec, nact, local_nCV});

    using ma::ph_excited_2body_energy_dense_cholesky_Tpan;
    RUNTIME_CHECK(Kl.stride(0) == Kl.size(1)*Kl.size(2), "");
    RUNTIME_CHECK(Kl.stride(1) == Kl.size(2), "");           
    RUNTIME_CHECK(Kl.stride(2) == 1, ""); 
    RUNTIME_CHECK(Kl.size(0) == ndet, "");
    RUNTIME_CHECK(Kl.size(1) == nwalk, "");
    RUNTIME_CHECK(Kl.size(2) == local_nCV, "");
    ph_excited_2body_energy_dense_cholesky_Tpan(iexcit, refc, Twian, R, wgt, 
      E.rotated()[1], E.rotated()[2], Kl);
  }

};

} // namespace afqmc

} // namespace sfqmc

#endif
