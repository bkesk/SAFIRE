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

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_SPARSETENSOR_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_SPARSETENSOR_HPP

#include <vector>
#include <type_traits>

//#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "SparseMatrix/csr_matrix_construct.hpp"
#include "Numerics/csr_blas.hpp"
#include "Numerics/ma_operations.hpp"

#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/taskgroup.h"

#include "Numerics/batched_operations.hpp"
#include "Utilities/FairDivide.hpp"

namespace sfqmc
{
namespace afqmc
{
template<bool MP, class LikType, class V2XType, class LakType>
class SparseTensor
{
  using SpV2XType = typename to_working_precision<MP, V2XType>::type;   // Type of EX integrals
  using SpLikType = typename to_working_precision<MP, LikType>::type; // Type of Likn
  using SpLakType = typename to_working_precision<MP, LakType>::type; // Type of Lnak

  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;

  template<class T>
  using device_alloc_type = DeviceBufferManager::template allocator_t<T>;
  template<class T>
  using shm_alloc_type = LocalTGBufferManager::template allocator_t<T>;

  // pointers
  using pointer    = typename std::allocator_traits<device_alloc_type<ComplexType>>::pointer;
  using sp_pointer = typename std::allocator_traits<device_alloc_type<SPComplexType>>::pointer;

  using const_pointer    = typename std::allocator_traits<device_allocator<ComplexType>>::const_pointer;
  using const_sp_pointer = typename std::allocator_traits<device_allocator<SPComplexType>>::const_pointer;

  template<class U, int N>
  using DevArray = StaticArray_<N, device_alloc_type<U>>;
  template<class U, int N>
  using DevArray_ref = Array_ref_<N, typename std::allocator_traits<device_alloc_type<U>>::pointer>;
  template<class U, int N>
  using DevArray_cref = multi::array_cref<U, N, typename std::allocator_traits<device_allocator<U>>::const_pointer>;

  // arrays on shared work space
  // remember that this is device memory when built with accelerator support
  template<class U, int N>
  using ShmArray = multi::static_array<U, N, shm_alloc_type<U>>;

  // arrays on node allocator, for fixed arrays, e.g. Luv, Piu, ...
  // remember that this is device memory when built with accelerator support
  template<class U, int N>
  using nodeArray = Array_<N, node_allocator<U>>;

  using mpi3CMatrix = Matrix_<shared_allocator<ComplexType>>;

  // on GPU, Index type should be int for now. 
  // cuda's SPMM/SPMV can handle different index types with some algorithms, check later if it works
#if defined(ENABLE_DEVICE)
  using csrIndexType = int;
#else
  using csrIndexType = size_t;
#endif

public:
  static const HamiltonianTypes HamOpType = FactorizedSparse;
  HamiltonianTypes getHamType() const { return HamOpType; }

  template<class csrM1, class csrM2, class csrM3> 
  SparseTensor(afqmc::TaskGroup_& tg_,
               WALKER_TYPES type,
               mpi3CMatrix&& hij_,
               mpi3CMatrix&& h1,
               mpi3CMatrix&& vn0_,
               std::vector<csrM1>&& v2,
               csrM2&& vn,
               std::vector<csrM3>&& vnT,
               ComplexType e0_,
               int gncv,
	       bool use_transpose) :
        TG(tg_),
        device_buffer_manager(), 
	localTG_buffer_manager(),
        walker_type(type),
        global_nCV(gncv),
        E0(e0_),
	use_Lnik(use_transpose and h1.size(0) > 1),
        hij(std::move(hij_)),
        haj(std::move(h1)),
        vn0(std::move(vn0_)),
        Vakbl(move_vector<local_csr_Matrix<SpV2XType, csrIndexType>>(std::move(v2))),
	Lnik( use_Lnik ?  // only used if ndet>1 and use_transpose=true 
	      local_csr_Matrix<SpLikType, csrIndexType>(
		std::move(csr::shm::transpose<typename std::decay_t<csrM2>>(vn))) :
	      local_csr_Matrix<SpLikType, csrIndexType>{
			std::tuple<std::size_t, std::size_t>{0ul,0ul},
			std::tuple<std::size_t, std::size_t>{0ul,0ul},
		    	0,make_localTG_allocator<SpLikType>(TG)}	
	    ),  
        Likn(std::move(vn)),
        Lnak(move_vector<local_csr_Matrix<SpLakType, csrIndexType>>(std::move(vnT)))
#if !defined(ENABLE_DEVICE)
	,Likn_view(csr::shm::local_balanced_partition(Likn, TG))
	,Lnik_view( use_Lnik ?  
		   csr::shm::local_balanced_partition(Lnik, TG) :
		   Likn[std::array<size_t, 4>{0,1,0,Likn.size(1)}] 
		 )
#endif
  {
    // right now, npol is assumed to be 1, but this is how it should be with SO enabled!
    // if NONCOLLINEAR, then npol = 2, nspin = 1 and ndn=nelec[1]=0
    // hij   [ nspin*npol*NMO, npol*NMO ]
    // haj   [ ndet ][ (nup+ndn)*npol*NMO ]
    // vn0   [ NMO ][ NMO ]
    // Vakbl [ nspin*ndet ][ nelec[i]*npol*NMO ][ nelec[i]*npol*NMO ] 
    // Likn  [ npol*NMO*npol*NMO ][ nchol ]
    // Lnik  [ nchol ][ npol*NMO*npol*NMO ]
    // Lnak  [ nspin*ndet ][ nchol ][ nelec[i]*npol*NMO ]
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO = hij.size(1)/npol;
    int ndet = haj.size(0);
    int nchol = Lnak[0].size(0);
    RUNTIME_CHECK(Lnak.size() == nspin*ndet, "");
    size_t nel[2] = {Lnak[0].size(1)/NMO/npol, Lnak[nspin-1].size(1)/NMO/npol};
    if(walker_type != COLLINEAR) nel[1]=0;

    // hij   [ nspin*npol*NMO, npol*NMO ]
    RUNTIME_CHECK((hij.size(0) == nspin*npol*NMO) or
	    (hij.size(0) == npol*NMO), "");
    RUNTIME_CHECK(hij.size(1) == npol*NMO, "");
    // haj   [ ndet ][ (nup+ndn)*npol*NMO ]
    RUNTIME_CHECK(haj.size(0) == ndet, "");
    RUNTIME_CHECK(haj.size(1) == (nel[0]+nel[1])*npol*NMO, "");
    // vn0   [ NMO ][ NMO ]
    RUNTIME_CHECK(vn0.size(0) == NMO, "");
    RUNTIME_CHECK(vn0.size(1) == NMO, "");
    // Likn  [ npol*NMO*npol*NMO ][ nchol ]
    RUNTIME_CHECK(Likn.size(0) == npol*NMO*npol*NMO, "");
    RUNTIME_CHECK(Likn.size(1) == nchol, "");
    // Lnik  [ nchol ][ npol*NMO*npol*NMO ]
    if(use_Lnik) {
      RUNTIME_CHECK(Lnik.size(0) == nchol, "");
      RUNTIME_CHECK(Lnik.size(1) == npol*NMO*npol*NMO, "");
    }
    // Vakbl [ nspin*ndet ][ nelec[i]*npol*NMO ][ nelec[i]*npol*NMO ] 
    // Lnak  [ nspin*ndet ][ nchol ][ nelec[i]*npol*NMO ]
    RUNTIME_CHECK(Vakbl.size() == nspin*ndet, "");
    RUNTIME_CHECK(Lnak.size() == nspin*ndet, "");
    for(int d=0, ds=0; d<ndet; ++d)	
      for(int s=0; s<nspin; ++s, ++ds) {
        RUNTIME_CHECK(Vakbl[ds].size(0) == nel[s]*npol*NMO, "");
        RUNTIME_CHECK(Vakbl[ds].size(1) == nel[s]*npol*NMO, "");
        RUNTIME_CHECK(Lnak[ds].size(0) == nchol, "");
        RUNTIME_CHECK(Lnak[ds].size(1) == nel[s]*npol*NMO, "");
      }

#if !defined(ENABLE_DEVICE)
    Lnak_view.reserve(nspin*ndet);
    Vakbl_view.reserve(nspin*ndet);
    for(int d=0, ds=0; d<ndet; ++d, ds+=nspin) {	
      Lnak_view.emplace_back(csr::shm::local_balanced_partition(Lnak[ds], TG));	
      Vakbl_view.emplace_back(csr::shm::local_balanced_partition(Vakbl[ds], TG));	
      if(nspin == 2) {
	// keep partitioning over 'n' consistent between spins, to avoid extra sync!
	size_t n0 = Lnak_view[ds].local_origin()[0]; 
	size_t n1 = n0 + Lnak_view[ds].size(0);
        Lnak_view.emplace_back( Lnak[ds+1][std::array<size_t, 4>{n0,n1,0,Lnak[ds+1].size(1)}] );
        if(nel[0]==nel[1]) {
	  n0 = Vakbl_view[ds].local_origin()[0]; 
	  n1 = n0 + Vakbl_view[ds].size(0);
          Vakbl_view.emplace_back( Vakbl[ds+1][std::array<size_t, 4>{n0,n1,0,Vakbl[ds+1].size(1)}] );
        } else {
          Vakbl_view.emplace_back(csr::shm::local_balanced_partition(Vakbl[ds+1], TG));	
        }
      }	
    }
#endif

  }

  ~SparseTensor() {}

  SparseTensor(const SparseTensor& other) = delete;
  SparseTensor& operator=(const SparseTensor& other) = delete;
  SparseTensor(SparseTensor&& other)                 = default;
  SparseTensor& operator=(SparseTensor&& other) = default;

  Matrix<ComplexType> getOneBodyPropagatorMatrix(TaskGroup_& TG_, double dt, Vector<ComplexType> const& vMF)
  {
    if(walker_type == NONCOLLINEAR)
      APP_ABORT("Error: Noncollinear not yet implemented in SparseTensor.\n ");
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO = hij.size(1)/npol;

    RUNTIME_CHECK((hij.size(0) == npol*NMO) || (hij.size(0) == nspin*npol*NMO), "");
    if( hij.size(0) == 2*NMO )
      RUNTIME_CHECK(walker_type == COLLINEAR, "");
    int I0 = (hij.size(0) == 2*NMO) ? NMO : 0;

    ShmArray<ComplexType, 1> vMF_(vMF, localTG_buffer_manager.get_generator().template get_allocator<ComplexType>());
    ShmArray<ComplexType, 1> P1D(iextensions<1u>{NMO * NMO}, ComplexType(0),
                                 localTG_buffer_manager.get_generator().template get_allocator<ComplexType>());

    vHS(vMF_, P1D, dt);
    if (TG_.TG_Cores().size() > 1 && TG_.TG_local().root())
      TG_.TG_Cores().all_reduce_in_place_n(raw_pointer_cast(P1D.origin()), P1D.num_elements(), std::plus<>());
    TG_.TG().barrier();

    Matrix<ComplexType> H1({nspin * NMO, NMO});
    copy_n(P1D.origin(), NMO * NMO, H1.origin());

    if(walker_type == COLLINEAR)
      copy_n(P1D.origin(), NMO*NMO, H1[NMO].origin());

    // add hij + vn0 and symmetrize
    using ma::conj;

    for (int i = 0; i < NMO; i++)
    {
      H1[i][i] += dt * (hij[i][i] + vn0[i][i]);
      if(walker_type == COLLINEAR)
        H1[NMO+i][i] += dt * (hij[I0+i][i] + vn0[i][i]);
      for (int j = i + 1; j < NMO; j++)
      {
        H1[i][j] += dt * (hij[i][j] + vn0[i][j]);
        H1[j][i] += dt * (hij[j][i] + vn0[j][i]);
        // This is really cutoff dependent!!!
        if (std::abs(H1[i][j] - ma::conj(H1[j][i])) > 1e-6)
        {
          app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 is not hermitian. ");
          app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[i][j],H1[j][i]);
          app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[i][j],hij[j][i]);
        }
        H1[i][j] = 0.5 * (H1[i][j] + ma::conj(H1[j][i]));
        H1[j][i] = ma::conj(H1[i][j]);
        if(walker_type == COLLINEAR) {
          H1[NMO+i][j] += dt * (hij[I0+i][j] + vn0[i][j]);
          H1[NMO+j][i] += dt * (hij[I0+j][i] + vn0[j][i]);
          // This is really cutoff dependent!!!
          if (std::abs(H1[NMO+i][j] - ma::conj(H1[NMO+j][i])) > 1e-6)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (beta) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",i,j,H1[NMO+i][j],H1[NMO+j][i]);
            app_warning("             hij[I,J]:{}, hij[J,I]:{} ",hij[I0+i][j],hij[I0+j][i]);
          }
          H1[NMO+i][j] = 0.5 * (H1[NMO+i][j] + ma::conj(H1[NMO+j][i]));
          H1[NMO+j][i] = ma::conj(H1[NMO+i][j]);
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

  template<class Mat, class MatB,
           typename = typename std::enable_if_t<(std::decay_t<Mat>::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay_t<MatB>::dimensionality == 2)>
	  >
  void energy(Mat&& E,
              MatB const& Gc,
              int k,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nwalk = Gc.size(1);
    int NMO   = hij.size(1)/npol;

    RUNTIME_CHECK(E.size(0) == nwalk, "");
    RUNTIME_CHECK(E.size(1) >= 3, "");
    RUNTIME_CHECK(k >= 0 && k < haj.size(), "");
    RUNTIME_CHECK(nspin*k+nspin-1 < Vakbl.size(), "");
    RUNTIME_CHECK(nspin*k+nspin-1 < Lnak.size(), "");

    size_t nel[2] = {Lnak[nspin*k].size(1)/NMO/npol, 0}; 
    if(walker_type == COLLINEAR) nel[1]=Lnak[nspin*k+1].size(1)/NMO/npol;

    RUNTIME_CHECK(Gc.size(0) == (nel[0]+nel[1]) * npol * NMO, "");

    ma::fill(E({0,nwalk},{0,3}),ComplexType(0.0));
    // one-body contribution
    if (addH1)
    {
      ma::fill(E({0,nwalk},0),ComplexType(E0));
      ma::product(ComplexType(1.), ma::T(Gc), haj[k], ComplexType(1.), E(E.extension(0), 0));
    }
    if(not addEJ and not addEXX) return; 

    if constexpr (std::is_same_v<SPComplexType, 
		                 typename std::decay_t<typename MatB::element_type>
				>) {
      energy2B_impl(E,Gc,k,addEJ,addEXX);
    } else {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(TG.getLocalTGRank()), 
					    long(Gc.size(0)), long(TG.getNCoresPerTG()));
      ShmArray<SPComplexType, 2> Gsp(Gc.extensions(),
              localTG_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      if(iN>i0)
        ma::copy_n_cast(Gc.sliced(i0,iN),Gsp.sliced(i0,iN));
      TG.TG_local().barrier();
      energy2B_impl(E,Gsp,k,addEJ,addEXX);
    }
    TG.TG_local().barrier();
  }

  // This routine expects the GF of a single spin component (special only in COLLINEAR case)
  template<class Mat, class MatB, class MatC>
  void energy(SpinTypes spin_component,
              Mat&& E,
              MatB const& Gc,
              int k,
              MatC&& Kl,
              bool addH1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int ispin = (spin_component == Alpha ? 0 : 1);
    int nwalk = Gc.size(1);
    int NMO   = hij.size(1)/npol;

    RUNTIME_CHECK(E.size(0) == nwalk, "");
    RUNTIME_CHECK(E.size(1) >= 3, "");
    RUNTIME_CHECK(k >= 0 && k < haj.size(), "");
    RUNTIME_CHECK(nspin*k+ispin < Vakbl.size(), "");
    RUNTIME_CHECK(nspin*k+ispin < Lnak.size(), "");

    size_t nel[2] = {Lnak[nspin*k].size(1)/NMO/npol, 0}; 
    if(walker_type == COLLINEAR) nel[1]=Lnak[nspin*k+1].size(1)/NMO/npol;
    RUNTIME_CHECK(Gc.size(0) == Lnak[nspin*k+ispin].size(1), "");

    ma::fill(E({0,nwalk},{0,3}),ComplexType(0.0));
    // one-body contribution
    if (addH1)
    {
      ma::fill(E({0,nwalk},0),ComplexType(E0));
      if(ispin == 0) {
        ma::product(ComplexType(1.), ma::T(Gc), haj[k].sliced(0,nel[0]*npol*NMO),
                    ComplexType(1.), E(E.extension(0), 0));
      } else if(ispin==1) {
        ma::product(ComplexType(1.), ma::T(Gc), haj[k].sliced(nel[0]*npol*NMO,(nel[0]+nel[1])*npol*NMO),
                    ComplexType(1.), E(E.extension(0), 0));
      }
    }
    if(not addEJ and not addEXX) return; 

    if constexpr (std::is_same_v<SPComplexType, 
		                 typename std::decay_t<typename MatB::element_type>
				>) {
      energy2B_impl(ispin,E,Gc,Kl,k,addEJ,addEXX);
    } else {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(TG.getLocalTGRank()), 
					    long(Gc.size(0)), long(TG.getNCoresPerTG()));
      ShmArray<SPComplexType, 2> Gsp(Gc.extensions(),
              localTG_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      if(iN>i0)
        ma::copy_n_cast(Gc.sliced(i0,iN),Gsp.sliced(i0,iN));
      TG.TG_local().barrier();
      energy2B_impl(ispin,E,Gsp,Kl,k,addEJ,addEXX);
    }
    TG.TG_local().barrier();
  }

  template<class... Args>
  void fast_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: fast_energy not implemented in SparseTensor. ");
  }

  template<class... Args>
  void ph_reference_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_reference_energy not implemented yet. ");
  }

  template<class... Args>
  void ph_excited_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_excited_energy not implemented yet. ");
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay_t<MatA>::dimensionality <= 2)>,
           typename = typename std::enable_if_t<(std::decay_t<MatB>::dimensionality <= 2)>>
  void vHS(MatA&& X, MatB&& v, double dt, double a = 1., double c = 0.)
  {
    using AType = typename std::decay_t<typename std::decay_t<MatA>::element_type>;
    using BType = typename std::decay_t<MatB>::element_type;
    // check dimensionality at compile time
    if constexpr (std::decay_t<MatA>::dimensionality==1) {
      vHS(X.partitioned(X.size(0)),std::forward<MatB>(v),dt,a,c); 
    } else if constexpr (std::decay_t<MatB>::dimensionality==1) {
      vHS(std::forward<MatA>(X),v.partitioned(v.size(0)),dt,a,c); 
    } else if constexpr (not std::is_same_v<AType,SPComplexType>) {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(TG.getLocalTGRank()), 
					    long(X.size(0)), long(TG.getNCoresPerTG()));
      ShmArray<SPComplexType, 2> X_(X.extensions(),
                     localTG_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      if(iN>i0) 
        ma::copy_n_cast(X.sliced(i0,iN),X_.sliced(i0,iN));
      TG.TG_local().barrier();
      vHS(X_,std::forward<MatB>(v),dt,a,c);
    } else if constexpr (not std::is_same_v<BType,SPComplexType>) {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(TG.getLocalTGRank()), 
					    long(v.size(0)), long(TG.getNCoresPerTG()));
      ShmArray<SPComplexType, 2> v_(v.extensions(),
                     localTG_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      if(iN>i0 and std::abs(c) > 1e-12) 
        ma::copy_n_cast(v.sliced(i0,iN),v_.sliced(i0,iN));
      TG.TG_local().barrier();
      vHS_impl(std::forward<MatA>(X),v_,dt,a,c);
      if(iN>i0)
        ma::copy_n_cast(v_.sliced(i0,iN), v.sliced(i0,iN));
      TG.TG_local().barrier();
    } else {
      vHS_impl(X,v,dt,a,c); 
    } 
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay_t<MatA>::dimensionality <= 2)>,
           typename = typename std::enable_if_t<(std::decay_t<MatB>::dimensionality <= 2)>>
  void vbias(const MatA& G, MatB&& v, double dt, double a = 1., double c = 0., int k = 0)
  {
    using AType = typename std::decay_t<typename MatA::element_type>;
    using BType = typename std::decay_t<MatB>::element_type;
    // check dimensionality at compile time
    if constexpr (MatA::dimensionality==1) {
      vbias(G.partitioned(G.size(0)),std::forward<MatB>(v),dt,a,c,k); 
    } else if constexpr (std::decay_t<MatB>::dimensionality==1) {
      vbias(G,v.partitioned(v.size(0)),dt,a,c,k); 
    } else if constexpr (not std::is_same_v<AType,SPComplexType>) {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(TG.getLocalTGRank()), 
					    long(G.size(0)), long(TG.getNCoresPerTG()));
      ShmArray<SPComplexType, 2> G_(G.extensions(),
                     localTG_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      if(iN>i0) 
        ma::copy_n_cast(G.sliced(i0,iN),G_.sliced(i0,iN));
      TG.TG_local().barrier();
      vbias(G_,std::forward<MatB>(v),dt,a,c,k);
    } else if constexpr (not std::is_same_v<BType,SPComplexType>) {
      long i0, iN;
      std::tie(i0, iN) = FairDivideBoundary(long(TG.getLocalTGRank()), 
					    long(v.size(0)), long(TG.getNCoresPerTG()));
      ShmArray<SPComplexType, 2> v_(v.extensions(),
                     localTG_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      if(iN>i0 and std::abs(c) > 1e-12) 
        ma::copy_n_cast(v.sliced(i0,iN),v_.sliced(i0,iN));
      TG.TG_local().barrier();
      vbias_impl(G,v_,dt,a,c);
      if(iN>i0)
        ma::copy_n_cast(v_.sliced(i0,iN), v.sliced(i0,iN));
      TG.TG_local().barrier();
    } else {
      vbias_impl(G,v,dt,a,c); 
    } 
  }

  template<class Mat, class MatB>
  void generalizedFockMatrix([[maybe_unused]] Mat&& G, [[maybe_unused]] MatB&& Fp, [[maybe_unused]] MatB&& Fm)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

  bool distribution_over_cholesky_vectors() const { return true; }
  int number_of_ke_vectors() const { return Likn.size(1); }
  int local_number_of_cholesky_vectors() const { return Likn.size(1); }
  int global_number_of_cholesky_vectors() const { return global_nCV; }
  int global_origin_cholesky_vector() const { return Likn.global_origin()[1]; }

  // transpose=true means G[nwalk][ik], false means G[ik][nwalk]
  bool transposed_G_for_vbias() const { return false; }
  bool transposed_G_for_E() const { return false; }
  // transpose=true means vHS[nwalk][ik], false means vHS[ik][nwalk]
  bool transposed_vHS() const { return false; }

  bool fast_ph_energy() const { return false; }
  bool spin_dependent_vHS() const { return false; }

  boost::multi::array<ComplexType, 2> getHSPotentials()
  {
    boost::multi::array<ComplexType, 2> HSPot({Likn.size(1), Likn.size(0)});
    ma::Matrix2MA('T', Likn, HSPot);
    return HSPot;
  }

private:

  afqmc::TaskGroup_& TG;

  DeviceBufferManager device_buffer_manager;
  LocalTGBufferManager localTG_buffer_manager;

  WALKER_TYPES walker_type;

  int global_nCV;

  ComplexType E0;

  const bool use_Lnik;

  // bare one body hamiltonian
  mpi3CMatrix hij;

  // (potentially half rotated) one body hamiltonian
  nodeArray<ComplexType, 2> haj;

  // one-body piece of Hamiltonian factorization
  nodeArray<ComplexType, 2> vn0;

  // sparse 2-body 2-electron exchange integrals in matrix form
  // Vakbl.size() == nspins * ndet
  std::vector<local_csr_Matrix<SpV2XType, csrIndexType>> Vakbl;

  // Cholesky factorization of 2-electron integrals in sparse matrix form
  // Only used if haj.size() > 1 and use_Lnik
  // Declare ahead of Lnik, to be able to use constructor argument to build
  // before it is "moved" to Likn
  local_csr_Matrix<SpLikType, csrIndexType> Lnik;

  // Cholesky factorization of 2-electron integrals in sparse matrix form
  local_csr_Matrix<SpLikType, csrIndexType> Likn;

  // Cholesky factorization of 2-electron integrals in sparse matrix form
  std::vector<local_csr_Matrix<SpLakType, csrIndexType>> Lnak; 
  
#if !defined(ENABLE_DEVICE)
  // sparse sub-matrix views on CPU
  typename local_csr_Matrix<SpLikType, csrIndexType>::template matrix_view<int> Likn_view;
  typename local_csr_Matrix<SpLikType, csrIndexType>::template matrix_view<int> Lnik_view;
  std::vector<typename local_csr_Matrix<SpV2XType, csrIndexType>::template matrix_view<int>> Vakbl_view;
  std::vector<typename local_csr_Matrix<SpLakType, csrIndexType>::template matrix_view<int>> Lnak_view;
#endif

  template<class MatA, class MatB>
  void vbias_impl(MatA&& G, MatB&& v, double dt, double a, double c)
  {
    static_assert( std::decay_t<MatA>::dimensionality == 2, "Wrong dimensionality" );
    static_assert( std::decay_t<MatB>::dimensionality == 2, "Wrong dimensionality" );
    static_assert( std::is_same_v<typename std::decay_t<typename std::decay_t<MatA>::element_type>,
                                  SPComplexType>, "Wrong element_type" );
    static_assert( std::is_same_v<typename std::decay_t<MatB>::element_type,
                                  SPComplexType>, "Wrong element_type" );
    // scale a by sqrt(dt)
    a *= std::sqrt(dt);
    if (walker_type == CLOSED)
      a *= 2.0;
    RUNTIME_CHECK(v.size(1) == G.size(1), "");
    
#if defined(ENABLE_DEVICE)
    local_csr_Matrix<SpLikType, csrIndexType>& Likn_v(Likn);
    local_csr_Matrix<SpLikType, csrIndexType>& Lnik_v(Lnik);
    std::vector<local_csr_Matrix<SpLakType, csrIndexType>>& Lnak_v(Lnak);
#else
    typename local_csr_Matrix<SpLikType, csrIndexType>::template matrix_view<int>& Likn_v(Likn_view);
    typename local_csr_Matrix<SpLikType, csrIndexType>::template matrix_view<int>& Lnik_v(Lnik_view);
    std::vector<typename local_csr_Matrix<SpLakType, csrIndexType>::template matrix_view<int>>& Lnak_v(Lnak_view); 
#endif

    if( haj.size(0) == 1 ) {
      // single determinant calculation, can use Lnak!
      RUNTIME_CHECK(v.size(0) == Lnak[0].size(0), "");
      long n0 = long(Lnak_v[0].local_origin()[0]); 
      long n1 = n0 + long(Lnak_v[0].size(0));
      if (walker_type == COLLINEAR)
      {
        RUNTIME_CHECK(Lnak.size() == 2, "");
        RUNTIME_CHECK(Lnak_v.size() == 2, "");
        RUNTIME_CHECK(Lnak_v[0].local_origin()[0] == Lnak_v[1].local_origin()[0], "");
        RUNTIME_CHECK(Lnak_v[0].size(0) == Lnak_v[1].size(0), "");
        RUNTIME_CHECK(G.size(0) == Lnak[0].size(1) + Lnak[1].size(1), "");
        // views are constructed to have consistent partitioning in both spin channels
        ma::product(SpLakType(SPRealType(a)), Lnak_v[0], G.sliced(0, Lnak[0].size(1)), 
		    SpLakType(SPRealType(c)), v.sliced(n0,n1));
        if (std::abs(c) < 1e-8) c = 1.0;
        ma::product(SpLakType(SPRealType(a)), Lnak_v[1], G.sliced(Lnak[0].size(1), G.size(0)), 
		    SpLakType(SPRealType(c)), v.sliced(n0,n1));
      } else {
        RUNTIME_CHECK(Lnak.size() == 1, ""); 
        RUNTIME_CHECK(Lnak_v.size() == 1, "");
        RUNTIME_CHECK(G.size(0) == Lnak[0].size(1), "");  
        ma::product(SpLakType(SPRealType(a)), Lnak_v[0], G, SpLakType(SPRealType(c)), v.sliced(n0,n1));
      }
    } else {
      if( use_Lnik ) {
        RUNTIME_CHECK(G.size(0) == Lnik.size(1), "");  
        RUNTIME_CHECK(v.size(0) == Lnik.size(0), "");
        long n0 = long(Lnik_v.local_origin()[0]); 
        long n1 = n0 + long(Lnik_v.size(0));
        ma::product(SpLikType(SPRealType(a)), Lnik_v, G, SpLikType(SPRealType(c)), v.sliced(n0,n1));
      } else {
        RUNTIME_CHECK(G.size(0) == Likn.size(0), "");  
        RUNTIME_CHECK(v.size(0) == Likn.size(1), "");
	// would need a temporary storage for result and then sum_reduce into v
	// just disable for now
	if(TG.TG_local().size() > 1) 
	  APP_ABORT(" Error: ncores>1 not allowed with ndet>1 in SparseTensor.");
        ma::product(SpLikType(SPRealType(a)), ma::T(Likn_v), G, SpLikType(SPRealType(c)), v);
      }
    }
    TG.TG_local().barrier();
  }

  template<class MatA, class MatB>
  void vHS_impl(MatA&& X, MatB&& v, double dt, double a, double c)
  {
    static_assert( std::decay_t<MatA>::dimensionality == 2, "Wrong dimensionality" );
    static_assert( std::decay_t<MatB>::dimensionality == 2, "Wrong dimensionality" );
    static_assert( std::is_same_v<typename std::decay_t<typename std::decay_t<MatA>::element_type>,
				  SPComplexType>, "Wrong element_type" );
    static_assert( std::is_same_v<typename std::decay_t<MatB>::element_type,
				  SPComplexType>, "Wrong element_type" );
    // scale a by sqrt(dt)
    a *= std::sqrt(dt);
    RUNTIME_CHECK(Likn.size(1) == X.size(0), "");
    RUNTIME_CHECK(Likn.size(0) == v.size(0), "");
    RUNTIME_CHECK(X.size(1) == v.size(1), "");

#if defined(ENABLE_DEVICE)
    local_csr_Matrix<SpLikType, csrIndexType>& Likn_v(Likn);
#else
    typename local_csr_Matrix<SpLikType, csrIndexType>::template matrix_view<int>& Likn_v(Likn_view);
#endif

    long n0 = long(Likn_v.local_origin()[0]); 
    long n1 = n0 + long(Likn_v.size(0));
    ma::product(SpLikType(SPRealType(a)), Likn_v, X, SpLikType(SPRealType(c)), v.sliced(n0,n1));
    TG.TG_local().barrier();
  }

  template<class Mat, class MatB>
  void energy2B_impl(Mat&& E, MatB&& Gc, int k, bool addEJ, bool addEXX)
  { 
    static_assert( std::decay_t<Mat>::dimensionality == 2, "Wrong dimensionality" );
    static_assert( std::decay_t<MatB>::dimensionality == 2, "Wrong dimensionality" );
    static_assert( std::is_same_v<typename std::decay_t<typename std::decay_t<MatB>::element_type>,
				  SPComplexType>, "Wrong element_type" );

#if defined(ENABLE_DEVICE)
    std::vector<local_csr_Matrix<SpV2XType, csrIndexType>>& Vakbl_v(Vakbl);
    std::vector<local_csr_Matrix<SpLakType, csrIndexType>>& Lnak_v(Lnak);
#else
    std::vector<typename local_csr_Matrix<SpV2XType, csrIndexType>::template matrix_view<int>>& Vakbl_v(Vakbl_view);
    std::vector<typename local_csr_Matrix<SpLakType, csrIndexType>::template matrix_view<int>>& Lnak_v(Lnak_view);
#endif

    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nwalk = Gc.size(1);
    int NMO   = hij.size(1)/npol;
    size_t nel[2] = {Lnak[nspin*k].size(1)/NMO/npol, 0}; 
    if(walker_type == COLLINEAR) nel[1]=int(Lnak[nspin*k+1].size(1)/NMO/npol);

    // move factor of 2 in CLOSED EXX here!
    if (addEXX)
    {
      RealType scl = 0.5 * (walker_type == CLOSED ? 2.0 : 1.0);
      size_t ak = std::max( Vakbl_v[nspin*k].size(0), Vakbl_v[nspin*k+nspin-1].size(0) );
      DevArray<SPComplexType, 2> Vak({ak, nwalk},
                     device_buffer_manager.get_generator().template get_allocator<SPComplexType>());

      for(int is=0, is0=0; is<nspin; is++) {	
        ma::product(Vakbl_v[nspin*k+is], Gc.sliced(is0,is0+nel[is]*NMO), 
		    Vak.sliced(0,Vakbl_v[nspin*k+is].size(0)));
        // for n in [0,N), y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} op(A)[n,m] * op(B)[n,m]
	ma::dot('T','T',ComplexType(scl),Gc.sliced(is0,is0+nel[is]*NMO),
	        Vak.sliced(0,Vakbl_v[nspin*k+is].size(0)),ComplexType(1.0),
		E.rotated()[1]);
        is0 += nel[is]*NMO;
      }
    }

    if (addEJ)
    {
      RealType scl = 0.5 * (walker_type == CLOSED ? 4.0 : 1.0);
      DevArray<SPComplexType, 2> Vn({Lnak_v[nspin*k].size(0), nwalk}, SPComplexType(0.0),
                     device_buffer_manager.get_generator().template get_allocator<SPComplexType>());

      // Vn = sum_is Lnak[is]*Gak[is]
      for(int is=0, is0=0; is<nspin; is++) {	
        ma::product(SpLakType(1.0), Lnak_v[nspin*k+is], Gc.sliced(is0,is0+nel[is]*NMO),
		    SpLakType(1.0), Vn);
        is0 += nel[is]*NMO;
      }	
      // E[w] += scl * sum_n Vn[v][w] * Vn[n][w] 
      ma::dot('T', 'T', ComplexType(scl), Vn, Vn, ComplexType(1.0), E.rotated()[2]); 
    }
  }

  template<class Mat, class MatB, class MatC>
  void energy2B_impl(int ispin, Mat&& E, MatB&& Gc, MatC&& Kl, int k, bool addEJ, bool addEXX)
  { 
    static_assert( std::decay_t<Mat>::dimensionality == 2, "Wrong dimensionality" );
    static_assert( std::decay_t<MatB>::dimensionality == 2, "Wrong dimensionality" );
    static_assert( std::decay_t<MatC>::dimensionality == 2, "Wrong dimensionality" );
    static_assert( std::is_same_v<typename std::decay_t<typename std::decay_t<MatB>::element_type>,
				  SPComplexType>, "Wrong element_type" );
    static_assert( std::is_same_v<typename std::decay_t<typename std::decay_t<MatC>::element_type>,
				  SPComplexType>, "Wrong element_type" );

#if defined(ENABLE_DEVICE)
    std::vector<local_csr_Matrix<SpV2XType, csrIndexType>>& Vakbl_v(Vakbl);
    std::vector<local_csr_Matrix<SpLakType, csrIndexType>>& Lnak_v(Lnak);
#else
    std::vector<typename local_csr_Matrix<SpV2XType, csrIndexType>::template matrix_view<int>>& Vakbl_v(Vakbl_view);
    std::vector<typename local_csr_Matrix<SpLakType, csrIndexType>::template matrix_view<int>>& Lnak_v(Lnak_view);
#endif

    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    //int npol = (walker_type == NONCOLLINEAR ? 2 : 1);
    int nwalk = Gc.size(1);
    //int NMO   = hij.size(1)/npol;
    int nchol = Lnak[nspin*k].size(0); 
    //size_t nel[2] = {Lnak[nspin*k].size(1)/NMO/npol, 0}; 
    //if(walker_type == COLLINEAR) nel[1]=int(Lnak[nspin*k+1].size(1)/NMO/npol);

    // move factor of 2 in CLOSED EXX here!
    if (addEXX)
    {
      RealType scl = 0.5 * (walker_type == CLOSED ? 2.0 : 1.0);
      size_t ak = Vakbl_v[nspin*k+ispin].size(0);
      DevArray<SPComplexType, 2> Vak({ak, nwalk},
                     device_buffer_manager.get_generator().template get_allocator<SPComplexType>());

      ma::product(Vakbl_v[nspin*k+ispin], Gc, Vak); 
      // for n in [0,N), y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} op(A)[n,m] * op(B)[n,m]
      ma::dot('T','T',ComplexType(scl),Gc, Vak,ComplexType(1.0), E.rotated()[1]);
    }

    if (addEJ)
    {
      RealType scl = 0.5 * (walker_type == CLOSED ? 4.0 : 1.0);
      RUNTIME_CHECK(Kl.size(0) == nwalk, "");
      RUNTIME_CHECK(Kl.size(1) == nchol, "");
      int n0 = int(Lnak_v[nspin*k+ispin].local_origin()[0]);
      int nc = int(Lnak_v[nspin*k+ispin].size(0));
      DevArray<SPComplexType, 2> Vn({nc, nwalk}, SPComplexType(0.0),
                     device_buffer_manager.get_generator().template get_allocator<SPComplexType>());

      // Vn = sum_is Lnak[is]*Gak[is]
      ma::product(Lnak_v[nspin*k+ispin], Gc, Vn);
      // E[w] += scl * sum_n Vn[v][w] * Vn[n][w] 
      ma::dot('T', 'T', ComplexType(scl), Vn, Vn, ComplexType(1.0), E.rotated()[2]); 
      // Kl[w][n] = Vn[n][w]	
      ma::transpose(Vn,Kl(boost::multi::ALL, {n0,n0+nc}));
    }
    TG.TG_local().barrier();
  }
};

} // namespace afqmc

} // namespace sfqmc

#endif
