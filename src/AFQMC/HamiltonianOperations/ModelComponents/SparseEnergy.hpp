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

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_SPARSEENERGY_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_SPARSEENERGY_HPP

#include <vector>
#include <type_traits>

//#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "mpi3/shared_communicator.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "Numerics/ma_operations.hpp"
#include "Numerics/batched_operations.hpp"

#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/taskgroup.h"

namespace sfqmc
{
namespace afqmc
{
template<bool MP, bool REAL>
class SparseEnergy
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;

  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  using SPValueType   = typename to_working_precision<MP,ValueType  >::type;

  using device_alloc_type  = DeviceBufferManager::template allocator_t<SPComplexType>;
  using StaticMatrix  = boost::multi::static_array<SPComplexType, 2, device_alloc_type>;

public:

  template<class csrM1, class csrM2, class csrM3, class IVec>
// requires: {Psi(std::move(psi_)) is valid}, {hij(std::move(hij_)) is valid}, ... 
  SparseEnergy(afqmc::TaskGroup_& tg_,
                          WALKER_TYPES type,
                          bool full_g_,
                          std::vector<csrM1>&& hij_,
                          std::vector<csrM2>&& vj_,
                          std::vector<csrM2>&& vx_,
                          csrM3&& u_,
                          csrM3&& j_,
                          IVec&& n2ij_,
                          ComplexType e0 = 0.0
                )
      : TG(tg_),
        walker_type(type),
        fullG(full_g_),
        n2IJ_host(std::move(n2ij_)),
        n2IJ_dev(n2IJ_host),
        E0(e0),
        hij(std::move(move_vector<local_csr_Matrix<SPValueType>>(std::move(hij_)))),
        SpVJ(std::move(move_vector<local_csr_Matrix<SPValueType>>(std::move(vj_)))),
        SpVX(std::move(move_vector<local_csr_Matrix<SPValueType>>(std::move(vx_)))),
        U(std::move(u_)),
        J(std::move(j_))
  {
    // expect hij as a sparse Matrix with a single row (to reuse csr_matrix class) 
    if(type == COLLINEAR ) {
      RUNTIME_CHECK(SpVJ.size() == 3, "");
      RUNTIME_CHECK(SpVX.size() == 2, "");
      RUNTIME_CHECK(hij[0].size(0) == 1, "");
      RUNTIME_CHECK(hij[1].size(0) == 1, "");
      RUNTIME_CHECK(SpVJ[0].size(0) == n2IJ_host.size(), "");
      RUNTIME_CHECK(SpVJ[0].size(0) == SpVJ[0].size(1), ""); 
      RUNTIME_CHECK(SpVJ[1].size(0) == n2IJ_host.size(), "");
      RUNTIME_CHECK(SpVJ[1].size(0) == SpVJ[1].size(1), ""); 
      RUNTIME_CHECK(SpVJ[2].size(0) == n2IJ_host.size(), "");
      RUNTIME_CHECK(SpVJ[2].size(0) == SpVJ[2].size(1), ""); 
      RUNTIME_CHECK(SpVX[0].size(0) == n2IJ_host.size(), "");
      RUNTIME_CHECK(SpVX[0].size(0) == SpVX[0].size(1), "");
      RUNTIME_CHECK(SpVX[1].size(0) == n2IJ_host.size(), "");
      RUNTIME_CHECK(SpVX[1].size(0) == SpVX[1].size(1), "");
    } else if(type == NONCOLLINEAR ) {
      RUNTIME_CHECK(SpVJ.size() == 1, "");
      RUNTIME_CHECK(SpVX.size() == 1, "");
      RUNTIME_CHECK(hij[0].size(0) == 1, "");
      RUNTIME_CHECK(SpVJ[0].size(0) == SpVJ[0].size(1), ""); 
      RUNTIME_CHECK(SpVX[0].size(0) == SpVX[0].size(1), "");
      RUNTIME_CHECK(SpVJ[0].size(0) == n2IJ_host.size(), "");
      RUNTIME_CHECK(SpVX[0].size(0) == n2IJ_host.size(), "");
    }
    for(int n=0; n<SpVJ.size(); n++) 
      nnz_VJ[n]  = SpVJ[n].num_non_zero_elements();
    for(int n=0; n<SpVX.size(); n++) 
      nnz_VXX[n] = SpVX[n].num_non_zero_elements();
  }

  ~SparseEnergy() {}

  SparseEnergy(const SparseEnergy& other) = delete;
  SparseEnergy& operator=(const SparseEnergy& other) = delete;
  SparseEnergy(SparseEnergy&& other)                 = default;
  SparseEnergy& operator=(SparseEnergy&& other) = delete;

  bool expects_fullG() const { return fullG; }

  Vector<size_t> const* get_n2IJ() const { return std::addressof(n2IJ_host); };
  Vector<size_t, device_allocator<size_t>> const* 
                    get_n2IJ_dev() const { return std::addressof(n2IJ_dev); };

  template<class Mat>
  void addOneBodyPropagatorMatrix([[maybe_unused]] TaskGroup_& TG_, Mat&& H1, double dt)
  {
    using std::copy_n; 
    static_assert(std::decay_t<Mat>::dimensionality == 2,"Incorrect dimensions.");
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = H1.size(1) / npol; 

    RUNTIME_CHECK(H1.size(0) == nspin * npol * NMO, "");

    auto nnz0 = hij[0].num_non_zero_elements(0);
    auto nnz1 = hij[nspin-1].num_non_zero_elements(0);
    if(nnz0+nnz1 == 0) return;

    boost::multi::array_ref<ComplexType, 1> H1D( H1.origin(), 
                                                 {H1.num_elements()} ); 

    {
      // hij is in device memory, make a copy on host 
      Vector<int> indexes(iextensions<1u>{nnz0});
      Vector<SPValueType> values(iextensions<1u>{nnz0});
      copy_n( hij[0].non_zero_indices2_data(0), nnz0, indexes.origin() );
      copy_n( hij[0].non_zero_values_data(0), nnz0, values.origin() );
  
      if(fullG) {
        RUNTIME_CHECK(H1.num_elements() == nspin * hij[0].size(1), "");
        for( size_t iz=0; iz<size_t(nnz0); iz++)
          H1D[ indexes[iz] ] += ComplexType(dt)*static_cast<ComplexType>(values[iz]);
      } else { 
        for( size_t iz=0; iz<size_t(nnz0); iz++)
          H1D[ n2IJ_host[indexes[iz]] ] += ComplexType(dt)*static_cast<ComplexType>(values[iz]);
      }
    }

    // beta component
    if( walker_type == COLLINEAR ) {
      Vector<int> indexes(iextensions<1u>{nnz1});
      Vector<SPValueType> values(iextensions<1u>{nnz1});
      copy_n( hij[1].non_zero_indices2_data(0), nnz1, indexes.origin() );
      copy_n( hij[1].non_zero_values_data(0), nnz1, values.origin() );

      if(fullG) {
        RUNTIME_CHECK(H1.num_elements() == nspin * hij[1].size(1), "");
        for( size_t iz=0; iz<size_t(nnz1); iz++)
          H1D[ indexes[iz] + NMO*NMO ] += ComplexType(dt)*static_cast<ComplexType>(values[iz]);
      } else {
        for( size_t iz=0; iz<size_t(nnz1); iz++)
          H1D[ n2IJ_host[indexes[iz]] + NMO*NMO ] += ComplexType(dt)*static_cast<ComplexType>(values[iz]);
      }   
    }
  }

  template<class Mat, class MatB, class MatC>
  void accumulate_energy(int ispin,
              Mat&& E,
              MatB const& G,
              MatC&& EJn,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    using EJType = typename std::decay_t<MatC>::element_type;
    static_assert(std::is_same_v<SPComplexType,EJType>, "Wrong type.");
    if(fullG)
      energy_fullG(ispin,E,G,std::forward<MatC>(EJn),addE1,addEJ,addEXX);
    else
      energy_spG(ispin,E,G,std::forward<MatC>(EJn),addE1,addEJ,addEXX);      
  }

  template<class Mat, class MatB>
  void generalizedFockMatrix([[maybe_unused]] Mat&& G, [[maybe_unused]] MatB&& Fp, [[maybe_unused]] MatB&& Fm)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

private:

  afqmc::TaskGroup_& TG;

  WALKER_TYPES walker_type;

  /* Whether we expect a full Greens function or not. */
  bool fullG;

  /* If fullG=false, this defines the compact ordering */
  Vector<size_t> n2IJ_host;  
  Vector<size_t, device_allocator<size_t>> n2IJ_dev;  

  ComplexType E0;

  int nnz_VJ[3];
  int nnz_VXX[2];

  // bare one body hamiltonian
  // [0]:(alpha,alpha), [1]:(beta,beta)
  std::vector<local_csr_Matrix<SPValueType>> hij;

  // coulomb 2-e matrix elements 
  // [0]:(alpha,alpha), [1]:(beta,beta), [2]:[ (beta,alpha) + T( (alpha,beta) ) ]
  std::vector<local_csr_Matrix<SPValueType>> SpVJ;

  // exchange 2-e matrix elements 
  // [0]:(alpha,alpha), [1]:(beta,beta)
  std::vector<local_csr_Matrix<SPValueType>> SpVX;

  // combined U matrices 
  local_csr_Matrix<SPValueType> U;

  // combined J matrices 
  local_csr_Matrix<SPValueType> J;

  template<class Mat, class MatB, class MatC>
  void energy_spG(int ispin,
	      Mat&& E, 
	      MatB const& G,
              MatC&& EJn,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    static_assert(std::decay_t<Mat>::dimensionality == 2,"Incorrect dimensions.");
    static_assert(std::decay_t<MatB>::dimensionality == 2,"Incorrect dimensions.");
    static_assert(std::decay_t<MatC>::dimensionality == 2,"Incorrect dimensions.");
    int nwalk  = G.size(1);

    RUNTIME_CHECK(E.size(0) == nwalk, "");
    RUNTIME_CHECK(E.size(1) >= 3, "");

    if( addE1 ) {
      if(ispin == 0) ma::add_scalar(E0, E({0, nwalk},0));
      ma::spAi_Bij_yj(hij[ispin][0], G, E.rotated()[0].unrotated()); 		
    }

    if( (not addEJ) and
        (not addEXX or nnz_VXX[ispin] == 0) )
      return;

    // buffer allocators
    DeviceBufferManager device_buffer_manager{};
    int nIJ = n2IJ_dev.size();
    RUNTIME_CHECK(G.size(0) == nIJ, "");
    StaticMatrix VG( {nIJ, nwalk},
                     device_buffer_manager.get_generator().template get_allocator<SPComplexType>());

    if( addEJ ) { 
      if( nnz_VJ[ispin] > 0 ) {
        ma::product(SpVJ[ispin], G, VG);
        // y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} opA(A)[n,m] * opB(B)[n,m]  
        ma::dot('T','T', ComplexType(1.0), G, VG, 
    		         ComplexType(1.0), E.rotated()[2].unrotated());
      }	
      if(walker_type == COLLINEAR) {
        RUNTIME_CHECK(EJn.size(0) == nwalk, "");
        RUNTIME_CHECK(EJn.size(1) == nIJ, "");
        if(ispin==0) {
          if( nnz_VJ[2] > 0 ) {
            // SpVJ[2] = VJ(beta,alpha) + T( VJ(alpha,beta) )
            ma::product(SpVJ[2], G, VG);
	    ma::transpose(VG,EJn);
	  } else {
	    ma::fill(EJn, SPComplexType(0.0));
	  }
        } else {
	  ma::transpose(G,EJn);
	}
      }
    }

    if( addEXX and nnz_VXX[ispin] > 0 ) {
      ma::product(SpVX[ispin], G, VG); 
      // y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} opA(A)[n,m] * opB(B)[n,m]  
      ma::dot('T','T', ComplexType(1.0), G, VG, 
		       ComplexType(1.0), E.rotated()[1].unrotated());
    }
  }

  // use G (full GF) for E1, generate GIJ for EJ and/or EX
  template<class Mat, class MatB, class MatC>
  void energy_fullG(int ispin,
	      Mat&& E, 
	      MatB const& G, 
              MatC&& EJn,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    static_assert(std::decay_t<Mat>::dimensionality == 2,"Incorrect dimensions.");
    static_assert(std::decay_t<MatB>::dimensionality == 2,"Incorrect dimensions.");
    static_assert(std::decay_t<MatC>::dimensionality == 2,"Incorrect dimensions.");
    int nwalk  = G.size(1);
    
    RUNTIME_CHECK(E.size(0) == nwalk, "");
    RUNTIME_CHECK(E.size(1) >= 3, "");

    DeviceBufferManager device_buffer_manager{};

    if( addE1 ) {
      if(ispin==0) ma::add_scalar(E0, E({0, nwalk},0));
      ma::spAi_Bij_yj(hij[ispin][0], G, E.rotated()[0].unrotated()); 		
    }

    if( not addEJ and not addEXX ) return; 

    // buffer allocators
    int nIJ = n2IJ_dev.size();
    StaticMatrix GIJ( {nIJ, nwalk},
                     device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    StaticMatrix VG( {nIJ, nwalk},
                     device_buffer_manager.get_generator().template get_allocator<SPComplexType>());

    // GIJ[n][:] = G[ I[n] ][:]
    ma::copy_select(G, GIJ, n2IJ_dev, false);

    if( addEJ ) {
      if(nnz_VJ[ispin] > 0) {
        ma::product(SpVJ[ispin], GIJ, VG);
        // y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} opA(A)[n,m] * opB(B)[n,m]  
        ma::dot('T','T', ComplexType(1.0), GIJ, VG, 
			 ComplexType(1.0), E.rotated()[2].unrotated());
      }
      if(walker_type == COLLINEAR) {
        RUNTIME_CHECK(EJn.size(0) == nwalk, "");
        RUNTIME_CHECK(EJn.size(1) == nIJ, "");
        if(ispin==0) {
          // SpVJ[2] = VJ(beta,alpha) + T( VJ(alpha,beta) )
          if(nnz_VJ[2] > 0) {   	
            ma::product(SpVJ[2], GIJ, VG);
            ma::transpose(VG,EJn);
	  } else {
	    ma::fill(EJn, SPComplexType(0.0));
	  }
        } else {
          ma::transpose(GIJ,EJn);
        }
      }          
    } 
  
    if( addEXX and nnz_VXX[ispin] > 0 ) {
      ma::product(SpVX[ispin], GIJ, VG);
      // y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} opA(A)[n,m] * opB(B)[n,m]  
      ma::dot('T','T', ComplexType(1.0), GIJ, VG, 
		       ComplexType(1.0), E.rotated()[1].unrotated());
    } 

  }

};

} // namespace afqmc

} // namespace sfqmc

#endif
