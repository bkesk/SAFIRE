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

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_MODELHAMOPS_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_MODELHAMOPS_HPP

#include <vector>
#include <type_traits>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "mpi3/shared_communicator.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "Numerics/ma_operations.hpp"
//#include "Memory/buffer_managers.h"

#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/taskgroup.h"

#include "AFQMC/HamiltonianOperations/ModelComponents/SparseEnergy.hpp"
#include "AFQMC/HamiltonianOperations/ModelComponents/ModelComponent.hpp"

namespace sfqmc
{
namespace afqmc
{

template<bool MP, bool REAL, class OrbitalMatrixType>
class ModelHamOps
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;

  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  using SPValueType   = typename to_working_precision<MP,ValueType  >::type;

  using device_alloc_type  = DeviceBufferManager::template allocator_t<SPComplexType>;  

  using pointer                 = typename std::allocator_traits<device_allocator<SPComplexType>>::pointer;
  using const_pointer           = typename std::allocator_traits<device_allocator<SPComplexType>>::const_pointer;

  using StaticVector  = StaticArray<SPComplexType, 1, device_alloc_type>;
  using StaticMatrix  = StaticArray<SPComplexType, 2, device_alloc_type>;
  using Static3Tensor = StaticArray<SPComplexType, 3, device_alloc_type>;
  using Static4Tensor = StaticArray<SPComplexType, 4, device_alloc_type>;

  using CMatrix_ref   = Array_ref<SPComplexType, 2, pointer>; 
  using CMatrix_cref   = Array_ref<SPComplexType const, 2, const_pointer>; 
  using C4Tensor_ref  = Array_ref<SPComplexType , 4, pointer>;
  using C4Tensor_cref  = Array_ref<SPComplexType const, 4, const_pointer>;

public:
  static const HamiltonianTypes HamOpType = ModelHamiltonian; 
  HamiltonianTypes getHamType() const { return ModelHamiltonian; }

  template<class MatO, class IVec>
// requires: {Psi(std::move(psi_)) is valid}, {hij(std::move(hij_)) is valid}, ... 
  ModelHamOps(afqmc::TaskGroup_& tg_,
                          WALKER_TYPES type,
                          std::vector<MatO>&& psi_,
                          SparseEnergy<MP,REAL>&& et_,
                          std::vector<ModelComponent<MP,REAL>>&& h_,
                          IVec&& n2ij_,
                          bool sparse_g_eval_ = true 
                )
      : TG(tg_),
        walker_type(type),
        PsiC(std::move(move_vector<OrbitalMatrixType>(std::move(psi_)))),
        ET(std::move(et_)),
        Hams(std::move(h_)),
        n2IJ(std::move(n2ij_)),
        n2IJ_dev(n2IJ),
        field_pos(iextensions<1u>{ Hams.size() + 1 }),
        ke_pos(iextensions<1u>{ Hams.size() + 1 }),
        sparse_G_eval(sparse_g_eval_)
  {
    static_assert(OrbitalMatrixType::dimensionality == 2,"Invalid array type.");
    TG.Node().barrier();
    // don't allow ncores>1 for now, since it is unlikely useful in this case.
    // reenable later if needed!
    if(TG.TG_local().size() > 1)
      APP_ABORT(" Error: ncores>1 not allowed with ModelHamOps. \n\n");

    num_ke_vectors=0;
    local_nCV=0;
    field_pos[0] = 0;
    ke_pos[0] = 0;
    int pos=0;
    for(auto const& v: Hams) {
      num_ke_vectors += v.number_of_ke_vectors();
      local_nCV += v.local_number_of_cholesky_vectors();
      field_pos[++pos] = local_nCV; 
      ke_pos[pos] = num_ke_vectors; 
    }

    nIJ_first_beta=-1;
    size_t M(PsiC[0].size(0));
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO   = PsiC[0].size(0) / npol;
    for(int n=0; n<n2IJ.size(); ++n) {
      if( n2IJ[n]/M >= NMO ) {
        nIJ_first_beta=n;
        break;
      }
    } 

    TG.Node().barrier();
  }

  ~ModelHamOps() {}

  ModelHamOps(const ModelHamOps& other) = delete;
  ModelHamOps& operator=(const ModelHamOps& other) = delete;
  ModelHamOps(ModelHamOps&& other)                 = default;
  ModelHamOps& operator=(ModelHamOps&& other) = delete;

  boost::multi::array<ComplexType, 2> getOneBodyPropagatorMatrix(TaskGroup_& TG_, double dt, 
                                     boost::multi::array<ComplexType, 1> const& vMF)
  {
    RUNTIME_CHECK(vMF.size(0) == local_nCV, ""); 
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC[0].size(0) / npol;

    boost::multi::array<ComplexType, 2> H1({nspin * npol * NMO, npol * NMO});
    std::fill_n(H1.origin(), H1.num_elements(), ComplexType(0.0));

    ET.addOneBodyPropagatorMatrix(TG_,H1,dt);

    for(int i=0; i<Hams.size(); i++) 
      Hams[i].addOneBodyPropagatorMatrix(TG_,H1,dt,
                            vMF.sliced(field_pos[i],field_pos[i+1]), n2IJ);

    // symmetrize
    for (int I = 0; I < npol * NMO; I++)
    {
      for (int J = I + 1; J < npol * NMO; J++)
      {
        // This is really cutoff dependent!!!
        if (std::abs(H1[I][J] - ma::conj(H1[J][I])) * 2.0 > 1e-5)
        {
          app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 is not hermitian. ");
          app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",I,J,H1[I][J],H1[J][I]);
        }
        H1[I][J] = 0.5 * (H1[I][J] + ma::conj(H1[J][I]));
        H1[J][I] = ma::conj(H1[I][J]);
      }
    }
    if (walker_type == COLLINEAR)
    {
      for (int I = 0; I < NMO; I++)
      {
        for (int J = I + 1; J < NMO; J++)
        {
        // This is really cutoff dependent!!!
          if (std::abs(H1[NMO+I][J] - ma::conj(H1[NMO+J][I])) * 2.0 > 1e-5)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 (beta) is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",I,J,H1[NMO+I][J],H1[NMO+J][I]);
          }
          H1[NMO+I][J] = 0.5 * (H1[NMO+I][J] + ma::conj(H1[NMO+J][I]));
          H1[NMO+J][I] = ma::conj(H1[NMO+I][J]);
        }
      }
    }

    return H1;
  }

  template<class TVec>
  void getFieldTypes(TVec&& v) {
    RUNTIME_CHECK(v.size() == local_nCV, "");
    for(int i=0; i<Hams.size(); i++) 
      Hams[i].getFieldTypes(v.sliced(field_pos[i],field_pos[i+1]));
  }
 
  template<class Vec, class Vec2>
  void update_potentials(double dt, Vec&& nMF, Vec2&& vMF, bool natural_shift)
  {
    if(IJ2n.size()==0) {
      IJ2n.reserve(n2IJ.size());
      for(int n=0; n<n2IJ.size(); n++)
        IJ2n.insert(std::make_pair(n2IJ[n], n));
    } 
    for(int i=0; i<Hams.size(); i++) 
      Hams[i].update(dt,std::forward<Vec>(nMF),n2IJ,IJ2n,
		     vMF.sliced(field_pos[i],field_pos[i+1]),natural_shift);
  }

  template<class Mat, class MatB>
  void energy(Mat&& E,
              MatB const& Gc,
              int nd,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC[0].size(0) / npol;
    int nwalk = Gc.size(0);
    for (int n = 0; n < nwalk; n++)
      std::fill_n(E[n].origin(), 3, ComplexType(0.));    
    RUNTIME_CHECK(PsiC.size() >= nspin*nd + (nspin-1), "");
    long nel[2] = {PsiC[ nspin * nd ].size(1), 0l};
    if( walker_type == COLLINEAR ) nel[1] = PsiC[2*nd+1].size(1); 

    bool fullG(ET.expects_fullG());
    // buffer allocators
    DeviceBufferManager device_buffer_manager{};
    // ET.get_n2IJ() runs over [0,M^2) in COLLINEAR case 
    int nIJ(ET.get_n2IJ()->size());
    bool allocate_EJn (addEJ and walker_type==COLLINEAR);
    Static3Tensor EJn({nspin, (allocate_EJn?nwalk:1) , (allocate_EJn?nIJ:1)}, 
                  device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    
    if(fullG) {

      // generate full G
      StaticMatrix Gfull({nspin * npol * NMO * npol * NMO, nwalk}, 
                  device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      getGFull(Gc,Gfull,PsiC[ nspin * nd ],PsiC[ nspin * nd + (nspin-1)]);
      for(int is=0, is0=0; is<nspin; is++ ) {
        ET.accumulate_energy(is, E, Gfull.sliced(is0,is0+npol*NMO*npol*NMO), 
			     EJn[is], addE1, addEJ, addEXX); 
 	is0 += npol*NMO*npol*NMO;
      }

    } else {

      StaticMatrix GIJ( {nIJ, nwalk}, SPComplexType(0.0),
        device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      auto ET_n2IJ= ET.get_n2IJ_dev();

      // generate GIJ with custom mapping
      if( sparse_G_eval ) {
	auto&& G3D= Gc.rotated().partitioned(nel[0]+nel[1]).unrotated();
        for(int is=0, is0=0; is<nspin; is++ ) {
          ma::getGIJ(G3D.rotated().sliced(is0,is0+nel[is]).unrotated(),GIJ,PsiC[ nspin * nd + is ], *ET_n2IJ);
          ET.accumulate_energy(is, E, GIJ, EJn[is], addE1, addEJ, addEXX); 
          is0 += nel[is]; 
        }
      } else {

        // generate full G
        StaticMatrix Gfull({nspin * npol * NMO * npol * NMO, nwalk},
              device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
        getGFull(Gc,Gfull,PsiC[ nspin * nd ],PsiC[ nspin * nd + (nspin-1) ]);

        for(int is=0, is0=0; is<nspin; is++ ) {
          // B[n][:] = A[ I[n] ][:] 
          ma::copy_select(Gfull.sliced(is0,is0+npol*NMO*npol*NMO), GIJ, *ET_n2IJ, false);
          ET.accumulate_energy(is, E, GIJ, EJn[is], addE1, addEJ, addEXX); 
	  is0 += npol*NMO*npol*NMO;
        }

      }

    }

    // opposite spin EJ contribution
    if(addEJ and walker_type == COLLINEAR)
      ma::dot('N','N', ComplexType(1.0), EJn[0], EJn[1],
                       ComplexType(1.0), E.rotated()[2].unrotated());

  }

  template<class Mat, class MatB, class MatC,
	typename = std::enable_if_t<std::is_same_v<SPComplexType,typename std::decay_t<MatC>::element_type>>>
  void energy(SpinTypes spin_component,
              Mat&& E,
              MatB const& Gc,
              int nd,
              MatC&& EJn,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int ispin = ( spin_component==Alpha ? 0 : 1 );
    int NMO   = PsiC[0].size(0) / npol;
    int nel = PsiC[ nspin * nd + ispin].size(1);
    int nwalk = Gc.size(0);
    int nIJ(ET.get_n2IJ()->size());
    for (int n = 0; n < nwalk; n++)
      std::fill_n(E[n].origin(), 3, ComplexType(0.));    
    RUNTIME_CHECK(PsiC.size() >= nspin*nd + (nspin-1), "");
    if(addEJ and walker_type==COLLINEAR) {
      RUNTIME_CHECK(EJn.size(0) == nwalk, "");
      RUNTIME_CHECK(EJn.size(1) == nIJ, "");
    }

    bool fullG(ET.expects_fullG());
    // buffer allocators
    DeviceBufferManager device_buffer_manager{};
    
    if(fullG) {

      // generate full G
      StaticMatrix Gfull({npol * NMO * npol * NMO, nwalk}, 
                  device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      getGFull(Gc,Gfull,PsiC[ nspin * nd + ispin]);

      ET.accumulate_energy(ispin, E, Gfull, EJn, addE1, addEJ, addEXX); 

    } else {

      StaticMatrix GIJ( {nIJ, nwalk}, SPComplexType(0.0),
        device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      auto ET_n2IJ = ET.get_n2IJ_dev();

      // generate GIJ with custom mapping
      if( sparse_G_eval ) {

	auto&& G3D = Gc.rotated().partitioned(nel).unrotated();
        ma::getGIJ(G3D,GIJ,PsiC[ nspin * nd + ispin], *ET_n2IJ);

      } else {

        // generate full G
        StaticMatrix Gfull({npol * NMO * npol * NMO, nwalk},
              device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
        getGFull(Gc,Gfull,PsiC[ nspin * nd + ispin]);

        // B[n][:] = A[ I[n] ][:] 
        ma::copy_select(Gfull, GIJ, *ET_n2IJ, false);

      }
      ET.accumulate_energy(ispin, E, GIJ, EJn, addE1, addEJ, addEXX); 

    }
  }

  template<class... Args>
  void fast_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: fast_energy not implemented in ModelHamOps. ");
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
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 1)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 1)>,
           typename = void>
  void vHS(MatA& X, MatB&& v, double dt, double a = 1., double c = 0.)
  {
    using XType = typename std::decay_t<typename MatA::element>;
    using vType = typename std::decay<MatB>::type::element;
    Matrix_ref<vType, decltype(v.origin())> v_(v.origin(), {v.size(0), 1});
    Matrix_ref<XType const, decltype(X.origin())> X_(X.origin(), {X.size(0), 1});
    vHS(X_, v_, dt, a, c);
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>
          >
  void vHS(MatA& X, MatB&& v, double dt, double a = 1., double c = 0.)
  {
    using XType = typename std::decay_t<typename MatA::element>;
    using vType = typename std::decay<MatB>::type::element;
    using vType_t = typename vType::value_type;
    RUNTIME_CHECK(X.size(1) == v.size(1), "");

    // buffer allocators
    DeviceBufferManager device_buffer_manager{};

    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC[0].size(0) / npol;
    int nwalk = X.size(1);
    int nIJ = n2IJ.size();

    // sanity checks!
    RUNTIME_CHECK(X.size(0) == local_nCV, "");
    RUNTIME_CHECK(v.size(0) == nspin * npol * npol * NMO * NMO, "");
    RUNTIME_CHECK(v.size(1) == nwalk, ""); 

    // before anything, apply "c" scalar
    if( std::abs(c) < 1e-8 )
      fill_n(v.origin(), v.num_elements(), vType(0.0));
    else
      ma::scal(vType_t(c), v);

    /*************************************************************/
    // usual trick to cast if needed only 
    pointer Xptr(nullptr);
    size_t Xmem(0);
    using sfqmc::afqmc::reinterpret_pointer_cast;
    if( not std::is_same<XType,SPComplexType>::value ) Xmem += X.num_elements();
    StaticVector Buff( iextensions<1u>{Xmem},
          device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    if( std::is_same<XType,SPComplexType>::value ) {
      Xptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(X.origin()));
    } else {
      Xptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(Buff.origin()));
      copy_n_cast(make_device_ptr(X.origin()), X.num_elements(), Xptr);
    }
    /*************************************************************/

    StaticMatrix localV( {nIJ, nwalk}, SPComplexType(0.0), 
          device_buffer_manager.get_generator().template get_allocator<SPComplexType>());

    /*   reference to correct memory    */
    CMatrix_ref Xsp( Xptr, X.extensions() );

    for(int i=0; i<Hams.size(); i++)
      Hams[i].vHS(Xsp.sliced(field_pos[i],field_pos[i+1]), localV, dt, a);

    // B[ I[n] ][:] += A[n][:] 
    ma::copy_select(SPComplexType(1.0), localV, SPComplexType(1.0), v, n2IJ_dev);
  }

  template<class MatA,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>
          >
  auto vHS_sparse(MatA& X, double dt, double a = 1.)
  {
    using XType = typename std::decay_t<typename MatA::element>;

    // buffer allocators
    DeviceBufferManager device_buffer_manager{};

    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC[0].size(0) / npol;
    int nwalk = X.size(1);
    int nIJ = n2IJ.size();

    // sanity checks!
    RUNTIME_CHECK(X.size(0) == local_nCV, "");

    if( spvHS.size() != nspin or
        spvHS[0].size(0) != nwalk*npol*NMO or spvHS[0].size(1) != nwalk*npol*NMO or 
        spvHS[nspin-1].size(0) != nwalk*npol*NMO or spvHS[nspin-1].size(1) != nwalk*npol*NMO ) 
        make_csr_vHS(nwalk);
    RUNTIME_CHECK(spvHS[0].size(0) == nwalk*npol*NMO and spvHS[0].size(1) == nwalk*npol*NMO, "");
    RUNTIME_CHECK(spvHS[nspin-1].size(0) == nwalk*npol*NMO and spvHS[nspin-1].size(1) == nwalk*npol*NMO, "");
    if(walker_type == COLLINEAR) {
      RUNTIME_CHECK(spvHS[0].capacity() == nwalk*nIJ_first_beta, "");
      RUNTIME_CHECK(spvHS[1].capacity() == nwalk*(nIJ-nIJ_first_beta), "");
    } else if(walker_type == NONCOLLINEAR) {
      RUNTIME_CHECK(spvHS[0].capacity() == nwalk*nIJ, "");
    }

    /*************************************************************/
    // usual trick to cast if needed only 
    pointer Xptr(nullptr);
    size_t Xmem(0);
    using sfqmc::afqmc::reinterpret_pointer_cast;
    if( not std::is_same<XType,SPComplexType>::value ) Xmem += X.num_elements();
    StaticVector Buff( iextensions<1u>{Xmem},
          device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    if( std::is_same<XType,SPComplexType>::value ) {
      Xptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(X.origin()));
    } else {
      Xptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(Buff.origin()));
      copy_n_cast(make_device_ptr(X.origin()), X.num_elements(), Xptr);
    }
    /*************************************************************/

    StaticMatrix localV( {nIJ, nwalk}, SPComplexType(0.0), 
          device_buffer_manager.get_generator().template get_allocator<SPComplexType>());

    /*   reference to correct memory    */
    CMatrix_ref Xsp( Xptr, X.extensions() );

    for(int i=0; i<Hams.size(); i++)
      Hams[i].vHS(Xsp.sliced(field_pos[i],field_pos[i+1]), localV, dt, a);

    if constexpr (MP) {
      using C_dev_alloc_type  = DeviceBufferManager::template allocator_t<ComplexType>;
      boost::multi::static_array<ComplexType, 2, C_dev_alloc_type> vbuff( {nIJ, nwalk},
            device_buffer_manager.get_generator().template get_allocator<ComplexType>());
      copy_n_cast(localV.origin(), localV.num_elements(), vbuff.origin());
      if(walker_type == COLLINEAR) {
        Matrix_ref<ComplexType, device_ptr<ComplexType>>
              vup( spvHS[0].non_zero_values_data(), {nwalk, nIJ_first_beta} );      
        Matrix_ref<ComplexType, device_ptr<ComplexType>>
              vdn( spvHS[1].non_zero_values_data(), {nwalk, nIJ-nIJ_first_beta} );
        ma::transpose( vbuff.sliced(0, nIJ_first_beta), vup);
        ma::transpose( vbuff.sliced(nIJ_first_beta, nIJ), vdn);
      } else {
        Matrix_ref<ComplexType, device_ptr<ComplexType>>
              v( spvHS[0].non_zero_values_data(), {nwalk, nIJ} );      
        ma::transpose( vbuff, v );
      }
    } else {
      if(walker_type == COLLINEAR) {
        Matrix_ref<ComplexType, device_ptr<ComplexType>>
              vup( spvHS[0].non_zero_values_data(), {nwalk, nIJ_first_beta} );
        Matrix_ref<ComplexType, device_ptr<ComplexType>>
              vdn( spvHS[1].non_zero_values_data(), {nwalk, nIJ-nIJ_first_beta} );
        ma::transpose( localV.sliced(0, nIJ_first_beta), vup);
        ma::transpose( localV.sliced(nIJ_first_beta, nIJ), vdn);
      } else {
        Matrix_ref<ComplexType, device_ptr<ComplexType>>
              v( spvHS[0].non_zero_values_data(), {nwalk, nIJ} );
        ma::transpose( localV, v );
      }
    }

    return std::tuple<dev_csr_Matrix<ComplexType> const*, dev_csr_Matrix<ComplexType> const*>{spvHS.data(),spvHS.data()+nspin-1}; 
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 1)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 1)>,
           typename = void>
  void vbias(const MatA& G, MatB&& v, double dt, double a = 1., double c = 0., int k = 0)
  {
    using BType = typename std::decay_t<MatB>::element;
    using AType = typename std::decay_t<MatA>::element;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int nci = PsiC.size() / nspin;
    if( nci == 1 ) {
      Matrix_ref<BType, decltype(v.origin())> v_(v.origin(), {v.size(0), 1});
      Matrix_ref<AType const, decltype(G.origin())> G_(G.origin(), {1, G.size(0)});
      vbias(G_, v_, dt, a, c, k);
    } else {
      Matrix_ref<BType, decltype(v.origin())> v_(v.origin(), {v.size(0), 1});
      Matrix_ref<AType const, decltype(G.origin())> G_(G.origin(), {G.size(0), 1});
      vbias(G_, v_, dt, a, c, k);
    }
  }

  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>>
  void vbias(const MatA& G, MatB&& v, double dt, double a = 1., double c = 0., [[maybe_unused]] int nd = 0)
  {
    // NOTE: nd is irrelevant. For multi-determinant calculations the GF used for
    //       vbias is not compact 
    //using GType = typename std::decay_t<typename MatA::element>;
    using vType = typename std::decay<MatB>::type::element;
    using vType_t = typename vType::value_type; // vType is generally std::complex

    // buffer allocators
    DeviceBufferManager device_buffer_manager{};

    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC[0].size(0) / npol;
    int nwalk = v.size(1);
    int nci = PsiC.size() / nspin; 
    int nIJ = n2IJ.size(); 

    RUNTIME_CHECK(v.size(0) == local_nCV, "");

    // before anything, apply "c" scalar
    if( std::abs(c) < 1e-8 ) 
      fill_n(v.origin(), v.num_elements(), vType(0.0));
    else 
      ma::scal(vType_t(c), v); 

    /*************************************************************/
    // usual trick to cast if needed only 
    pointer vptr(nullptr);
    size_t vmem(0);
    using sfqmc::afqmc::reinterpret_pointer_cast;
    if( not std::is_same<vType,SPComplexType>::value ) vmem += v.num_elements();
    StaticVector vbuff( iextensions<1u>{vmem},
                        device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    if( std::is_same<vType,SPComplexType>::value ) { 
      vptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(v.origin())); 
    } else {
      vptr = reinterpret_pointer_cast<SPComplexType>(make_device_ptr(vbuff.origin())); 
      copy_n_cast(make_device_ptr(v.origin()), v.num_elements(), vptr);
    }
    /*************************************************************/

    StaticMatrix GIJ( {nIJ, nwalk}, SPComplexType(0.0),
          device_buffer_manager.get_generator().template get_allocator<SPComplexType>());

    /*   reference to correct memory    */
    CMatrix_ref vsp( vptr, v.extensions() );

    if( nci == 1 ) {
      long nel[2] = {PsiC[0].size(1), 0l};
      if( walker_type == COLLINEAR ) nel[1] = PsiC[1].size(1);

      if( sparse_G_eval ) {

	auto&& G3D = G.rotated().partitioned(nel[0]+nel[1]).unrotated();
        if( walker_type == COLLINEAR )	
          ma::getGIJ(G3D,GIJ,PsiC[0],PsiC[1],n2IJ_dev);
        else
          ma::getGIJ(G3D,GIJ,PsiC[0],n2IJ_dev);

      } else {

        // generate full G
        StaticMatrix Gfull({nspin * npol * NMO * npol * NMO, nwalk}, 
              device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
        getGFull(G,Gfull,PsiC[ 0 ],PsiC[ nspin-1 ]);

        // B[n][:] = A[ I[n] ][:] 
        ma::copy_select(Gfull, GIJ, n2IJ_dev, false);

      }
        
      for(int i=0; i<Hams.size(); i++) 
        Hams[i].vbias(GIJ, vsp.sliced(field_pos[i],field_pos[i+1]), dt, a); 
 
    } else {

      // if nci > 1, we expect [...][nwalk]
      RUNTIME_CHECK(G.size(0)    == nspin * npol * NMO * npol * NMO, "");
      RUNTIME_CHECK(G.size(1)    == nwalk, "");

      // B[n][:] = A[ I[n] ][:] 
      ma::copy_select(G, GIJ, n2IJ_dev, false);

      for(int i=0; i<Hams.size(); i++) 
        Hams[i].vbias(GIJ, vsp.sliced(field_pos[i],field_pos[i+1]), dt, a );

    }

    if( not std::is_same<vType,SPComplexType>::value ) 
      copy_n_cast( vsp.origin(), vsp.num_elements(), make_device_ptr(v.origin()) );
  }

  template<class Mat, class MatB>
  void generalizedFockMatrix([[maybe_unused]] Mat&& G, [[maybe_unused]] MatB&& Fp, [[maybe_unused]] MatB&& Fm)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

  bool distribution_over_cholesky_vectors() const { return true; }
  int local_number_of_cholesky_vectors() const { return local_nCV; }
  int global_number_of_cholesky_vectors() const { return local_nCV; }
  int global_origin_cholesky_vector() const { return 0; }
  int number_of_ke_vectors() const { return ET.get_n2IJ()->size(); }

  // transpose=true means [nwalk][ik], false means [ik][nwalk]
  bool transposed_G_for_vbias() const { 
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int nci = PsiC.size() / nspin;
    return (nci==1); 
  }
  bool transposed_G_for_E() const { return true; }
  bool transposed_vHS() const { return false; }
  bool fast_ph_energy() const { return false; }
  bool spin_dependent_vHS() const { return true; }

  boost::multi::array<ComplexType, 2> getHSPotentials() { 
    return boost::multi::array<ComplexType, 2>{}; 
  }

private:

  afqmc::TaskGroup_& TG;

  WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE;

  // (conjugated) orbital Matrix
  // This should be in node memory
  // PsiC[ ndets * npin ][i][a] = std::conj( PsiTrial[ ndets * npin ][i][a] )  
  std::vector<OrbitalMatrixType> PsiC;

  // One body Hamiltonian
  SparseEnergy<MP,REAL> ET; 

  // list of ModelComponents
  // By convention, Hams[0] is a 1-body term.
  // All others (n>0) must be interacting terms. 
  std::vector<ModelComponent<MP,REAL>> Hams;

  // local csr_matrix matrix that will store vHS_sparse if needed
  std::vector<dev_csr_Matrix<ComplexType>> spvHS;

  // Vector with an ordered list of all IJ terms ( c^I cJ ) present in any term in the Hamiltonian,
  // n2IJ[n] = (I + si*NMO) * (npol * NMO) + J 
  // I/J are in range [0,NMO] for CLOSED/COLLINEAR and [0,2*NMO] for NONCOLLINEAR
  Vector<size_t> n2IJ;
  Vector<size_t, device_allocator<size_t>> n2IJ_dev;
  int nIJ_first_beta;
  // used to update discrete factorizations
  std::unordered_map<size_t, int> IJ2n;

  Vector<int> field_pos;
  Vector<int> ke_pos;

  // container for vHS in case vHS is requested in sparse form
  // sparse_vHS_t sparse_vHS;
  // sparse_vHS should be a csr_matrix in device memory. 

  int num_ke_vectors = 0;
  int local_nCV = 0;  

  bool sparse_G_eval = true;

  template<class MatA, class MatB>
  void getGFull(MatA const& Gc, MatB& Gfull, OrbitalMatrixType& PsiCA, OrbitalMatrixType& PsiCB)
  {
    using AType = typename std::decay<MatA>::type::element;
    using BType = typename std::decay<MatB>::type::element;
    static_assert(std::is_same<BType,SPComplexType>::value,
                  "Invalid value_type in getGFull. Should not happen.");
    static_assert(MatA::dimensionality == 2, "Incorrect matrix dimension.");
    static_assert(MatB::dimensionality == 2, "Incorrect matrix dimension.");
    int npol   = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO    = PsiCA.size(0) / npol;
    int nwalk  = Gc.size(0);
    int nel_a  = PsiCA.size(1);
    int nel_b  = ( walker_type == COLLINEAR ) ? PsiCB.size(1) : 0;
    RUNTIME_CHECK(PsiCA.size(0) == npol * NMO, "");
    RUNTIME_CHECK(PsiCB.size(0) == npol * NMO, "");
    RUNTIME_CHECK(Gc.size(0)    == nwalk, "");
    RUNTIME_CHECK(Gc.size(1)    == ( nel_a + nel_b ) * npol * NMO, "");
    RUNTIME_CHECK(Gfull.size(0)    == nspin * npol * NMO * npol * NMO, "");
    RUNTIME_CHECK(Gfull.size(1)    == nwalk, "");

    DeviceBufferManager device_buffer_manager{};
    Static3Tensor Gt_({nwalk, nspin * npol * NMO, npol * NMO}, 
                    device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    auto G2D = Gt_.rotated().flatted().unrotated();

    if( std::is_same<AType,SPComplexType>::value ) {
      using sfqmc::afqmc::reinterpret_pointer_cast;
      Array_ref<SPComplexType const, 3, const_pointer> G_( 
                        reinterpret_pointer_cast<SPComplexType const>(make_device_ptr(Gc.origin())),
                        {nwalk, (nel_a+nel_b), npol * NMO} );   

      ma::productStridedBatched(PsiCA, G_(G_.extension(0),{0,nel_a},G_.extension(2)), 
                        Gt_(Gt_.extension(0),{0,npol*NMO},Gt_.extension(2)));
      if( walker_type == COLLINEAR ) {
        ma::productStridedBatched(PsiCB, G_(G_.extension(0),{nel_a, nel_a+nel_b},G_.extension(2)), 
                        Gt_(Gt_.extension(0),{NMO, 2*NMO},Gt_.extension(2)));
      }
      ma::transpose(G2D,Gfull);  

    } else { 
      // buffer allocators

      Static3Tensor G_({nwalk, (nel_a+nel_b), npol * NMO}, 
                      device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      copy_n_cast( make_device_ptr(Gc.origin()), Gc.num_elements(), G_.origin());

      ma::productStridedBatched(PsiCA, G_(G_.extension(0),{0,nel_a},G_.extension(2)),
                        Gt_(Gt_.extension(0),{0,npol*NMO},Gt_.extension(2)));
      if( walker_type == COLLINEAR ) {
        ma::productStridedBatched(PsiCB, G_(G_.extension(0),{nel_a, nel_a+nel_b},G_.extension(2)),
                        Gt_(Gt_.extension(0),{NMO, 2*NMO},Gt_.extension(2)));
      }
      ma::transpose(G2D,Gfull);

    }

  }

  template<class MatA, class MatB>
  void getGFull(MatA const& Gc, MatB& Gfull, OrbitalMatrixType& PsiCA)
  {
    using AType = typename std::decay<MatA>::type::element;
    using BType = typename std::decay<MatB>::type::element;
    static_assert(std::is_same<BType,SPComplexType>::value,
                  "Invalid value_type in getGFull. Should not happen.");
    static_assert(MatA::dimensionality == 2, "Incorrect matrix dimension.");
    static_assert(MatB::dimensionality == 2, "Incorrect matrix dimension.");
    int npol   = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO    = PsiCA.size(0) / npol;
    int nwalk  = Gc.size(0);
    int nel  = PsiCA.size(1);
    RUNTIME_CHECK(PsiCA.size(0) == npol * NMO, "");
    RUNTIME_CHECK(Gc.size(0)    == nwalk, "");
    RUNTIME_CHECK(Gc.size(1)    == nel * npol * NMO, "");
    RUNTIME_CHECK(Gfull.size(0)    == npol * NMO * npol * NMO, "");
    RUNTIME_CHECK(Gfull.size(1)    == nwalk, "");

    DeviceBufferManager device_buffer_manager{};
    Static3Tensor Gt_({nwalk, npol * NMO, npol * NMO}, 
                    device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    auto G2D = Gt_.rotated().flatted().unrotated();

    if constexpr ( std::is_same<AType,SPComplexType>::value ) {
      using sfqmc::afqmc::reinterpret_pointer_cast;
      auto&& G_=Gc.rotated().partitioned(nel).unrotated();
      ma::productStridedBatched(PsiCA, G_, Gt_); 
      ma::transpose(G2D,Gfull);  
    } else { 
      // buffer allocators
      Static3Tensor G_({nwalk, nel, npol * NMO}, 
                      device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      ma::copy_n_cast(Gc,G_.rotated().flatted().unrotated());
      ma::productStridedBatched(PsiCA, G_, Gt_);
      ma::transpose(G2D,Gfull);
    }

  }

  void make_csr_vHS(int nwalk) 
  {
    int npol   = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO    = PsiC[0].size(0) / npol;
    size_t M = size_t(npol*NMO);

    spvHS.clear();
    spvHS.reserve(2);

    
    if(walker_type == COLLINEAR) {

      std::vector<int> cnt(2*NMO,0);
      for( auto IJ: n2IJ ) cnt[ IJ/M ]++;
 
      std::vector<int> nnz_a(nwalk*NMO,0);
      std::vector<int> nnz_b(nwalk*NMO,0);

      auto ita=nnz_a.begin();
      auto itb=nnz_b.begin();
      for(int w=0; w<nwalk; w++, ita+=NMO, itb+=NMO)
      { 
        std::copy_n( cnt.begin(),     NMO, ita);
        std::copy_n( cnt.begin()+NMO, NMO, itb);
      }

      // setting up in host for simplicity
      ma::sparse::csr_matrix<ComplexType, int, int>
          va_h(tp_ul_ul{nwalk*NMO,nwalk*NMO},tp_ul_ul{0,0},nnz_a);
      ma::sparse::csr_matrix<ComplexType, int, int>
          vb_h(tp_ul_ul{nwalk*NMO,nwalk*NMO},tp_ul_ul{0,0},nnz_b);
       
      for( auto IJ: n2IJ ) {

        int I(int(IJ/M)), J(int(IJ%M));

        // n2IJ is guaranteed to be ordered, so we can use emplace_back safely  
        if( I < NMO ) {
          for(int w=0; w<nwalk; w++)
            va_h.emplace_back( { w*NMO + I, w*NMO + J }, 0.0 );
        } else {
          for(int w=0; w<nwalk; w++)
            vb_h.emplace_back( { w*NMO + (I-NMO), w*NMO + J }, 0.0 );
        } 

      }  

      spvHS.emplace_back( std::move(va_h) );
      spvHS.emplace_back( std::move(vb_h) );

    } else {

      std::vector<int> cnt(npol*NMO,0);
      for( auto IJ: n2IJ ) cnt[ IJ/M ]++;

      std::vector<int> nnz(nwalk*npol*NMO,0);

      auto ita=nnz.begin();
      for(int w=0; w<nwalk; w++, ita+=npol*NMO)
        std::copy_n( cnt.begin(), npol*NMO, ita);

      ma::sparse::csr_matrix<ComplexType, int, int>
          v_h(tp_ul_ul{nwalk*npol*NMO,nwalk*npol*NMO},tp_ul_ul{0,0},nnz);

      for( auto IJ: n2IJ ) {

        int I(int(IJ/M)), J(int(IJ%M));
        for(int w=0; w<nwalk; w++)
          v_h.emplace_back( { w*npol*NMO + I, w*npol*NMO + J }, 0.0 );

      }  

      spvHS.emplace_back( std::move(v_h) );

    }

  }
  

};

} // namespace afqmc

} // namespace sfqmc

#endif
