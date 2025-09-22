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

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_CONTINUOUS_GENERALUJ_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_CONTINUOUS_GENERALUJ_HPP

#include <type_traits>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "mpi3/shared_communicator.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "Numerics/ma_operations.hpp"
//#include "Numerics/tensor_operations.hpp"

#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/taskgroup.h"

namespace sfqmc
{
namespace afqmc
{

template<bool MP>
class Continuous_GeneralUJ
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;

  // MAM: since the expectation is that the models are very sparse, 
  //      I'm operating under the assumption that all these sparse matrices are
  //      quite small (memory usage). So I'm keeping everything in device memory for now...
  template<class T>
  using csrMat = ma::sparse::csr_matrix<T, int, int, device_allocator<T>>;

  using device_alloc_type  = DeviceBufferManager::template allocator_t<SPComplexType>;
  using StaticMatrix  = boost::multi::static_array<SPComplexType, 2, device_alloc_type>;

public:

  template<class Vec, class csrM1, class csrM2>
// requires: {Psi(std::move(psi_)) is valid}, {hij(std::move(hij_)) is valid}, ... 
  Continuous_GeneralUJ(afqmc::TaskGroup_& tg_,
                          WALKER_TYPES type,
                          PropagatorTypes ptype,  
                          Vec&& h0_,
                          csrM1&& vn_,
                          csrM2&& vnT_,
                          bool shift_ = false,  
                          ComplexType e0 = 0
                )
      : TG(tg_),
        walker_type(type),
        propg_type(ptype),
        local_nCV(0),
        shift_one_body_terms(shift_),
        E0(e0),
        h0(std::move(h0_)),
        SpVn(std::move(vn_)),
        SpVnT(std::move(vnT_))
  {
    if(propg_type != ContinuousChargePropagator and 
       propg_type != ContinuousSpinPropagator) 
      APP_ABORT(" Error: Wrong PropagatorTypes argument in Continuous_GeneralUJ. ");      
    local_nCV = SpVn.size(1);
    RUNTIME_CHECK(SpVn.size(0) == SpVnT.size(1), "");
    RUNTIME_CHECK(SpVn.size(1) == SpVnT.size(0), "");
    RUNTIME_CHECK(SpVn.size(0) == h0.size(), "");
  }

  ~Continuous_GeneralUJ() {}

  Continuous_GeneralUJ(const Continuous_GeneralUJ& other)            = delete;
  Continuous_GeneralUJ& operator=(const Continuous_GeneralUJ& other) = delete;
  Continuous_GeneralUJ(Continuous_GeneralUJ&& other)                 = default;
  Continuous_GeneralUJ& operator=(Continuous_GeneralUJ&& other)      = delete;

  /*
   * n2IJ maps an index in the ordering of the sparse structures to the ordering 
   * of H1 (the generic spin ordering of 1-body operators)
   * n2IJ is expected in host memory. 
   */
  template<class Mat, class map_t>
  void addOneBodyPropagatorMatrix([[maybe_unused]] TaskGroup_& TG_, Mat&& H1, double dt,
                                  boost::multi::array<ComplexType, 1> const& vMF,
                                  map_t& n2IJ)
  {
    static_assert(std::decay_t<Mat>::dimensionality == 2,"Incorrect dimensions.");
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO = H1.size(1) / npol;

    RUNTIME_CHECK(H1.size(0) == nspin * npol * NMO, "");
    RUNTIME_CHECK(n2IJ.size() == h0.size(), "");

    Matrix<SPComplexType, device_allocator<SPComplexType>> v_({n2IJ.size(), 1}, SPComplexType(0.0));
    Matrix<SPComplexType, device_allocator<SPComplexType>> vMF_( {vMF.size(0), 1} ); 
    copy_n_cast(vMF.origin(), vMF.num_elements(), vMF_.origin());
    // turn off 1-body terms in vHS temporarily 
    bool shift_one_body_terms_ = shift_one_body_terms;
    shift_one_body_terms = true;
    vHS(vMF_, v_, dt);
    shift_one_body_terms = shift_one_body_terms_;

    SPComplexType Cdt = SPComplexType(SPRealType(dt));
    boost::multi::array_ref<ComplexType, 1> H1D( H1.origin(),
                                                 {H1.num_elements()} );
    Vector<SPComplexType> v_host(iextensions<1u>{v_.size(0)});
    copy_n(v_.origin(),v_.size(0),v_host.origin());
    if(shift_one_body_terms) {
      Vector<SPComplexType> h0_host(h0.extensions());
      copy_n(h0.origin(),h0.size(),h0_host.origin());
      for( size_t n=0; n<n2IJ.size(); n++) 
        H1D[ n2IJ[n] ] += static_cast<ComplexType>(Cdt * h0_host[n] + v_host[n]);
    } else
      for( size_t n=0; n<n2IJ.size(); n++) 
        H1D[ n2IJ[n] ] += static_cast<ComplexType>(v_host[n]);
  }

  template<class TVec>
  void getFieldTypes(TVec&& v) {
    RUNTIME_CHECK(v.size() == local_nCV, "");
    using std::fill_n;
    fill_n( v.origin(), v.size(), propg_type );
  }

  // nothing to update in Continuous case!
  template<class... Args> void update([[maybe_unused]] Args&&... args) {}

  // v(IJ,w) = sum_n Vn(IJ,n) X(n,w)
  template<class MatX,
           class MatV,
           typename = typename std::enable_if_t<(std::decay<MatX>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatV>::type::dimensionality == 2)>,
           typename = void
          >
  void vHS(MatX&& X, MatV&& v, double dt, double a = 1.)
  {
    RUNTIME_CHECK(X.size(1) == v.size(1), "");
    RUNTIME_CHECK(SpVn.size(1) == X.size(0), "");
    RUNTIME_CHECK(SpVn.size(0) == v.size(0), "");
    RUNTIME_CHECK(v.size(0) == h0.size(0), "");

    SPComplexType ia = SPComplexType(SPRealType(std::sqrt(dt)*a));
    ma::product(ia, SpVn, X, SPComplexType(1.0,0.0), v);

    if(shift_one_body_terms) return;
  
    // multiply by '-i' to compensate for factor of 'i' implicit in the propagator
    // factor of -1 coming from the minus sign in exp(- t V) = exp( i * vHS )
    ia = SPComplexType(0.0, SPRealType(dt*a));

    // v[n][iw] += ia*h0[n];
    ma::elementwise(ma::TOp_PLUS, 0, ia, h0, v);
  }

  // v(n,w) = sum_IJ VnT(n,IJ) G(w,IJ)
  template<class MatA,
           class MatB,
           typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality == 2)>,
           typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality == 2)>>
  void vbias(const MatA& G, MatB&& v, double dt, double a = 1.)
  {
    SPComplexType ia = SPComplexType(SPRealType(std::sqrt(dt)*a)); 
    RUNTIME_CHECK(SpVnT.size(0) == v.size(0), "");
    RUNTIME_CHECK(SpVnT.size(1) == G.size(0), "");
    RUNTIME_CHECK(G.size(1) == v.size(1), "");

    ma::product(ia, SpVnT, G, SPComplexType(1.0,0.0), v);
  }

  template<class Mat, class MatB>
  void generalizedFockMatrix([[maybe_unused]] Mat&& G, [[maybe_unused]] MatB&& Fp, [[maybe_unused]] MatB&& Fm)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

  int number_of_ke_vectors() const { return local_nCV; }
  int local_number_of_cholesky_vectors() const { return local_nCV; }

private:

  afqmc::TaskGroup_& TG;

  WALKER_TYPES walker_type;

  PropagatorTypes propg_type;

  int local_nCV;
  // if shift_one_body_terms=true, h0 is added to the one body propagator.
  // otherwise it is added through vHS
  bool shift_one_body_terms = false;

  ComplexType E0 = 0.0;

  // All sparse matrices have a compact representation of IJ, as defined
  // by n2IJ in ModelHamOps. 

  // constant one-body term associated with the 
  // interacting term.
  Vector<SPComplexType, device_allocator<SPComplexType>> h0;

  // HS operator 
  csrMat<SPComplexType> SpVn;

  // transposed HS operator 
  csrMat<SPComplexType> SpVnT;

};

} // namespace afqmc

} // namespace sfqmc

#endif
