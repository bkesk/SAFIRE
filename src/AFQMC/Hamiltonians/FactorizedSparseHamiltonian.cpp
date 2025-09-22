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

#include <cstdlib>
#include <algorithm>
#include <complex>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <vector>
#include <numeric>
#include <functional>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "AFQMC/Utilities/Utils.hpp"
#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "Numerics/ma_blas_extensions.hpp"
#include "SparseMatrix/array_partition.hpp"

#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"

#include "AFQMC/Hamiltonians/FactorizedSparseHamiltonian.h"

#include "AFQMC/Hamiltonians/rotateHamiltonian.hpp"
#include "AFQMC/SlaterDeterminantOperations/rotate.hpp"
#include "AFQMC/Hamiltonians/HSPotential_Helpers.hpp"

#include "AFQMC/HamiltonianOperations/SparseTensor.hpp"
//#include "AFQMC/HamiltonianOperations/SparseTensorIO.hpp"

namespace sfqmc
{
namespace afqmc
{

template<typename VType> mpi3_csr_matrix<VType> 
FactorizedSparseHamiltonian::calculateHSPotentials(double cut,
                                                   TaskGroup_& TGprop,
						   mpi3_csr_matrix<VType>& Likn)	
{
  using Alloc = shared_allocator<VType>;
  if (TG.getNumberOfTGs() > 1)
    APP_ABORT("Error: HSPotential not implemented with distributed Hamiltonian. ");

  if (TG.getNumberOfTGs() > 1)
  {
    APP_ABORT(" Finish HS. ");
    return mpi3_csr_matrix<VType>(tp_ul_ul{0, 0}, tp_ul_ul{0, 0}, 0, Alloc(TG.Node()));
  }
  else
  {
    // you always need to count since some vectors might be empty
    auto nnz_per_cv = HamHelper::count_nnz_per_cholvec(cut, TG, Likn, NMO);

    // count vectors and make mapping from 2*n/2*n+1 to actual value
    std::vector<int> map_;
    map_.reserve(nnz_per_cv.size());
    int cnt = 0;
    for (auto& n : nnz_per_cv)
      map_.emplace_back((n > 0) ? (cnt++) : (-1));

    // partition and
    std::size_t cv0, cvN;
    if (TGprop.getNGroupsPerTG() == 1)
    { // Spvn is not distributed
      cv0 = 0;
      cvN = nnz_per_cv.size();
      if (TG.Global().root())
      {
        app_log(2, "\nPartition of cholesky vector over nodes in TG: ");
        app_log(2, "{} ", std::count_if(nnz_per_cv.begin(), nnz_per_cv.begin() + cvN, [](std::size_t i) { return i > 0; }));
        app_log(2, "");;
        app_log(2,"Number of terms in Cholesky Matrix per node in TG: ");
        app_log(2, "{} ", std::accumulate(nnz_per_cv.begin(), nnz_per_cv.begin() + cvN, std::size_t(0)));
        app_log(2, "");
      }
    }
    else
    {
      std::vector<std::size_t> cv_boundaries(TGprop.getNGroupsPerTG() + 1);
      simple_matrix_partition<TaskGroup_, std::size_t, double> split(Likn.size(0), nnz_per_cv.size(), cut);
      // no need for all cores to do this
      if (TG.Global().root())
        split.partition(TGprop, false, nnz_per_cv, cv_boundaries);
      TG.Global().broadcast_n(cv_boundaries.begin(), cv_boundaries.size());
      cv0 = cv_boundaries[TGprop.getLocalGroupNumber()];
      cvN = cv_boundaries[TGprop.getLocalGroupNumber() + 1];
      // no need for all cores to do this
      if (TG.Global().root())
      {
        app_log(2, "\nPartition of cholesky vector over nodes in TG: ");
        for (int i = 0; i < TGprop.getNGroupsPerTG(); i++)
          app_log(2, "{} ", std::count_if(nnz_per_cv.begin() + cv_boundaries[i], 
					      nnz_per_cv.begin() + cv_boundaries[i + 1],
                                              [](std::size_t i_) { return i_ > 0; }));
        app_log(2, "Number of terms in Cholesky Matrix per node in TG: ");
        for (int i = 0; i < TGprop.getNGroupsPerTG(); i++)
          app_log(2, "{} ", std::accumulate(nnz_per_cv.begin() + cv_boundaries[i], 
						nnz_per_cv.begin() + cv_boundaries[i + 1],
                                       		std::size_t(0)));
        app_log(2, "\n"); 
      }
    }

    auto nnz_per_ik = HamHelper::count_nnz_per_ik(cut, TG, Likn, NMO, cv0, cvN);

    std::size_t nvec =
        std::count_if(nnz_per_cv.begin() + cv0, nnz_per_cv.begin() + cvN, [](std::size_t const& i) { return i > 0; });

    std::size_t cv_origin =
        std::count_if(nnz_per_cv.begin(), nnz_per_cv.begin() + cv0, [](std::size_t const& i) { return i > 0; });

    // can build csr directly since cores work on non-overlapping rows
    // and can use emplace_back
    mpi3_csr_matrix<VType> csr(tp_ul_ul{NMO * NMO, nvec}, tp_ul_ul{0, cv_origin}, nnz_per_ik, Alloc(TG.Node()));

    HamHelper::generateHSPotential(csr, map_, cut, TG, Likn, NMO, cv0, cvN);
    TG.Node().barrier();

    return csr;
  }
}

// this routine should rely on Likn and not on Likn_base! Rewrite rotateHijkl!
template<typename CType, class PsiT_Mat, class csrMat>
mpi3_csr_matrix<CType> FactorizedSparseHamiltonian::halfRotatedHijkl(WALKER_TYPES type,
                                                                     TaskGroup_& TGHam,
                                                                     PsiT_Mat* Alpha,
                                                                     PsiT_Mat* Beta,
								     csrMat const& Likn_base, 
                                                                     RealType const cut)
{
  using Alloc = shared_allocator<CType>;
  RUNTIME_CHECK(Alpha != nullptr, "");
  if (type == COLLINEAR)
    RUNTIME_CHECK(Beta != nullptr, "");
  std::size_t nr = Alpha->size(0) * Alpha->size(1);
  if (type == COLLINEAR)
    nr = (Alpha->size(0) + Beta->size(0)) * Alpha->size(1);
  bool addCoulomb = false; // legacy parameter from old code, remove at some point!
  if (TGHam.getNGroupsPerTG() > 1)
  {
    using tvec = std::vector<std::tuple<int, int, CType>>;
    tvec tmat;
    tmat.reserve(100000); // reserve some space
    rotateHijkl<CType>(factorizedHalfRotationType, type, addCoulomb, TG, tmat, Alpha, Beta, Likn_base, cut,
                      maximum_buffer_size, false, false);
    TG.Node().barrier();
    return csr::shm::construct_distributed_csr_matrix_from_distributed_containers<mpi3_csr_matrix<CType>>(tmat, nr, nr,
                                                                                                          TGHam);
  }
  else
  {
    using ucsr_mat = typename mpi3_csr_matrix<CType>::base;
    ucsr_mat ucsr(tp_ul_ul{nr, nr}, tp_ul_ul{0, 0}, 0, Alloc(TG.Node()));
    if (TG.getTotalNodes() > 1)
      rotateHijkl<CType>(factorizedHalfRotationType, type, addCoulomb, TG, ucsr, Alpha, Beta, Likn_base, cut,
                            maximum_buffer_size, true, true);
    else
      rotateHijkl_single_node<CType>(factorizedHalfRotationType, type, addCoulomb, TG, ucsr, Alpha, Beta, Likn_base,
                                        cut, maximum_buffer_size, true);
    TG.Node().barrier();
    return csr::shm::construct_csr_matrix_from_distributed_ucsr<mpi3_csr_matrix<CType>, TaskGroup_>(std::move(ucsr),
                                                                                                    TG);
  }
}

template<bool MP, typename LikType, typename LakType, class PsiT_Type> HamiltonianOperations<MP> 
FactorizedSparseHamiltonian::getHamiltonianOperations_impl(WALKER_TYPES type,
                                                         std::vector<PsiT_Matrix>& PsiT,
                                                         std::vector<PsiT_Type>& PsiTsp,
                                                         TaskGroup_& TGprop,   
                                                         TaskGroup_& TGwfn,    
                                                         hdf_archive& hdf_restart)
{
  using SPLikType   = typename to_working_precision<MP,LikType  >::type;
  using SPLakType   = typename to_working_precision<MP,LakType  >::type;
  using SPRealType = typename to_working_precision<MP,RealType>::type;

  using shmCMatrix    = Matrix_<shared_allocator<ComplexType>>;

  RUNTIME_CHECK(PsiT.size() == PsiTsp.size(), ""); 
  if (type == COLLINEAR)
    RUNTIME_CHECK(PsiT.size() % 2 == 0, "");

  int ndet = ((type != COLLINEAR) ? (PsiT.size()) : (PsiT.size() / 2));
  int nspins = ((type == COLLINEAR) ? 2 : 1);
  int npol = ((type == NONCOLLINEAR) ? 2 : 1);
  size_t nel[2] = {PsiT[0].size(0), ((type == COLLINEAR) ? (PsiT[1].size(0)) : 0ul )};

  // check nel consistency
  for(int d=0, ds=0; d>ndet; ++d) {
    for(int s=0; s<nspins; ++s, ++ds) {
      RUNTIME_CHECK(PsiT[ds].size(0) == nel[s], "");
      RUNTIME_CHECK(PsiT[ds].size(1) == npol*NMO, "");
      RUNTIME_CHECK(PsiT[ds].size(0) == PsiTsp[ds].size(0), ""); 
      RUNTIME_CHECK(PsiT[ds].size(1) == PsiTsp[ds].size(1), ""); 
    }
  }

  // write restart file?
  bool write_hdf = false;
  if (TGwfn.Global().root())
    write_hdf = !hdf_restart.closed();
  TGwfn.Global().broadcast_value(write_hdf);

  std::string base_error(" Error in FactorizedSparseHamiltonian::getHamiltonianOperations(): ");

  hdf_archive dump;
  std::vector<int> Idata(8);
  int ncores = TG.getTotalCores();
  int nread = (n_reading_cores <= 0) ? (ncores) : (std::min(n_reading_cores, ncores));
  if (TG.getCoreID() < nread)
  { 
    if (!dump.open(fileName, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening integral file in FactorizedSparseHamiltonian. ");
    if (dump.push("Hamiltonian", false)<0)
      APP_ABORT(base_error + " Group Hamiltonian not found. ");
  }
  if(TG.Node().root())
    if (!dump.readEntry(Idata, "dims"))
      APP_ABORT(base_error + "Problems reading dims. ");
  TG.Node().broadcast(Idata.begin(), Idata.end());
  int nvecs = Idata[7];       // # of cholesky vectors
  int int_blocks = Idata[2];  // # of integral blocks

  shmCMatrix H1({npol*NMO, npol*NMO}, ComplexType(0.0), shared_allocator<ComplexType>{TG.Node()});
  if (TG.Node().root()) {
    Matrix_ref_<ComplexType*> H1_(raw_pointer_cast(H1.origin()), H1.extensions());
    read_one_body_hamiltonian(dump, H1_);  
  }

  // number of cores reading from hdf file
  if(TG.getCoreID() < nread) {
    if(dump.push("Factorized", false)<0)
      APP_ABORT(base_error + "Error opening Factorized dataset.");
  }
  mpi3_csr_matrix<SPLikType> Likn_bare = read_V2fact<SPLikType>(dump, TG, nread, NMO, nvecs, 1e-8, int_blocks);
 
  app_log(1," Memory used by factorized 2-el integral table (on head node): {} MB.",
               (SPRealType(Likn_bare.capacity() * (sizeof(SPLikType) + sizeof(IndexType)) +
                Likn_bare.size(0) * (2 * sizeof(std::size_t)))) / 1024.0 / 1024.0);

  if (not dump.closed())
  {
    dump.pop();
    dump.pop();
    dump.close();
  }

  // half-rotate H1
  shmCMatrix haj({ndet, (nel[0]+nel[1])*npol*NMO}, ComplexType(0.0), shared_allocator<ComplexType>{TG.Node()});
  {
    auto&& haj3D=haj.rotated().partitioned(nel[0]+nel[1]).unrotated();
    double scl = (type == CLOSED ? 2.0 : 1.0);
    for (int d=0; d<ndet; d++) {
      if (d % TG.Node().size() != TG.Node().rank())
        continue;
      ma::product(ComplexType(scl), PsiT[nspins*d], H1, 
  		ComplexType(0.0), haj3D[d].sliced(0,nel[0]));
      if( type == COLLINEAR )
        ma::product(ComplexType(1.0), PsiT[2*d+1], H1, 
  		  ComplexType(0.0), haj3D[d].sliced(nel[0],nel[0]+nel[1]));
    }
  }

  auto H2B = [&](IndexType I, IndexType J, IndexType K, IndexType L) 
  {
    int ik        = I * NMO + K;
    int lj        = L * NMO + J;
    return ComplexType(csr::csrvv<ComplexType>('N', 'C', Likn_bare.sparse_row(ik), Likn_bare.sparse_row(lj)));
  };

  shmCMatrix vn0({NMO, NMO}, ComplexType(0.0), shared_allocator<ComplexType>{TG.Node()});
  for (int i = 0, cnt = 0; i < NMO; i++)
    for (int l = i; l < NMO; l++, cnt++)
    { 
      if (cnt % TG.Global().size() != TG.Global().rank())
        continue;
      ComplexType vl(0.0);
      for (int j = 0; j < NMO; j++)
        vl += H2B(i, j, j, l);
      vn0[i][l] -= 0.5 * vl;
      if (i != l) 
        vn0[l][i] -= 0.5 * std::conj(vl);
    }
  TG.Node().barrier();
  if(TG.Node().root())
    TG.Cores().all_reduce_in_place_n(raw_pointer_cast(vn0.origin()), vn0.num_elements(), std::plus<>());

  // generate Likn
  mpi3_csr_matrix<SPLikType> Likn(calculateHSPotentials<SPLikType>(cutoff_cholesky, TGprop, Likn_bare));
  // If Likn_bare is not used anymore, find a way to destroy it here!

  // trick: the last node always knows what the total # of chol vecs is
  int global_ncvecs = 0;
  if (TG.getNodeID() == TG.getTotalNodes() - 1 && TG.getCoreID() == 0)
    global_ncvecs = Likn.global_origin()[1] + Likn.size(1);
  global_ncvecs = (TG.Global() += global_ncvecs);

  ComplexType E0 = NuclearCoulombEnergy + FrozenCoreEnergy;

  // now build Lnak
  std::vector<mpi3_csr_matrix<SPLakType>> Lnak;
  Lnak.reserve(ndet*nspins);
  // Lnak[ds](n,aj) = sum_i Psi[ds](a,i) * Likn(ij,n)	
  for ( auto& v : PsiTsp )
    Lnak.emplace_back(sparse_rotate::halfRotateCholeskyMatrixForBias<SPLakType>(CLOSED, TGprop, 
	std::addressof(v), std::addressof(v), Likn, cutoff_cholesky));

  std::vector<mpi3_csr_matrix<SPLakType>> V2;
  V2.reserve(ndet*nspins);
  for ( auto& v : PsiTsp )
    V2.emplace_back(halfRotatedHijkl<SPLakType>(CLOSED, TGwfn, std::addressof(v), std::addressof(v), 
						Likn_bare, cutoff_exx));

//  if (write_hdf)
//    writeSparseTensor(dump, type, NMO, NAEA, NAEB, TGprop, TGwfn, H1, V2, Spvn, vn0, E0, global_ncvecs, 21);

  return HamiltonianOperations<MP>(SparseTensor<MP, LikType, LakType, LakType>(TGwfn, type, 
		std::move(H1), std::move(haj), std::move(vn0), std::move(V2), std::move(Likn), 
		std::move(Lnak), E0, global_ncvecs, (use_transpose and ndet>1)));
}

template<bool MP> HamiltonianOperations<MP> 
FactorizedSparseHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
                                                      std::vector<PsiT_Matrix>& PsiT,
                                                      TaskGroup_& TGprop,   
                                                      TaskGroup_& TGwfn,    
                                                      hdf_archive& hdf_restart)
{
  bool RealLik = false;
  bool RealPsiT = false;
  bool pureSD = false;
 
  std::string base_error(" Error in FactorizedSparseHamiltonian::getHamiltonianOperations(): ");
  if(TG.Global().root()) {
    // deduce Real
    hdf_archive dump;
    if (!dump.open(fileName, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening integral file in FactorizedSparseHamiltonian. ");
    if (dump.push("Hamiltonian", false)<0)
      APP_ABORT(base_error + " Group Hamiltonian not found. ");
    if(dump.push("Factorized", false)<0)
      APP_ABORT(base_error + "Error opening Factorized dataset.");
    std::vector<int> shape;
    if (!dump.getShape<RealType>("vals_0", shape))
        APP_ABORT(base_error + " getShape(vals_0) returned error. ");

    if( shape.size() == 2 ) RealLik = false;
    else if( shape.size() == 1 ) RealLik = true;
    else APP_ABORT(base_error + "Inconsistent data shape. ");

    dump.pop();
    dump.pop();
    dump.close();
  }
  TG.Global().broadcast_n(&RealLik, 1, 0);
  TG.Global().broadcast_n(&RealPsiT, 1, 0);
  TG.Global().broadcast_n(&pureSD, 1, 0);

  // implement pureSD and <RealType,RealType> case if needed
  //if(pureSD) {
    // specialized routine for pureSD
    //return getHamiltonianOperations_impl<MP,ComplexType,ComplexType>(
    //				type, PsiT, TGprop, TGwfn, hdf_restart);
  //  return HamiltonianOperations<MP>{};				  
  //} else 
  if(not RealLik) {
    if constexpr (MP) {
      auto PsiTsp=csr::shm::CSRvector_to_single_precision<PsiT_Matrix_t<ComplexFloat>>(PsiT);
      return getHamiltonianOperations_impl<true,ComplexType,ComplexType>(
				type, PsiT, PsiTsp, TGprop, TGwfn, hdf_restart);
    } else {
      return getHamiltonianOperations_impl<false,ComplexType,ComplexType>(
				type, PsiT, PsiT, TGprop, TGwfn, hdf_restart);
    }
  //} else if(RealPsiT){
    // make copy of PsiT with RealType
    //return getHamiltonianOperations_impl<MP,RealType,RealType>(
    //				type, PsiT, TGprop, TGwfn, hdf_restart);
  } else {
    if constexpr (MP) {
      auto PsiTsp=csr::shm::CSRvector_to_single_precision<PsiT_Matrix_t<ComplexFloat>>(PsiT);
      return getHamiltonianOperations_impl<true,RealType,ComplexType>(
				type, PsiT, PsiTsp, TGprop, TGwfn, hdf_restart);
    } else {
      return getHamiltonianOperations_impl<false,RealType,ComplexType>(
				type, PsiT, PsiT, TGprop, TGwfn, hdf_restart);
    }
  }

}

// instantiate templates
template HamiltonianOperations<true>
FactorizedSparseHamiltonian::getHamiltonianOperations<true>(WALKER_TYPES,
                std::vector<PsiT_Matrix>&,TaskGroup_&,TaskGroup_&,hdf_archive&);
template HamiltonianOperations<false>
FactorizedSparseHamiltonian::getHamiltonianOperations<false>(WALKER_TYPES,
                std::vector<PsiT_Matrix>&,TaskGroup_&,TaskGroup_&,hdf_archive&);


} // namespace afqmc
} // namespace sfqmc
