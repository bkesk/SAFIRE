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

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "Utilities/type_traits/container_traits_multi.h"
#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/Utils.hpp"
#include "THCHamiltonian.h"
#include "AFQMC/Hamiltonians/rotateHamiltonian.hpp"
#include "AFQMC/HamiltonianOperations/THCOpsIO.hpp"

namespace sfqmc
{
namespace afqmc
{
// Right now, cutvn, cutvn2, TGprop and TGwfn are completely ignored.
// Note: addCoulomb only has meaning on the sparse hamiltonians, not in THC
template<bool MP, bool REAL> HamiltonianOperations<MP> 
THCHamiltonian::getHamiltonianOperations_impl(WALKER_TYPES type,
                                              std::vector<PsiT_Matrix>& PsiT,
                                              TaskGroup_& TGprop,   
                                              TaskGroup_& TGwfn,    
                                              hdf_archive& hdf_restart)
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;

  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  using SPValueType   = typename to_working_precision<MP,ValueType  >::type;

  using shmCMatrix    = Matrix_<shared_allocator<ComplexType>>;
  using shmSPVMatrix  = Matrix_<shared_allocator<SPValueType>>;
  using shmSPCMatrix  = Matrix_<shared_allocator<SPComplexType>>;

  std::string base_error(" Error in THCHamiltonian::getHamiltonianOperations():\n   ");

  size_t nspin = (type == COLLINEAR?2:1);
  size_t npol = (type == NONCOLLINEAR?2:1);
  size_t nspin_in_file = 0;

  // hack until parallel hdf is in place
  bool write_hdf = false;
  if (TGwfn.Global().root())
    write_hdf = !hdf_restart.closed();
  //  if(TGwfn.Global().root()) write_hdf = (hdf_restart.file_id != hdf_archive::is_closed);
  TGwfn.Global().broadcast_value(write_hdf);

  if (type == COLLINEAR)
    RUNTIME_CHECK(PsiT.size() % 2 == 0, "");
  int ndet = ((type != COLLINEAR) ? (PsiT.size()) : (PsiT.size() / 2));

  if (ndet > 1)
    APP_ABORT("Error: ndet > 1 not yet implemented in THCHamiltonian::getHamiltonianOperations.");

  // this communicator needs to be used for data structures that are distributed
  // over multiple nodes in a TG. When built with accelerator support, multiple
  // members of the TG will reside in the same node, so the Node() communicator will lead
  // to wrong results. For host only builds, this will just be a copy of Node.
  auto distNode=TG.Node().split(TGwfn.getLocalGroupNumber(), TG.Node().rank());
  if (TGwfn.getLocalGroupNumber() != TGprop.getLocalGroupNumber())
    APP_ABORT(" Error: nnodes in wavefunction must match value in Propagator. ");

  size_t gnmu, gnnv, grotnmu, nmu, rotnmu, nmu0, nmuN, rotnmu0, rotnmuN;
  hdf_archive dump(TGwfn.Global());
  // right now only Node.root() reads
  if (distNode.root())
  {
    if (!dump.open(fileName, H5F_ACC_RDONLY))
      APP_ABORT(base_error + "Error opening integral file");
  }

  // read H1, always in Complex
  shmCMatrix H1({nspin*npol*NMO, npol*NMO}, shared_allocator<ComplexType>{TG.Node()});
  if (TG.Node().root())
  {
    if (dump.push("System", false)<0)
      APP_ABORT(base_error + "Group /System not found");
    std::vector<int> shape;
    if (!dump.getShape<RealType>("H0", shape))
      APP_ABORT(" Error in  THCHamiltonian::getHamiltonianOperations(): getShape(H0) returned error. ");
    nspin_in_file = shape[0];
    // Convert to current run if possible
    // H1_skij
    if((shape[0]!=1 and shape[0]!=2) or shape[1] != 1 or shape[2] != npol*NMO or shape[3] != npol*NMO) 
      APP_ABORT(" Error: Dimension mismatch in /System/H0."); 
    if(nspin == 1 and shape[0]==2) 
      APP_ABORT(" Error: nspin==2 in /System/H0 requires a COLLINEAR calculation."); 
    if(shape.size() == 5) {
      if(shape[4]!=2)
        APP_ABORT(" Error: Inconsistent format of /System/H0 (wrong format for complex numbers).");
      Array_ref_<4,ComplexType*> h1_(raw_pointer_cast(H1.origin()), {shape[0],1,npol*NMO,npol*NMO});
      if (!dump.readEntry(h1_, std::string("H0")))
        APP_ABORT(base_error + "Problems reading /System/H0. ");
      if(nspin==2 and shape[0] == 1) 
        std::copy_n(H1.origin(), npol*npol*NMO*NMO, raw_pointer_cast(H1[npol*NMO].origin()));
    } else if(shape.size() == 4) {
      Array<RealType,4> h1_({shape[0],1,npol*NMO,npol*NMO},RealType(0.0));
      if (!dump.readEntry(h1_, std::string("H0")))
        APP_ABORT(base_error + "Problems reading /System/H0. ");
      std::copy_n(h1_.origin(), npol*npol*NMO*NMO, raw_pointer_cast(H1.origin()));
      if(nspin==2 and shape[0] == 1)
        std::copy_n(h1_.origin(), npol*npol*NMO*NMO, raw_pointer_cast(H1[npol*NMO].origin()));
    } else {
      APP_ABORT(" Error: Inconsistent shape of hcore in /Hamiltonian/hcore. ");
    }
    dump.pop();
  }
  TG.Global().broadcast_value(nspin_in_file);

  if (distNode.root())
  { 
    if (dump.push("Interaction", false)<0)
      APP_ABORT(base_error + " Group THC not found. ");
  }
  // MAM: Add possibility to have a rectangular Luv
  if (TG.Global().root())
  {
    std::vector<int> shape;

    if (!dump.getShape<RealType>("factorized_coulomb_matrix", shape))
      APP_ABORT(base_error + "Problems reading /Interaction/factorized_coulomb_matrix. ");
    if(shape[0]!=1) 
      APP_ABORT(" Error: Found /Interaction/factorized_coulomb_matrix with nq>1."); 
    gnmu    = size_t(shape[1]);
    gnnv    = size_t(shape[2]);

    if(dump.getShape<RealType>("coulomb_matrix", shape)) {
      if(shape[0]!=1 or shape[1]!=gnmu or shape[2]!=gnmu)
        APP_ABORT(base_error + " Shape mismatch in /Interaction/coulomb_matrix");
      grotnmu = gnmu;
    } else if(dump.getShape<RealType>("half_rotated_coulomb_matrix", shape)) {
      if(shape[0]!=1 or shape[2]!=shape[1])
        APP_ABORT(base_error + " Shape mismatch in /Interaction/half_rotated_coulomb_matrix");
      grotnmu = shape[1];
    } else {
      APP_ABORT(base_error + "THC h5 requires either coulomb_matrix or half_rotated_coulomb_matrix");
    } 
  }
  TG.Global().broadcast_value(gnmu);
  TG.Global().broadcast_value(gnnv);
  TG.Global().broadcast_value(grotnmu);

  // setup partition, in general matrices are partitioned asize_t 'u'
  {
    int node_number            = TGwfn.getLocalGroupNumber();
    int nnodes_prt_TG          = TGwfn.getNGroupsPerTG();
    std::tie(rotnmu0, rotnmuN) = FairDivideBoundary(size_t(node_number), grotnmu, size_t(nnodes_prt_TG));
    rotnmu                     = rotnmuN - rotnmu0;

    node_number          = TGprop.getLocalGroupNumber();
    nnodes_prt_TG        = TGprop.getNGroupsPerTG();
    std::tie(nmu0, nmuN) = FairDivideBoundary(size_t(node_number), gnmu, size_t(nnodes_prt_TG));
    nmu                  = nmuN - nmu0;
    RUNTIME_CHECK(nmu > 0, "too many Nodes for number of columns in collocation matrix, {}. Use no more than one node per column.", gnmu);
  }

  // Until I figure something else, rotPiu and rotcXau are not distributed because a full copy is needed
  // distribution:  size,  global,  offset
  //   - rotMuv:    {rotnmu,grotnmu},{grotnmu,grotnmu},{rotnmu0,0}
  //   - rotPiu:    {nspin*npol*NMO,grotnmu},{nspin*npol*NMO,grotnmu},{0,0}
  //   - rotcXau    {nel,grotnmu},{nel,grotnmu},{0,0}
  //   - Piu:       {nspin*npol*NMO,nmu},{nspin*npol*NMO,gnmu},{0,nmu0}
  //   - Luv:       {nmu,gnmu},{gnmu,gnmu},{nmu0,0}
  //   - cXau       {nel,nmu},{nel,gnmu},{nmu0,0}
  //
  size_t nel = PsiT[0].size(0) + ((type == COLLINEAR) ? PsiT[1].size(0) : 0);
  shmSPVMatrix rotMuv({rotnmu, grotnmu}, shared_allocator<SPValueType>{distNode});
  shmSPVMatrix rotPiu({size_t(nspin * npol * NMO), grotnmu}, shared_allocator<SPValueType>{distNode});
  std::vector<shmSPCMatrix> rotcXau;
  rotcXau.reserve(ndet);
  for (int i = 0; i < ndet; i++)
    rotcXau.emplace_back(shmSPCMatrix({nel,grotnmu}, shared_allocator<SPComplexType>{distNode}));
  shmSPVMatrix Piu({nspin * npol * NMO, nmu}, shared_allocator<SPValueType>{distNode});
  shmSPVMatrix Luv({nmu, gnmu}, shared_allocator<SPValueType>{distNode});
  if (distNode.root())
  {
    /***************************************/
    // X_skiu
    Array_ref_<4,SPValueType*> Piu_(raw_pointer_cast(Piu.origin()), {nspin_in_file,1,npol*NMO,nmu});
    hyperslab_proxy<decltype(Piu_), 4> hslab(Piu_, 
                std::array<size_t, 4>{size_t(nspin_in_file), 1ul, size_t(npol*NMO), gnmu},
                std::array<size_t, 4>{size_t(nspin_in_file), 1ul, size_t(npol*NMO), nmu}, 
                std::array<size_t, 4>{0, 0, 0, nmu0});
    if (!dump.readEntry(hslab, "collocation_matrix"))
      APP_ABORT(base_error+" Problems reading collocation_matrix. ");
    if(nspin == 2 and nspin_in_file == 1)
      std::copy_n(raw_pointer_cast(Piu.origin()),npol*NMO*nmu,raw_pointer_cast(Piu[npol*NMO].origin())); 
    /***************************************/
    // Vqun
    Array_ref_<3,SPValueType*> Luv_(raw_pointer_cast(Luv.origin()), {1,nmu,gnmu});
    hyperslab_proxy<decltype(Luv_), 3> hslab2(Luv_, 
                std::array<size_t, 3>{1,gnmu,gnmu}, 
                std::array<size_t, 3>{1,nmu,gnmu},
                std::array<size_t, 3>{0,nmu0,0});
    if (!dump.readEntry(hslab2, "factorized_coulomb_matrix"))
      APP_ABORT(base_error+" Problems reading factorized_coulomb_matrix ");
    /***************************************/
  }
  TG.Global().barrier();

  if (distNode.root())
  { 
    using ma::conj;
    std::vector<int> shape;
    std::string coul_name, coll_name; 
    if(dump.getShape<RealType>("coulomb_matrix", shape)) {
      coul_name = "coulomb_matrix";
      coll_name = "collocation_matrix"; 
    } else {
      coul_name = "half_rotated_coulomb_matrix";
      coll_name = "collocation_matrix_half_rotated"; 
    }
    /***************************************/
    // Mquv
    /***************************************/
    Array_ref_<3,SPValueType*> rotMuv_(raw_pointer_cast(rotMuv.origin()), {1,rotnmu,grotnmu});
    hyperslab_proxy<decltype(rotMuv_), 3> hslab2(rotMuv_, 
                std::array<size_t, 3>{1,grotnmu,grotnmu},  
                std::array<size_t, 3>{1,rotnmu,grotnmu},
                std::array<size_t, 3>{0,rotnmu0,0});
    if (!dump.readEntry(hslab2, coul_name))
      APP_ABORT(base_error+" Problems reading " + coul_name);
    // X_skiu
    Array_ref_<4,SPValueType*> rotPiu_(raw_pointer_cast(rotPiu.origin()), {nspin_in_file,1,npol*NMO,grotnmu});
    if (!dump.readEntry(rotPiu_, coll_name))
      APP_ABORT(base_error+" Problems reading " + coll_name);
    if(nspin == 2 and nspin_in_file == 1)
      std::copy_n(raw_pointer_cast(rotPiu.origin()),npol*NMO*grotnmu,raw_pointer_cast(rotPiu[npol*NMO].origin())); 
    /***************************************/
  }
  TG.Global().barrier();

  shmCMatrix v0({nspin * NMO, NMO}, shared_allocator<ComplexType>{TG.Node()});
  if (TGprop.getNGroupsPerTG() > 1)
  {
    // MAM: doing this in single precision to be consistent with non-distributed case
    // very inefficient, find better work distribution for calculation of v0
    // that doesn't so much temporary space!!!
    boost::multi::array<SPValueType, 2> v0_({nspin * NMO, NMO});
    // TOO MUCH MEMORY, FIX FIX FIX!!!
    shmSPVMatrix Piu__({size_t(nspin * npol * NMO), gnmu}, shared_allocator<SPValueType>{TG.Node()});
    shmSPVMatrix Luv__({gnmu, gnmu}, shared_allocator<SPValueType>{TG.Node()});
    if (TG.Node().root())
    {
      /***************************************/
      // X_skiu
      Array_ref_<4,SPValueType*> Piu_(raw_pointer_cast(Piu__.origin()), {nspin_in_file,1,npol*NMO,gnmu});
      if (!dump.readEntry(Piu_, "collocation_matrix"))
        APP_ABORT(base_error+" Problems reading collocation_matrix. ");
      if(nspin == 2 and nspin_in_file == 1)
        std::copy_n(raw_pointer_cast(Piu__.origin()),npol*NMO*gnmu,raw_pointer_cast(Piu__[npol*NMO].origin())); 
      /***************************************/
      // Vqun
      Array_ref_<3,SPValueType*> Luv_(raw_pointer_cast(Luv__.origin()), {1,gnmu,gnmu});
      if (!dump.readEntry(Luv_, "factorized_coulomb_matrix"))
        APP_ABORT(base_error+" Problems reading factorized_coulomb_matrix ");
      /***************************************/
    }
    TG.Node().barrier();

    using ma::conj;
    using ma::H;
    using ma::T;
    size_t c0, cN, nc;
    std::tie(c0, cN) = FairDivideBoundary(size_t(TG.Global().rank()), gnmu, size_t(TG.Global().size()));
    nc               = cN - c0;
    RUNTIME_CHECK(nc > 0, "too many MPI tasks for number of columns in collocation matrix, {}. MPI rank {} has <{}> working columns. Use no more than one task per column. ", gnmu, TG.Global().rank(), nc);
    boost::multi::array<SPValueType, 2> Tuv({gnmu, nc});
    boost::multi::array<SPValueType, 2> Muv({gnmu, nc});

    // Muv = Luv * H(Luv)
    // This can benefit from 2D split of work
    ma::product(Luv__, H(Luv__.sliced(c0, cN)), Muv);

    // since generating v0 takes some effort and temporary space,
    // v0(s,i,l) = -0.5*sum_j <i,j|j,l>
    //         = -0.5 sum_j,u,v ma::conj(Piu(s,i,u)) ma::conj(Piu(s,j,v)) Muv Piu(s,j,u) Piu(s,l,v)
    //         = -0.5 sum_u,v ma::conj(Piu(s,i,u)) W(s,u,v) Piu(s,l,v), where
    // W(s,u,v) = Muv(u,v) * sum_j Piu(s,j,u) ma::conj(Piu(s,j,v))
    boost::multi::array<SPValueType, 2> T_({nc, NMO});
    if(type == COLLINEAR) {
      for(size_t is=0, is0=0; is<nspin; is++, is0 += NMO) {
        auto&& P_iu = Piu({is0, is0 + NMO}, {0, gnmu});
        auto&& P_iv = Piu({is0, is0 + NMO}, {c0, cN});
        ma::product(H(P_iu), P_iv, Tuv);
        auto itM = Muv.origin();
        auto itT = Tuv.origin();
        for (size_t i = 0; i < Muv.num_elements(); ++i, ++itT, ++itM)
          *(itT) = ma::conj(*itT) * (*itM);
        ma::product(T(Tuv), H(P_iu), T_);
        ma::product(SPValueType(-0.5), T(T_), T(P_iv),
                    SPValueType(0.0), v0_( {is0, is0 + NMO}, {0, NMO} ));
      }
    } else if(type == NONCOLLINEAR) {
      for(size_t ip=0, ip0=0; ip<npol; ip++, ip0 += NMO) {
        auto&& P_iu = Piu({ip0, ip0 + NMO}, {0, gnmu});
        auto&& P_iv = Piu({ip0, ip0 + NMO}, {c0, cN});
        ma::product(H(P_iu), P_iv, Tuv);
        auto itM = Muv.origin();
        auto itT = Tuv.origin();
        for (size_t i = 0; i < Muv.num_elements(); ++i, ++itT, ++itM)
          *(itT) = ma::conj(*itT) * (*itM);
        ma::product(T(Tuv), H(P_iu), T_);
        ma::product(SPValueType(-0.5), T(T_), T(P_iv),
                    SPValueType(0.0), v0_( {ip0, ip0 + NMO}, {0, NMO} ));
      }
    } else {
      auto&& P_iu = Piu({0, NMO}, {0, gnmu});
      auto&& P_iv = Piu({0, NMO}, {c0, cN});
      ma::product(H(P_iu), P_iv, Tuv);
      auto itM = Muv.origin();
      auto itT = Tuv.origin();
      for (size_t i = 0; i < Muv.num_elements(); ++i, ++itT, ++itM)
        *(itT) = ma::conj(*itT) * (*itM);
      ma::product(T(Tuv), H(P_iu), T_);
      ma::product(SPValueType(-0.5), T(T_), T(P_iv),
                  SPValueType(0.0), v0_);
    }

    // reduce over Global
    TG.Global().all_reduce_in_place_n(v0_.origin(), v0_.num_elements(), std::plus<>());
    if (TG.Node().root())
    {
      copy_n_cast(v0_.origin(), nspin * NMO * NMO, raw_pointer_cast(v0.origin()));
      // MAM: Since Muv gets large, might have problems with the check for hermicity below
      // fixing here 
      for (size_t is = 0, is0 = 0; is < nspin; is++, is0 += NMO)
        for (size_t i = 0; i < NMO; i++)
          for (size_t j = i + 1; j < NMO; j++)
          { 
            v0[is0+i][j] = 0.5 * (v0[is0+i][j] + ma::conj(v0[is0+j][i]));
            v0[is0+j][i] = ma::conj(v0[is0+i][j]);
          }
    }
    TG.Node().barrier();
  }
  else
  {
    // very inefficient, find better work distribution for calculation of v0
    // that doesn't so much temporary space!!!
    boost::multi::array<SPValueType, 2> v0_({nspin * NMO, NMO});
    // very simple partitioning until something more sophisticated is in place!!!
    using ma::conj;
    using ma::H;
    using ma::T;
    size_t c0, cN, nc;
    std::tie(c0, cN) = FairDivideBoundary(size_t(TG.Global().rank()), gnmu, size_t(TG.Global().size()));
    nc               = cN - c0;
    RUNTIME_CHECK(nc > 0, "too many MPI tasks for number of columns in collocation matrix, {}. MPI rank {} has <{}> working columns. Use no more than one task per column. ", gnmu, TG.Global().rank(), nc);
    boost::multi::array<SPValueType, 2> Tuv({gnmu, nc});
    boost::multi::array<SPValueType, 2> Muv({gnmu, nc});

    // Muv = Luv * H(Luv)
    // This can benefit from 2D split of work
    ma::product(Luv, H(Luv.sliced(c0, cN)), Muv);

    // since generating v0 takes some effort and temporary space,
    // v0(s,i,l) = -0.5*sum_j <i,j|j,l>
    //         = -0.5 sum_j,u,v ma::conj(Piu(s,i,u)) ma::conj(Piu(s,j,v)) Muv Piu(s,j,u) Piu(s,l,v)
    //         = -0.5 sum_u,v ma::conj(Piu(s,i,u)) W(s,u,v) Piu(s,l,v), where
    // W(s,u,v) = Muv(u,v) * sum_j Piu(s,j,u) ma::conj(Piu(s,j,v))
    boost::multi::array<SPValueType, 2> T_({nc, NMO});
    if(type == COLLINEAR) {
      for(size_t is=0, is0=0; is<nspin; is++, is0 += NMO) {
        auto&& P_iu = Piu({is0, is0 + NMO}, {0, gnmu});
        auto&& P_iv = Piu({is0, is0 + NMO}, {c0, cN});
        ma::product(H(P_iu), P_iv, Tuv);
        auto itM = Muv.origin();
        auto itT = Tuv.origin();
        for (size_t i = 0; i < Muv.num_elements(); ++i, ++itT, ++itM)
          *(itT) = ma::conj(*itT) * (*itM);
        ma::product(T(Tuv), H(P_iu), T_);
        ma::product(SPValueType(-0.5), T(T_), T(P_iv), 
                    SPValueType(0.0), v0_( {is0, is0 + NMO}, {0, NMO} ));
      }
    } else if(type == NONCOLLINEAR) {
      for(size_t ip=0, ip0=0; ip<npol; ip++, ip0 += NMO) {
        auto&& P_iu = Piu({ip0, ip0 + NMO}, {0, gnmu});
        auto&& P_iv = Piu({ip0, ip0 + NMO}, {c0, cN});
        ma::product(H(P_iu), P_iv, Tuv);
        auto itM = Muv.origin();
        auto itT = Tuv.origin();
        for (size_t i = 0; i < Muv.num_elements(); ++i, ++itT, ++itM)
          *(itT) = ma::conj(*itT) * (*itM);
        ma::product(T(Tuv), H(P_iu), T_);
        ma::product(SPValueType(-0.5), T(T_), T(P_iv), 
                    SPValueType(0.0), v0_( {ip0, ip0 + NMO}, {0, NMO} ));
      }
    } else {
      auto&& P_iu = Piu({0, NMO}, {0, gnmu});
      auto&& P_iv = Piu({0, NMO}, {c0, cN});
      ma::product(H(P_iu), P_iv, Tuv);
      auto itM = Muv.origin();
      auto itT = Tuv.origin();
      for (size_t i = 0; i < Muv.num_elements(); ++i, ++itT, ++itM)
        *(itT) = ma::conj(*itT) * (*itM);
      ma::product(T(Tuv), H(P_iu), T_);
      ma::product(SPValueType(-0.5), T(T_), T(P_iv),
                  SPValueType(0.0), v0_);
    }

    // reduce over Global
    TG.Global().all_reduce_in_place_n(v0_.origin(), v0_.num_elements(), std::plus<>());
    if (TG.Node().root())
    {
      copy_n_cast(v0_.origin(), nspin * NMO * NMO, raw_pointer_cast(v0.origin()));
      // MAM: Since Muv gets large, might have problems with the check for hermicity below
      // fixing here
      for (size_t is = 0, is0 = 0; is < nspin; is++, is0 += NMO)
        for (size_t i = 0; i < NMO; i++)
          for (size_t j = i + 1; j < NMO; j++)
          {
            v0[is0+i][j] = 0.5 * (v0[is0+i][j] + ma::conj(v0[is0+j][i]));
            v0[is0+j][i] = ma::conj(v0[is0+i][j]);
          }
    }
    TG.Node().barrier();
  }
  TG.Global().barrier();

  size_t nup = PsiT[0].size(0);
  size_t ndown = ((type == COLLINEAR) ? PsiT.back().size(0) : 0);

  // half-rotated Pia
  std::vector<shmSPCMatrix> cXau;
  cXau.reserve(ndet);
  for (int i = 0; i < ndet; i++)
    cXau.emplace_back(shmSPCMatrix({nel, nmu}, shared_allocator<SPComplexType>{distNode}));
  if (distNode.root())
  {
    // simple
    using ma::H;
    // cXau = ma::conj(Aai * Piu) = T( H(Piu), H(Aai) ), 
    //                 where PsiT = conj(Aai), so conj(Aia) = T(PsiT)  
    if (type == COLLINEAR)
    {
      boost::multi::array<SPComplexType, 2> A({NMO, nup});
      boost::multi::array<SPComplexType, 2> T1({std::max(nmu,grotnmu), nup});
      auto&& Piu_up = Piu({0,NMO}, {0, nmu});
      auto&& Piu_dn = Piu({NMO,2*NMO}, {0, nmu});
      auto&& rotPiu_up = rotPiu({0,NMO}, {0, grotnmu});
      auto&& rotPiu_dn = rotPiu({NMO,2*NMO}, {0, grotnmu});
      for (int i = 0; i < ndet; i++)
      {
        // alpha
        ma::Matrix2MA('T', PsiT[2 * i], A);

        ma::product(H(Piu_up), A, T1({0,nmu},{0,nup}));
        ma::transpose(T1({0,nmu},{0,nup}),cXau[i].sliced(0, nup));

        ma::product(H(rotPiu_up), A, T1({0,grotnmu},{0,nup})); 
        ma::transpose(T1({0,grotnmu},{0,nup}),rotcXau[i].sliced(0, nup));

        // beta
        ma::Matrix2MAREF('T', PsiT[2 * i + 1], A({0,NMO},{0,ndown}));

        ma::product(H(Piu_dn), A({0,NMO},{0,ndown}), T1({0,nmu},{0,ndown})); 
        ma::transpose(T1({0,nmu},{0,ndown}),cXau[i].sliced(nup, nel));

        ma::product(H(rotPiu_dn), A({0,NMO},{0,ndown}), T1({0,grotnmu},{0,ndown})); 
        ma::transpose(T1({0,grotnmu},{0,ndown}),rotcXau[i].sliced(nup, nel));
      }
    }
    else if (type == NONCOLLINEAR)
    {
      APP_ABORT("Finish THC noncollinear.");
    }
    else
    {
      boost::multi::array<SPComplexType, 2> A({PsiT[0].size(1), PsiT[0].size(0)});
      boost::multi::array<SPComplexType, 2> T1({std::max(nmu,grotnmu), nup});
      for (int i = 0; i < ndet; i++)
      {
        ma::Matrix2MA('T', PsiT[i], A);
        ma::product(H(Piu), A, T1.sliced(0,nmu)); 
        ma::transpose(T1.sliced(0,nmu),cXau[i]);
        ma::product(H(rotPiu), A, T1.sliced(0,grotnmu)); 
        ma::transpose(T1.sliced(0,grotnmu),rotcXau[i]);
      }
    }
  }
  TG.Node().barrier();

  ComplexType E0 = NuclearCoulombEnergy + FrozenCoreEnergy;

  shmCMatrix hij({ndet, (nup+ndown) * npol * NMO}, shared_allocator<ComplexType>{TG.Node()});
  if (TG.Node().root())
  {
    // dense one body hamiltonian
    int skp = ((type == COLLINEAR) ? 1 : 0);
    for (int n = 0, nd = 0; n < ndet; ++n, nd += (skp + 1))
    {
      check_wavefunction_consistency(type, &PsiT[nd], &PsiT[nd + skp], NMO, nup, ndown);
      auto hij_=rotateHij(type, &PsiT[nd], &PsiT[nd + skp], H1);
      std::copy_n(hij_.origin(), hij_.num_elements(), raw_pointer_cast(hij[n].origin()));
    }
  }
  TG.Node().barrier();

//  if (write_hdf)
//    writeTHCOps(hdf_restart, type, NMO, nup, ndown, nmu0, rotnmu0, ndet, TGprop, TGwfn, H1, rotPiu, rotMuv, Piu, Luv,
//                v0, E0);

  if (distNode.root())
  {
    dump.pop();
    dump.close();
  }

  return HamiltonianOperations<MP>(THCOps<MP, REAL>(TGwfn.TG_local(), NMO, nup, ndown, type, nmu0, rotnmu0, std::move(H1),
                                      std::move(hij), std::move(rotMuv), std::move(rotPiu), std::move(rotcXau),
                                      std::move(Luv), std::move(Piu), std::move(cXau), std::move(v0), E0));
}

template<bool MP> HamiltonianOperations<MP> 
THCHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
                                         std::vector<PsiT_Matrix>& PsiT,
                                         TaskGroup_& TGprop,   
                                         TaskGroup_& TGwfn,    
                                         hdf_archive& hdf_restart)
{
  bool Real; 
 
  // factorized_coulomb_matrix(q,u,n): required
  // collocation_matrix(s,k,i,u): required
  // Option 1: Provide L * dagger(L) where L=factorized_coulomb_matrix. Reuse collocation_matrix.
  // coulomb_matrix(q,u,n) 
  // Option 2: Independently generated THC factorization for half-rotated orbitals 
  // half_rotated_coulomb_matrix(q,u',v') 
  // collocation_matrix_half_rotated(s,k,i,u)
  if(TG.Global().root()) { 

    hdf_archive dump{};
    if (!dump.open(fileName, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening integral file in THCHamiltonian. ");
    if (dump.push("Interaction", false)<0)
      APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): Group Interaction not found. ");
    std::vector<int> shape;
    if (!dump.getShape<RealType>("factorized_coulomb_matrix", shape))
      APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): getShape(factorized_coulomb_matrix) returned error. ");
    if( shape.size() == 4 ) Real = false;
    else if( shape.size() == 3 ) Real = true;
    else APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): Inconsistent shape of HalfTransformedFullOrbitals. "); 

    // check others
    shape.clear();
    if (!dump.getShape<RealType>("collocation_matrix", shape))
      APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): getShape(collocation_matrix) returned error. ");
    if( (shape.size() != 4 and shape.size() != 5) or 
        (shape.size() == 5 and Real ) or
	(shape.size() == 4 and not Real) )
      APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): Inconsistent shape between factorized_coulomb_matrix and collocation_matrix. ");

    shape.clear();
    if (dump.getShape<RealType>("coulomb_matrix", shape))  {

      // provide Muv, reuse collocation_matrix
      if( (shape.size() != 3 and shape.size() != 4) or 
          (shape.size() == 4 and Real ) or
          (shape.size() == 3 and not Real) )
        APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): Inconsistent shape between coulomb_matrix and factorized_coulomb_matrix. ");

    } else if(dump.getShape<RealType>("half_rotated_coulomb_matrix", shape)) {

      // provide half rotated Muv and X 
      if( (shape.size() != 3 and shape.size() != 4) or 
          (shape.size() == 4 and Real ) or
          (shape.size() == 3 and not Real) )
        APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): Inconsistent shape between factorized_coulomb_matrix and half_rotated_coulomb_matrix. ");

      if (!dump.getShape<RealType>("collocation_matrix_half_rotated", shape))
        APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): getShape(collocation_matrix_half_rotated) returned error. ");
      if( (shape.size() != 4 and shape.size() != 5) or
          (shape.size() == 5 and Real ) or
          (shape.size() == 4 and not Real) )
        APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): Inconsistent shape between factorized_coulomb_matrix and collocation_matrix_half_rotated"); 

    } else {
      APP_ABORT(" Error in THCHamiltonian::getHamiltonianOperations(): Missing datasets in THC h5 file. File must contain either coulomb_matrix or {half_rotated_coulomb_matrix + collocation_matrix_half_rotated}"); 
    }

    dump.pop();
    dump.close(); 
  }
  TG.Global().broadcast_n(&Real, 1, 0);

  if(Real) 
    return getHamiltonianOperations_impl<MP,true>(type, PsiT, TGprop, TGwfn, hdf_restart);
  else
    return getHamiltonianOperations_impl<MP,false>(type, PsiT, TGprop, TGwfn, hdf_restart);
}

template HamiltonianOperations<true> THCHamiltonian::getHamiltonianOperations<true>(
		WALKER_TYPES, std::vector<PsiT_Matrix>&,TaskGroup_&,TaskGroup_&,hdf_archive&);
template HamiltonianOperations<false> THCHamiltonian::getHamiltonianOperations<false>(
		WALKER_TYPES, std::vector<PsiT_Matrix>&,TaskGroup_&,TaskGroup_&,hdf_archive&);

} // namespace afqmc
} // namespace sfqmc
