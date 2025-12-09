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

#include <iostream>
#include <vector>
#include <string>
#include <tuple>

#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "utilities/check_strides.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"
//#include "AFQMC/SlaterDeterminantOperations/

namespace sfqmc
{
namespace afqmc
{
/*
 * Class that implements a multi-Slater determinant trial wave-function.
 * Single determinant wfns are also allowed. 
 * No relation between different determinants in the expansion is assumed.
 * Designed for non-orthogonal MSD expansions. 
 * For particle-hole orthogonal MSD wfns, use FastMSD.
 */
template<MEMORY_SPACE MEM, class devPsiT>
class NOMSD : public AFQMCInfo
{

public:

  NOMSD() {
    utils::check(false,"Default constructor for NOMSD disabled.");
  }

  NOMSD(AFQMCInfo& info,
        ptree pt_in,
        WALKER_TYPES wlk,
        std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
        HamiltonianOperations<MEM>&& hop_,
        nda::array<ComplexType,1>&& ci_,
        nda::array<devPsiT,2>&& orbs_,
        ComplexType nce,
        [[maybe_unused]] int targetNW = 1)
      : AFQMCInfo(info),
        mpi(_mpi),
        walker_type(wlk),
        HamOp(std::move(hop_)),
        ci(std::move(ci_)),
        OrbMats(std::move(orbs_)),
        RefOrbMats(0, 0, 0, 0),
        NuclearCoulombEnergy(nce)
  {
    utils::check(OrbMats.extent(0) == ci.size(), "Size mismatch");
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nNOMSD input:\n{}\n",io::to_string(pt));
    // initialize using verbose input
    bool rediag;
    rediag = pt.get<bool>("rediag");

    if (rediag) {
      utils::check(false,"finish");
      //recompute_ci();
    }
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    auto rediag      = pt0.get<bool>("rediag", false);
    auto dense_trial = pt0.get<bool>("dense_trial", true);
    // create verbose internal inputs
    ptree pt1;
    pt1.put("rediag", rediag);
    pt1.put("dense_trial", dense_trial);
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "name",
      "ndets_to_read",
      "filename",
      "compute",
      "dense_trial"
    };
    io::compare_known_keys("Non-orthogonal multi-Slater det. (NOMSD) Wavefunction",pt1, pt0,pass_through_keys);
    return pt1;
  }

  ~NOMSD() = default; 

  NOMSD(NOMSD const& other) = delete;
  NOMSD& operator=(NOMSD const& other) = delete;
  NOMSD(NOMSD&& other)                 = default;
  NOMSD& operator=(NOMSD&& other) = delete;

  int number_of_cholesky_vectors() const { return HamOp.number_of_cholesky_vectors(); }

  WALKER_TYPES getWalkerType() const { return walker_type; }

  /*
   * Returns the memory space.
   */
  constexpr auto get_memory_space() const { return MEM; }

  /*
   *  Performs runtime optimizations.
   */       
  template<class WlkSet>
  void runtime_optimization(WlkSet& wset)
  {
    const int nw   = wset.size();
    const int nel = (walker_type==COLLINEAR ? nup+ndown : nup );
    const int nspin = (walker_type==COLLINEAR ? 2 : 1 );
    const int npol = (walker_type==NONCOLLINEAR ? 2 : 1 );
    memory::array<MEM,ComplexType,2> G(nw,nel*npol*NMO);
    // don't use buffered_array!!!
    HamOp.runtime_optimization(G);
  }

  /*
   * Expectation value of Hubbard-Stratonovich potential with respect to trial wave-function.
   */
  void vMF(nda::MemoryVector auto&& v, double dt);

  /*
   * Green function of the trial wave-funtion. 
   */
  template<MEMORY_SPACE M>
  auto G_MF();

/*
  SlaterDetOperations* getSlaterDetOperations() { return std::addressof(SDetOp); }
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    HamOp.generalizedFockMatrix(std::forward<Args>(args)...); 
    TG.TG_local().barrier();
  }
*/

  HamiltonianTypes getHamType() const { return HamOp.getHamType(); }

  auto getFieldTypes()
  {
    return HamOp.getFieldTypes();
  }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    HamOp.update_potentials(std::forward<Args>(args)...);
  }

  auto vHS_dims() const { return HamOp.vHS_dims(); }

  template<class... Args>
  auto getOneBodyPropagatorMatrix(Args&&... args)
  {
    return HamOp.getOneBodyPropagatorMatrix(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto vHS_sparse(Args&&... args)
  {
    return HamOp.vHS_sparse(std::forward<Args>(args)...);
  }

  /*
   * Calculates the bias potential.
   */
  template<class WlkSet>
  void vbias(WlkSet& wset, nda::MemoryMatrix auto && v, double dt)
  {
    memory::check_memory_space<MEM>(v);
    AFQMCTimer.start(G_for_vbias_timer);
    bool compact_G_for_vbias = (ci.size() == 1); 
    int nel   = (walker_type==COLLINEAR ? nup+ndown : nup);
    int nspin = (walker_type==COLLINEAR ? 2 : 1);
    int npol  = (walker_type==NONCOLLINEAR ? 2 : 1);
    int nw = wset.size();
    int nc = (compact_G_for_vbias ? nel*npol*NMO : nspin*npol*NMO*npol*NMO );
    utils::check(v.shape() == std::array<long,2>{nw,HamOp.number_of_cholesky_vectors()}, 
                 "Shape mismatch");
    memory::buffered_array<MEM,ComplexType,2> G(nw,nc);
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw);
    MixedDensityMatrix(wset, G, ovlp, compact_G_for_vbias);
    AFQMCTimer.stop(G_for_vbias_timer);
    AFQMCTimer.start(vbias_timer);
    v() = ComplexType(0.0);
    HamOp.vbias(G, v, dt);
    AFQMCTimer.stop(vbias_timer);
  }

  /*
   * Returns the Hubbard-Stratonovoch potential. 
   *  Input:
   *    - X: [# chol vecs][nW]
   *  Output:
   *    - vHS
   */
  template<nda::MemoryMatrix X>
  auto vHS(X && x, double dt )
  {
    utils::check(x.extent(1) == HamOp.number_of_cholesky_vectors(), "Shape mismatch");
    return HamOp.vHS(std::forward<X>(x), dt);
  }

  /*
   * Calculates the local energy and overlaps of all the walkers in the set and stores
   * them in the wset data
   */
  template<class WlkSet>
  void Energy(WlkSet& wset)
  {
    auto all = nda::range::all;
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    memory::buffered_array<MEM,ComplexType,2> eloc(nw,3);
    eloc() = ComplexType(0.0);
    Energy(wset, eloc(), ovlp());
    wset.setProperty(OVLP, ovlp);
    wset.setProperty(E1_, eloc(all, 0));
    wset.setProperty(EXX_, eloc(all, 1));
    wset.setProperty(EJ_, eloc(all, 2));
  }

  /*
   * Calculates the local energy and overlaps of all the walkers in the set and 
   * returns them in the appropriate data structures
   */
  template<class WlkSet,  nda::MemoryMatrix TMat, nda::MemoryVector TVec>
  void Energy(const WlkSet& wset, TMat&& E, TVec&& Ov);

  /*
   * Calculates the mixed density matrix for all walkers in the walker set. 
   * Options:
   *  - compact:   If true (default), returns compact form with Dim: [NEL*NMO], 
   *                 otherwise returns full form with Dim: [NMO*NMO]. 
   */ 
  template<class WlkSet, nda::MemoryMatrix MatG>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, bool compact = true)
  {
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    MixedDensityMatrix(wset, std::forward<MatG>(G), ovlp, compact);
  }

  template<class WlkSet, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true);

  /*
   * Calculates the density matrix with respect to a given Reference
   * for all walkers in the walker set. 
   */
  template<class WlkSet, nda::MemoryVector RVec, nda::MemoryMatrix MatG, nda::MemoryVector TVec>
  void DensityMatrix(const WlkSet& wset, RVec&& Ref, MatG&& G, TVec&& Ov, 
                     bool compact = true, bool herm = true);

  /*
   * Calculates the overlaps of all walkers in the set. Returns values in arrays. 
   */
  template<class WlkSet, nda::MemoryArrayOfRank<1> TVec>
  void Overlap(const WlkSet& wset, TVec && Ov);

  /*
   * Calculates the overlaps of all walkers in the set. Updates values in wset. 
   */
  template<class WlkSet>
  void Overlap(WlkSet& wset)
  {
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    Overlap(wset, ovlp);
    wset.setProperty(OVLP, ovlp);
  }
/*
  template<class... Args>
  void accumulate_estimators(Args&&... args)
  {
    if(ci.size()>1)
      accumulate_estimators_general_impl(std::forward<Args>(args)...);
    else
      accumulate_estimators_single_ref_impl(std::forward<Args>(args)...);    
  }
*/
  /*
   * Returns the number of reference Slater Matrices needed for back propagation.  
   */
  int number_of_references_for_back_propagation() const
  {
    return OrbMats.extent(0);
  }

  ComplexType getReferenceWeight(int i) const { return ci[i]; }

  /*
    * Returns the reference Slater Matrices needed for back propagation.  
    */
  auto getReferences() const
  {
    return OrbMats;
  }

protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  // type of walker/wfn
  WALKER_TYPES walker_type;

  HamiltonianOperations<MEM> HamOp;

  nda::array<ComplexType, 1> ci;

  // OrbMats[ndet][nspin](nel,NMO)
  nda::array<devPsiT,2> OrbMats;

  // RefOrbMats[ndet][nspin][nel][NMO]
  memory::array<MEM,ComplexType,4> RefOrbMats;

  ComplexType NuclearCoulombEnergy;

/*
  void recompute_ci();

  template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3, class Observable>
  void accumulate_estimators_general_impl(int iav, WlkSet& wset, TVec& wgt, 
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
        Mat1 const& X, Mat2 const& Y, Mat3 const& M, bool time_evolved, bool importanceSampling);

  template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3, class Observable>
  void accumulate_estimators_single_ref_impl(int iav, WlkSet& wset, TVec& wgt, 
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
        Mat1 const& X, Mat2 const& Y, Mat3 const& M, bool time_evolved, bool importanceSampling);
*/
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Wavefunctions/NOMSD.icc"

