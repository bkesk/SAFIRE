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

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <tuple>

#include "AFQMC/config.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/Utilities/type_conversion.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

#include "AFQMC/Wavefunctions/phmsd_helpers.hpp"
#include "AFQMC/Wavefunctions/Excitations.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Class that implements particle-hole multi-Slater determinant expansions trial wave-functions.
 * All determinants in the expansion are related to the reference determinant 
 * by a list of single particle excitations.
 */
template<MEMORY_SPACE MEM>
class PHMSD : public AFQMCInfo
{

public:
  // temporary
  PHMSD(AFQMCInfo& info,
        ptree pt_in,
        WALKER_TYPES wlk,
        std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
        HamiltonianOperations<MEM>&& hop_)
      : AFQMCInfo(info),
        mpi(_mpi),
        walker_type(wlk),
        HamOp(std::move(hop_))
  {}

  template<class csrM>
  PHMSD(AFQMCInfo& info,
        ptree pt_in,
        WALKER_TYPES wlk,
        std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
        HamiltonianOperations<MEM>&& hop_,
        std::map<int, int>&& acta2mo_,
        std::map<int, int>&& actb2mo_,
        ph_excitations<int, ComplexType>&& abij_,
        nda::array<csrM,1>&& op_spin_det_coupling_,
        nda::array<csrM,1>&& orbs_,
        ComplexType nce,
        [[maybe_unused]] int targetNW = 1)
      : AFQMCInfo(info),
        mpi(_mpi),
        walker_type(wlk),
        HamOp(std::move(hop_)),
        acta2mo(std::move(acta2mo_)),
        actb2mo(std::move(actb2mo_)),
        abij(std::move(abij_)),
        OpSpinDetCouplings(std::move(op_spin_det_coupling_)),
        OrbMats(std::move(orbs_)),
        number_of_references(-1),
        NuclearCoulombEnergy(nce) 
//        maxnactive(std::max(OrbMats[0].size(0), OrbMats[1].size(0))),
//        max_exct_n(std::max(abij.maximum_excitation_number()[0], abij.maximum_excitation_number()[1]))
  {
    /* To me, PHMSD is not compatible with walker_type=CLOSED unless
     * the MSD expansion is symmetric with respect to spin. For this, 
     * it is better to write a specialized class that assumes either spin symmetry
     * or e.g. Perfect Pairing.
     */
    if (walker_type == CLOSED)
      APP_ABORT("Error: PHMSD requires walker_type != CLOSED.");

    // FINISH!!!
    if (walker_type == NONCOLLINEAR)
      APP_ABORT("PHMSD has not yet been implemented for NONCOLLINEAR walkers.");

    if (walker_type == FULLYPOLARIZED)
      APP_ABORT("PHMSD has not yet been implemented for FULLYPOLARIZED walkers.");

    // setup device structures

    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nPHMSD input:\n{}\n",io::to_string(pt));
    // initialize using verbose input
    number_of_references = pt.get<int>("number_of_references");

    // optional
    if( auto val = pt.get_optional<int>("algorithm") ) {
      energy_algorithm = *val;
    } else {
      if(HamOp.getHamType() == RealDenseFactorized)
        energy_algorithm=1;  // add others as they get implemented...
      else 
        energy_algorithm=0;
    }

    if(energy_algorithm==0)
      app_log(1, " Using default (slow) energy algorithm. ");
    else if(energy_algorithm==1)
      app_log(1, " Using energy algorithm 1. ");
    else if(energy_algorithm==2)
      app_log(1, " Using energy algorithm 2. ");
    else
      APP_ABORT(" Error in PHMSD constructor: Unknown algorithm. \n\n");
    // check that refc is appropriate for the selected algorithm
    if(energy_algorithm==1) {
      auto refc=abij.reference_configuration();
      for(int i=0; i<nup; i++)
        if( refc[i] != i ) 
        utils::check(refc[i] == i, " Error: PHMSD algorithm=1 requires refc[i]==i.\n\n");
      for(int i=0; i<ndown; i++)
        utils::check(refc[nup+i] == i, " Error: PHMSD algorithm=1 requires refc[i]==i.\n\n");
    }
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    int number_of_references = pt0.get<int>("number_of_references", -1);
    // create verbose internal inputs
    ptree pt1;
    pt1.put("number_of_references", number_of_references);
    // leave as a true optional, to bypass issue with default value
    if( auto val = pt0.get_optional<int>("algorithm") )
      pt1.put("algorithm", *val);
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "name",
      "ndets_to_read",
      "restart_file",
      "filename",
      "rediag"
    };
    io::compare_known_keys("particle-hole multi-Slater det. (PHMSD) Wavefunction", pt1, pt0,pass_through_keys);
    return pt1;
  }

  ~PHMSD() = default; 

  PHMSD(PHMSD const& other) = default;
  PHMSD& operator=(PHMSD const& other) = default;
  PHMSD(PHMSD&& other)                 = default;
  PHMSD& operator=(PHMSD&& other) = default;

  int number_of_cholesky_vectors() const { return HamOp.number_of_cholesky_vectors(); }

  WALKER_TYPES getWalkerType() const { return walker_type; }

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
// This needs to depend on algorithm!!!
    HamOp.runtime_optimization(G);
  }

  /*
   * Returns the memory space.
   */
  constexpr auto get_memory_space() const { return MEM; }

  /*
   * Expectation value of Hubbard-Stratonovich potential with respect to trial wave-function.
   */
  void vMF(nda::MemoryVector auto&& v, double dt); 

  /*
   * Green function of the trial wave-funtion. 
   */
  auto G_MF();

  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    HamOp.generalizedFockMatrix(std::forward<Args>(args)...);
  }

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
    int nact  = OrbMats(0).extent(0) + (walker_type==COLLINEAR ? 0 : OrbMats(1).extent(0));
    int nspin = (walker_type==COLLINEAR ? 2 : 1);
    int npol  = (walker_type==NONCOLLINEAR ? 2 : 1);
    int nw = wset.size();
    utils::check(v.shape() == std::array<long,2>{nw,HamOp.number_of_cholesky_vectors()},
                 "Shape mismatch");
    memory::buffered_array<MEM,ComplexType,2> G(nw,nact*npol*NMO);
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw);
    MixedDensityMatrix(wset, G, ovlp);
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
  void Energy(const WlkSet& wset, TMat&& E, TVec&& Ov)
  {
    if(energy_algorithm==0)
      energy_alg0(wset,E,Ov);
    else if(energy_algorithm==1)
      energy_alg1(wset,E,Ov);
    else if(energy_algorithm==2)
      energy_alg2(wset,E,Ov);
    else
      utils::check(false," Error: Unknown energy_algorithm. \n\n");
  }

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
                     bool compact = true, bool herm = true)
  {}

  /*
   * Calculates the overlaps of all walkers in the set. Returns values in arrays. 
   */
  template<class WlkSet, nda::MemoryArrayOfRank<1> TVec>
  void Log_Overlap(const WlkSet& wset, TVec && Ov);

  /*
   * Calculates the overlaps of all walkers in the set. Updates values in wset. 
   */
  template<class WlkSet>
  void Log_Overlap(WlkSet& wset)
  {
    int nw = wset.size();
    memory::buffered_array<MEM,ComplexType,1> ovlp(nw,ComplexType(0.0));
    Log_Overlap(wset, ovlp);
    wset.setProperty(OVLP, ovlp);
  }

/*
  template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, TVec& wgt,
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
        Mat1 const& X, Mat2 const& Y, Mat3 const& M, bool time_evolved, bool importanceSampling);
*/

  /*
   * Returns the number of reference Slater Matrices needed for back propagation.  
   */
  int number_of_references_for_back_propagation() const
  {
    if (number_of_references > 0)
      return number_of_references;
    else
      return abij.number_of_configurations();
  }

  ComplexType getReferenceWeight(int i) const { return 0.0; } //std::get<2>(*abij.configuration(i)); }

  /*
   * Returns the reference Slater Matrices needed for back propagation.  
   */
  auto getReferences() const
  {
    utils::check(false,"finish");
    return  OrbMats; // This is wrong!!!
/*
    static_assert(std::decay<Mat>::type::dimensionality == 2, "Wrong dimensionality");
    int ndet = number_of_references_for_back_propagation();
    RUNTIME_CHECK(A.size(0) == ndet, "");
    if (RefOrbMats.size(0) == 0)
    {
      TG.Node().barrier(); // for safety
      int nrow(NMO * ((walker_type == NONCOLLINEAR) ? 2 : 1));
      int ncol(nup + ndown); //careful here, spins are stored contiguously
      RefOrbMats = mpi3CMatrix({ndet, nrow * ncol}, RefOrbMats.get_allocator());
      TG.Node().barrier(); // for safety
      if (TG.Node().root())
      {
        boost::multi::array<ComplexType, 2> OA_({OrbMats[0].size(1), OrbMats[0].size(0)});
        boost::multi::array<ComplexType, 2> OB_({0, 0});
        if (OrbMats.size() > 1)
          OB_.reextent({OrbMats[1].size(1), OrbMats[1].size(0)});
        ma::Matrix2MAREF('H', OrbMats[0], OA_);
        if (OrbMats.size() > 1)
          ma::Matrix2MAREF('H', OrbMats[1], OB_);
        std::vector<int> Ac(nup);
        std::vector<int> Bc(ndown);
        for (int i_det = 0; i_det < ndet; ++i_det)
        {
          auto c=abij.configuration(i_det);
          abij.get_configuration(0, std::get<0>(*c), Ac);
          abij.get_configuration(1, std::get<1>(*c), Bc);
          boost::multi::array_ref<ComplexType, 2> A_(raw_pointer_cast(RefOrbMats[i_det].origin()), {NMO, nup});
          boost::multi::array_ref<ComplexType, 2> B_(A_.origin() + A_.num_elements(), {NMO, ndown});
          for (int i = 0, ia = 0; i < NMO; ++i)
            for (int a = 0; a < nup; ++a, ia++)
              A_[i][a] = OA_[i][Ac[a]];
          if (OrbMats.size() > 1)
          {
            for (int i = 0, ia = 0; i < NMO; ++i)
              for (int a = 0; a < ndown; ++a, ia++)
                B_[i][a] = OB_[i][Bc[a]];
          }
          else if(walker_type == COLLINEAR)
          {
            for (int i = 0, ia = 0; i < NMO; ++i)
              for (int a = 0; a < ndown; ++a, ia++)
                B_[i][a] = OA_[i][Bc[a]];
          }
        }
      }                    // TG.Node().root()
      TG.Node().barrier(); // for safety
    }
    RUNTIME_CHECK(RefOrbMats.size(0) == ndet, "");
    RUNTIME_CHECK(RefOrbMats.size(1) == A.size(1), "");
    auto&& RefOrbMats_=boost::multi::static_array_cast<ComplexType, ComplexType*>(RefOrbMats);
    auto&& A_=boost::multi::static_array_cast<ComplexType, Ptr>(A);
    using std::copy_n;
    int n0, n1;
    std::tie(n0, n1) = FairDivideBoundary(TG.getLocalTGRank(), int(A.size(1)), TG.getNCoresPerTG());
    for (int i = 0; i < ndet; i++)
      copy_n(RefOrbMats_[i].origin() + n0, n1 - n0, A_[i].origin() + n0);
    TG.TG_local().barrier();
*/
  }

protected:

  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  // type of walker/wfn
  WALKER_TYPES walker_type;  

  HamiltonianOperations<MEM> HamOp;

  // MAM: use enum when this is settled...
  // 0: loop over unique configurations, calculate G and evaluate E from scratch
  // 1: use ph_reference_energy and ph_excited_energy, which requires compact R matrix
  // 2: calculate Fapbq and call ph_energy_Fapbq
  int energy_algorithm = 0;

  std::map<int, int> acta2mo;
  std::map<int, int> actb2mo;

  ph_excitations<int, ComplexType> abij; 

  // sparse matrix with opposite spin determinant couplings
  nda::array<PsiT_Matrix<MEM>,1> OpSpinDetCouplings;

  nda::array<PsiT_Matrix<MEM>,1> OrbMats;
  int number_of_references;

  ComplexType NuclearCoulombEnergy;

  // shared memory arrays for temporary calculations
  size_t maxnactive;        // maximum number of states in active space
  size_t max_exct_n;        // maximum excitation number (number of electrons excited simultaneously)

  /* Implementation of various energy evaluation algorithms. */
  template<class WlkSet,  nda::MemoryMatrix Mat, nda::MemoryVector TVec>
  void energy_alg0(const WlkSet& wset, Mat&& E, TVec&& Ov);

  template<class WlkSet,  nda::MemoryMatrix Mat, nda::MemoryVector TVec>
  void energy_alg1(const WlkSet& wset, Mat&& E, TVec&& Ov)
  { utils::check(false, "fiinish"); }

  template<class WlkSet,  nda::MemoryMatrix Mat, nda::MemoryVector TVec>
  void energy_alg2(const WlkSet& wset, Mat&& E, TVec&& Ov)
  { utils::check(false, "fiinish"); }

};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Wavefunctions/PHMSD.icc"

