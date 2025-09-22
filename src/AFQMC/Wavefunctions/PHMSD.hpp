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

#ifndef SFQMC_AFQMC_PHMSD_HPP
#define SFQMC_AFQMC_PHMSD_HPP

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <tuple>
#include <boost/optional.hpp>

#include "io/ptree/ptree_utilities.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/config.h"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "Memory/buffer_allocators.hpp"
#include "SparseMatrix/array_of_sequences.hpp"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.hpp"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"

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
template<bool MP>
class PHMSD : public AFQMCInfo
{
  using SPRealType = typename to_working_precision<MP,RealType>::type;
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;

  // allocators
  using Allocator          = device_allocator<ComplexType>;
  using SPAllocator        = device_allocator<SPComplexType>;
  using IAllocator         = device_allocator<int>;
  using Allocator_shared   = localTG_allocator<ComplexType>;
  using SPAllocator_shared = localTG_allocator<SPComplexType>;

  // type defs
  using pointer              = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer        = typename std::allocator_traits<Allocator>::const_pointer;
  using Ipointer             = typename std::allocator_traits<IAllocator>::pointer;
  using pointer_shared       = typename std::allocator_traits<Allocator_shared>::pointer;
  using const_pointer_shared = typename std::allocator_traits<Allocator_shared>::const_pointer;

  using IVector       = boost::multi::array<int, 1, IAllocator>;
  using CMatrix       = boost::multi::array<ComplexType, 2, Allocator>;
  using SPCMatrix     = boost::multi::array<SPComplexType, 2, SPAllocator>;
  using CMatrix_ref   = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CMatrix_cref  = boost::multi::array_ref<const ComplexType, 2, const_pointer>;
  using shmCVector    = boost::multi::array<ComplexType, 1, Allocator_shared>;
  using shmCMatrix    = boost::multi::array<ComplexType, 2, Allocator_shared>;
  using shmC3Tensor   = boost::multi::array<ComplexType, 3, Allocator_shared>;
  using shmSPCMatrix  = boost::multi::array<SPComplexType, 2, SPAllocator_shared>;
  using index_aos     = ma::sparse::array_of_sequences<int, int, shared_allocator<int>, ma::sparse::is_root>;

  using mpi3CVector = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3CMatrix = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;

  using buffer_alloc_type       = DeviceBufferManager::template allocator_t<ComplexType>;
  using Ibuffer_alloc_type      = DeviceBufferManager::template allocator_t<int>;
  using shm_buffer_alloc_type   = LocalTGBufferManager::template allocator_t<ComplexType>;
  using shm_Ibuffer_alloc_type  = LocalTGBufferManager::template allocator_t<int>;
  using shm_Lbuffer_alloc_type  = LocalTGBufferManager::template allocator_t<long>;
  using shm_SPbuffer_alloc_type = LocalTGBufferManager::template allocator_t<SPComplexType>;

  using StaticIVector  = boost::multi::static_array<int, 1, Ibuffer_alloc_type>;
  using StaticVector  = boost::multi::static_array<ComplexType, 1, buffer_alloc_type>;
  using StaticMatrix  = boost::multi::static_array<ComplexType, 2, buffer_alloc_type>;
  using Static3Tensor = boost::multi::static_array<ComplexType, 3, buffer_alloc_type>;

  using StaticSHMIVector = boost::multi::static_array<int, 1, shm_Ibuffer_alloc_type>;
  using StaticSHMLVector = boost::multi::static_array<long, 1, shm_Lbuffer_alloc_type>;
  using StaticSHMVector = boost::multi::static_array<ComplexType, 1, shm_buffer_alloc_type>;
  using StaticSHMMatrix = boost::multi::static_array<ComplexType, 2, shm_buffer_alloc_type>;
  using StaticSHM3Tensor = boost::multi::static_array<ComplexType, 3, shm_buffer_alloc_type>;
  using StaticSHM4Tensor = boost::multi::static_array<ComplexType, 4, shm_buffer_alloc_type>;

  using StaticSHMSPMatrix = boost::multi::static_array<SPComplexType, 2, shm_SPbuffer_alloc_type>;
  using StaticSHMSP3Tensor = boost::multi::static_array<SPComplexType, 3, shm_SPbuffer_alloc_type>;

public:
  template<class csrMat, class PH_EXCIT>
  PHMSD(AFQMCInfo& info,
        ptree pt_in,
        afqmc::TaskGroup_& tg_,
        SlaterDetOperations&& sdet_,
        HamiltonianOperations<MP>&& hop_,
        std::map<int, int>&& acta2mo_,
        std::map<int, int>&& actb2mo_,
        PH_EXCIT&& abij_,
        std::vector<csrMat>&& op_spin_det_coupling_,
        std::vector<PsiT_Matrix>&& orbs_,
        WALKER_TYPES wlk,
        ComplexType nce,
        [[maybe_unused]] int targetNW = 1)
      : AFQMCInfo(info),
        TG(tg_),
        SDetOp(std::move(sdet_)),
        buffer_manager(), 
        shm_buffer_manager(),
        HamOp(std::move(hop_)),
        acta2mo(std::move(acta2mo_)),
        actb2mo(std::move(actb2mo_)),
        abij(std::move(abij_)),
        refc_dev(iextensions<1u>{NAEA+NAEB}),
        OpSpinDetCouplings(move_vector<local_csr_Matrix<ComplexType>>(std::move(op_spin_det_coupling_))),
        OpSpinDetCouplings_sp(make_vector<local_csr_Matrix<SPComplexType>>(OpSpinDetCouplings,
                                make_node_allocator<SPComplexType>(TG))),
        OrbMats(move_vector<local_csr_Matrix<ComplexType>>(std::move(orbs_))),
        RefOrbMats({0, 0}, shared_allocator<ComplexType>{TG.Node()}),
        number_of_references(-1),
        walker_type(wlk),
        NuclearCoulombEnergy(nce),
        maxn_unique_confg(std::max(abij.number_of_unique_excitations()[0], abij.number_of_unique_excitations()[1])),
        maxnactive(std::max(OrbMats[0].size(0), OrbMats[1].size(0))),
        max_exct_n(std::max(abij.maximum_excitation_number()[0], abij.maximum_excitation_number()[1]))
  {
    /* To me, PHMSD is not compatible with walker_type=CLOSED unless
       * the MSD expansion is symmetric with respect to spin. For this, 
       * it is better to write a specialized class that assumes either spin symmetry
       * or e.g. Perfect Pairing.
       */
    if (walker_type == CLOSED)
      APP_ABORT("Error: PHMSD requires walker_type != CLOSED.");

    if (walker_type == NONCOLLINEAR)
      APP_ABORT("PHMSD has not yet been implemented for NONCOLLINEAR walkers.");

    if (walker_type == FULLYPOLARIZED)
      APP_ABORT("PHMSD has not yet been implemented for FULLYPOLARIZED walkers.");


    // setup device structures
    using std::copy_n;
    copy_n(abij.reference_configuration(), NAEA+NAEB, refc_dev.origin());

    compact_G_for_vbias     = true;
    transposed_G_for_vbias_ = HamOp.transposed_G_for_vbias();
    transposed_G_for_E_     = HamOp.transposed_G_for_E();
    transposed_vHS_         = HamOp.transposed_vHS();

    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nPHMSD input:\n{}\n",io::to_string(pt));
    // initialize using verbose input
    nbatch = pt.get<int>("nbatch");
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
      for(int i=0; i<NAEA; i++)
        if( refc[i] != i ) 
          APP_ABORT(" Error: PHMSD algorithm=1 requires refc[i]==i.\n\n");
      for(int i=0; i<NAEB; i++)
        if( refc[NAEA+i] != i ) 
          APP_ABORT(" Error: PHMSD algorithm=1 requires refc[i]==i.\n\n");
    }
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    int nbatch_default    = ((number_of_devices() > 0) ? -1 : 1);
    int nbatch    = pt0.get<int>("nbatch", nbatch_default);
    int number_of_references = pt0.get<int>("number_of_references", -1);
    // validate inputs
    if ((omp_get_num_threads() > 1) && (nbatch == 0))
    {
      app_warning(" WARNING!!!: Found OMP_NUM_THREADS > 1 with nbatch=0.");
      app_warning("             This will lead to low performance. Set nbatch. ");
    }
    // create verbose internal inputs
    ptree pt1;
    pt1.put("nbatch", nbatch);
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

  PHMSD(PHMSD const& other) = delete;
  PHMSD& operator=(PHMSD const& other) = delete;
  PHMSD(PHMSD&& other)                 = default;
  PHMSD& operator=(PHMSD&& other) = delete;

  int local_number_of_cholesky_vectors() const { return HamOp.local_number_of_cholesky_vectors(); }
  int global_number_of_cholesky_vectors() const { return HamOp.global_number_of_cholesky_vectors(); }
  int global_origin_cholesky_vector() const { return HamOp.global_origin_cholesky_vector(); }
  bool distribution_over_cholesky_vectors() const { return HamOp.distribution_over_cholesky_vectors(); }
  bool spin_dependent_vHS() const { return HamOp.spin_dependent_vHS(); };

  int size_of_G_for_vbias() const { return dm_size(!compact_G_for_vbias); }

  bool transposed_G_for_vbias() const { return transposed_G_for_vbias_; }
  bool transposed_G_for_E() const { return transposed_G_for_E_; }
  bool transposed_vHS() const { return transposed_vHS_; }
  WALKER_TYPES getWalkerType() const { return walker_type; }

  template<class Vec>
  void vMF(Vec&& v, double dt);

  template<class Mat>
  void G_MF(Mat&& G);

  SlaterDetOperations* getSlaterDetOperations() { return std::addressof(SDetOp); }
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    HamOp.generalizedFockMatrix(std::forward<Args>(args)...);
    TG.TG_local().barrier();
  }

  HamiltonianTypes getHamType()
  {
    return HamOp.getHamType();
  }

  template<class... Args>
  void getFieldTypes(Args&&... args)
  {
    HamOp.getFieldTypes(std::forward<Args>(args)...);
  }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    HamOp.update_potentials(std::forward<Args>(args)...);
  }

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
     * local contribution to vbias for the Green functions in G 
     * G: [size_of_G_for_vbias()][nW]
     * v: [local # Chol. Vectors][nW]
     */
  template<class MatG, class MatA>
  void vbias(const MatG& G, MatA&& v, double dt, double a = 1.0)
  {
    if (transposed_G_for_vbias_)
    {
      RUNTIME_CHECK(G.size(0) == v.size(1), "");
      RUNTIME_CHECK(G.size(1) == size_of_G_for_vbias(), "");
    }
    else
    {
      RUNTIME_CHECK(G.size(0) == size_of_G_for_vbias(), "");
      RUNTIME_CHECK(G.size(1) == v.size(1), "");
    }
    RUNTIME_CHECK(v.size(0) == HamOp.local_number_of_cholesky_vectors(), "");
    HamOp.vbias(G, std::forward<MatA>(v), dt, a);
    TG.TG_local().barrier();
  }

  /*
     * local contribution to vHS for the Green functions in G 
     * X: [# chol vecs][nW]
     * v: [NMO^2][nW] / [nW]NMO^2] depending on layout
     * For spin dependent interactions: 
     * v: [2][NMO^2][nW] / [2][nW]NMO^2] depending on layout
     * Dimensionality of v determines the assumed spin dependency (or lack of)
     */
  template<class MatX, class MatA>
  void vHS(MatX&& X, MatA&& v, double dt, double a = 1.0)
  {
    RUNTIME_CHECK(X.size(0) == HamOp.local_number_of_cholesky_vectors(), "");
    int nspin = (spin_dependent_vHS()?2:1);
    if (transposed_vHS_)
      RUNTIME_CHECK(X.size(1)*nspin == v.size(0), "");
    else
      RUNTIME_CHECK(X.size(1)*nspin == v.size(1), "");
    HamOp.vHS(std::forward<MatX>(X), std::forward<MatA>(v), dt, a);
    TG.TG_local().barrier();
  }

  /*
     * Calculates the local energy and overlaps of all the walkers in the set and stores
     * them in the wset data
     */
  template<class WlkSet>
  void Energy(WlkSet& wset)
  {
    int nw = wset.size();
    StaticVector ovlp(iextensions<1u>{nw}, 
                      buffer_manager.get_generator().template get_allocator<ComplexType>());
    StaticMatrix eloc({nw, 3}, 
                      buffer_manager.get_generator().template get_allocator<ComplexType>());
    Energy(wset, eloc, ovlp);
    TG.TG_local().barrier();
    if (TG.getLocalTGRank() == 0)
    {
      wset.setProperty(OVLP, ovlp);
      wset.setProperty(E1_, eloc(eloc.extension(), 0));
      wset.setProperty(EXX_, eloc(eloc.extension(), 1));
      wset.setProperty(EJ_, eloc(eloc.extension(), 2));
    }
    TG.TG_local().barrier();
  }

  /*
     * Calculates the local energy and overlaps of all the walkers in the set and 
     * returns them in the appropriate data structures
     */
  template<class WlkSet, class Mat, class TVec>
  void Energy(const WlkSet& wset, Mat&& E, TVec&& Ov)
  {
    if (TG.getNGroupsPerTG() > 1)
      Energy_distributed(wset, std::forward<Mat>(E), std::forward<TVec>(Ov));
    else
      Energy_shared(wset, std::forward<Mat>(E), std::forward<TVec>(Ov));
  }

  /*
     * Calculates the mixed density matrix for all walkers in the walker set. 
     * Options:
     *  - compact:   If true (default), returns compact form with Dim: [NEL*NMO], 
     *                 otherwise returns full form with Dim: [NMO*NMO]. 
     *  - transpose: If false (default), returns standard form with Dim: [XXX][nW]
     *                 otherwise returns the transpose with Dim: [nW][XXX}
     */
  template<class WlkSet, class MatG>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, bool compact = true, bool transpose = false)
  {
    int nw = wset.size();
    StaticVector ovlp(iextensions<1u>{nw}, 
                      buffer_manager.get_generator().template get_allocator<ComplexType>());
    MixedDensityMatrix(wset, std::forward<MatG>(G), ovlp, compact, transpose);
  }

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true, 
                          bool transpose = false)
  {  
    int nw = wset.size();
    int nspins  = (walker_type==COLLINEAR?2:1);
    StaticSHM3Tensor local_ov({nspins, maxn_unique_confg, nw},
                      shm_buffer_manager.get_generator().template get_allocator<ComplexType>());
    MixedDensityMatrix_impl(wset,G,Ov,local_ov,compact,transpose);
  }

  /*
     * Calculates the density matrix with respect to a given Reference
     * for all walkers in the walker set. 
     */
  template<class WlkSet, class MatA, class MatB, class MatG, class TVec>
  void DensityMatrix(const WlkSet& wset,
                     MatA&& RefA,
                     MatB&& RefB,
                     MatG&& G,
                     TVec&& Ov,
                     bool herm,
                     bool compact,
                     bool transposed)
  {
    /*
      if(nbatch != 0)
        DensityMatrix_batched(wset,std::forward<MatA>(RefA),std::forward<MatB>(RefB),
                                std::forward<MatG>(G),std::forward<TVec>(Ov),
                                herm,compact,transposed);
      else
*/
    DensityMatrix_shared(wset, std::forward<MatA>(RefA), std::forward<MatB>(RefB), std::forward<MatG>(G),
                         std::forward<TVec>(Ov), herm, compact, transposed);
  }

  /*
     * Calculates the mixed density matrix for all walkers in the walker set
     *   with a format consistent with (and expected by) the vbias routine.
     * This is implementation dependent, so this density matrix should ONLY be used
     * in conjunction with vbias. 
     */
  template<class WlkSet, class MatG>
  void MixedDensityMatrix_for_vbias(const WlkSet& wset, MatG&& G)
  {
    int nw = wset.size();
    StaticVector ovlp(iextensions<1u>{nw}, 
                      buffer_manager.get_generator().template get_allocator<ComplexType>());
    MixedDensityMatrix(wset, std::forward<MatG>(G), ovlp, compact_G_for_vbias, transposed_G_for_vbias_);
  }

  /*
     * Calculates the overlaps of all walkers in the set. Returns values in arrays. 
     */
  template<class WlkSet, class TVec>
  void Overlap(const WlkSet& wset, TVec&& Ov);

  /*
     * Calculates the overlaps of all walkers in the set. Updates values in wset. 
     */
  template<class WlkSet>
  void Overlap(WlkSet& wset)
  {
    int nw = wset.size();
    StaticVector ovlp(iextensions<1u>{nw}, 
                      buffer_manager.get_generator().template get_allocator<ComplexType>());
    Overlap(wset, ovlp);
    TG.TG_local().barrier();
    if (TG.getLocalTGRank() == 0)
    {
      wset.setProperty(OVLP, ovlp);
    }
    TG.TG_local().barrier();
  }

  template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, TVec& wgt,
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
        Mat1 const& X, Mat2 const& Y, Mat3 const& M, bool time_evolved, bool importanceSampling);

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

  ComplexType getReferenceWeight(int i) const { return std::get<2>(*abij.configuration(i)); }

  /*
   * Returns the reference Slater Matrices needed for back propagation.  
   */
  template<class Mat, class Ptr = ComplexType*>
  void getReferencesForBackPropagation(Mat&& A)
  {
    static_assert(std::decay<Mat>::type::dimensionality == 2, "Wrong dimensionality");
    int ndet = number_of_references_for_back_propagation();
    RUNTIME_CHECK(A.size(0) == ndet, "");
    if (RefOrbMats.size(0) == 0)
    {
      TG.Node().barrier(); // for safety
      int nrow(NMO * ((walker_type == NONCOLLINEAR) ? 2 : 1));
      int ncol(NAEA + NAEB); //careful here, spins are stored contiguously
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
        std::vector<int> Ac(NAEA);
        std::vector<int> Bc(NAEB);
        for (int i_det = 0; i_det < ndet; ++i_det)
        {
          auto c=abij.configuration(i_det);
          abij.get_configuration(0, std::get<0>(*c), Ac);
          abij.get_configuration(1, std::get<1>(*c), Bc);
          boost::multi::array_ref<ComplexType, 2> A_(raw_pointer_cast(RefOrbMats[i_det].origin()), {NMO, NAEA});
          boost::multi::array_ref<ComplexType, 2> B_(A_.origin() + A_.num_elements(), {NMO, NAEB});
          for (int i = 0, ia = 0; i < NMO; ++i)
            for (int a = 0; a < NAEA; ++a, ia++)
              A_[i][a] = OA_[i][Ac[a]];
          if (OrbMats.size() > 1)
          {
            for (int i = 0, ia = 0; i < NMO; ++i)
              for (int a = 0; a < NAEB; ++a, ia++)
                B_[i][a] = OB_[i][Bc[a]];
          }
          else if(walker_type == COLLINEAR)
          {
            for (int i = 0, ia = 0; i < NMO; ++i)
              for (int a = 0; a < NAEB; ++a, ia++)
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
  }

protected:
  TaskGroup_& TG;

  //SlaterDetOperations_shared<ComplexType> SDetOp;
  SlaterDetOperations SDetOp;

  DeviceBufferManager buffer_manager;
  LocalTGBufferManager shm_buffer_manager;

  HamiltonianOperations<MP> HamOp;

  // MAM: use enum when this is settled...
  // 0: loop over unique configurations, calculate G and evaluate E from scratch
  // 1: use ph_reference_energy and ph_excited_energy, which requires compact R matrix
  // 2: calculate Fapbq and call ph_energy_Fapbq
  int energy_algorithm = 0;

  // number of walkers in batched Ov, DM, etc...
  int nbatch;

  std::map<int, int> acta2mo;
  std::map<int, int> actb2mo;

  ph_excitations<int, ComplexType, shared_allocator<int>, ma::sparse::is_root, 
                 device_allocator<int>> abij;
  IVector refc_dev;

  // sparse matrix with opposite spin determinant couplings
  // unfortunately, need to keep 2 copies in case mixed precision is used.
  // not sure how to avoid this, so just do it, :-(
  std::vector<local_csr_Matrix<ComplexType>> OpSpinDetCouplings;
  std::vector<local_csr_Matrix<SPComplexType>> OpSpinDetCouplings_sp;

  // eventually switched from CMatrix to SMHSparseMatrix(node)
  std::vector<local_csr_Matrix<ComplexType>> OrbMats;
  mpi3CMatrix RefOrbMats;
  int number_of_references;

  // in both cases below: closed_shell=0, UHF/ROHF=1, GHF=2
  WALKER_TYPES walker_type;

  bool compact_G_for_vbias;

  // in the 3 cases, true means [nwalk][...], false means [...][nwalk]
  bool transposed_G_for_vbias_;
  bool transposed_G_for_E_;
  bool transposed_vHS_;

  ComplexType NuclearCoulombEnergy;

  // shared memory arrays for temporary calculations
  size_t maxn_unique_confg; // maximum number of unque configurations
  size_t maxnactive;        // maximum number of states in active space
  size_t max_exct_n;        // maximum excitation number (number of electrons excited simultaneously)

  /*
     * Calculates the local energy and overlaps of all the walkers in the set and 
     * returns them in the appropriate data structures
     */
  template<class WlkSet, class Mat, class TVec>
  void Energy_shared(const WlkSet& wset, Mat&& E, TVec&& Ov)
  {
    if(energy_algorithm==0)
      energy_shared_alg0(wset,E,Ov);
    else if(energy_algorithm==1)
      energy_shared_alg1(wset,E,Ov);
    else if(energy_algorithm==2)
      energy_shared_alg2(wset,E,Ov);
    else
      APP_ABORT(" Error: Unknown energy_algorithm. \n\n"); 
  }

  /*
     * Calculates the local energy and overlaps of all the walkers in the set and 
     * returns them in the appropriate data structures
     */
  template<class WlkSet, class Mat, class TVec>
  void Energy_distributed(const WlkSet& wset, Mat&& E, TVec&& Ov);

  /* Implementation of various energy evaluation algorithms. */
  template<class WlkSet, class Mat, class TVec>
  void energy_shared_alg0(const WlkSet& wset, Mat&& E, TVec&& Ov);

  template<class WlkSet, class Mat, class TVec>
  void energy_shared_alg1(const WlkSet& wset, Mat&& E, TVec&& Ov);

  template<class WlkSet, class Mat, class TVec>
  void energy_shared_alg2(const WlkSet& wset, Mat&& E, TVec&& Ov);

  /* 
     * Computes the density matrix with respect to a given reference. 
     * Intended to be used in combination with the energy evaluation routine.
     * G and Ov are expected to be in shared memory.
     */
  template<class WlkSet, class MatA, class MatB, class MatG, class TVec>
  void DensityMatrix_shared(const WlkSet& wset,
                            MatA&& RefsA,
                            MatB&& RefsB,
                            MatG&& G,
                            TVec&& Ov,
                            bool herm,
                            bool compact,
                            bool transposed);

/*
    template<class WlkSet, class MatA, class MatB, class MatG, class TVec>
    void DensityMatrix_batched(const WlkSet& wset, MatA&& RefsA, MatB&& RefsB, MatG&& G,
                                TVec&& Ov, bool herm, bool compact, bool transposed);
*/

  template<class WlkSet, class MatG, class TVec, class MatOv>
  void MixedDensityMatrix_impl(const WlkSet& wset,
                          MatG&& G,
                          TVec&& Ov,
                          MatOv&& ovlps,
                          bool compact = true,
                          bool transpose = false);

  int dm_size(bool full) const
  {
    switch (walker_type)
    {
    case CLOSED: // closed-shell RHF
      return (full) ? (NMO * NMO) : (OrbMats[0].size(0) * NMO);
      break;
    case COLLINEAR:
      return (full) ? (2 * NMO * NMO) : ((OrbMats[0].size(0) + OrbMats[1].size(0)) * NMO);
      break;
    case NONCOLLINEAR:
      return (full) ? (4 * NMO * NMO) : ((OrbMats[0].size(0)) * 2 * NMO);
      break;
    default:
      APP_ABORT(" Error: Unknown walker_type in dm_size. ");
      return -1;
    }
  }
  // dimensions for each component of the DM.
  std::pair<int, int> dm_dims(bool full, SpinTypes sp = Alpha) const
  {
    using arr = std::pair<int, int>;
    switch (walker_type)
    {
    case CLOSED: // closed-shell RHF
      return (full) ? (arr{NMO, NMO}) : (arr{OrbMats[0].size(0), NMO});
      break;
    case COLLINEAR:
      return (full) ? (arr{NMO, NMO})
                    : ((sp == Alpha) ? (arr{OrbMats[0].size(0), NMO}) : (arr{OrbMats[1].size(0), NMO}));
      break;
    case NONCOLLINEAR:
      return (full) ? (arr{2 * NMO, 2 * NMO}) : (arr{OrbMats[0].size(0), 2 * NMO});
      break;
    default:
      APP_ABORT(" Error: Unknown walker_type in dm_size. ");
      return arr{-1, -1};
    }
  }
  std::pair<int, int> dm_dims_ref(bool full, SpinTypes sp = Alpha) const
  {
    using arr = std::pair<int, int>;
    switch (walker_type)
    {
    case CLOSED: // closed-shell RHF
      return (full) ? (arr{NMO, NMO}) : (arr{NAEA, NMO});
      break;
    case COLLINEAR:
      return (full) ? (arr{NMO, NMO}) : ((sp == Alpha) ? (arr{NAEA, NMO}) : (arr{NAEB, NMO}));
      break;
    case NONCOLLINEAR:
      return (full) ? (arr{2 * NMO, 2 * NMO}) : (arr{NAEA, 2 * NMO});
      break;
    default:
      APP_ABORT(" Error: Unknown walker_type in dm_size. ");
      return arr{-1, -1};
    }
  }
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Wavefunctions/PHMSD.icc"

#endif
