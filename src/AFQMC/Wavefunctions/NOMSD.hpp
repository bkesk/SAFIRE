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

#ifndef SFQMC_AFQMC_NOMSD_H
#define SFQMC_AFQMC_NOMSD_H

#include <vector>
//#include <map>
#include <string>
//#include <iostream>
#include <tuple>

#include "io/ptree/ptree_utilities.hpp"
//#include "AFQMC/Utilities/readWfn.h"
#include "AFQMC/config.h"
#include "mpi3/shm/mutex.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"
//#include "Memory/buffer_managers.h"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.hpp"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"

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
template<bool MP, class devPsiT>
class NOMSD : public AFQMCInfo
{
  // Note:
  // if number_of_devices > 0, nextra should always be 0,
  // so code doesn't need to be portable in places guarded by if(nextra>0)

  // allocators
  using Allocator        = device_allocator<ComplexType>;
  using Allocator_shared = localTG_allocator<ComplexType>;

  // type defs
  using pointer              = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer        = typename std::allocator_traits<Allocator>::const_pointer;
  using pointer_shared       = typename std::allocator_traits<Allocator_shared>::pointer;
  using const_pointer_shared = typename std::allocator_traits<Allocator_shared>::const_pointer;

  using buffer_alloc_type     = DeviceBufferManager::template allocator_t<ComplexType>;
  using shm_buffer_alloc_type = LocalTGBufferManager::template allocator_t<ComplexType>;

  using CVector      = boost::multi::array<ComplexType, 1, Allocator>;
  using CMatrix      = boost::multi::array<ComplexType, 2, Allocator>;
  using CTensor      = boost::multi::array<ComplexType, 3, Allocator>;
  using CVector_ref  = boost::multi::array_ref<ComplexType, 1, pointer>;
  using CMatrix_ref  = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CMatrix_ptr  = boost::multi::array_ptr<ComplexType, 2, pointer>;
  using CMatrix_cref = boost::multi::array_ref<const ComplexType, 2, const_pointer>;
  using CTensor_ref  = boost::multi::array_ref<ComplexType, 3, pointer>;
  using CTensor_cref = boost::multi::array_ref<const ComplexType, 3, const_pointer>;
  using shmCVector   = boost::multi::array<ComplexType, 1, Allocator_shared>;
  using shmCMatrix   = boost::multi::array<ComplexType, 2, Allocator_shared>;
  using shared_mutex = boost::mpi3::shm::mutex;

  using stdCVector  = boost::multi::array<ComplexType, 1>;
  using stdCMatrix  = boost::multi::array<ComplexType, 2>;
  using stdCTensor  = boost::multi::array<ComplexType, 3>;
  using mpi3CVector = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3CMatrix = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using mpi3CTensor = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;

  using stdCMatrix_ref = boost::multi::array_ref<ComplexType, 2>;

  using StaticVector  = boost::multi::static_array<ComplexType, 1, buffer_alloc_type>;
  using StaticMatrix  = boost::multi::static_array<ComplexType, 2, buffer_alloc_type>;
  using Static3Tensor = boost::multi::static_array<ComplexType, 3, buffer_alloc_type>;

  using StaticSHMVector = boost::multi::static_array<ComplexType, 1, shm_buffer_alloc_type>;
  using StaticSHMMatrix = boost::multi::static_array<ComplexType, 2, shm_buffer_alloc_type>;
  using StaticSHM3Tensor = boost::multi::static_array<ComplexType, 3, shm_buffer_alloc_type>;
  using StaticSHM4Tensor = boost::multi::static_array<ComplexType, 4, shm_buffer_alloc_type>;

public:
  template<class MType>
  NOMSD(AFQMCInfo& info,
        ptree pt_in,
        afqmc::TaskGroup_& tg_,
        SlaterDetOperations&& sdet_,
        HamiltonianOperations<MP>&& hop_,
        std::vector<ComplexType>&& ci_,
        std::vector<MType>&& orbs_,
        WALKER_TYPES wlk,
        ComplexType nce,
        [[maybe_unused]] int targetNW = 1)
      : AFQMCInfo(info),
        TG(tg_),
        alloc_(), // right now device_allocator is default constructible
        alloc_shared_(make_localTG_allocator<ComplexType>(TG)),
        buffer_manager(),
        shm_buffer_manager(),
        host_buffer_manager(),
        SDetOp(std::move(sdet_)),
        HamOp(std::move(hop_)),
        ci(std::move(ci_)),
        OrbMats(move_vector<devPsiT>(std::move(orbs_))),
        RefOrbMats({0, 0}, shared_allocator<ComplexType>{TG.Node()}),
        mutex(std::make_unique<shared_mutex>(TG.TG_local())),
        walker_type(wlk),
        nspins((walker_type == COLLINEAR) ? (2) : (1)),
        number_of_references(-1),
        NuclearCoulombEnergy(nce),
        last_number_extra_tasks(-1),
        last_task_index(-1),
        local_group_comm(),
        shmbuff_for_G(nullptr)
  {
    compact_G_for_vbias     = (ci.size() == 1); // this should be input, since it is determined by HOps
    transposed_G_for_vbias_ = HamOp.transposed_G_for_vbias();
    transposed_G_for_E_     = HamOp.transposed_G_for_E();
    transposed_vHS_         = HamOp.transposed_vHS();

    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nNOMSD input:\n{}\n",io::to_string(pt));
    // initialize using verbose input
    bool rediag;
    nbatch = pt.get<int>("nbatch");
    number_of_references = pt.get<int>("number_of_references");
    rediag = pt.get<bool>("rediag");

    if (rediag)
      recompute_ci();

  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    bool rediag; 
    int number_of_references, nbatch; 
    int nbatch_default    = ((number_of_devices() > 0) ? -1 : 0);
    nbatch    = pt0.get<int>("nbatch", nbatch_default);
    number_of_references = pt0.get<int>("number_of_references", -1);
    rediag         = pt0.get<bool>("rediag", false);
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
    pt1.put("rediag", rediag);
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "name",
      "ndets_to_read",
      "restart_file",
      "filename"
    };
    io::compare_known_keys("Non-orthogonal multi-Slater det. (NOMSD) Wavefunction",pt1, pt0,pass_through_keys);
    return pt1;
  }

  ~NOMSD() {}

  NOMSD(NOMSD const& other) = delete;
  NOMSD& operator=(NOMSD const& other) = delete;
  NOMSD(NOMSD&& other)                 = default;
  NOMSD& operator=(NOMSD&& other) = delete;

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
  //multi::array<ComplexType, 2> getOneBodyPropagatorMatrix(Args&&... args)
  {
    return HamOp.getOneBodyPropagatorMatrix(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto vHS_sparse(Args&&... args)
  //std::tuple<dev_csr_Matrix<ComplexType> const*, dev_csr_Matrix<ComplexType> const*> vHS_sparse(Args&&... args)
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
    // special case:
    if( (HamOp.getHamType() != ModelHamiltonian) and
        (walker_type == COLLINEAR)  and
        (ci.size() > 1) ) {
      // expects either alpha or beta in spin independent HamOps, so must be called twice
      if (transposed_G_for_vbias_)
      {
        HamOp.vbias(G(G.extension(0) ,{0, NMO * NMO}), std::forward<MatA>(v), dt, a, 0.0);
        HamOp.vbias(G(G.extension(0) ,{NMO * NMO, 2 * NMO * NMO} ), std::forward<MatA>(v), dt, a, 1.0);
      } else {
        HamOp.vbias(G.sliced(0, NMO * NMO), std::forward<MatA>(v), dt, a, 0.0);
        HamOp.vbias(G.sliced(NMO * NMO, 2 * NMO * NMO), std::forward<MatA>(v), dt, a, 1.0);
      }
    } else {
      HamOp.vbias(G, std::forward<MatA>(v), dt, a);
    }
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
  void vHS(MatX&& X, MatA&& v, double dt, double a = 1.0 )
  {
    RUNTIME_CHECK(X.size(0) == HamOp.local_number_of_cholesky_vectors(), "");
    [[maybe_unused]] int nspin = (spin_dependent_vHS()?2:1);
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
    StaticVector ovlp(iextensions<1u>{nw}, buffer_manager.get_generator().template get_allocator<ComplexType>());
    MixedDensityMatrix(wset, std::forward<MatG>(G), ovlp, compact, transpose);
  }

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true, bool transpose = false)
  {
    if (nbatch != 0) {
      if( ci.size() > 1 ) {
        MixedDensityMatrix_batched(wset, std::forward<MatG>(G), std::forward<TVec>(Ov), 
                                   compact, transpose);
      } else {
        int ispin = (walker_type == COLLINEAR ? 1 : 0);
        DensityMatrix_batched(wset, OrbMats[0], OrbMats[ispin], std::forward<MatG>(G), 
                              std::forward<TVec>(Ov), true, compact, transpose);
      }
    } else {
      MixedDensityMatrix_shared(wset, std::forward<MatG>(G), std::forward<TVec>(Ov), 
                                compact, transpose);
    }
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
    if (nbatch != 0)
      DensityMatrix_batched(wset, std::forward<MatA>(RefA), std::forward<MatB>(RefB), std::forward<MatG>(G),
                            std::forward<TVec>(Ov), herm, compact, transposed);
    else
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
    StaticVector ovlp(iextensions<1u>{nw}, buffer_manager.get_generator().template get_allocator<ComplexType>());
    MixedDensityMatrix(wset, std::forward<MatG>(G), ovlp, compact_G_for_vbias, transposed_G_for_vbias_);
  }

  /*
     * Calculates the overlaps of all walkers in the set. Returns values in arrays. 
     */
  template<class WlkSet, class TVec>
  void Overlap(const WlkSet& wset, TVec&& Ov)
  {
    if (nbatch != 0) {
      if(ci.size() > 1) {
        Overlap_batched(wset, std::forward<TVec>(Ov));
      } else {
        Overlap_batched_single_det(wset, std::forward<TVec>(Ov));
      }
    } else {
      Overlap_shared(wset, std::forward<TVec>(Ov));
    }
  }

  /*
     * Calculates the overlaps of all walkers in the set. Updates values in wset. 
     */
  template<class WlkSet>
  void Overlap(WlkSet& wset)
  {
    int nw = wset.size();
    StaticVector ovlp(iextensions<1u>{nw}, buffer_manager.get_generator().template get_allocator<ComplexType>());
    Overlap(wset, ovlp);
    TG.TG_local().barrier();
    if (TG.getLocalTGRank() == 0)
    {
      wset.setProperty(OVLP, ovlp);
    }
    TG.TG_local().barrier();
  }

  template<class... Args>
  void accumulate_estimators(Args&&... args)
  {
    if(ci.size()>1)
      accumulate_estimators_general_impl(std::forward<Args>(args)...);
    else
      accumulate_estimators_single_ref_impl(std::forward<Args>(args)...);    
  }

  /*
     * Returns the number of reference Slater Matrices needed for back propagation.  
     */
  int number_of_references_for_back_propagation() const
  {
    if (number_of_references > 0)
      return number_of_references;
    else
      return ((walker_type == COLLINEAR) ? OrbMats.size() / 2 : OrbMats.size());
  }

  ComplexType getReferenceWeight(int i) const { return ci[i]; }

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
      int ncol(NAEA + ((walker_type == CLOSED) ? 0 : NAEB)); //careful here, spins are stored contiguously
      RefOrbMats = mpi3CMatrix({ndet, nrow * ncol}, RefOrbMats.get_allocator());
      TG.Node().barrier(); // for safety
      if (TG.Node().root())
      {
        if (walker_type != COLLINEAR)
        {
          for (int i = 0; i < ndet; ++i)
          {
            boost::multi::array_ref<ComplexType, 2> A_(raw_pointer_cast(RefOrbMats[i].origin()), {nrow, ncol});
            ma::Matrix2MAREF('H', OrbMats[i], A_);
          }
        }
        else
        {
          for (int i = 0; i < ndet; ++i)
          {
            boost::multi::array_ref<ComplexType, 2> A_(raw_pointer_cast(RefOrbMats[i].origin()), {NMO, NAEA});
            ma::Matrix2MAREF('H', OrbMats[2 * i], A_);
            boost::multi::array_ref<ComplexType, 2> B_(A_.origin() + A_.num_elements(), {NMO, NAEB});
            ma::Matrix2MAREF('H', OrbMats[2 * i + 1], B_);
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

  Allocator alloc_;
  Allocator_shared alloc_shared_;

  DeviceBufferManager buffer_manager;
  LocalTGBufferManager shm_buffer_manager;
  HostBufferManager host_buffer_manager;

  //SlaterDetOperations_shared<ComplexType> SDetOp;
  SlaterDetOperations SDetOp;

  HamiltonianOperations<MP> HamOp;

  std::vector<ComplexType> ci;

  // eventually switched from CMatrix to SMHSparseMatrix(node)
  std::vector<devPsiT> OrbMats;
  mpi3CMatrix RefOrbMats;

  std::unique_ptr<shared_mutex> mutex;

  // in both cases below: closed_shell=0, UHF/ROHF=1, GHF=2
  WALKER_TYPES walker_type;
  int nspins;

  int number_of_references;

  int nbatch;

  bool compact_G_for_vbias;

  // in the 3 cases, true means [nwalk][...], false means [...][nwalk]
  bool transposed_G_for_vbias_;
  bool transposed_G_for_E_;
  bool transposed_vHS_;

  ComplexType NuclearCoulombEnergy;

  // not elegant, but reasonable for now
  int last_number_extra_tasks;
  int last_task_index;

  // shared_communicator for parallel work within TG_local()
  //std::unique_ptr<shared_communicator> local_group_comm;
  shared_communicator local_group_comm;
  std::unique_ptr<mpi3CVector> shmbuff_for_G;

  void recompute_ci();

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

  template<class WlkSet, class MatA, class MatB, class MatG, class TVec>
  void DensityMatrix_batched(const WlkSet& wset,
                             MatA&& RefsA,
                             MatB&& RefsB,
                             MatG&& G,
                             TVec&& Ov,
                             bool herm,
                             bool compact,
                             bool transposed);

  template<class MatSM, class MatG, class TVec>
  void MixedDensityMatrix_for_E_from_SM(const MatSM& SM, MatG&& G, TVec&& Ov, int nd, double LogOverlapFactor);

  /*
     * Calculates the local energy and overlaps of all the walkers in the set and 
     * returns them in the appropriate data structures
     */
  template<class WlkSet, class Mat, class TVec>
  void Energy_shared(const WlkSet& wset, Mat&& E, TVec&& Ov);

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix_shared(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true, bool transpose = false);

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix_batched(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true, bool transpose = false);

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix_batched_single_det(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true, bool transpose = false);

  template<class WlkSet, class TVec>
  void Overlap_shared(const WlkSet& wset, TVec&& Ov);

  template<class WlkSet, class TVec>
  void Overlap_batched(const WlkSet& wset, TVec&& Ov);

  template<class WlkSet, class TVec>
  void Overlap_batched_single_det(const WlkSet& wset, TVec&& Ov);

  /*
     * Calculates the local energy and overlaps of all the walkers in the set and 
     * returns them in the appropriate data structures
     */
  template<class WlkSet, class Mat, class TVec>
  void Energy_distributed(const WlkSet& wset, Mat&& E, TVec&& Ov)
  {
    if (ci.size() == 1)
      Energy_distributed_singleDet(wset, std::forward<Mat>(E), std::forward<TVec>(Ov));
    else
      Energy_distributed_multiDet(wset, std::forward<Mat>(E), std::forward<TVec>(Ov));
  }

  template<class WlkSet, class Mat, class TVec>
  void Energy_distributed_singleDet(const WlkSet& wset, Mat&& E, TVec&& Ov);

  template<class WlkSet, class Mat, class TVec>
  void Energy_distributed_multiDet(const WlkSet& wset, Mat&& E, TVec&& Ov);

  template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3, class Observable>
  void accumulate_estimators_general_impl(int iav, WlkSet& wset, TVec& wgt, 
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
        Mat1 const& X, Mat2 const& Y, Mat3 const& M, bool time_evolved, bool importanceSampling);

  template<class WlkSet, class TVec, class Mat1, class Mat2, class Mat3, class Observable>
  void accumulate_estimators_single_ref_impl(int iav, WlkSet& wset, TVec& wgt, 
        std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
        Mat1 const& X, Mat2 const& Y, Mat3 const& M, bool time_evolved, bool importanceSampling);

  int dm_size(bool full) const
  {
    switch (walker_type)
    {
    case CLOSED: // closed-shell RHF
      return (full) ? (NMO * NMO) : (NAEA * NMO);
      break;
    case COLLINEAR:
      return (full) ? (2 * NMO * NMO) : ((NAEA + NAEB) * NMO);
      break;
    case NONCOLLINEAR:
      return (full) ? (4 * NMO * NMO) : (NAEA * 2 * NMO);
      break;
    case FULLYPOLARIZED:
      return (full) ? (NMO * NMO) : (NAEA * NMO);
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
      return (full) ? (arr{NMO, NMO}) : (arr{NAEA, NMO});
      break;
    case COLLINEAR:
      return (full) ? (arr{NMO, NMO}) : ((sp == Alpha) ? (arr{NAEA, NMO}) : (arr{NAEB, NMO}));
      break;
    case NONCOLLINEAR:
      return (full) ? (arr{2 * NMO, 2 * NMO}) : (arr{NAEA, 2 * NMO});
      break;
        case FULLYPOLARIZED:
      return (full) ? (arr{NMO, NMO}) : (arr{NAEA, NMO});
      break;
    default:
      APP_ABORT(" Error: Unknown walker_type in dm_size. ");
      return arr{-1, -1};
    }
  }
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Wavefunctions/NOMSD.icc"

#endif
