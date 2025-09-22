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

#ifndef SFQMC_AFQMC_BASEPROPAGATOR_H
#define SFQMC_AFQMC_BASEPROPAGATOR_H

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <tuple>

#include "hdf/hdf_archive.h"
#include "io/ptree/ptree_utilities.hpp"
#include "Utilities/Random.hpp"
#include "Utilities/app_loggers.h"

#include "AFQMC/config.h"
#include "Memory/buffer_managers.h"
#include "Memory/device_rng.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"
#include "AFQMC/Propagators/generate1BodyPropagator.hpp"

#include "AFQMC/Wavefunctions/Wavefunction.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Base class for AFQMC propagators.
 */
template<bool SP>
class AFQMCBasePropagator : public AFQMCInfo
{
protected:
  using SPRealType = typename to_working_precision<SP,RealType>::type;
  using SPComplexType = typename to_working_precision<SP,ComplexType>::type;  

  // allocator for local memory
  using allocator = device_allocator<ComplexType>;
  using pointer   = device_ptr<ComplexType>;
  // allocator for memory shared by all cores in working local group
  using aux_allocator = localTG_allocator<ComplexType>;
  using sp_pointer    = typename std::allocator_traits<device_allocator<SPComplexType>>::pointer;

  using stack_alloc_type   = LocalTGBufferManager::template allocator_t<ComplexType>;
  using stack_alloc_SPtype = LocalTGBufferManager::template allocator_t<SPComplexType>;
  using dev_stack_alloc_type   = DeviceBufferManager::template allocator_t<ComplexType>;

  using StaticVector    = boost::multi::static_array<ComplexType, 1, stack_alloc_type>;
  using StaticMatrix    = boost::multi::static_array<ComplexType, 2, stack_alloc_type>;
  using Static3Tensor   = boost::multi::static_array<ComplexType, 3, stack_alloc_type>;
  using StaticSPVector  = boost::multi::static_array<SPComplexType, 1, stack_alloc_SPtype>;
  using StaticSPMatrix  = boost::multi::static_array<SPComplexType, 2, stack_alloc_SPtype>;
  using StaticSP3Tensor = boost::multi::static_array<SPComplexType, 3, stack_alloc_SPtype>;

  using CVector          = boost::multi::array<ComplexType, 1, allocator>;
  using CMatrix          = boost::multi::array<ComplexType, 2, allocator>;
  using C3Tensor         = boost::multi::array<ComplexType, 3, allocator>;
  using CVector_ref      = boost::multi::array_ref<ComplexType, 1, pointer>;
  using CMatrix_ref      = boost::multi::array_ref<ComplexType, 2, pointer>;
  using SPCMatrix_ref    = boost::multi::array_ref<SPComplexType, 2, sp_pointer>;
  using C3Tensor_ref     = boost::multi::array_ref<ComplexType, 3, pointer>;
  using C4Tensor_ref     = boost::multi::array_ref<ComplexType, 4, pointer>;
  using sharedCVector    = ComplexVector<aux_allocator>;
  using stdCVector       = boost::multi::array<ComplexType, 1>;
  using stdCMatrix       = boost::multi::array<ComplexType, 2>;
  using stdC3Tensor      = boost::multi::array<ComplexType, 3>;
  using stdCVector_ref   = boost::multi::array_ref<ComplexType, 1>;
  using stdSPCVector_ref = boost::multi::array_ref<SPComplexType, 1>;
  using stdCMatrix_ref   = boost::multi::array_ref<ComplexType, 2>;
  using stdSPCMatrix_ref = boost::multi::array_ref<SPComplexType, 2>;
  using stdC3Tensor_ref  = boost::multi::array_ref<ComplexType, 3>;
  using CMatrix_ptr      = boost::multi::array_ptr<ComplexType, 2, pointer>;

  using node3CTensor   = boost::multi::array<ComplexType, 3, node_allocator<ComplexType>>;

  using mpi3CVector   = boost::multi::array<ComplexType, 1, shared_allocator<ComplexType>>;
  using mpi3SPCVector = boost::multi::array<SPComplexType, 1, shared_allocator<SPComplexType>>;
  using mpi3CMatrix   = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using mpi3CTensor   = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;

public:
  AFQMCBasePropagator(AFQMCInfo& info,
                      ptree pt_in,
                      afqmc::TaskGroup_& tg_,
                      Wavefunction& wfn_,
                      utils::DeviceRandomGenerator_t* r)
      : AFQMCInfo(info),
        TG(tg_),
        buffer_manager(),
        device_buffer_manager(),
        alloc_(),
        aux_alloc_(make_localTG_allocator<ComplexType>(TG)),
        wfn(wfn_),
        H1ext({2, 1, 1}, shared_allocator<ComplexType>{TG.Node()}),
        P1d({0,0,0},make_node_allocator<ComplexType>(TG)),
        P1d_inv({0,0,0},make_node_allocator<ComplexType>(TG)),
        vMF(iextensions<1u>{wfn.local_number_of_cholesky_vectors()}),
        rng(r),
        rng_block_size(vMF.size(0)),  // by default, set to nCV
        SDetOp(wfn.getSlaterDetOperations()),
        local_group_comm(),
        old_dt(-123456.789),
        last_nextra(-1),
        last_task_index(-1),
        order(6),
        external_H1(false)
  {
    P1s.reserve(2);
    P1s.emplace_back(P1Type(tp_ul_ul{0, 0}, tp_ul_ul{0, 0}, 0, aux_alloc_));
    P1s.emplace_back(P1Type(tp_ul_ul{0, 0}, tp_ul_ul{0, 0}, 0, aux_alloc_));
    P1s_inv.reserve(2);
    P1s_inv.emplace_back(P1Type(tp_ul_ul{0, 0}, tp_ul_ul{0, 0}, 0, aux_alloc_));
    P1s_inv.emplace_back(P1Type(tp_ul_ul{0, 0}, tp_ul_ul{0, 0}, 0, aux_alloc_));
    transposed_vHS_ = wfn.transposed_vHS();
    transposed_G_   = wfn.transposed_G_for_vbias();
    nspins_in_vHS      = (wfn.spin_dependent_vHS() ? 2 : 1);
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nBasePropagator input:\n\n{}\n",io::to_string(pt));
    // initialize using verbose input
    int i_, a_;
    std::string external_field, excited_file;
    double external_field_scale;
    nbatched_propagation = pt.get<int>("nbatch");
    nbatched_qr = pt.get<int>("nbatch_qr");
    i_ = pt.get<int>("i");
    a_ = pt.get<int>("a");
    vbias_bound          = pt.get<double>("vbias_bound");
    external_field_scale = pt.get<double>("external_field_scale");
    upper_cutoff_scale = pt.get<double>("upper_cutoff_scale");
    lower_cutoff_scale = pt.get<double>("lower_cutoff_scale");
    apply_constrain     = pt.get<bool>("apply_constrain");
    importance_sampling = pt.get<bool>("importance_sampling");
    substractMF          = pt.get<bool>("substractMF");
    hybrid              = pt.get<bool>("hybrid");
    external_field      = pt.get<std::string>("external_field");
    printP1eV           = pt.get<bool>("printP1eigval");
    free_projection     = pt.get<bool>("free_projection");
    denseP1             = pt.get<bool>("denseP1");
    excited_file        = pt.get<std::string>("excited");
    debug_verbosity     = pt.get<bool>("debug_verbosity");
    // checks that can only be performed upon instantiation
    if ((number_of_devices() > 0) && NMO < 2048 && NAEA < 1024 && nbatched_qr != -1)
    {
      app_error("system too small to benefit from batched QR on GPU");
      APP_ABORT("BasePropagator: set nbatch_qr to -1");
    }
    auto hamtype(wfn.getHamType());
    if ((hamtype == KPFactorized || hamtype == KPTHC) && denseP1)
    {
      app_error("dense Ham. with kpoints");
      APP_ABORT("BasePropagator: set denseP1 to false");
    }

    app_log(1,"\n\n --------------- Constructing Propagator ------------------ \n");

    if (nbatched_propagation != 0)
      app_log(1," Using batched propagation with a batch size: {}",nbatched_propagation);
    else
      app_log(1," Using sequential propagation. ");
    app_log(1," vbias_bound: {}", vbias_bound); 
    if(denseP1)
      app_log(1," Using dense 1-body propagator");
    else
      app_log(1," Using sparse 1-body propagator");

    if(nspins_in_vHS>1) 
      app_log(1, " Using a spin-dependent vHS.");

    if (!importance_sampling && !free_projection)
      app_log(1," WARNING: importance_sampling=no without free projection does not make sense. ");

    if (hybrid)
      app_log(1," Using hybrid method to calculate the weights during the propagation.");
    else
      app_log(1," Using local energy method to calculate the weights during the propagation.");

    if (debug_verbosity)
    {
      app_log(1,"[WARNING] Using debug verbosity. THIS WILL GENERATE A LOT OF OUTPUT.");
      app_log(1,"Intended for debugging purposes with a few walkers only.");
    }

    // read orbital matrix if excited state propagator
    excitedState = false;
    if (excited_file != "" && i_ >= 0 && a_ >= 0)
    {
      if (i_ < NMO && a_ < NMO)
      {
        if (i_ >= NAEA || a_ < NAEA)
          APP_ABORT(" Errors: Inconsistent excited orbitals for alpha electrons. ");
        excitedState        = true;
        maxOccupExtendedMat = {a_, NAEB};
        numExcitations      = {1, 0};
        excitations.push_back({i_, a_});
      }
      else if (i_ >= NMO && a_ >= NMO)
      {
        if (i_ >= NMO + NAEB || a_ < NMO + NAEB)
          APP_ABORT(" Errors: Inconsistent excited orbitals for beta electrons. ");
        excitedState        = true;
        maxOccupExtendedMat = {NAEA, a_ - NMO};
        numExcitations      = {0, 1};
        excitations.push_back({i_ - NMO, a_ - NMO});
      }
      else
      {
        APP_ABORT(" Errors: Inconsistent excited orbitals. ");
      }
      stdC3Tensor excitedOrbMat_;
      APP_ABORT(" Error: Finish implementation. ");	
      // read from hdf5
      //readWfn(excited_file, excitedOrbMat_, NMO, maxOccupExtendedMat.first, maxOccupExtendedMat.second);
      excitedOrbMat = excitedOrbMat_;
    }
    // MAM: hardcoded for only collinear structure, generalize later to all cases
    //      or simply eliminate! 
    if (external_field != std::string(""))
    {
      //    read_external_field(H1ext);

      external_H1 = true;
      H1ext = mpi3CTensor({2, NMO, NMO}, ComplexType(0.0), shared_allocator<ComplexType>{TG.Node()});
      TG.Node().barrier();
      if (TG.Node().root())
      {
        std::ifstream in(external_field.c_str());
        for (int i = 0; i < NMO; i++)
          for (int j = 0; j < NMO; j++)
            in >> H1ext[0][i][j];
        for (int i = 0; i < NMO; i++)
          for (int j = 0; j < NMO; j++)
            in >> H1ext[1][i][j];
        if (in.fail())
          APP_ABORT(" Error: Problems with external field.");
        ma::scal(ComplexType(external_field_scale), H1ext[0]);
        ma::scal(ComplexType(external_field_scale), H1ext[1]);
      }
      TG.Node().barrier();
    }
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    int nbatch_default    = ((number_of_devices() > 0) ? -1 : 0);
    int nbatch_qr_default = ((number_of_devices() > 0) ? -1 : 0);
    int nbatch    = pt0.get<int>("nbatch", nbatch_default);
    int nbatch_qr = pt0.get<int>("nbatch_qr", nbatch_qr_default);
    int i_ = pt0.get<int>("i", -1);
    int a_ = pt0.get<int>("a", -1);
    double vbias_bound = pt0.get<double>("vbias_bound", 50.0);
    double external_field_scale = pt0.get<double>("external_field_scale", 1.0);
    double upper_cutoff_scale = pt0.get<double>("upper_cutoff_scale", 10.0);
    double lower_cutoff_scale = pt0.get<double>("lower_cutoff_scale", 1.0);
    bool apply_constrain        = pt0.get<bool>("apply_constrain", true);
    bool importance_sampling    = pt0.get<bool>("importance_sampling", true);
    bool substractMF            = pt0.get<bool>("substractMF", true);
    bool hybrid                 = pt0.get<bool>("hybrid", true);
    bool printP1eigval          = pt0.get<bool>("printP1eigval", false);
    bool free_projection        = pt0.get<bool>("free_projection", false);
    bool denseP1                = pt0.get<bool>("denseP1", false);
    bool debug_verbosity        = pt0.get<bool>("debug_verbosity", false);
    std::string external_field  = pt0.get<std::string>("external_field", "");
    std::string excited_file    = pt0.get<std::string>("excited", "");
    // validate inputs
    if ((omp_get_num_threads() > 1) && (nbatch == 0))
    {
      app_warning(" WARNING!!!: Found OMP_NUM_THREADS > 1 with nbatch=0.");
      app_warning("             This will lead to low performance. Set nbatch. ");
    }
    if (free_projection)
    {
      if (importance_sampling || !hybrid || apply_constrain)
      {
        app_error("Free projection requires:");
        app_error(" importance_sampling = no, currently {}", importance_sampling);
        app_error(" hybrid = yes, currently {}", hybrid);
        app_error(" apply_constrain = no, currently {}", apply_constrain); 
        APP_ABORT("BasePropagator: free_projection");
      }
    }
    // create verbose internal inputs
    ptree pt1;
    pt1.put("nbatch", nbatch);
    pt1.put("nbatch_qr", nbatch_qr);
    pt1.put("i", i_);
    pt1.put("a", a_);
    pt1.put("vbias_bound", vbias_bound);
    pt1.put("external_field_scale", external_field_scale);
    pt1.put("upper_cutoff_scale", upper_cutoff_scale);
    pt1.put("lower_cutoff_scale", lower_cutoff_scale);
    pt1.put("apply_constrain", apply_constrain);
    pt1.put("importance_sampling", importance_sampling);
    pt1.put("substractMF", substractMF);
    pt1.put("hybrid", hybrid);
    pt1.put("printP1eigval", printP1eigval);
    pt1.put("free_projection", free_projection);
    pt1.put("denseP1", denseP1);
    pt1.put("external_field", external_field);
    pt1.put("excited", excited_file);
    pt1.put("debug_verbosity", debug_verbosity);
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "name",
      "debug"
    };
    io::compare_known_keys("Propagator",pt1, pt0,pass_through_keys);
    return pt1;
  }

  ~AFQMCBasePropagator() {}

  AFQMCBasePropagator(AFQMCBasePropagator const& other) = delete;
  AFQMCBasePropagator& operator=(AFQMCBasePropagator const& other) = delete;
  AFQMCBasePropagator(AFQMCBasePropagator&& other)                 = default;
  AFQMCBasePropagator& operator=(AFQMCBasePropagator&& other) = delete;

  template<class WlkSet>
  void Propagate(int steps, WlkSet& wset, RealType E1, RealType dt, int fix_bias = 1)
  {
    int nblk   = steps / fix_bias;
    int nextra = steps % fix_bias;
    for (int i = 0; i < nblk; i++)
    {
      step(fix_bias, wset, E1, dt);
      // MAM: I need to update buffer generators here, otherwise the
      //      first block would be quite slow
      update_memory_managers();
    }
    if (nextra > 0)
      step(nextra, wset, E1, dt);
    TG.TG_local().barrier();
  }

  template<class WlkSet, class CTens, class CMat>
  void BackPropagate(int steps, int nStabalize, WlkSet& wset, CTens&& Refs, CMat&& logdetR);

  template<class WlkSet, class Mat1, class Mat2, class Mat3> 
  void PropagateOperators(int steps, WlkSet& wset,  Mat1&& X, Mat2&& Y, Mat3&& M);

  bool hybrid_propagation() { return hybrid; }

  bool free_propagation() { return free_projection; }

  int global_number_of_cholesky_vectors() const { return wfn.global_number_of_cholesky_vectors(); }

  // constructs the 1-body hamiltonian for propagation and generates the propagator
  // if Pinv = true, the routine builds the invere of the propagator and stores it in P_inv
  void generateP1(double dt, WALKER_TYPES walker_type, bool Pinv = false);

  template<class WlkSet>
  void Orthogonalize(WlkSet& wset);

  void set_rng_block_size(int sz) { rng_block_size = sz; }

protected:
  TaskGroup_& TG;

  LocalTGBufferManager buffer_manager;
  DeviceBufferManager device_buffer_manager;

  allocator alloc_;

  aux_allocator aux_alloc_;

  Wavefunction& wfn;

  mpi3CTensor H1ext;

  // 1Body propagator in sparse and dense forms
  bool denseP1 = false;
  std::vector<P1Type> P1s;
  node3CTensor P1d;

  // used to propagate operator orbitals  
  std::vector<P1Type> P1s_inv;
  node3CTensor P1d_inv;

  CVector vMF;

  utils::DeviceRandomGenerator_t* rng;

  // number of random numbers to generate for each walker at each step.
  // In general, rng_block_size will be set to the number of cholesky vectors.
  // In correlated sampling calculations, this will be the maximum number of cholesky
  // vectors in all systems, which is needed to keep the generators synchronized. 
  int rng_block_size = 0;

  SlaterDetOperations* SDetOp;

  shared_communicator local_group_comm;

  RealType old_dt;
  int last_nextra = 0;
  int last_task_index = 0;
  int order = 0;
  int nbatched_propagation = 1;
  int nbatched_qr = 1;
  bool external_H1 = false; 
  bool printP1eV = false;

  RealType vbias_bound;
  bool substractMF = true;

  // type of propagation
  bool free_projection = false;
  bool hybrid = true;
  bool importance_sampling = true;
  bool apply_constrain = true;
  double upper_cutoff_scale = 10.0;
  double lower_cutoff_scale = 1.0;

  bool transposed_vHS_ = false;
  bool transposed_G_ = false;
  int nspins_in_vHS = 1;

  bool debug_verbosity = false;

  // used in propagator step
  // intead of doing this, should use TBuff to transpose vHS4D and only have propagation
  // with vHD3D[nwalk*nsteps,...]
  CMatrix local_vHS;
  CMatrix local_vHS2;

  CVector new_overlaps;
  CMatrix new_energies;

  CMatrix MFfactor;
  CMatrix hybrid_weight;

  boost::multi::array<ComplexType, 2> work;

  // excited state propagator
  bool excitedState = false;
  std::vector<std::pair<int, int>> excitations;
  C3Tensor excitedOrbMat;
  CMatrix extendedMatAlpha;
  CMatrix extendedMatBeta;
  std::pair<int, int> maxOccupExtendedMat;
  std::pair<int, int> numExcitations;

  template<class WlkSet>
  void step(int steps, WlkSet& wset, RealType E1, RealType dt);

  template<class MatA, class MatB, class MatC, class MatD>
  void assemble_X(size_t nsteps,
                  size_t nwalk,
                  RealType sqrtdt,
                  MatA&& X,
                  MatB&& vbias,
                  MatC&& MF,
                  MatD&& HWs,
                  bool addRAND = true);

  void reset_nextra(int nextra);

  template<class WSet>
  void apply_propagators(char TA, WSet& wset, int ni, int tk0, int tkN, int ntask_total_serial, C4Tensor_ref& vHS4D);

  template<class Mat1, class Mat2>
  void apply_propagators_batched_impl(char TA, WALKER_TYPES walker_type, int ni,
                Mat1&& Pup, Mat1&& Pdn, Mat2&& SMup, Mat2&& SMdn, C4Tensor_ref& vHS4D);

  template<class Mat>
  void apply_propagators_batched(char TA, WALKER_TYPES walker_type, int ni,
                Mat&& SMup, Mat&& SMdn, C4Tensor_ref& vHS4D, bool P1inv = false) 
  {
    int nspin = (walker_type == COLLINEAR ? 2 : 1);
    if(P1inv) {
      if(denseP1) {
        apply_propagators_batched_impl(TA,walker_type,ni,P1d_inv[0],P1d_inv[nspin-1],
                std::forward<Mat>(SMup),std::forward<Mat>(SMdn),vHS4D);
      } else {
        apply_propagators_batched_impl(TA,walker_type,ni,P1s_inv[0],P1s_inv[nspin-1],
                std::forward<Mat>(SMup),std::forward<Mat>(SMdn),vHS4D);
      }
    } else {
      if(denseP1) {
        apply_propagators_batched_impl(TA,walker_type,ni,P1d[0],P1d[nspin-1],
                std::forward<Mat>(SMup),std::forward<Mat>(SMdn),vHS4D);
      } else {
        apply_propagators_batched_impl(TA,walker_type,ni,P1s[0],P1s[nspin-1],
                std::forward<Mat>(SMup),std::forward<Mat>(SMdn),vHS4D);
      }
    }
  }

  template<typename T>
  T apply_bound_vbias(T v, RealType sqrtdt)
  {
    // explict caste to avoid compiler warnings when T is std::complex<float>.
    return (std::abs(v) > std::abs(static_cast<T>(static_cast<SPRealType>(vbias_bound * sqrtdt))))
        ? (v / (std::abs(v) / static_cast<T>(static_cast<SPRealType>(vbias_bound * sqrtdt))))
        : (v);
  }

  /* Note: Orthogonalize_impl accumulates ln(det(R)) in logdetR, 
   *       since this is needed for back propagation  
   */
  template<class WlkSet, class CMat>
  void Orthogonalize_impl(WlkSet& wset, CMat&& logdetR);

  template<class WlkSet>
  void Orthogonalize_impl(WlkSet& wset);

  template<class WlkSet>
  void Orthogonalize_excited_impl(WlkSet& wset);

};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMCBasePropagator.icc"

#endif
