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

#ifndef SFQMC_AFQMC_MODELPROPAGATOR_H
#define SFQMC_AFQMC_MODELPROPAGATOR_H

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <tuple>

#include "hdf/hdf_archive.h"
#include "Utilities/Random.hpp"

#include "AFQMC/config.h"
#include "Memory/buffer_managers.h"
#include "Memory/device_rng.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"
#include "AFQMC/Propagators/generate1BodyPropagator.hpp"

#include "AFQMC/Wavefunctions/Wavefunction.hpp"

// MAM: Too many things are shared between the ab-initio and model hamiltonians
//      Make a "true" base class and derive everything from it. 
//      Define abinitio and model propagators based on the new base.

namespace sfqmc
{
namespace afqmc
{
/*
 * AFQMC propagator for model hamiltonians.
 * This propagator assumes a spin dependent HS potential.
 * Both sparse and dense HS potentials (will be...) are implemented. 
 *
 */
template<bool SP>
class AFQMCModelPropagator : public AFQMCInfo
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
  using Rallocator = device_allocator<RealType>;
  using Rpointer   = device_ptr<RealType>;
  using Iallocator = device_allocator<int>;

  using stack_alloc_type   = LocalTGBufferManager::template allocator_t<ComplexType>;
  using stack_alloc_SPtype = LocalTGBufferManager::template allocator_t<SPComplexType>;
  using dev_stack_alloc_type   = DeviceBufferManager::template allocator_t<ComplexType>;

  using StaticCVector   = StaticVector_<stack_alloc_type>;
  using StaticCMatrix   = StaticMatrix_<stack_alloc_type>;
  using Static3Tensor   = StaticArray_<3, stack_alloc_type>;
  using StaticSPVector  = StaticVector_<stack_alloc_SPtype>;
  using StaticSPMatrix  = StaticMatrix_<stack_alloc_SPtype>;
  using StaticSP3Tensor = StaticArray_<3, stack_alloc_SPtype>;

  using CVector          = Vector_<allocator>;
  using RVector          = Vector_<Rallocator>;
  using IVector          = Vector_<Iallocator>;
  using CMatrix          = Matrix_<allocator>;
  using C3Tensor         = Array_<3, allocator>;
  using CVector_ref      = Vector_ref_<pointer>;
  using CMatrix_ref      = Matrix_ref_<pointer>;
  using SPCMatrix_ref    = Matrix_ref_<sp_pointer>;
  using C3Tensor_ref     = Array_ref_<3, pointer>;
  using SPC3Tensor_ref    = Array_ref_<3, sp_pointer>;
  using C4Tensor_ref     = Array_ref_<4, pointer>;
  using sharedCVector    = Vector_<aux_allocator>;
  using CMatrix_ptr      = multi::array_ptr<ComplexType, 2, pointer>;

  using node3CTensor   = Array_<3, node_allocator<ComplexType>>;
  using mpi3CTensor   = Array_<3, shared_allocator<ComplexType>>;

  using csrMat = dev_csr_Matrix<ComplexType>;

public:
  template< class IVec>
  AFQMCModelPropagator(AFQMCInfo& info,
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
        vMF(iextensions<1u>{wfn.local_number_of_cholesky_vectors()}),
        FieldTypes(wfn.getFieldTypes(FieldTypes)),
        rng(r),
        uniformRNptr(nullptr),
	rng_block_size(vMF.size(0)),
        SDetOp(wfn.getSlaterDetOperations()),
        local_group_comm(),
        old_dt(-123456.789),
        last_nextra(-1),
        last_task_index(-1),
        order(6),
        external_H1(false)
  {
    if(TG.TG_local().size() > 1)
      APP_ABORT(" Error: ncores>1 not allowed in AFQMCModelPropagator.");

    P1s.reserve(2);
    P1s.emplace_back(P1Type(tp_ul_ul{0, 0}, tp_ul_ul{0, 0}, 0, aux_alloc_));
    P1s.emplace_back(P1Type(tp_ul_ul{0, 0}, tp_ul_ul{0, 0}, 0, aux_alloc_));
    transposed_vHS_ = wfn.transposed_vHS();
    transposed_G_   = wfn.transposed_G_for_vbias();
    // convert user input to verbose input
    ptree pt = interpret_inputs(pt_in);
    app_log(2,"\nModelPropagator input:\n{}\n",io::to_string(pt));
    // initialize using verbose input
    nbatched_propagation = pt.get<int>("nbatch");
    nbatched_qr = pt.get<int>("nbatch_qr");
    auto i_ = pt.get<int>("i");
    auto a_ = pt.get<int>("a");
    vbias_bound          = pt.get<double>("vbias_bound");
    auto external_field_scale = pt.get<double>("external_field_scale");
    upper_cutoff_scale = pt.get<double>("upper_cutoff_scale");
    lower_cutoff_scale = pt.get<double>("lower_cutoff_scale");
    apply_constrain     = pt.get<bool>("apply_constrain");
    use_cp_constraint   = pt.get<bool>("use_cp_constraint");
    use_real_vbias      = pt.get<bool>("use_real_vbias");
    importance_sampling = pt.get<bool>("importance_sampling");
    substractMF          = pt.get<bool>("substractMF");
    hybrid              = pt.get<bool>("hybrid");
    auto external_field      = pt.get<std::string>("external_field");
    printP1eV           = pt.get<bool>("printP1eigval");
    free_projection     = pt.get<bool>("free_projection");
    denseP1             = pt.get<bool>("denseP1");
    auto excited_file        = pt.get<std::string>("excited");
    order		= pt.get<int>("order");
    natural_shift       = pt.get<bool>("natural_shift");
    symmetric_split     = pt.get<bool>("symmetric_split");
    debug_verbosity     = pt.get<bool>("debug_verbosity");
    // checks that can only be performed upon instantiation
    if (NMO >= 128 && denseP1)
    {
      // heuristic (guessing) for now
      // should really be determined by the fraction of non-zero elements
      // also easy to test and decide during setup, since P1 is fixed
      //denseP1 = (NMO < 128); 
      app_log(1,"system large enough to benefit from sparse 1-body propagator");
      APP_ABORT("ModelPropagator: set denseP1 to false");
    }
    app_log(1,"\n\n --------------- Constructing Propagator ------------------ \n");

    if (nbatched_propagation != 0)
      app_log(1," Using batched propagation with a batch size: {}", nbatched_propagation);
    else
      app_log(1," Using sequential propagation. ");
    app_log(1," vbias_bound: {}", vbias_bound);
    if(denseP1)
      app_log(1," Using dense 1-body propagator");
    else
      app_log(1," Using sparse 1-body propagator");

    if (!importance_sampling && !free_projection)
      app_log(1," WARNING: importance_sampling=no without free projection does not make sense. ");

    if (hybrid)
      app_log(1," Using hybrid method to calculate the weights during the propagation.");
    else
      app_log(1," Using local energy method to calculate the weights during the propagation.");
    if(natural_shift)
      app_log(1, "Using natural shifts with discrete propagators. ");
    if(not symmetric_split)
      app_log(1, "Not using symmetric split of walker weight update.");

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
      Array<ComplexType, 3> excitedOrbMat_;
      APP_ABORT(" Error: Finish implementation. ");
      //readWfn(excited_file, excitedOrbMat_, NMO, maxOccupExtendedMat.first, maxOccupExtendedMat.second);
      excitedOrbMat = excitedOrbMat_;
    }
    // MAM: hardcoded for only collinear structure, generalize later to all cases
    //      or simply eliminate! 
    if (external_field != std::string(""))
    {
      //    read_external_field(H1ext);

      external_H1 = true;
      H1ext.reextent({2, NMO, NMO});
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
    int nbatch_default    = -1; // use batched by default even on GPU
    int nbatch_qr_default = ((number_of_devices() > 0) ? -1 : 0);
    auto nbatch    = pt0.get<int>("nbatch", nbatch_default);
    auto nbatch_qr = pt0.get<int>("nbatch_qr", nbatch_qr_default);
    auto vbias_bound = pt0.get<double>("vbias_bound", 100.0);
    // excited state parameters (i, a, excited*)
    auto i_ = pt0.get<int>("i", -1);
    auto a_ = pt0.get<int>("a", -1);
    auto external_field_scale = pt0.get<double>("external_field_scale", 1.0);
    auto upper_cutoff_scale  = pt0.get<double>("upper_cutoff_scale", 50.0); 
    auto lower_cutoff_scale  = pt0.get<double>("lower_cutoff_scale", 50.0); 
    auto apply_constrain     = pt0.get<bool>("apply_constrain", true);
    auto use_cp_constraint   = pt0.get<bool>("use_cp_constraint", false);
    auto use_real_vbias      = pt0.get<bool>("use_real_vbias", false);
    auto importance_sampling = pt0.get<bool>("importance_sampling", true);
    auto substractMF         = pt0.get<bool>("substractMF", true);
    auto hybrid              = pt0.get<bool>("hybrid", true);
    auto printP1eigval       = pt0.get<bool>("printP1eigval", false);
    auto free_projection     = pt0.get<bool>("free_projection", false);
    auto denseP1             = pt0.get<bool>("denseP1", false);
    auto denseP2             = pt0.get<bool>("denseP2", false); 
    auto order		     = pt0.get<int>("order", 6);
    auto excited_file   = pt0.get<std::string>("excited", "");
    auto external_field = pt0.get<std::string>("external_field", "");
    auto natural_shift       = pt0.get<bool>("natural_shift",true);
    auto symmetric_split     = pt0.get<bool>("symmetric_split",false);
    auto debug_verbosity     = pt0.get<bool>("debug_verbosity", false);
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
    pt1.put("use_cp_constraint", use_cp_constraint);
    pt1.put("use_real_vbias", use_real_vbias);
    pt1.put("importance_sampling", importance_sampling);
    pt1.put("substractMF", substractMF);
    pt1.put("hybrid", hybrid);
    pt1.put("external_field", external_field);
    pt1.put("printP1eigval", printP1eigval);
    pt1.put("free_projection", free_projection);
    pt1.put("denseP1", denseP1);
    pt1.put("denseP2", denseP2);
    pt1.put("excited", excited_file);
    pt1.put("order", order);
    pt1.put("natural_shift",natural_shift);
    pt1.put("symmetric_split",symmetric_split);
    pt1.put("debug_verbosity", debug_verbosity);
    std::unordered_set<std::string> pass_through_keys = {
      "system",
      "name",
      "debug"
    };
    io::compare_known_keys("Lattice model propagator",pt1, pt0,pass_through_keys);
    return pt1;
  }

  ~AFQMCModelPropagator() {}

  AFQMCModelPropagator(AFQMCModelPropagator const& other) = delete;
  AFQMCModelPropagator& operator=(AFQMCModelPropagator const& other) = delete;
  AFQMCModelPropagator(AFQMCModelPropagator&& other)                 = default;
  AFQMCModelPropagator& operator=(AFQMCModelPropagator&& other) = delete;

  template<class WlkSet>
  void Propagate(int steps, WlkSet& wset, RealType E1, RealType dt, int fix_bias = 1)
  {
    // sparse P2 does not allow fix_bias>1
    if(not denseP2 and fix_bias > 1) {
      if(fix_bias_comment) {
	fix_bias_comment=false;
        app_warning(" fix_bias > 1 not allowed with Model Hamiltonians combined with denseP2=false. Setting fix_bias to 1.");
      }
      fix_bias=1;
    }
    int nblk   = steps / fix_bias;
    int nextra = steps % fix_bias;
    for (int i = 0; i < nblk; i++)
    {
      if(denseP2) step_denseP2 (fix_bias, wset, E1, dt);
      else        step_sparseP2(fix_bias, wset, E1, dt);
      // MAM: I need to update buffer generators here, otherwise the
      //      first block would be quite slow
      update_memory_managers();
    }
    if (nextra > 0) {
      if(denseP2) step_denseP2 (nextra, wset, E1, dt);
      else        step_sparseP2(nextra, wset, E1, dt);
    }
    TG.TG_local().barrier();
  }

  template<class WlkSet, class CTens, class CMat>
  void BackPropagate(int steps, int nStabalize, WlkSet& wset, CTens&& Refs, CMat&& logdetR)
  {
    if(denseP2) BackPropagate_denseP2(steps,nStabalize,wset,std::forward<CTens>(Refs), std::forward<CMat>(logdetR));
    else BackPropagate_sparseP2(steps,nStabalize,wset,std::forward<CTens>(Refs), std::forward<CMat>(logdetR));
  }

  template<class WlkSet, class Mat1, class Mat2, class Mat3>
  void PropagateOperators(int steps, WlkSet& wset,  Mat1&& X, Mat2&& Y, Mat3&& M);

  bool hybrid_propagation() { return hybrid; }

  bool free_propagation() { return free_projection; }

  int global_number_of_cholesky_vectors() const { return wfn.global_number_of_cholesky_vectors(); }

  // in case P1 needs to exist before call to Propagate is executed
  void generateP1(double dt, WALKER_TYPES walker_type);

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
  std::vector<P1Type> P1s;
  node3CTensor P1d;

  CVector vMF;

  bool natural_shift = true;
  bool symmetric_split = false;

// This should really go in Node memory (or TGlocal memory if distributing fields)
// to avoid unnecessary copies
  IVector FieldTypes; 

  utils::DeviceRandomGenerator_t* rng;
  RVector uniformRNbuffer;  
  Rpointer uniformRNptr;  

  // number of random numbers to generate for each walker at each step.
  // In general, rng_block_size will be set to the number of cholesky vectors.
  // In correlated sampling calculations, this will be the maximum number of cholesky
  // vectors in all systems, which is needed to keep the generators synchronized. 
  int rng_block_size=0;

  SlaterDetOperations* SDetOp;

  shared_communicator local_group_comm;

  RealType old_dt = 0.0;
  int last_nextra = 0;
  int last_task_index = 0;
  int order = 0;
  int nbatched_propagation = 1;
  int nbatched_qr = 1;
  bool external_H1 = false; 
  bool printP1eV = false;
  double upper_cutoff_scale = 50.0;
  double lower_cutoff_scale = 50.0;

  RealType vbias_bound = 100.0;
  bool substractMF = true;

  // type of propagation
  bool free_projection = false;
  bool hybrid = true;
  bool importance_sampling = true;
  bool apply_constrain = true;
  bool use_cp_constraint = false;
  bool use_real_vbias = false;
  bool denseP1 = false;
  bool denseP2 = false;
  bool fix_bias_comment = true;

  bool transposed_vHS_ = false;
  bool transposed_G_ = false;

  bool debug_verbosity = false;

  // used in propagator step
  // intead of doing this, should use TBuff to transpose vHS3D and only have propagation
  // with vHD3D[nwalk*nsteps,...]
  CMatrix local_vHS;

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
  void step_denseP2(int steps, WlkSet& wset, RealType E1, RealType dt);

  template<class WlkSet>
  void step_sparseP2(int steps, WlkSet& wset, RealType E1, RealType dt);

  template<class WlkSet, class CTens, class CMat>
  void BackPropagate_denseP2(int steps, int nStabalize, WlkSet& wset, CTens&& Refs, CMat&& logdetR);

  template<class WlkSet, class CTens, class CMat>
  void BackPropagate_sparseP2(int steps, int nStabalize, WlkSet& wset, CTens&& Refs, CMat&& logdetR);

  template<class MatA, class MatB, class MatC, class MatD>
  void assemble_X(size_t nsteps,
                  size_t nwalk,
                  RealType sqrtdt,
                  MatA&& X,
                  MatB&& vbias,
                  MatC&& MF,
                  MatD&& HWs,
                  bool addRAND = true);

  template<class WSet>
  void apply_propagators(char TA, WSet& wset, int ni, C4Tensor_ref& v);

  template<class WSet>
  void apply_propagators(char TA, WSet& wset, int ni, csrMat const& vup, csrMat const& vdn);

  template<typename T>
  T apply_bound_vbias(T v, RealType sqrtdt)
  {
    return (std::abs(v) > std::abs(static_cast<T>(static_cast<SPRealType>(vbias_bound * sqrtdt))))
        ? (v / (std::abs(v) / static_cast<T>(static_cast<SPRealType>(vbias_bound * sqrtdt))))
        : (v);
  }

  template<class WlkSet, class CMat>
  void Orthogonalize_impl(WlkSet& wset, CMat&& logdetR);

  template<class WlkSet>
  void Orthogonalize_impl(WlkSet& wset);

  template<class WlkSet>
  void Orthogonalize_excited_impl(WlkSet& wset);

};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMCModelPropagator.icc"

#endif
