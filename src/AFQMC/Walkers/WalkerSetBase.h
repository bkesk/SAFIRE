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

#include <random>
#include <type_traits>
#include <memory>

#include "config.h"
#include "IO/AppAbort.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/mpi_context.h"
#include "utilities/type_traits.hpp"

#include "nda/tensor.hpp"

#include "AFQMC/config.h"
#include "IO/app_loggers.h"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Utilities/type_conversion.hpp"

#include "AFQMC/Walkers/Walkers.hpp"
#include "AFQMC/Walkers/WalkerControl.hpp"
#include "AFQMC/Walkers/WalkerConfig.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Class that contains and handles walkers.
 * Implements communication, load balancing, and I/O operations.   
 * Walkers are always accessed through the handler.
 */
template<MEMORY_SPACE _MEM_>
class WalkerSetBase : public AFQMCInfo
{
protected:

public:
  static const MEMORY_SPACE MEM    = _MEM_;

public:
  // contiguous_walker = true means that all the data of a walker is continguous in memory
  static const bool contiguous_walker = true;
  // contiguous_storage = true means that the data of all walker is continguous in memory
  static const bool contiguous_storage = true;
  static const bool fixed_population   = true;

  using reference = walker<MEM>;
  using iterator  = walker_iterator<MEM>;
  using const_reference = reference;     // MAM: do I need an actual const version??? 
  using const_iterator  = iterator; 

  WalkerSetBase() = default;

  /// constructor
  WalkerSetBase(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi_,
                ptree pt,
                AFQMCInfo& info,
                std::shared_ptr<utils::RandomGenerator_t> r
               )
      : AFQMCInfo(info),
        mpi(_mpi_),
        rng(r),
        walker_size(1),
        walker_memory_usage(0),
        bp_walker_size(0),
        bp_walker_memory_usage(0),
        bp_pos(-1),
        history_pos(0),
        walkerType(UNDEFINED_WALKER_TYPE),
        tot_num_walkers(0),
        walker_buffer(0, 1),
        bp_buffer(0, 0),
        load_balance(UNDEFINED_LOAD_BALANCE),
        pop_control(UNDEFINED_BRANCHING),
        min_weight(0.05),
        max_weight(4.0)
  {
    parse(pt);
    setup();
  }

  /// destructor
  ~WalkerSetBase() {}

  WalkerSetBase(WalkerSetBase const& other) = delete;
  WalkerSetBase(WalkerSetBase&& other)      = default;
  WalkerSetBase& operator=(WalkerSetBase const& other) = delete;
  WalkerSetBase& operator=(WalkerSetBase&& other) = delete;

  /*
   * Returns the current number of walkers in the set.
   */
  int size() const { return tot_num_walkers; }

  /*
   * Returns the maximum number of walkers in the set that can be stored without reallocation.
   */
  int capacity() const { return int(walker_buffer.extent(0)); }

  /*
   * Returns the maximum number of fields in the set that can be stored without reallocation. 
   */
  int NumBackProp() const { return wlk_desc[3]; }
  /*
   * Returns the maximum number of cholesky vectors in the set that can be stored without reallocation. 
   */
  int NumCholVecs() const { return wlk_desc[4]; }
  /*
   * Returns the length of the history buffers. 
   */
  int HistoryBufferLength() const { return wlk_desc[6]; }

  /*
   * Returns the position of the insertion point in the BP stack. 
   */
  int getBPPos() const { return bp_pos; }
  void setBPPos(int p) { bp_pos = p; }
  void advanceBPPos() { bp_pos++; }

  /*
   * Returns, sets and advances the position of the insertion point in the History circular buffers. 
   */
  int getHistoryPos() const { return history_pos; }
  void setHistoryPos(int p) { history_pos = p % wlk_desc[6]; }
  void advanceHistoryPos() { history_pos = (history_pos + 1) % wlk_desc[6]; }


  /*
   * Returns iterator to the first walker in the set
   */
  iterator begin()
  {
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return iterator(0, walker_buffer, data_displ, wlk_desc);
  }

  /*
   * Returns iterator to the first walker in the set
   */
/*
  const_iterator begin() const
  {
    utils::check(walker_buffer.size(1) == walker_size, "");
    return const_iterator(0, boost::multi::static_array_cast<element, pointer>(walker_buffer), data_displ, wlk_desc);
  }
*/


  /*
   * Returns iterator to the past-the-end walker in the set
   */
  iterator end()
  {
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return iterator(tot_num_walkers, walker_buffer, data_displ, wlk_desc);
  }

  /*
   * Returns a reference to a walker
   */
  reference operator[](int i)
  {
    utils::check(i>=0 and i<tot_num_walkers, "error: index out of bounds.");
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return reference(walker_buffer(i,nda::range::all), data_displ, wlk_desc);
  }

  /*
   * Returns a reference to a walker
   */
  const_reference operator[](int i) const
  {
    utils::check(i>=0 and i<tot_num_walkers, "error: index out of bounds.");
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return const_reference(walker_buffer(i,nda::range::all), data_displ, wlk_desc);
  }

  // cleans state of object.
  //   -erases allocated memory
  bool clean();

  /*
   * Increases the capacity of the containers to n.
   */
  void reserve(int n);

  /*
   * Adds/removes the number of walkers in the set to match the requested value.
   * Walkers are removed from the end of the set 
   *     and buffer capacity remains unchanged in this case.
   * New walkers are initialized from already existing walkers in a round-robin fashion. 
   * If the set is empty, calling this routine will abort. 
   * Capacity is increased if necessary.
   * Target Populations are set to n.
   */
  void resize(int n);

  /*
   * Adds/removes the number of walkers in the set to match the requested value.
   * Walkers are removed from the end of the set 
   *     and buffer capacity remains unchanged in this case.
   * New walkers are initialized from the supplied matrix. 
   * Capacity is increased if necessary.
   * Target Populations are set to n.
   */
  template<class MatA, class MatB>
  void resize(int n, MatA&& A, MatB&& B);

  /*
   * Resizes back propagation buffers.
   * Must be called before any call to bp-related routines.
   */     
  void resize_bp(int nbp, int nCV, int nref);

  // perform and report tests/timings
  void benchmark(std::string& blist, int maxnW, int delnW, int repeat);

  int get_target_population() const { return targetN_per_rank; }
  int get_global_target_population() const { return targetN; }

  std::pair<int, int> walker_dims() const { return std::pair<int, int>{wlk_desc[0], wlk_desc[1]}; }

  int GlobalPopulation() const
  {
    int res = 0;
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch.");
    res += tot_num_walkers;
    return (mpi->comm += res);
  }

  RealType GlobalWeight() const
  {
    RealType res = 0;
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch.");
    nda::array<ComplexType, 1> buff(tot_num_walkers,ComplexType(0.0));
    getProperty(WEIGHT, buff);
    for (int i = 0; i < tot_num_walkers; i++)
      res += std::abs(buff(i));
    return (mpi->comm += res);
  }

/*
  auto SlaterMatrices( SpinTypes s )  
//  -> decltype( W.rotated().partitioned(1) )
  {
    auto&& W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    auto i0{data_displ[SM]};
    auto dx{wlk_desc[0]*wlk_desc[1]};
    if( s == Alpha )
      return W({0,tot_num_walkers}, {i0, i0+dx}).rotated().partitioned(wlk_desc[0]).unrotated();
    else {
      auto dx2{dx + wlk_desc[0]*wlk_desc[2]};
      return W({0,tot_num_walkers}, {i0+dx, i0+dx2}).rotated().partitioned(wlk_desc[0]).unrotated();
    }
  } 

  auto SlaterMatrices( SpinTypes s ) const
//  -> decltype( W.rotated().partitioned(1) )
  {
    auto&& W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    auto i0{data_displ[SM]};
    auto dx{wlk_desc[0]*wlk_desc[1]};
    if( s == Alpha )
      return W({0,tot_num_walkers}, {i0, i0+dx}).rotated().partitioned(wlk_desc[0]).unrotated();
    else {
      auto dx2{dx + wlk_desc[0]*wlk_desc[2]};
      return W({0,tot_num_walkers}, {i0+dx, i0+dx2}).rotated().partitioned(wlk_desc[0]).unrotated();
    }
  }

  auto SlaterMatricesN( SpinTypes s )
//  -> decltype( W.rotated().partitioned(1) )
  {
    if (data_displ[SMN] < 0)
      APP_ABORT("error in SlaterMatricesN: access to uninitialized BP sector. ");
    auto&& W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    auto i0{data_displ[SMN]};
    auto dx{wlk_desc[0]*wlk_desc[1]};
    if( s == Alpha )
      return W({0,tot_num_walkers}, {i0, i0+dx}).rotated().partitioned(wlk_desc[0]).unrotated();
    else {
      auto dx2(dx + wlk_desc[0]*wlk_desc[2]);
      return W({0,tot_num_walkers}, {i0+dx, i0+dx2}).rotated().partitioned(wlk_desc[0]).unrotated();
    }
  }

  auto SlaterMatricesN( SpinTypes s ) const
//  -> decltype( W.rotated().partitioned(1) )
  {
    if (data_displ[SMN] < 0)
      APP_ABORT("error in SlaterMatricesN: access to uninitialized BP sector. ");
    auto&& W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    auto i0{data_displ[SMN]};
    auto dx{wlk_desc[0]*wlk_desc[1]};
    if( s == Alpha )
      return W({0,tot_num_walkers}, {i0, i0+dx}).rotated().partitioned(wlk_desc[0]).unrotated();
    else {
      auto dx2(dx + wlk_desc[0]*wlk_desc[2]);
      return W({0,tot_num_walkers}, {i0+dx, i0+dx2}).rotated().partitioned(wlk_desc[0]).unrotated();
    }
  }

  auto SlaterMatricesAux( SpinTypes s )
//  -> decltype( W.rotated().partitioned(1) )
  {
    if (data_displ[SM_AUX] < 0)
      APP_ABORT("error in SlaterMatricesAux: access to uninitialized BP sector. ");
    auto&& W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    auto i0{data_displ[SM_AUX]};
    auto dx(wlk_desc[0]*wlk_desc[1]);
    if( s == Alpha )
      return W({0,tot_num_walkers}, {i0, i0+dx}).rotated().partitioned(wlk_desc[0]).unrotated();
    else {
      auto dx2(dx + wlk_desc[0]*wlk_desc[2]);
      return W({0,tot_num_walkers}, {i0+dx, i0+dx2}).rotated().partitioned(wlk_desc[0]).unrotated();
    }
  }

  auto SlaterMatricesAux( SpinTypes s ) const
//  -> decltype( W.rotated().partitioned(1) )
  {
    if (data_displ[SM_AUX] < 0)
      APP_ABORT("error in SlaterMatricesAux: access to uninitialized BP sector. ");
    auto&& W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    auto i0{data_displ[SM_AUX]};
    auto dx(wlk_desc[0]*wlk_desc[1]);
    if( s == Alpha )
      return W({0,tot_num_walkers}, {i0, i0+dx}).rotated().partitioned(wlk_desc[0]).unrotated();
    else {
      auto dx2(dx + wlk_desc[0]*wlk_desc[2]);
      return W({0,tot_num_walkers}, {i0+dx, i0+dx2}).rotated().partitioned(wlk_desc[0]).unrotated();
    }
  }
*/

  void processWalkerData(std::vector<ComplexType>& curData);

  void popControl();

  // population control algorithm
  // Note: the following overload is deprecated
  void popControl(std::vector<ComplexType>& curData, bool skip = false);

//  template<class Mat>
//  void push_walkers(Mat&& M);

//  template<class Mat>
//  void pop_walkers(Mat&& M);

  // given a list of new weights and counts, add/remove walkers and reassign weight accordingly
  template<class It>
  void branch(It itbeg, It itend, nda::MemoryArrayOfRank<2> auto&& M); 

  template<class T>
  void scaleWeight(const T& w0, bool scale_last_history = false)
  {
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    nda::blas::scal(ComplexType(w0), walker_buffer(nda::range(tot_num_walkers), data_displ[WEIGHT]));
    if (scale_last_history)
    {
      int his_pos = ((history_pos == 0) ? wlk_desc[6] - 1 : history_pos - 1);
      if (wlk_desc[6] > 0 && his_pos >= 0 && his_pos < wlk_desc[6])
      {
        nda::blas::scal(ComplexType(w0), 
                bp_buffer(data_displ[WEIGHT_HISTORY] + his_pos, nda::range(tot_num_walkers)));
      }
    }
  }

  void scaleWeightsByOverlap()
  {
    using nda::tensor::op::MUL;
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    nda::range r(tot_num_walkers);
    nda::array<ComplexType,1> ov(tot_num_walkers);  // on host
    nda::array<ComplexType,1> buff(tot_num_walkers);  // on host
    memory::array<MEM,ComplexType,1> buff_d(tot_num_walkers);  // on device
    getProperty(OVLP, ov);
    for (int i = 0; i < tot_num_walkers; i++)
      buff(i) = ComplexType(1.0 / std::abs(ov[i]), 0.0);
    buff_d() = buff(); // to device
    // A(i) = A(i) * x(i)
    nda::tensor::elementwise(buff_d,"i",walker_buffer(r,data_displ[WEIGHT]),"i",MUL);
    for (int i = 0; i < tot_num_walkers; i++)
      buff[i] = std::exp(ComplexType(0.0, -std::arg(ov[i])));
    buff_d() = buff();  // to device
    // A(i) = A(i) * x(i)
    nda::tensor::elementwise(buff_d,"i",walker_buffer(r,data_displ[PHASE]),"i",MUL);
    nda::tensor::elementwise(buff_d,"i",walker_buffer(r,data_displ[PHASE1]),"i",MUL);
    nda::tensor::elementwise(buff_d,"i",walker_buffer(r,data_displ[PHASE2]),"i",MUL);
    nda::tensor::elementwise(buff_d,"i",walker_buffer(r,data_displ[PHASE3]),"i",MUL);
  }

  auto get_mpi() const { return mpi; }

  int single_walker_memory_usage() const { return walker_memory_usage; }
  int single_walker_size() const { return walker_size; }
  int single_walker_bp_memory_usage() const { return (wlk_desc[3] > 0) ? bp_walker_memory_usage : 0; }
  int single_walker_bp_size() const { return (wlk_desc[3] > 0) ? bp_walker_size : 0; }

  WALKER_TYPES getWalkerType() const { return walkerType; }
  std::tuple<BRANCHING_ALGORITHM,int,int> population_control_parameters() const 
  { return std::make_tuple(pop_control,min_weight,max_weight); }

  int walkerSizeIO()
  {
    if (walkerType == COLLINEAR)
      return wlk_desc[0] * (wlk_desc[1] + wlk_desc[2]) + 10;
    else // since NAEB = 0 in both CLOSED and NONCOLLINEAR cases
      return wlk_desc[0] * wlk_desc[1] + 10;
    return 0;
  }

/*
  // I am going to assume that the relevant data to be copied is continuous,
  // careful not to break this in the future
  template<class Vec>
  void copyToIO(Vec&& x, int n)
  {
    utils::check(n < tot_num_walkers, "");
    utils::check(x.size() >= walkerSizeIO(), "");
    utils::check(walker_buffer.size(1) == walker_size, "");
    auto W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    using std::copy_n;
    copy_n(W[n].origin(), walkerSizeIO(), x.origin());
  }

  template<class Vec>
  void copyFromIO(Vec&& x, int n)
  {
    utils::check(n < tot_num_walkers, "");
    utils::check(x.size() >= walkerSizeIO(), "");
    utils::check(walker_buffer.size(1) == walker_size, "");
    auto W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    using std::copy_n;
    copy_n(x.origin(), walkerSizeIO(), W[n].origin());
  }
*/

  void getProperty(walker_data id, nda::MemoryArrayOfRank<1> auto&& v) const
  {
    utils::check(v.size() >= tot_num_walkers, " Shape mismatch");
    v(nda::range(tot_num_walkers)) = walker_buffer(nda::range(tot_num_walkers),data_displ[id]);
  }

  void setProperty(walker_data id, nda::MemoryArrayOfRank<1> auto&& v)
  {
    utils::check(v.size() >= tot_num_walkers, " Shape mismatch");
    walker_buffer(nda::range(tot_num_walkers),data_displ[id]) = v(nda::range(tot_num_walkers));
  }

/*
  void resetWeights()
  {
    TG.TG_local().barrier();
    if (TG.TG_local().root())
    {
      boost::multi::array<element, 1> w_(iextensions<1u>{tot_num_walkers}, ComplexType(1.0));
      setProperty(WEIGHT, w_);
    }
    TG.TG_local().barrier();
  }

  // Careful!!! This matrix returns an array_ptr, NOT a copy!!!
  stdBPCMatrix_ptr getFields(int ip)
  {
    if (ip < 0 || ip > wlk_desc[3])
      APP_ABORT(" Error: index out of bounds in getFields. ");
    int skip = (data_displ[FIELDS] + ip * wlk_desc[4]) * bp_buffer.size(1);
    return stdBPCMatrix_ptr(raw_pointer_cast(bp_buffer.origin()) + skip, {wlk_desc[4], bp_buffer.size(1)});
  }

  stdBPCTensor_ptr getFields()
  {
    return stdBPCTensor_ptr(raw_pointer_cast(bp_buffer.origin()) + data_displ[FIELDS] * bp_buffer.size(1),
                          {wlk_desc[3], wlk_desc[4], bp_buffer.size(1)});
  }

  template<class Mat>
  void storeFields(int ip, Mat&& V)
  {
    static_assert(std::decay<Mat>::type::dimensionality == 2, "Wrong dimensionality");
    auto&& F{*getFields(ip)};
    if (V.stride(0) == V.size(1))
    {
      using std::copy_n;
      copy_n(V.origin(), F.num_elements(), F.origin());
    }
    else
      F = V;
  }

  stdBPCMatrix_ptr getWeightFactors()
  {
    return stdBPCMatrix_ptr(raw_pointer_cast(bp_buffer.origin()) + data_displ[WEIGHT_FAC] * bp_buffer.size(1),
                          {wlk_desc[6], bp_buffer.size(1)});
  }

  stdBPCMatrix_ptr getWeightHistory()
  {
    return stdBPCMatrix_ptr(raw_pointer_cast(bp_buffer.origin()) + data_displ[WEIGHT_HISTORY] * bp_buffer.size(1),
                          {wlk_desc[6], bp_buffer.size(1)});
  }
*/ 

  double getLogOverlapFactor() const { return LogOverlapFactor; }

/**
 * @brief Updates the WalkerSetBase::LogOverlapFactor
 *
 * @details adjustLogOverlapFactor adjusts the LogOverlapFactor. 
 *  If F = e^(-f) is the "linear" overlap to include and nx is a factor described below, 
 *  then the new LogOverlapFactor is updated as LogOverlapFactor = LogOverlapFactor + log(F)/nx.
 *  
 *  nx= {2:CLOSED&&COLLINEAR, 1:NONCOLLINEAR }
 *  before: OV_full = exp( nx*LogOverlapFactor ) * OVold
 *  after: OV_full = exp( nx*LogOverlapFactor+f ) * OVnew
 *  OVnew = OVold * exp( -f )
 *  LogOverlapFactor_new = LogOverlapFactor + f/nx
 * 
 * @param f const double f is factor to include in the current LogOverlapFactor. It is assumed that f = log(F).
 */
  void adjustLogOverlapFactor(const double f)
  {
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    double nx = (walkerType == NONCOLLINEAR or walkerType == FULLYPOLARIZED ? 1.0 : 2.0);
    nda::blas::scal(ComplexType(std::exp(-f)), walker_buffer(nda::range(tot_num_walkers), data_displ[OVLP]));
    LogOverlapFactor += f / nx;
    mpi->comm.barrier();
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    // read inputs with default options
    std::string name, walker_type, load_balance_type, pop_control_type;
    double min_weight, max_weight;
    name              = pt0.get<std::string>("name", "wset0");
    walker_type       = pt0.get<std::string>("walker_type", "collinear");
    load_balance_type = pt0.get<std::string>("load_balance_type", "async");
    pop_control_type  = pt0.get<std::string>("pop_control_type", "pair");
    min_weight        = pt0.get<double>("min_weight", 0.05);
    max_weight        = pt0.get<double>("max_weight", 4.0);
  
    // check input validity
    if (min_weight < 1e-2) APP_ABORT("min_weight too small");
    //std::map<std::string, std::set<std::string> > options = {
    //  {"walker_type", {"closed", "collinear", "noncollinear"}}
    //};
    //if (options["walker_type"].count(tolower(walker_type)) < 0)
    //{
    //  app_log() << walker_type << std::endl;
    //  APP_ABORT("unknown walker_type");
    //}
  
    // create verbose internal inputs
    ptree pt1;
    pt1.put("name", name);
    pt1.put("walker_type", walker_type);
    pt1.put("load_balance_type", load_balance_type);
    pt1.put("pop_control_type", pop_control_type);
    pt1.put("min_weight", min_weight);
    pt1.put("max_weight", max_weight);
    std::unordered_set<std::string> pass_through_keys = {
      "system"
    };
    io::compare_known_keys("Walker set",pt1, pt0,pass_through_keys);
    return pt1;
  }

/*
  // load balancing algorithm
  template<class Mat>
  void loadBalance(Mat&& M,  std::vector<int> const& nwalk_counts_old,  std::vector<int> const& nwalk_counts_new)
  {
    if (load_balance == SIMPLE)
    {
      if (TG.TG_local().root())
        afqmc::swapWalkersSimple(*this, std::forward<Mat>(M), nwalk_counts_old, nwalk_counts_new, TG.TG_heads());
    }
    else if (load_balance == ASYNC)
    {
      if (TG.TG_local().root())
        afqmc::swapWalkersAsync(*this, std::forward<Mat>(M), nwalk_counts_old, nwalk_counts_new, TG.TG_heads());
    }
    TG.TG_local().barrier();
    // since tot_num_walkers is local, you need to sync it
    if (TG.TG_local().size() > 1)
      TG.TG_local().broadcast_n(&tot_num_walkers, 1, 0);
  }

  utils::RandomGenerator_t* getRNG() { return rng; }

*/
protected:
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  std::shared_ptr<utils::RandomGenerator_t> rng;

  int LoadBalance_timer;
  int Branching_timer;

  int walker_size, walker_memory_usage;
  int bp_walker_size, bp_walker_memory_usage;
  int bp_pos;
  int history_pos;

  // wlk_descriptor: {nmo, naea, naeb, nback_prop, nCV, nRefs, nHist}
  wlk_descriptor wlk_desc;
  wlk_indices data_displ;

  WALKER_TYPES walkerType;

  int targetN_per_rank;
  int targetN;
  int tot_num_walkers;

  // Stores an overall scaling factor for walker weights (assumed to be
  // consistent over all walker groups).
  // The actual overlap of a walker is exp(OverlapFactor)*wset[i].weight()
  // Notice that overlap ratios (which are always what matters) are
  // independent of this value.
  // If this value is changed, the overlaps of the walkers must be adjusted
  // This is needed for stability reasons in large systems
  // Note that this is stored on a "per spin" capacity
  double LogOverlapFactor = 0.0;

  // Contains main walker data needed for propagation
  memory::array<MEM, ComplexType, 2> walker_buffer;

  // Contains stack of fields and slater matrix references for back propagation
  memory::array<MEM, ComplexType, 2> bp_buffer;

  // performs setup
  void parse(ptree cur);
  void setup();

  // load balance algorithm
  LOAD_BALANCE_ALGORITHM load_balance;

  // branching algorithm
  BRANCHING_ALGORITHM pop_control;
  [[maybe_unused]] double min_weight, max_weight;
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Walkers/WalkerSetBase.icc"

