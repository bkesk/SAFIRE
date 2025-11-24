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

#include "configuration.hpp"
#include "config.h"
#include "IO/AppAbort.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/mpi_context.h"
#include "utilities/type_traits.hpp"

#include "nda/nda.hpp"
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
 *
 */

/*
 * IMPLEMENTATION NOTE:
 * Routines that return an array view to local memory are now templated on the
 * MEMORY_SPACE. This is needed to be able to use the member functions as visitors,
 * since they must return the same type. If the memory spaces are incompatible, 
 * the call will abort the execution. 
 */
template<MEMORY_SPACE _MEM_>
class WalkerSetBase : public AFQMCInfo
{
protected:

public:
  static constexpr MEMORY_SPACE MEM    = _MEM_;

public:
  // contiguous_walker = true means that all the data of a walker is continguous in memory
  static const bool contiguous_walker = true;
  // contiguous_storage = true means that the data of all walker is continguous in memory
  static const bool contiguous_storage = true;
  static const bool fixed_population   = true;

  using reference = walker<ComplexType>;
  using iterator  = walker_iterator<ComplexType>;
  using const_reference = walker<const ComplexType>;
  using const_iterator  = walker_iterator<const ComplexType>;

  WalkerSetBase() 
  {
    utils::check(false, "Default initialization of WalkerSetBase is not allowed.");
  } 

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

  WalkerSetBase(WalkerSetBase const& other) = default;
  WalkerSetBase(WalkerSetBase&& other)      = default;
  WalkerSetBase& operator=(WalkerSetBase const& other) = default;
  WalkerSetBase& operator=(WalkerSetBase&& other) = default;

  /*
   * Returns the memory space.
   */
  constexpr auto get_memory_space() const { return MEM; }

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
  auto begin()
  {
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return iterator(0, walker_buffer, data_displ, wlk_desc);
  }

  /*
   * Returns iterator to the first walker in the set
   */
  auto begin() const
  {
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return const_iterator(0, walker_buffer, data_displ, wlk_desc);
  }

  /*
   * Returns iterator to the past-the-end walker in the set
   */
  auto end()
  {
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return iterator(tot_num_walkers, walker_buffer, data_displ, wlk_desc);
  }

  /*
   * Returns iterator to the past-the-end walker in the set
   */
  auto end() const
  {
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return const_iterator(tot_num_walkers, walker_buffer, data_displ, wlk_desc);
  } 

  /*
   * Returns a reference to a walker
   */
  auto operator[](int i)
  {
    utils::check(i>=0 and i<tot_num_walkers, "error: index out of bounds.");
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    return reference(walker_buffer(i,nda::range::all), data_displ, wlk_desc);
  }

  /*
   * Returns a reference to a walker
   */
  auto operator[](int i) const
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
  void resize(int n, nda::MemoryArrayOfRank<3> auto const& A);

  /*
   * Resizes back propagation buffers.
   * Must be called before any call to bp-related routines.
   */     
  void resize_bp(int nbp, int nCV, int nref);

  // perform and report tests/timings
  void benchmark(std::string& blist, int maxnW, int delnW, int repeat);

  auto get_target_population() const { return targetN_per_rank; }
  auto get_global_target_population() const { return targetN; }

  auto walker_dims() const { return std::pair<int, int>{wlk_desc[0], wlk_desc[1]}; }

  auto GlobalPopulation() const
  {
    int res = 0;
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch.");
    res += tot_num_walkers;
    return (mpi->comm += res);
  }

  auto GlobalWeight() const
  {
    RealType res = 0;
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch.");
    nda::array<ComplexType, 1> buff(tot_num_walkers,ComplexType(0.0));
    getProperty(WEIGHT, buff);
    for (int i = 0; i < tot_num_walkers; i++)
      res += std::abs(buff(i));
    return (mpi->comm += res);
  }

  private:

  template<MEMORY_SPACE _M_, walker_data D>
  auto extract_SM( SpinTypes s ) {
    static_assert(D == SM or D == SMN or D == SM_AUX, "Invalid enum");
    utils::check(_M_ == MEM, "Incompatible memory space");
    auto i0 = (s==Alpha?data_displ[D]:data_displ[D]+wlk_desc[0]*wlk_desc[1]);
    auto nc = (s==Alpha?wlk_desc[1]:wlk_desc[2]);
    std::array<long,3> shape = {tot_num_walkers,wlk_desc[0],nc};
    std::array<long,3> strides = {walker_buffer.strides()[0],nc,1};
    nda::idx_map<3, 0, nda::C_stride_order<3>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<_M_,ComplexType,3>(idxm, walker_buffer.data() + i0);     
  } 

  template<MEMORY_SPACE _M_, walker_data D>
  auto extract_SM( SpinTypes s ) const {
    static_assert(D == SM or D == SMN or D == SM_AUX, "Invalid enum");
    utils::check(_M_ == MEM, "Incompatible memory space");
    auto i0 = (s==Alpha?data_displ[D]:data_displ[D]+wlk_desc[0]*wlk_desc[1]);
    auto nc = (s==Alpha?wlk_desc[1]:wlk_desc[2]);
    std::array<long,3> shape = {tot_num_walkers,wlk_desc[0],nc};
    std::array<long,3> strides = {walker_buffer.strides()[0],nc,1};
    nda::idx_map<3, 0, nda::C_stride_order<3>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<_M_,const ComplexType,3>(idxm, walker_buffer.data() + i0); 
  }

  public:

  template<MEMORY_SPACE _M_>
  auto SlaterMatrices( SpinTypes s )  
  {
    return extract_SM<_M_,SM>(s);
  } 

  template<MEMORY_SPACE _M_>
  auto SlaterMatrices( SpinTypes s ) const
  {
    return extract_SM<_M_,SM>(s);
  }

  template<MEMORY_SPACE _M_>
  auto SlaterMatricesN( SpinTypes s )
  {
    utils::check(data_displ[SMN]>=0, "access to uninitialized BP sector. ");
    return extract_SM<_M_,SMN>(s);
  }

  template<MEMORY_SPACE _M_>
  auto SlaterMatricesN( SpinTypes s ) const
  {
    utils::check(data_displ[SMN]>=0, "access to uninitialized BP sector. ");
    return extract_SM<_M_,SMN>(s);
  }

  template<MEMORY_SPACE _M_>
  auto SlaterMatricesAux( SpinTypes s )
  {
    utils::check(data_displ[SM_AUX]>=0, "access to uninitialized BP sector. ");
    return extract_SM<_M_,SM_AUX>(s);
  }

  template<MEMORY_SPACE _M_>
  auto SlaterMatricesAux( SpinTypes s ) const
  {
    utils::check(data_displ[SM_AUX]>=0, "access to uninitialized BP sector. ");
    return extract_SM<_M_,SM_AUX>(s);
  }

  void processWalkerData(std::vector<ComplexType>& curData);

  void popControl();

  // population control algorithm
  // Note: the following overload is deprecated
  void popControl(std::vector<ComplexType>& curData, bool skip = false);

  void push_walkers(nda::MemoryArrayOfRank<2> auto&& M);

  void pop_walkers(nda::MemoryArrayOfRank<2> auto&& M);

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
                bp_buffer(nda::range(tot_num_walkers), data_displ[WEIGHT_HISTORY] + his_pos));
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
    nda::tensor::elementwise(ComplexType(1.0),buff_d,"i",
                             ComplexType(1.0),walker_buffer(r,data_displ[WEIGHT]),"i",MUL);
    for (int i = 0; i < tot_num_walkers; i++)
      buff[i] = std::exp(ComplexType(0.0, -std::arg(ov[i])));
    buff_d() = buff();  // to device
    // A(i) = A(i) * x(i)
    nda::tensor::elementwise(ComplexType(1.0),buff_d,"i",
                             ComplexType(1.0),walker_buffer(r,data_displ[PHASE]),"i",MUL);
    nda::tensor::elementwise(ComplexType(1.0),buff_d,"i",
                             ComplexType(1.0),walker_buffer(r,data_displ[PHASE1]),"i",MUL);
    nda::tensor::elementwise(ComplexType(1.0),buff_d,"i",
                             ComplexType(1.0),walker_buffer(r,data_displ[PHASE2]),"i",MUL);
    nda::tensor::elementwise(ComplexType(1.0),buff_d,"i",
                             ComplexType(1.0),walker_buffer(r,data_displ[PHASE3]),"i",MUL);
  }

  auto get_mpi() const { return mpi; }

  int single_walker_memory_usage() const { return walker_memory_usage; }
  int single_walker_size() const { return walker_size; }
  int single_walker_bp_memory_usage() const { return (wlk_desc[3] > 0) ? bp_walker_memory_usage : 0; }
  int single_walker_bp_size() const { return (wlk_desc[3] > 0) ? bp_walker_size : 0; }

  WALKER_TYPES getWalkerType() const { return walkerType; }
  std::tuple<BRANCHING_ALGORITHM,int,int> population_control_parameters() const 
  { return std::make_tuple(pop_control,min_weight,max_weight); }

  int walkerSizeIO() const
  {
    if (walkerType == COLLINEAR)
      return wlk_desc[0] * (wlk_desc[1] + wlk_desc[2]) + 10;
    else // since NAEB = 0 in both CLOSED and NONCOLLINEAR cases
      return wlk_desc[0] * wlk_desc[1] + 10;
    return 0;
  }

  // I am going to assume that the relevant data to be copied is continuous,
  // careful not to break this in the future
  void copyToIO(nda::MemoryArrayOfRank<1> auto&& x, int n) const
  {
    using nda::range;
    utils::check(n < tot_num_walkers, "Incorrect argument");
    utils::check(x.size() >= walkerSizeIO(), "Size mismatch");
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    x(range(walkerSizeIO())) = walker_buffer(n,range(walkerSizeIO()));
  }

  void copyFromIO(nda::MemoryArrayOfRank<1> auto&& x, int n)
  {
    using nda::range;
    utils::check(n < tot_num_walkers, "Incorrect argument");
    utils::check(x.size() >= walkerSizeIO(), "Size mismatch");
    utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch");
    walker_buffer(n,range(walkerSizeIO())) = x(range(walkerSizeIO()));
  }

  template<typename Arr>
  void getProperty(walker_data id, Arr&& v) const
  //void getProperty(walker_data id, nda::MemoryArrayOfRank<1> auto&& v) const
  {
    using nda::range;
    utils::check(v.size() >= tot_num_walkers, " Shape mismatch");
    v(range(tot_num_walkers)) = walker_buffer(range(tot_num_walkers),data_displ[id]);
  }

  void setProperty(walker_data id, nda::MemoryArrayOfRank<1> auto&& v)
  {
    using nda::range;
    utils::check(v.size() >= tot_num_walkers, " Shape mismatch");
    walker_buffer(range(tot_num_walkers),data_displ[id]) = v(range(tot_num_walkers));
  }

  void resetWeights()
  {
    mpi->comm.barrier();
    {
      memory::array<MEM, ComplexType, 1> w_(tot_num_walkers, ComplexType(1.0));
      setProperty(WEIGHT, w_);
    }
    mpi->comm.barrier();
  }

  template<MEMORY_SPACE _M_>
  auto getFields(int ip)
  {
    using nda::range;
    utils::check(_M_ == MEM, "Incompatible memory space");
    utils::check(ip>=0 and ip<wlk_desc[3], " Error: index out of bounds in getFields. ");
    long i0 = (data_displ[FIELDS] + ip * wlk_desc[4]);
    long nw = bp_buffer.extent(0);
    std::array<long,2> shape = {nw,wlk_desc[4]};
    nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm(shape,bp_buffer.strides());
    return memory::array_view<_M_,ComplexType,2>(idxm, bp_buffer.data() + i0);
  }

  template<MEMORY_SPACE _M_>
  auto getFields()
  {
    utils::check(_M_ == MEM, "Incompatible memory space");
    long i0 = data_displ[FIELDS];
    long nw = bp_buffer.extent(0);
    std::array<long,3> shape = {nw,wlk_desc[3],wlk_desc[4]};
    std::array<long,3> strides = {bp_buffer.strides()[0],wlk_desc[4],1};
    nda::idx_map<3, 0, nda::C_stride_order<3>, nda::layout_prop_e::none> idxm(shape,strides);
    return memory::array_view<_M_,ComplexType,3>(idxm, bp_buffer.data() + i0);
  }

  void storeFields(int ip, nda::MemoryArrayOfRank<2> auto&& V)
  {
    utils::check(ip>=0 and ip<wlk_desc[3], " Error: index out of bounds in getFields. ");
    long nw = bp_buffer.extent(0);
    utils::check(V.shape() == std::array<long,2>{nw,wlk_desc[4]}, 
                 "Shape mismatch");
    auto F = getFields<MEM>(ip);
    F() = V();
  }

  template<MEMORY_SPACE _M_>
  auto getWeightFactors()
  {
    using nda::range;
    utils::check(_M_ == MEM, "Incompatible memory space");
    long i0 = data_displ[WEIGHT_FAC];
    long nw = bp_buffer.extent(0);
    std::array<long,2> shape = {nw,wlk_desc[6]};
    nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm(shape,bp_buffer.strides());
    return memory::array_view<_M_,ComplexType,2>(idxm, bp_buffer.data() + i0);
  }

  template<MEMORY_SPACE _M_>
  auto getWeightHistory()
  {
    using nda::range;
    utils::check(_M_ == MEM, "Incompatible memory space");
    long i0 = data_displ[WEIGHT_HISTORY];
    long nw = bp_buffer.extent(0);
    std::array<long,2> shape = {nw,wlk_desc[6]};
    nda::idx_map<2, 0, nda::C_stride_order<2>, nda::layout_prop_e::none> idxm(shape,bp_buffer.strides());
    return memory::array_view<_M_,ComplexType,2>(idxm, bp_buffer.data() + i0);
  }

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

  // load balancing algorithm
  void loadBalance(nda::MemoryArrayOfRank<2> auto&& M,  
                   std::vector<int> const& nwalk_counts_old,  
                   std::vector<int> const& nwalk_counts_new)
  {
    if (load_balance == SIMPLE)
    {
      afqmc::swapWalkersSimple(*this, M, nwalk_counts_old, nwalk_counts_new, mpi->comm);
    }
    else if (load_balance == ASYNC)
    {
      afqmc::swapWalkersAsync(*this, M, nwalk_counts_old, nwalk_counts_new, mpi->comm);
    }
    mpi->comm.barrier();
  }

  std::shared_ptr<utils::RandomGenerator_t> getRNG() { return rng; }

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

