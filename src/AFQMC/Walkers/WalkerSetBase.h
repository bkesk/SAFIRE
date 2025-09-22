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

#ifndef SFQMC_AFQMC_WALKERSETBASE_H
#define SFQMC_AFQMC_WALKERSETBASE_H

#include <random>
#include <type_traits>
#include <memory>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "io/ptree/ptree_utilities.hpp"
#include "Utilities/Random.hpp"

#include "AFQMC/config.h"
#include "Utilities/app_loggers.h"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "Numerics/ma_blas.hpp"

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
template<class Alloc, class Ptr> //, class BPAlloc, class BPPtr>
class WalkerSetBase : public AFQMCInfo
{
protected:

public:
  using element       = typename std::pointer_traits<Ptr>::element_type;
  using element_type  = element; 
  using pointer       = Ptr;
  using bp_element       = std::complex<float>; //typename to_single_precision<ComplexType>::type; 
  using bp_element_value_type = typename std::decay<bp_element>::type::value_type;
  using bp_pointer       = bp_element*; 

protected:
  using const_element = const element;
  using const_pointer = const Ptr;
  using Allocator     = Alloc;

  // since there is no factory, I can't use the trick I've been using to 
  // choose precision runtime. Storing BP fields in single precision for now,
  // regardless of choice of precision. If this is a problem, make a factory and choose
  // at runtime appropriately.  
  using const_bp_element = const bp_element;
  using const_bp_pointer = const bp_pointer;
  using BPAllocator      = shared_allocator<bp_element>; //BPAlloc;

  using CMatrix       = boost::multi::array<element, 2, Allocator>;
  using BPCMatrix     = boost::multi::array<bp_element, 2, BPAllocator>;
  using BPCVector_ref = boost::multi::array_ref<bp_element, 1, bp_pointer>;
  using BPCMatrix_ref = boost::multi::array_ref<bp_element, 2, bp_pointer>;
  using BPCTensor_ref = boost::multi::array_ref<bp_element, 3, bp_pointer>;

  using stdBPCMatrix_ptr = boost::multi::array_ptr<bp_element, 2>;
  using stdBPCTensor_ptr = boost::multi::array_ptr<bp_element, 3>;

public:
  // contiguous_walker = true means that all the data of a walker is continguous in memory
  static const bool contiguous_walker = true;
  // contiguous_storage = true means that the data of all walker is continguous in memory
  static const bool contiguous_storage = true;
  static const bool fixed_population   = true;

  using reference = walker<pointer>;
  using iterator  = walker_iterator<pointer>;
  //using const_reference = const_walker<const_pointer>;
  //using const_iterator = const_walker_iterator<const_pointer>;
  using const_reference = walker<pointer>;
  using const_iterator  = walker_iterator<pointer>;

  /// constructor
  WalkerSetBase(afqmc::TaskGroup_& tg_,
                ptree pt,
                AFQMCInfo& info,
                utils::RandomGenerator_t* r,
                Allocator alloc_,
                BPAllocator bpalloc_)
      : AFQMCInfo(info),
        TG(tg_),
        rng(r),
        walker_size(1),
        walker_memory_usage(0),
        bp_walker_size(0),
        bp_walker_memory_usage(0),
        bp_pos(-1),
        history_pos(0),
        walkerType(UNDEFINED_WALKER_TYPE),
        tot_num_walkers(0),
        walker_buffer({0, 1}, alloc_),
        bp_buffer({0, 0}, bpalloc_),
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
  int capacity() const { return int(walker_buffer.size(0)); }

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
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    return iterator(0, boost::multi::static_array_cast<element, pointer>(walker_buffer), data_displ, wlk_desc);
  }

  /*
   * Returns iterator to the first walker in the set
   */
  const_iterator begin() const
  {
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    return const_iterator(0, boost::multi::static_array_cast<element, pointer>(walker_buffer), data_displ, wlk_desc);
  }


  /*
   * Returns iterator to the past-the-end walker in the set
   */
  iterator end()
  {
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    return iterator(tot_num_walkers, boost::multi::static_array_cast<element, pointer>(walker_buffer), data_displ,
                    wlk_desc);
  }

  /*
   * Returns a reference to a walker
   */
  reference operator[](int i)
  {
    if (i < 0 || i > tot_num_walkers)
      APP_ABORT("error: index out of bounds.");
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    return reference(boost::multi::static_array_cast<element, pointer>(walker_buffer)[i], data_displ, wlk_desc);
  }

  /*
   * Returns a reference to a walker
   */
  const_reference operator[](int i) const
  {
    if (i < 0 || i > tot_num_walkers)
      APP_ABORT("error: index out of bounds.");
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    return const_reference(boost::multi::static_array_cast<element, pointer>(walker_buffer)[i], data_displ, wlk_desc);
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

  int get_TG_target_population() const { return targetN_per_TG; }
  int get_global_target_population() const { return targetN; }

  std::pair<int, int> walker_dims() const { return std::pair<int, int>{wlk_desc[0], wlk_desc[1]}; }

  int GlobalPopulation() const
  {
    int res = 0;
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    if (TG.TG_local().root())
      res += tot_num_walkers;
    return (TG.Global() += res);
  }

  RealType GlobalWeight() const
  {
    RealType res = 0;
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    if (TG.TG_local().root())
    {
      boost::multi::array<ComplexType, 1> buff(iextensions<1u>{tot_num_walkers});
      getProperty(WEIGHT, buff);
      for (int i = 0; i < tot_num_walkers; i++)
        res += std::abs(buff[i]);
    }
    return (TG.Global() += res);
  }

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

  void processWalkerData(std::vector<ComplexType>& curData);

  void popControl();

  // population control algorithm
  // Note: the following overload is deprecated
  void popControl(std::vector<ComplexType>& curData, bool skip = false);

  template<class Mat>
  void push_walkers(Mat&& M);

  template<class Mat>
  void pop_walkers(Mat&& M);

  // given a list of new weights and counts, add/remove walkers and reassign weight accordingly
  template<class Mat>
  void branch(Vector<std::pair<double, int>>::iterator itbegin,
              Vector<std::pair<double, int>>::iterator itend,
              Mat& M);

  template<class T>
  void scaleWeight(const T& w0, bool scale_last_history = false)
  {
    if (!TG.TG_local().root())
      return;
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    auto W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    ma::scal(ComplexType(w0), W({0, tot_num_walkers}, data_displ[WEIGHT]));
    if (scale_last_history)
    {
      int his_pos = ((history_pos == 0) ? wlk_desc[6] - 1 : history_pos - 1);
      if (wlk_desc[6] > 0 && his_pos >= 0 && his_pos < wlk_desc[6])
      {
        auto BPW{boost::multi::static_array_cast<bp_element, bp_pointer>(bp_buffer)};
        ma::scal(bp_element(static_cast<bp_element_value_type>(w0)), BPW[data_displ[WEIGHT_HISTORY] + his_pos]);
      }
    }
  }

  void scaleWeightsByOverlap()
  {
    if (!TG.TG_local().root())
      return;
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    auto W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    boost::multi::array<ComplexType, 1> ov(iextensions<1u>{tot_num_walkers});
    boost::multi::array<ComplexType, 1> buff(iextensions<1u>{tot_num_walkers});
    getProperty(OVLP, ov);
    for (int i = 0; i < tot_num_walkers; i++)
      buff[i] = ComplexType(1.0 / std::abs(ov[i]), 0.0);
    ma::axty(ComplexType(1.0), buff, W({0, tot_num_walkers}, data_displ[WEIGHT]));
    for (int i = 0; i < tot_num_walkers; i++)
      buff[i] = std::exp(ComplexType(0.0, -std::arg(ov[i])));
    ma::axty(ComplexType(1.0), buff, W({0, tot_num_walkers}, data_displ[PHASE]));
    ma::axty(ComplexType(1.0), buff, W({0, tot_num_walkers}, data_displ[PHASE1]));
    ma::axty(ComplexType(1.0), buff, W({0, tot_num_walkers}, data_displ[PHASE2]));
    ma::axty(ComplexType(1.0), buff, W({0, tot_num_walkers}, data_displ[PHASE3]));
  }

  afqmc::TaskGroup_& getTG() const { return TG; }

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

  // I am going to assume that the relevant data to be copied is continuous,
  // careful not to break this in the future
  template<class Vec>
  void copyToIO(Vec&& x, int n)
  {
    RUNTIME_CHECK(n < tot_num_walkers, "");
    RUNTIME_CHECK(x.size() >= walkerSizeIO(), "");
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    auto W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    using std::copy_n;
    copy_n(W[n].origin(), walkerSizeIO(), x.origin());
  }

  template<class Vec>
  void copyFromIO(Vec&& x, int n)
  {
    RUNTIME_CHECK(n < tot_num_walkers, "");
    RUNTIME_CHECK(x.size() >= walkerSizeIO(), "");
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    auto W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    using std::copy_n;
    copy_n(x.origin(), walkerSizeIO(), W[n].origin());
  }

  template<class TVec>
  void getProperty(walker_data id, TVec&& v) const
  {
    static_assert(std::decay<TVec>::type::dimensionality == 1, "Wrong dimensionality");
    if (v.num_elements() < tot_num_walkers)
      APP_ABORT("Error: getProperty(v):: v.size < tot_num_walkers.");
    auto W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    ma::copy(W({0, tot_num_walkers}, data_displ[id]), v.sliced(0, tot_num_walkers));
  }

  template<class TVec>
  void setProperty(walker_data id, TVec&& v)
  {
    static_assert(std::decay<TVec>::type::dimensionality == 1, "Wrong dimensionality");
    if (v.num_elements() < tot_num_walkers)
      APP_ABORT("Error: setProperty(v):: v.size < tot_num_walkers.");
    auto W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
    ma::copy(v.sliced(0, tot_num_walkers), W({0, tot_num_walkers}, data_displ[id]));
  }

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
    RUNTIME_CHECK(walker_buffer.size(1) == walker_size, "");
    double nx = (walkerType == NONCOLLINEAR or walkerType == FULLYPOLARIZED ? 1.0 : 2.0);
    if (TG.TG_local().root())
    {
      auto W{boost::multi::static_array_cast<element, pointer>(walker_buffer)};
      ma::scal(ComplexType(std::exp(-f)), W({0, tot_num_walkers}, data_displ[OVLP]));
    }
    LogOverlapFactor += f / nx;
    TG.TG_local().barrier();
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

protected:
  afqmc::TaskGroup_& TG;

  utils::RandomGenerator_t* rng;

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

  int targetN_per_TG;
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
  CMatrix walker_buffer;

  // Contains stack of fields and slater matrix references for back propagation
  BPCMatrix bp_buffer;

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

#endif
