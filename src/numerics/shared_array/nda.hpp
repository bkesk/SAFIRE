/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#pragma once

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"
#include "mpi3/shared_window.hpp"

namespace math {
  namespace shm {

    namespace mpi3 = boost::mpi3;

    // use const_shared array instead!!
    
    /**
     * A simple wrapper for nda arrays and MPI shared memory (from boost::mpi3)
     * still in design stage...
     */
    template<::nda::MemoryArray Array_base_t>
    class shared_array {
    public:
      using Array_view_t = decltype(std::declval<std::decay_t < Array_base_t>>()());
      static constexpr int rank = ::nda::get_rank<Array_view_t>;
      using value_type = typename std::decay_t<Array_view_t>::value_type;

      static constexpr bool is_stride_order_Fortran() noexcept {
        return Array_view_t::layout_t::is_stride_order_Fortran();
      }
      static constexpr bool is_stride_order_C() noexcept {
        return Array_view_t::layout_t::is_stride_order_C();
      }

    private:
      static_assert ( Array_view_t::layout_t::is_stride_order_Fortran()
        or Array_view_t::layout_t::is_stride_order_C(), "Ordering mismatch.");
    public:
      /* 
       * Default constructor returns a shared_array in uninitialized state.
       */
      shared_array() = default;

      shared_array(std::shared_ptr<sfqmc::utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>> ctxt,
                   std::array<long, rank> shape):
          _mpi(ctxt),
          _size(std::accumulate(shape.cbegin(), shape.cend(), (mpi3::size_t)1, std::multiplies<>{})),
          _shape(shape),
          _win(std::make_unique<mpi3::shared_window<value_type>>(_mpi->node_comm, (_mpi->node_comm.root()) ? _size : 0))
      { 
        check_and_init();
      }

      shared_array(const shared_array &other) :
          _mpi(other.mpi()), 
          _size(other.size()),
          _shape(other.shape()),
          _win(std::make_unique<mpi3::shared_window<value_type>>(_mpi->node_comm, (_mpi->node_comm.root()) ? _size : 0))
      {
        check_and_init();
        node_sync();
        if (_mpi->node_comm.root()) 
          this->local() = other.local(); 
        node_sync(); 
      } 
      shared_array(shared_array &&other) = default;

      shared_array& operator=(const shared_array &other) {
        _mpi = other.mpi();
        _size = other.size();
        _shape = other.shape();
        _win = std::move(std::make_unique<mpi3::shared_window<value_type>>(_mpi->node_comm, (_mpi->node_comm.root()) ? _size : 0));
        node_sync();
        if (_mpi->node_comm.root()) 
          this->local() = other.local(); 
        node_sync(); 
        return *this;
      }
      shared_array& operator=(shared_array &&other) = default;

      ~shared_array() = default; 

      void check_and_init() {
        abort_if_empty();
        sfqmc::utils::check(_win->base(0) != nullptr, "shm::shared_array: win.base(0) == nullptr");
        sfqmc::utils::check(_win->size(0) == _size, "shm::shared_array: win.size(0) has incorrect dimension");
        if (_mpi->node_comm.size() > 1) {
          sfqmc::utils::check(_win->size(1) == 0, "shm::shared_array: win.size(!=0) has incorrect dimension");
        }
        // initialize array to 0.0
        set_zero();
      }

      void set_zero() {
        if (is_empty()) return; 
        node_sync();
        auto[origin_i, end_i] = itertools::chunk_range(0, _size, _mpi->node_comm.size(), _mpi->node_comm.rank());
        ::nda::range i_range(origin_i, end_i);
        auto _array = Array_view_t(_shape, (value_type*) _win->base(0));
        auto array_1D = ::nda::reshape(_array, std::array<long, 1>{_size});
        _win->fence();
        array_1D(i_range) = value_type(0.0);
        _win->fence();
        node_sync();
      }

      void all_reduce() {
        if (is_empty()) return; 
        node_sync();
        if (_mpi->node_comm.root()) {
          // split all_reduce() to avoid mpi count overflow
          for (size_t shift=0; shift<_size; shift+=size_t(1e9)) {
            value_type *start = (value_type*)_win->base(0) + shift;
            size_t count = (shift+size_t(1e9) < _size)? size_t(1e9) : _size-shift;
            _mpi->internode_comm.all_reduce_in_place_n(start, count, std::plus<>{});
          }
        }
        node_sync();
      }

      void broadcast_to_nodes(int src_node) {
        if (is_empty()) return; 
        node_sync();
        if (_mpi->node_comm.root()) {
          for (size_t shift=0; shift<_size; shift+=size_t(1e9)) {
            value_type *start = (value_type *)_win->base(0) + shift;
            size_t count = (shift+size_t(1e9) < _size) ? size_t(1e9) : _size-shift;
            _mpi->internode_comm.broadcast_n(start, count, src_node);
          }
        }
        node_sync();
      }

      void node_sync() {
        if (is_empty()) return; 
        _mpi->node_comm.barrier();
        _win->sync();
      }

      auto mpi() const { abort_if_empty(); return _mpi; }
      mpi3::shared_window<value_type>& win() { abort_if_empty(); return *_win; }

      auto extent(long i) const { return _shape[i]; }
      auto const& shape() const { return _shape; }
      auto const& global_shape() const { return _shape; }
      auto size() const { return _size; }
      auto data() { abort_if_empty(); return (value_type*) _win->base(0); }
      auto data() const { abort_if_empty(); return (value_type const*) _win->base(0); }

      auto local() { abort_if_empty(); return Array_view_t(_shape, (value_type*) _win->base(0)); }
      auto local() const { abort_if_empty(); return Array_view_t(_shape, (value_type*) _win->base(0)); }

      auto operator()() { return this->local(); }
      auto operator()() const { return this->local(); }

    protected:
      std::shared_ptr<sfqmc::utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>> _mpi;
      mpi3::size_t _size = 0;
      std::array<long, rank> _shape = {0};
      std::unique_ptr<mpi3::shared_window<value_type>> _win;

      bool is_empty() const {
        return (_size==0 or not is_initialized());
      }

      bool is_initialized() const {
        return (bool(_mpi) or bool(_win));
      }

      void abort_if_empty() const {
        sfqmc::utils::check(is_initialized(), "Error in shared_array: Calling with uninitialized array.");
      }

    };

    /**
     * Shared memory array, one copy per node
     */
    template<::nda::MemoryArray Array_base_t>
    auto make_shared_array(std::shared_ptr<sfqmc::utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>> ctxt,
             std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> shape) {
      using Array_t = shared_array<Array_base_t>;

      return Array_t(ctxt,shape);
    }    

  } // shm
} // math

