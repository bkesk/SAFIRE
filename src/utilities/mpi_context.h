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
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "nda/nda.hpp"

namespace mpi3 = boost::mpi3;
namespace sfqmc::utils {

template <typename comm_t = mpi3::communicator, 
          typename shm_comm_t = mpi3::shared_communicator> 
struct mpi_context_t {
  comm_t comm;
  shm_comm_t node_comm;
  comm_t internode_comm;

  mpi_context_t() = delete;
  mpi_context_t(comm_t const& c_, shm_comm_t const& s_, comm_t const& ic_)
    : comm(c_),node_comm(s_),internode_comm(ic_)
  {}

  mpi_context_t(comm_t && c_, shm_comm_t && s_, comm_t && ic_)
    : comm(std::move(c_)),node_comm(std::move(s_)),internode_comm(std::move(ic_))
  {}

  mpi_context_t(mpi_context_t const&) = default;
  mpi_context_t(mpi_context_t &&) = default;
  mpi_context_t& operator=(mpi_context_t const&) = default;
  mpi_context_t& operator=(mpi_context_t &&) = default;

  // some auxiliary functions for nda
  template<bool use_gpu = true>
  void broadcast(nda::Array auto&& A, int root = 0)
  {
    utils::check(A.is_contiguous(),"mpi_context_t::broadcast: Array must be contiguous.");
    comm.broadcast_n(A.data(),A.size(),root);
  } 

  template<typename Op, bool use_gpu = true>
  void all_reduce(nda::Array auto&& A, Op&& op)
  {
    utils::check(A.is_contiguous(),"mpi_context_t::all_reduce: Array must be contiguous.");
    comm.all_reduce_in_place_n(A.data(),A.size(),op);
  }

};

mpi_context_t<mpi3::communicator> make_mpi_context(); 
mpi_context_t<mpi3::communicator> make_mpi_context(mpi3::communicator& comm); 

}
