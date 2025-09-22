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

#ifndef SFQMC_AFQMC_HDF5_HELPERS_HPP
#define SFQMC_AFQMC_HDF5_HELPERS_HPP

#include <cstdlib>
#include <memory>
#include <algorithm>
#include <complex>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <vector>
#include <numeric>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "hdf/hdf_archive.h"
#include <boost/type_traits/is_complex.hpp>

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Utilities/Utils.hpp"
#include "SparseMatrix/hdf5_readers_new.hpp"
#include "SparseMatrix/array_partition.hpp"
#include "SparseMatrix/csr_hdf5_readers.hpp"
#include "SparseMatrix/matrix_emplace_wrapper.hpp"
#include "SparseMatrix/csr_matrix.hpp"
#include "SparseMatrix/coo_matrix.hpp"

namespace sfqmc
{
namespace afqmc
{

/*
 * Reads the one body hamiltonian from hdf5.
 * - Attempts to read both dense and sparse formats.
 * - Dynamically determines the type. If it is complex, will abort if template argument is incompatible.
 */
template<class Mat>
void read_one_body_hamiltonian(hdf_archive& dump, Mat&& H1)
{
  using VType = typename std::decay_t<Mat>::element_type;
  std::vector<int> shape;
  std::string base_error(" Error in read_one_body_hamiltonian: ");
  if (dump.getShape<RealType>("hcore", shape))  {
    // dense format
    if (shape.size() == 3)
    {
      if constexpr (not boost::is_complex<VType>::value)
        APP_ABORT(" Error in read_one_body_hamiltonian: Complex data with real container. ");
      Matrix_ref_<VType*> h1_(raw_pointer_cast(H1.origin()), H1.extensions());
      if (!dump.readEntry(h1_, std::string("hcore")))
        APP_ABORT(base_error + " Problems reading /Hamiltonian/hcore. ");
    } else if(shape.size() == 2) {
      Matrix<RealType> h1R(H1.extensions(), RealType(0.0));
      if (!dump.readEntry(h1R, std::string("hcore")))
        APP_ABORT(base_error + " Problems reading /Hamiltonian/hcore. ");
      std::copy_n(h1R.origin(), h1R.num_elements(), raw_pointer_cast(H1.origin()));
    } else {
      APP_ABORT(base_error + " Error: Inconsistent shape of hcore in /Hamiltonian/hcore. ");
    }
  } else if(dump.getShape<RealType>("H1", shape)) {
    std::vector<int> ivec;
    if (!dump.readEntry(ivec, "H1_indx"))
      APP_ABORT(base_error + " Problems reading /Hamiltonian/H1_indx.");
    if (shape.size() == 2) { // complex
      if constexpr (not boost::is_complex<VType>::value)
        APP_ABORT(base_error + " Complex data with real container. ");
      std::vector<ComplexType> vvec;
      if (!dump.readEntry(vvec, "H1"))
        APP_ABORT(base_error + " Problems reading /Hamiltonian/H1_indx.");
      if(2*vvec.size() != ivec.size())
        APP_ABORT(base_error + " /Hamiltonian/H1 and /Hamiltonian/H1_indx have inconsistent dimensions.");
      for (int i = 0; i < vvec.size(); i++)
      {
        if (ivec[2 * i] <= ivec[2 * i + 1])
        {
          H1[ivec[2 * i]][ivec[2 * i + 1]] = static_cast<VType>(vvec[i]);
          H1[ivec[2 * i + 1]][ivec[2 * i]] = static_cast<VType>(ma::conj(vvec[i]));
        } 
        else
        {
          H1[ivec[2 * i]][ivec[2 * i + 1]] = static_cast<VType>(ma::conj(vvec[i]));
          H1[ivec[2 * i + 1]][ivec[2 * i]] = static_cast<VType>(vvec[i]);
        } 
      } 
    } else if(shape.size() == 1) {
      std::vector<RealType> vvec;
      if (!dump.readEntry(ivec, "H1"))
        APP_ABORT(base_error + " Problems reading H1_indx.");
      if(2*vvec.size() != ivec.size())
        APP_ABORT(base_error + "H1 and H1_indx have inconsistent dimensions.");
      for (int i = 0; i < vvec.size(); i++)
      { 
        H1[ivec[2 * i]][ivec[2 * i + 1]] = static_cast<VType>(vvec[i]);
        H1[ivec[2 * i + 1]][ivec[2 * i]] = static_cast<VType>(vvec[i]);
      }
    } else {
      APP_ABORT(base_error + " Incorrect shape in H1 dataset.");
    }
  } else {
    APP_ABORT(base_error + "Could not find one body hamiltonian. ");
  }
}


template<typename VType>  
mpi3_csr_matrix<VType> read_V2fact(hdf_archive& dump,
                                   TaskGroup_& TG,
                                   int nread,
                                   int NMO,
                                   int nvecs,
                                   double cutoff1bar,
                                   [[maybe_unused]] int int_blocks)
{
  using counter = sfqmc::afqmc::sparse_matrix_element_counter;
  using Alloc   = shared_allocator<VType>;
  using ucsr_matrix =
      ma::sparse::ucsr_matrix<VType, int, std::size_t, shared_allocator<VType>, ma::sparse::is_root>;

  int min_i = 0;
  int max_i = nvecs;

  int nrows           = NMO * NMO;
  bool distribute_Ham = (TG.getNGroupsPerTG() < TG.getTotalNodes());
  std::vector<IndexType> row_counts(nrows);

  app_log(2," Reading sparse factorized two-electron integrals.");
  // calculate column range that belong to this node
  if (distribute_Ham)
  {
    // count number of non-zero elements along each column (Chol Vec)
    std::vector<IndexType> col_counts(nvecs);
    csr_hdf5::multiple_reader_global_count<VType>(dump, counter(false, nrows, nvecs, 0, nrows, 0, nvecs, cutoff1bar),
                                           col_counts, TG, nread);

    std::vector<IndexType> sets(TG.getNumberOfTGs() + 1);
    simple_matrix_partition<TaskGroup_, IndexType, RealType> split(nrows, nvecs, cutoff1bar);
    if (TG.getCoreID() < nread)
      split.partition_over_TGs(TG, false, col_counts, sets);

    if (TG.Global().rank() == 0)
    {
      app_log(2," Partitioning of (factorized) Hamiltonian Vectors: ");
      for (int i = 0; i <= TG.getNumberOfTGs(); i++)
        app_log(2, "{} ", sets[i]);
      app_log(2, "");
      app_log(2," Number of terms in each partitioning: ");
      for (int i = 0; i < TG.getNumberOfTGs(); i++)
        app_log(2, "{} ", accumulate(col_counts.begin() + sets[i], 
					 col_counts.begin() + sets[i + 1], 0));
      app_log(2, "");
    }

    TG.Node().broadcast(sets.begin(), sets.end());
    min_i = sets[TG.getTGNumber()];
    max_i = sets[TG.getTGNumber() + 1];

    csr_hdf5::multiple_reader_local_count<VType>(dump, counter(true, nrows, nvecs, 0, nrows, min_i, max_i, cutoff1bar),
                                          row_counts, TG, nread);
  }
  else
  {
    // should be faster if ham is not distributed
    csr_hdf5::multiple_reader_global_count<VType>(dump, counter(true, nrows, nvecs, 0, nrows, 0, nvecs, cutoff1bar),
                                           row_counts, TG, nread);
  }

  ucsr_matrix ucsr(tp_ul_ul{nrows, max_i - min_i}, tp_ul_ul{0, min_i}, row_counts, Alloc(TG.Node()));
  csr::matrix_emplace_wrapper<ucsr_matrix> csr_wrapper(ucsr, TG.Node());

  using mat_map = sfqmc::afqmc::matrix_map;
  csr_hdf5::multiple_reader_hdf5_csr<VType, int>(csr_wrapper,
                                                       mat_map(false, true, nrows, nvecs, 0, nrows, min_i, max_i,
                                                               cutoff1bar),
                                                       dump, TG, nread);
  csr_wrapper.push_buffer();
  TG.Node().barrier();

  return mpi3_csr_matrix<VType>(std::move(ucsr));
}

[[maybe_unused]] static HamiltonianTypes peekHamType(hdf_archive& dump, std::string format = "std")
{
  if (format  == "coqui") {
    // only format available, add choices as they are implemented
    // this is not enough, it could be THC, KPTHC, etc... Look for cholesky vectors...
    std::vector<int> shape;
    if (dump.getShape<RealType>("/Interaction/Vq0", shape))
      return KPFactorized;
    if (dump.getShape<RealType>("/Interaction/factorized_coulomb_matrix", shape))
    {
      if(shape[0]==1)
        return THC;
      else if(shape[0]>1)
        return KPTHC;
      else {
        APP_ABORT("  Error: Found Interaction/factorized_coulomb_matrix with dimension=0 "); 
        return UNKNOWN;
      } 
    }
  } else if(format == "std") {
    if (dump.is_group(std::string("/Hamiltonian/KPFactorized")))
    {
      return KPFactorized;
    }
    if (dump.is_group(std::string("/Hamiltonian/DenseFactorized")))
    {
      return RealDenseFactorized;
    }
    if (dump.is_group(std::string("/Hamiltonian/Factorized")))
    {
      return FactorizedSparse;
    }
    if (dump.is_group(std::string("/Hamiltonian/FactorizedSparse")))
    {
      return FactorizedSparse;
    }
    if (dump.is_group(std::string("/Hamiltonian/ModelHamiltonian")))
    {
      return ModelHamiltonian;
    }
  } else {
    APP_ABORT("  Error: Invalid format in peekHamType. ");
    return UNKNOWN;
  }
  APP_ABORT("  Error: Invalid hdf file format in peekHamType(hdf_archive). ");
  return UNKNOWN;
}

[[maybe_unused]] static std::string get_hamiltonian_format(hdf_archive& dump, communicator& comm)
{
  int code=-1;
  if(comm.root()) {
    if(dump.is_group(std::string("/Hamiltonian")))
      code=1;
    else if(dump.is_group(std::string("/System")) and dump.is_group(std::string("/Interaction")))
      code=2;
    else
      APP_ABORT("Error in get_hamiltonian_format: Invalid format");
  }
  comm.broadcast_n(&code,1,0);
  if(code==1)
    return "std";
  else if(code==2)
    return "coqui";
  else
    APP_ABORT("Error in get_hamiltonian_format: Invalid format (1)");
  return "";
}

} // namespace afqmc

} // namespace sfqmc

#endif
