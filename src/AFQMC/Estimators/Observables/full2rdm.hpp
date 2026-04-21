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

#ifndef SFQMC_AFQMC_FULL2RDM_HPP
#define SFQMC_AFQMC_FULL2RDM_HPP

#include "AFQMC/config.h"
#include <string>
#include <h5/group.hpp>
#include <nda/layout/range.hpp>
#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <mpi3/communicator.hpp>

#include "numerics/operations/product.hpp"
#include "utilities/check.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
/* 
 * Observable class that calculates the walker averaged 2 RDM.
 * The resulting RDM will be [3*spin][i][k][j][l]  
 * where x:2 for NONCOLLINEAR and 1 for everything else.
 * For collinear, the spin ordering is (a,a,a,a), (a,a,b,b), (b,b,b,b) 
 */
class full2rdm : public AFQMCInfo
{
public:
  full2rdm(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi_, AFQMCInfo& info, ptree pt, WALKER_TYPES wlk, int nave_ = 1, int bsize = 1)
      : AFQMCInfo{info},
        mpi{mpi_},
        walker_type{wlk},
        block_size{bsize},
        apply_rotation{false}
  {
    app_log(1,"  --  Adding 2RDM (TwoRDM) estimator. -- ");

    std::string rot_file, h5_path;
    rot_file = pt.get<std::string>("rotation", "");
    h5_path = pt.get<std::string>("path", "/");

    int dm_size{};
    if (rot_file != "")
    {
      apply_rotation = true;

      if (mpi->node_comm.root())
      {
        h5::file file(rot_file, 'r');
        h5::group grp = h5::group(file).open_group(h5_path);
        memory::default_array<ComplexType,2> R;
        h5::read(grp, "RotationMatrix", XRot);

        utils::check(XRot.shape(1) != NMO, "Rotation has wrong number of rows {} (expected {})", XRot.shape(1), NMO);
      }

      std::array<long int,2> dim = XRot.shape();
      mpi->node_comm.broadcast_n(dim.data(), dim.size(), 0);
      XRot.resize(dim);
      mpi->node_comm.broadcast_n(XRot.data(), XRot.size(), 0);

      dm_size = XRot.shape(0) * XRot.shape(0);
    }
    else
    {
      dm_size = NMO * NMO;
    }

    // (a,a,a,a), (a,a,b,b)
    int nspinblocks = 2;
    if (walker_type == COLLINEAR)
    {
      nspinblocks = 3; // (a,a,a,a), (a,a,b,b), (b,b,b,b)
    }
    else if (walker_type == NONCOLLINEAR)
      APP_ABORT(" Error: NONCOLLINEAR not yet implemented. \n\n");

    DMAverage.resize(nave_, nspinblocks, dm_size, dm_size);
    DMAverage() = 0;
  }

/*******   Interface for sum over references, e.g. NOMSD ********/
  auto accumulate(int iav, nda::MemoryArrayOfRank<4> auto&& G, nda::MemoryVector auto&& Xw, [[maybe_unused]] bool impsamp)
  {
    // assumes G[nwalk][spin][M][M]
    utils::check(G.shape(0) == Xw.shape(0), "G and Xw number of columns (walkers) mismatch: {} != {}", G.shape(0), Xw.shape(0));

    if (apply_rotation)
      acc_with_rotation(iav, G, Xw);
    else
      acc_no_rotation(iav, G, Xw);
  }

/*******   Interface for PHMSD-like wfns: Reference + excited configurations  *******/
  template<class... Args>
  void accumulate_reference_configuration([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_reference_configuration ");
  }

  template<class... Args>
  void accumulate_excited_configuration_first([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_first ");
  }

  template<class... Args>
  void accumulate_excited_configuration_second([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Finish: accumulate_excited_configuration_second ");
  }

  auto print(int iblock, h5::group *group, nda::Vector auto&& Wsum)
  {
    DMAverage() *= 1.0 / block_size;
    mpi->reduce(DMAverage, std::plus<>(), 0);
    if(mpi->comm.root())
    {
      assert(group);
      for (int i = 0; i < DMAverage.shape(0); ++i)
      {
        h5::group obs_group = group->open_group(std::format("FullTwoRDM/Average_{}", i));
        std::string padded_iblock = std::format("{:09}", iblock);
        h5::write(obs_group, "two_rdm_" + padded_iblock, nda::flatten(DMAverage(i, nda::ellipsis{})));
        h5::write(obs_group, "denominator_" + padded_iblock, Wsum[i]);
      }
    }
    DMAverage() = 0;
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  int block_size{};
  bool apply_rotation{};

  memory::default_array<ComplexType,2> XRot;
  memory::default_array<ComplexType,1> Grot;

  // DMAverage (nave, nspinblocks, x*NMO^2, x*NMO^2), x=(1:CLOSED/COLLINEAR, 2:NONCOLLINEAR)
  memory::default_array<ComplexType,4> DMAverage;

  auto acc_no_rotation(int iav, nda::MemoryArrayOfRank<4> auto&& G, nda::MemoryVector auto&& Xw)
  {
    using nda::ellipsis;
    // doing this 1 walker at a time and not worrying about speed
    int nw(G.shape(0));

    // (ikjl) = Gik * Gjl - (same spin) Gil Gjk
    if (walker_type == COLLINEAR)
    {
      size_t M2(NMO * NMO);
      size_t M4(M2 * M2);

      memory::default_array<ComplexType,2> R(NMO * NMO, NMO * NMO);
      memory::default_array<ComplexType,2> Q(NMO, NMO * NMO * NMO);
    
      memory::default_array<ComplexType,2> Gt(NMO, NMO);

      for (int iw = 0; iw < nw; iw++)
      {
        // same spin
        for(int ispin = 0; ispin < 2; ispin++) {
          auto Gv = reshape(G(iw, ispin, ellipsis{}), M2, 1);
          math::product<'N', 'T'>(Gv, Gv, R);
          DMAverage(iav, 2*ispin, ellipsis{}) += Xw(iw) * R;

          // reshape trick to get the -Gil Gjk term
          // Note: nda/tensor will not reshape transposed arrays correctly.
          // Therefore we make sure to materialize the transpose first
          Gt() = transpose(G(iw, ispin, ellipsis{}));
          math::product<'N', 'T'>(reshape(Gt, M2,1), reshape(Gt,M2,1), R);
          Q() = transpose(reshape(R, NMO*M2, NMO));
          DMAverage(iav, 2*ispin, ellipsis{}) -= Xw(iw) * reshape(Q, M2, M2);
        }
        // mixed spin: no exchange term
        auto Gv1 = reshape(G(iw, 0, ellipsis{}), M2, 1);
        auto Gv2 = reshape(G(iw, 1, ellipsis{}), M2, 1);
        math::product<'N', 'T'>(Gv1, Gv2, R);
        DMAverage(iav, 1, ellipsis{}) += Xw(iw) * R;
      }
    }
    else
    {
      APP_ABORT("Error: Complete full2rdm.");
    }
  }

  template<class MatG, class CVec>
  void acc_with_rotation([[maybe_unused]] int iav, [[maybe_unused]] MatG&& G, [[maybe_unused]] CVec&& Xw)
  {
    APP_ABORT("Error: Complete full2rdm.");
  }
};

} // namespace afqmc
} // namespace sfqmc

#endif
