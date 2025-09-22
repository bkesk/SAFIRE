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

#ifndef SFQMC_AFQMC_THCOPSIO_HPP
#define SFQMC_AFQMC_THCOPSIO_HPP

#include <fstream>

#include "Utilities/type_traits/container_traits_multi.h"
#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/config.h"
#include "SparseMatrix/ma_hdf5_readers.hpp"
#include "Numerics/ma_blas_extensions.hpp"

#include "AFQMC/HamiltonianOperations/THCOps.hpp"
#include "AFQMC/Hamiltonians/rotateHamiltonian.hpp"

namespace sfqmc
{
namespace afqmc
{
// distribution:  size,  global,  offset
//   - rotMuv:    {rotnmu,grotnmu},{grotnmu,grotnmu},{rotnmu0,0}
//   - rotPiu:    {size_t(NMO),grotnmu},{size_t(NMO),grotnmu},{0,0}
//   - rotcPua    {grotnmu,nel_},{grotnmu,nel_},{0,0}
//   - Piu:       {size_t(NMO),nmu},{size_t(NMO),gnmu},{0,nmu0}
//   - Luv:       {nmu,gnmu},{gnmu,gnmu},{nmu0,0}
//   - cPua       {nmu,nel_},{gnmu,nel_},{nmu0,0}


// Some code duplication with THCHamiltonian class.
template<bool MP, bool REAL>
inline THCOps<MP,REAL> loadTHCOps(hdf_archive& dump,
                         WALKER_TYPES type,
                         int NMO,
                         int NAEA,
                         int NAEB,
                         std::vector<PsiT_Matrix>& PsiT,
                         TaskGroup_& TGprop,
                         TaskGroup_& TGwfn,
                         [[maybe_unused]] RealType cutvn,
                         [[maybe_unused]] RealType cutv2)
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;

  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  using SPValueType   = typename to_working_precision<MP,ValueType  >::type;

  using shmCMatrix   = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using shmSPVMatrix = boost::multi::array<SPValueType, 2, shared_allocator<SPValueType>>;
  using shmSPCMatrix = boost::multi::array<SPComplexType, 2, shared_allocator<SPComplexType>>;

  if (type == COLLINEAR)
    RUNTIME_CHECK(PsiT.size() % 2 == 0, "");
  int ndet = ((type != COLLINEAR) ? (PsiT.size()) : (PsiT.size() / 2));
  if (ndet > 1)
    APP_ABORT(" Error in loadTHCOps: ndet > 1 not yet implemented in THCOps.");

  // fix later for multidet case
  std::vector<int> dims(10);
  ComplexType E0;
  std::size_t gnmu, grotnmu, nmu, rotnmu, nmu0, nmuN, rotnmu0, rotnmuN;

  // read from HDF

  if (dump.push("HamiltonianOperations", false)<0)
  {
    app_error(" Error in loadTHCOps: Group HamiltonianOperations not found. ");
    APP_ABORT("");
  }
  if (dump.push("THCOps", false)<0)
  {
    app_error(" Error in loadTHCOps: Group THCOps not found. ");
    APP_ABORT("");
  }
  if (TGwfn.Global().root())
  {
    if (!dump.readEntry(dims, "dims"))
    {
      app_error(" Error in loadTHCOps: Problems reading dataset. ");
      APP_ABORT("");
    }
    RUNTIME_CHECK(dims.size() == 7, "");
    if (dims[0] != NMO)
    {
      app_error(" Error in loadTHCOps: Inconsistent data in file: NMO. ");
      APP_ABORT("");
    }
    if (dims[1] != NAEA)
    {
      app_error(" Error in loadTHCOps: Inconsistent data in file: NAEA. ");
      APP_ABORT("");
    }
    if (dims[2] != NAEB)
    {
      app_error(" Error in loadTHCOps: Inconsistent data in file: NAEB. ");
      APP_ABORT("");
    }
    if (dims[3] != ndet)
    {
      app_error(" Error in loadTHCOps: Inconsistent data in file: ndet. ");
      APP_ABORT("");
    }
    if (type == CLOSED && dims[4] != 1)
    {
      app_error(" Error in loadTHCOps: Inconsistent data in file: walker_type. ");
      APP_ABORT("");
    }
    if (type == COLLINEAR && dims[4] != 2)
    {
      app_error(" Error in loadTHCOps: Inconsistent data in file: walker_type. ");
      APP_ABORT("");
    }
    if (type == NONCOLLINEAR && dims[4] != 3)
    {
      app_error(" Error in loadTHCOps: Inconsistent data in file: walker_type. ");
      APP_ABORT("");
    }
    std::vector<ComplexType> et;
    if (!dump.readEntry(et, "E0"))
    {
      app_error(" Error in loadTHCOps: Problems reading dataset. ");
      APP_ABORT("");
    }
    E0 = et[0];
  }
  TGwfn.Global().broadcast_n(dims.data(), 7);
  TGwfn.Global().broadcast_value(E0);
  gnmu    = size_t(dims[5]);
  grotnmu = size_t(dims[6]);

  // setup partition, in general matrices are partitioned along 'u'
  {
    int node_number            = TGwfn.getLocalGroupNumber();
    int nnodes_prt_TG          = TGwfn.getNGroupsPerTG();
    std::tie(rotnmu0, rotnmuN) = FairDivideBoundary(std::size_t(node_number), grotnmu, std::size_t(nnodes_prt_TG));
    rotnmu                     = rotnmuN - rotnmu0;

    node_number          = TGprop.getLocalGroupNumber();
    nnodes_prt_TG        = TGprop.getNGroupsPerTG();
    std::tie(nmu0, nmuN) = FairDivideBoundary(std::size_t(node_number), gnmu, std::size_t(nnodes_prt_TG));
    nmu                  = nmuN - nmu0;
  }

  // read 1-body hamiltonian and exchange potential (vn0)
  shmCMatrix H1({NMO, NMO}, shared_allocator<ComplexType>{TGwfn.Node()});
  shmCMatrix vn0({NMO, NMO}, shared_allocator<ComplexType>{TGwfn.Node()});
  if (TGwfn.Node().root())
  {
    boost::multi::array<ValueType, 2> H1_({NMO, NMO});
    if (!dump.readEntry(H1_, "H1"))
    {
      app_error(" Error in loadTHCOps: Problems reading dataset. ");
      APP_ABORT("");
    }
    copy_n_cast(H1_.origin(), NMO * NMO, raw_pointer_cast(H1.origin()));
    if (!dump.readEntry(vn0, "v0"))
    {
      app_error(" Error in loadTHCOps: Problems reading dataset. ");
      APP_ABORT("");
    }
  }

  // Until I figure something else, rotPiu and rotcPua are not distributed because a full copy is needed
  size_t nel_ = ((type == CLOSED) ? NAEA : (NAEA + NAEB));
  shmSPVMatrix rotMuv({rotnmu, grotnmu}, shared_allocator<SPValueType>{TGwfn.Node()});
  shmSPVMatrix rotPiu({size_t(NMO), grotnmu}, shared_allocator<SPValueType>{TGwfn.Node()});
  shmSPVMatrix Piu({size_t(NMO), nmu}, shared_allocator<SPValueType>{TGwfn.Node()});
  shmSPVMatrix Luv({nmu, gnmu}, shared_allocator<SPValueType>{TGwfn.Node()});

  // read Half transformed first
  if (TGwfn.Node().root())
  {
    /***************************************/
    if (!dump.readEntry(rotPiu, "HalfTransformedFullOrbitals"))
    {
      app_error(" Error in THCHamiltonian::getHamiltonianOperations():");
      app_error(" Problems reading HalfTransformedFullOrbitals. ");
      APP_ABORT("");
    }
    /***************************************/
    hyperslab_proxy<shmSPVMatrix, 2> hslab(rotMuv, std::array<size_t, 2>{grotnmu, grotnmu},
                                           std::array<size_t, 2>{rotnmu, grotnmu}, std::array<size_t, 2>{rotnmu0, 0});
    if (!dump.readEntry(hslab, "HalfTransformedMuv"))
    {
      app_error(" Error in THCHamiltonian::getHamiltonianOperations():");
      app_error(" Problems reading HalfTransformedMuv. ");
      APP_ABORT("");
    }
    /***************************************/
    hyperslab_proxy<shmSPVMatrix, 2> hslab2(Piu, std::array<size_t, 2>{size_t(NMO), gnmu},
                                            std::array<size_t, 2>{size_t(NMO), nmu}, std::array<size_t, 2>{0, nmu0});
    if (!dump.readEntry(hslab2, "Orbitals"))
    {
      app_error(" Error in THCHamiltonian::getHamiltonianOperations():");
      app_error(" Problems reading Orbitals. ");
      APP_ABORT("");
    }
    /***************************************/
    hyperslab_proxy<shmSPVMatrix, 2> hslab3(Luv, std::array<size_t, 2>{gnmu, gnmu}, std::array<size_t, 2>{nmu, gnmu},
                                            std::array<size_t, 2>{nmu0, 0});
    if (!dump.readEntry(hslab3, "Luv"))
    {
      app_error(" Error in THCHamiltonian::getHamiltonianOperations():");
      app_error(" Problems reading Luv. ");
      APP_ABORT("");
    }
    /***************************************/
  }
  TGwfn.Global().barrier();

  // half-rotated Pia
  std::vector<shmSPCMatrix> rotcPua;
  rotcPua.reserve(ndet);
  for (int i = 0; i < ndet; i++)
    rotcPua.emplace_back(shmSPCMatrix({grotnmu, nel_}, shared_allocator<SPComplexType>{TGwfn.Node()}));
  std::vector<shmSPCMatrix> cPua;
  cPua.reserve(ndet);
  for (int i = 0; i < ndet; i++)
    cPua.emplace_back(shmSPCMatrix({nmu, nel_}, shared_allocator<SPComplexType>{TGwfn.Node()}));
  if (TGwfn.Node().root())
  {
    // simple
    using ma::H;
    if (type == COLLINEAR)
    {
      boost::multi::array<SPComplexType, 2> A({NMO, NAEA});
      boost::multi::array<SPComplexType, 2> B({NMO, NAEB});
      for (int i = 0; i < ndet; i++)
      {
        // cPua = H(Piu) * conj(A)
        ma::Matrix2MA('T', PsiT[2 * i], A);
        ma::product(H(Piu), A, cPua[i](cPua[i].extension(0), {0, NAEA}));
        ma::product(H(rotPiu), A, rotcPua[i](cPua[i].extension(0), {0, NAEA}));
        ma::Matrix2MA('T', PsiT[2 * i + 1], B);
        ma::product(H(Piu), B, cPua[i](cPua[i].extension(0), {NAEA, NAEA + NAEB}));
        ma::product(H(rotPiu), B, rotcPua[i](cPua[i].extension(0), {NAEA, NAEA + NAEB}));
      }
    }
    else
    {
      boost::multi::array<SPComplexType, 2> A({PsiT[0].size(1), PsiT[0].size(0)});
      for (int i = 0; i < ndet; i++)
      {
        ma::Matrix2MA('T', PsiT[i], A);
        // cPua = H(Piu) * conj(A)
        ma::product(H(Piu), A, cPua[i]);
        ma::product(H(rotPiu), A, rotcPua[i]);
      }
    }
  }
  TGwfn.Node().barrier();

  // rotated 1 body hamiltonians
  shmCMatrix hij({ndet, (NAEA + NAEB) * NMO}, shared_allocator<ComplexType>{TGwfn.Node()});
  if (TGwfn.Node().root())
  {
    int skp = ((type == COLLINEAR) ? 1 : 0);
    for (int n = 0, nd = 0; n < ndet; ++n, nd += (skp + 1))
    {
      check_wavefunction_consistency(type, &PsiT[nd], &PsiT[nd + skp], NMO, NAEA, NAEB);
      auto hij_=rotateHij(type, &PsiT[nd], &PsiT[nd + skp], H1);
      std::copy_n(hij_.origin(), hij_.num_elements(), raw_pointer_cast(hij[n].origin()));
    }
  }
  TGwfn.Node().barrier();

  return THCOps<MP,REAL>(TGwfn.TG_local(), NMO, NAEA, NAEB, type, nmu0, rotnmu0, std::move(H1), std::move(hij),
                std::move(rotMuv), std::move(rotPiu), std::move(rotcPua), std::move(Luv), std::move(Piu),
                std::move(cPua), std::move(vn0), E0);
}

// single writer right now
template<class shmVMatrix, class shmCMatrix>
inline void writeTHCOps(hdf_archive& dump,
                        WALKER_TYPES type,
                        int NMO,
                        int NAEA,
                        int NAEB,
                        size_t nmu0,
                        size_t rotnmu0,
                        int ndet,
                        TaskGroup_& TGprop,
                        TaskGroup_& TGwfn,
                        shmCMatrix& H1,
                        shmVMatrix& rotPiu,
                        shmVMatrix& rotMuv,
                        shmVMatrix& Piu,
                        shmVMatrix& Luv,
                        shmCMatrix& vn0,
                        ComplexType E0)
{
  size_t gnmu(Luv.size(1));
  size_t grotnmu(rotMuv.size(1));
  if (TGwfn.Global().root())
  {
    dump.push("HamiltonianOperations");
    dump.push("THCOps");
    std::vector<int> dims{NMO, NAEA, NAEB, ndet, type, int(gnmu), int(grotnmu)};
    dump.write(dims, "dims");
    std::vector<ComplexType> et{E0};
    dump.write(et, "E0");
    dump.write(H1, "H1");
    dump.write(vn0, "v0");
    dump.write(rotPiu, "HalfTransformedFullOrbitals");
    ma_hdf5::write_distributed_MA(rotMuv, {rotnmu0, 0}, {grotnmu, grotnmu}, dump, "HalfTransformedMuv", TGwfn);
    ma_hdf5::write_distributed_MA(Piu, {0, nmu0}, {size_t(NMO), gnmu}, dump, "Orbitals", TGprop);
    ma_hdf5::write_distributed_MA(Luv, {nmu0, 0}, {gnmu, gnmu}, dump, "Luv", TGprop);
    dump.pop();
    dump.pop();
  }
  else
  {
    ma_hdf5::write_distributed_MA(rotMuv, {rotnmu0, 0}, {grotnmu, grotnmu}, dump, "HalfTransformedMuv", TGwfn);
    ma_hdf5::write_distributed_MA(Piu, {0, nmu0}, {size_t(NMO), gnmu}, dump, "Orbitals", TGprop);
    ma_hdf5::write_distributed_MA(Luv, {nmu0, 0}, {gnmu, gnmu}, dump, "Luv", TGprop);
  }
  TGwfn.Global().barrier();
}

} // namespace afqmc
} // namespace sfqmc

#endif
