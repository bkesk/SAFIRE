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

#include <random>
#include <boost/optional.hpp>
#include "utilities/h5_utils.hpp"

#include "AFQMC/Utilities/readWfn.h"
#include "WavefunctionFactory.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
//#include "AFQMC/Wavefunctions/Excitations.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
Wavefunction WavefunctionFactory::fromHDF5(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                           ptree pt_in,
                                           WALKER_TYPES walker_type,
                                           Hamiltonian& h,
                                           int targetNW)
{
  ptree pt = interpret_inputs(pt_in);

  std::string info = pt.get<std::string>("system");
  if (InfoMap.find(info) == InfoMap.end())
    utils::check(false,"ERROR: Undefined system in WavefunctionFactory. ");

  bool dense_trial;
  std::string name          = pt.get<std::string>("name");
  std::string filename      = pt.get<std::string>("filename");
  bool recompute_ci  = pt.get<bool>("rediag");
  int ndets_to_read  = pt.get<int>("ndets_to_read");
  boost::optional<bool> dense_trial_opt;// = pt.get_optional<bool>("dense_trial");
  if( auto node = pt.get_child_optional("dense_trial") )
    dense_trial_opt = node->get_value_optional<bool>(); 

  AFQMCInfo& AFinfo = InfoMap[info];
  ComplexType NCE     = 0.0;

  int NMO  = AFinfo.NMO;
  int nup = AFinfo.nup;
  int ndown = AFinfo.ndown;
  int nspin = (walker_type == COLLINEAR) ? 2 : 1;
  int npol = (walker_type == NONCOLLINEAR) ? 2 : 1;
  utils::check((walker_type != NONCOLLINEAR) or (ndown == 0),
    " Error in Wavefunctions/WavefunctionFactory::fromHDF5: noncollinear && ndown!=0. \n\n\n ");

  std::vector<int> excitations;
  WAVEFUNCTION_TYPES wfn_type; 
  if(mpi->comm.root()) { 
    wfn_type = afqmc::getWavefunctionType(filename); 
    int itype(wfn_type);
    mpi->comm.broadcast_n(&itype,1,0); 
  } else {
    int itype;
    mpi->comm.broadcast_n(&itype,1,0); 
    wfn_type = WAVEFUNCTION_TYPES(itype);
  }

  // everyone reading for now, change it problematic
  h5::file file(filename,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");

  if (wfn_type == NOMSD_WFN)
  {
    app_log(1," Wavefunction type: NOMSD");
    h5::group ngrp = wgrp.open_group("NOMSD");
    nda::array<ComplexType,1> ci;

    // Read common trial wavefunction input options.
    WALKER_TYPES input_wtype;
    getCommonInput(ngrp, NMO, nup, ndown, ndets_to_read, ci, input_wtype);
    
    // validation blocks
    utils::check(input_wtype != NONCOLLINEAR or walker_type == NONCOLLINEAR,
        "Error: Trial wavefunction is NONCOLLINEAR and requires NONCOLLINEAR walkers.");
    
    NCE = h.getNuclearCoulombEnergy();

    //mpi->comm.broadcast_n(ci.data(), ci.size());
    //mpi->comm.broadcast_value(NCE);

    // Create Trial wavefunction.
    auto PsiT = read_nomsd_wavefunction<MEM>(ngrp,ndets_to_read,walker_type,NMO,nup,ndown);
    utils::check(PsiT.shape() == std::array<long,2>{ndets_to_read,nspin}, "Shape mismatch");

    // Set initial walker's Slater matrix.
    getInitialGuess(ngrp, mpi, name, NMO, nup, ndown, walker_type);

    // if not set, get default based on HamTYpe
    // use sparse trial only on KP runs
    if (dense_trial_opt == boost::none)
    {
      dense_trial = true; 
      if (h.getHamType() == KPFactorized || h.getHamType() == KPTHC)
        dense_trial = false; 
    } else {
      dense_trial = *dense_trial_opt;
    }

    auto HOps = h.getHamiltonianOperations<MEM>(walker_type, mpi, PsiT);
    if (dense_trial)
    {
      using MType = memory::shared_array<MEM,ComplexType,2>; 
      nda::array<MType,2> PsiT_dense(ndets_to_read,nspin);
      // insert empty matrices 
      for(int id=0; id<ndets_to_read; ++id)
        for(int is=0; is<nspin; ++is)
          PsiT_dense(id,is) = memory::make_shared_array<MEM,ComplexType,2>(mpi,PsiT(id,is).shape());
      mpi->comm.barrier();
      if constexpr (MEM==HOST_MEMORY) {
        if(mpi->node_comm.root()) 
          for(int id=0; id<ndets_to_read; ++id)
            for(int is=0; is<nspin; ++is)
              PsiT_dense(id,is)() = math::sparse::to_array<ComplexType>(PsiT(id,is));
      } else {
        for(int id=0; id<ndets_to_read; ++id)
          for(int is=0; is<nspin; ++is)
            PsiT_dense(id,is)() = math::sparse::to_array<ComplexType>(PsiT(id,is));
      }
      mpi->comm.barrier();
      return Wavefunction(NOMSD<MEM,MType>(AFinfo, pt, walker_type, mpi, std::move(HOps), 
                                     std::move(ci), std::move(PsiT_dense),NCE,targetNW)); 
    }
    else
    {
      return Wavefunction(NOMSD<MEM,PsiT_Matrix<MEM>>(AFinfo, pt, walker_type, mpi, std::move(HOps), 
                                     std::move(ci), std::move(PsiT),NCE,targetNW)); 
    }
  }
  else if (wfn_type == PHMSD_WFN)
  {
/*
    app_log(1," Wavefunction type: PHMSD");

    / * Implementation notes:
     *  - PsiT: [Nact, NMO] where Nact is the number of active space orbitals,
     *                     those that participate in the ci expansion
     *  - The half rotation is done with respect to the supermatrix PsiT
     *  - Need to calculate Nact and create a mapping from orbital index to actice space index.
     *    Those orbitals in the corresponding virtual space (not in active) map to -1 as a precaution.
     * /

    // assuming walker_type==COLLINEAR for now, specialize a type for perfect pairing PHMSD
    std::vector<PsiT_Matrix> PsiT_MO;
    std::string wfn_type;
    if (dump.push("PHMSD", false)<0)
      APP_ABORT(" Error in WavefunctionFactory: Group PHMSD not found. ");
    std::vector<int> occbuff;
    std::vector<ComplexType> coeffs;
    // 1. Read occupancies and coefficients.
    app_log(1," Reading PHMSD wavefunction from {}", filename);
    read_ph_wavefunction_hdf(dump, coeffs, occbuff, ndets_to_read, walker_type, TGwfn.Node(), 
				NMO, nup, ndown, PsiT_MO, wfn_type);
    app_log(1," Finished reading PHMSD wavefunction ");
    boost::multi::array_ref<int, 2> occs(raw_pointer_cast(occbuff.data()), {ndets_to_read, nup + ndown});
    if(recompute_ci) {
      // 2. Compute Variational Energy / update coefficients
      app_log(1," Computing variational energy of trial wavefunction.");
      computeVariationalEnergyPHMSD(TGwfn, h, occs, coeffs, ndets_to_read, nup, ndown, NMO, recompute_ci);
      app_log(1," Finished computing variational energy of trial wavefunction.");
    }

    // build reference MOs (PsiT_MO) if needed...
    if (wfn_type == "occ")
    {
      build_PsiT_MO_phmsd(TGwfn,walker_type,NPOL,NMO,nup,ndown,ndets_to_read,coeffs,occbuff,PsiT_MO);
    }
    else if (wfn_type == "mixed")
    {
      // nothing to do
      for( auto& v: PsiT_MO)
        RUNTIME_CHECK(v.size(1) == NPOL*NMO, "");
    }
    else if (wfn_type == "matrix")
    {
      APP_ABORT("Error: wfn_type=matrix not allowed in WavefunctionFactory with PHMSD wavefunction.");
    }
    else
    {
      APP_ABORT("Error: Unknown wfn_type in WavefunctionFactory with MSD wavefunction.");
    }
    TGwfn.Node().barrier();

    // 3. Construct Structures.
    ph_excitations<int, ComplexType> abij = build_ph_struct(coeffs, occs, ndets_to_read, TGwfn.Node(), NPOL*NMO, nup, ndown);

    // find active space orbitals and create super trial matrix PsiT
    std::vector<PsiT_Matrix> PsiT;
    PsiT.reserve(2);
    // expect mapped over range [0-2*NMO], but alpha and beta sectors with 0-based active indexes
    std::map<int, int> mo2active(find_active_space(PsiT_MO.size() == 1, walker_type, abij, NMO, nup, ndown));
    std::map<int, int> acta2mo;
    std::map<int, int> actb2mo;
    std::vector<int> active_alpha;
    std::vector<int> active_beta;
    std::vector<int> active_combined;
    for (int i = 0; i < NPOL*NMO; i++)
    {
      if (mo2active[i] >= 0)
      {
        active_alpha.push_back(i);
        acta2mo[mo2active[i]] = i;
      }
      if (walker_type == COLLINEAR) {
        if(mo2active[i + NMO] >= 0)
        {
          active_beta.push_back(i);
          actb2mo[mo2active[i + NMO]] = i + NMO;
        }
        if (mo2active[i] >= 0 || mo2active[i + NMO] >= 0)
          active_combined.push_back(i);
      } else {
        if (mo2active[i] >= 0) 
          active_combined.push_back(i);
      }
    }
    if (PsiT_MO.size() == 1)
    {
      // RHF reference
      std::vector<size_t> nnzpr(get_nnz(PsiT_MO[0], active_combined.data(), active_combined.size(), 0));
      PsiT.emplace_back(PsiT_Matrix({active_combined.size(), NPOL*NMO}, 
				    {0, 0}, nnzpr, Alloc(TGwfn.Node())));
      if (TGwfn.Node().root())
      {
        for (int k = 0; k < active_combined.size(); k++)
        {
          size_t ki = active_combined[k]; // occupied state #k
          auto col  = PsiT_MO[0].non_zero_indices2_data(ki);
          auto val  = PsiT_MO[0].non_zero_values_data(ki);
          for (size_t ic = 0, icend = PsiT_MO[0].num_non_zero_elements(ki); ic < icend; ic++, ++col, ++val)
            PsiT[0].emplace_back({k, *col}, *val);
        }
      }
      // add second component
      if( walker_type == COLLINEAR )  
        PsiT.emplace_back(PsiT[0]);
    }
    else
    {
      // UHF reference
      std::vector<size_t> nnzpr(get_nnz(PsiT_MO[0], active_alpha.data(), active_alpha.size(), 0));
      PsiT.emplace_back(PsiT_Matrix({active_alpha.size(), NPOL*NMO}, 
				    {0, 0}, nnzpr, Alloc(TGwfn.Node())));
      if (TGwfn.Node().root())
      {
        for (int k = 0; k < active_alpha.size(); k++)
        {
          size_t ki = active_alpha[k]; // occupied state #k
          auto col  = PsiT_MO[0].non_zero_indices2_data(ki);
          auto val  = PsiT_MO[0].non_zero_values_data(ki);
          for (size_t ic = 0, icend = PsiT_MO[0].num_non_zero_elements(ki); ic < icend; ic++, ++col, ++val)
            PsiT.back().emplace_back({k, *col}, *val);
        }
      }
      nnzpr = get_nnz(PsiT_MO[1], active_beta.data(), active_beta.size(), 0);
      PsiT.emplace_back(PsiT_Matrix({active_beta.size(), NPOL*NMO}, 
				    {0, 0}, nnzpr, Alloc(TGwfn.Node())));
      if (TGwfn.Node().root())
      {
        for (int k = 0; k < active_beta.size(); k++)
        {
          size_t ki = active_beta[k]; // occupied state #k
          auto col  = PsiT_MO[1].non_zero_indices2_data(ki);
          auto val  = PsiT_MO[1].non_zero_values_data(ki);
          for (size_t ic = 0, icend = PsiT_MO[1].num_non_zero_elements(ki); ic < icend; ic++, ++col, ++val)
            PsiT[1].emplace_back({k, *col}, *val);
        }
      }
    }
    // now that mappings have been constructed, map indexes of excited state orbitals
    // to the corresponding active space indexes
    if (TGwfn.Node().root())
    {
      // map reference
      auto refc = abij.reference_configuration();
      for (int i = 0; i < nup + ndown; i++, ++refc)
        *refc = mo2active[*refc];
      for (int n = 1; n < abij.maximum_excitation_number()[0]; n++)
      {
        auto it  = abij.alpha_begin(n);
        auto ite = abij.alpha_end(n);
        for (; it < ite; ++it)
        {
          auto exct = (*it) + n; // only need to map excited state indexes
          for (int np = 0; np < n; ++np, ++exct)
            *exct = mo2active[*exct];
        }
      }
      for (int n = 1; n < abij.maximum_excitation_number()[1]; n++)
      {
        auto it  = abij.beta_begin(n);
        auto ite = abij.beta_end(n);
        for (; it < ite; ++it)
        {
          auto exct = (*it) + n; // only need to map excited state indexes
          for (int np = 0; np < n; ++np, ++exct)
            *exct = mo2active[*exct];
        }
      }
    }
    TGwfn.Node().barrier();

    getInitialGuess(dump, mpi, name, NMO, nup, ndown, walker_type);

    // csr matrix with determinant couplings in COLLINEAR case
    std::vector<PsiT_Matrix> det_coupling_matrix;
    det_coupling_matrix.reserve(2);

    auto n_unique(abij.number_of_unique_excitations());
    app_log(1," Number of unique determinants per spin channel: {} {} ",
                n_unique[0],n_unique[1]);
    std::vector<int> counts_alpha(n_unique[0]);
    std::vector<int> counts_beta(n_unique[1]);
    if (TGwfn.Node().root())
    {
      for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it)
      {
        ++counts_alpha[std::get<0>(*it)];
        ++counts_beta[std::get<1>(*it)];
      }
    }
    TGwfn.Node().broadcast_n(counts_alpha.begin(), counts_alpha.size());
    TGwfn.Node().broadcast_n(counts_beta.begin(), counts_beta.size());

    using ucsr_mat_t = ma::sparse::ucsr_matrix<ComplexType, int, int,
                                  shared_allocator<ComplexType>, ma::sparse::is_root>;
    std::vector<ucsr_mat_t> unsorted_det_coupling;
    unsorted_det_coupling.reserve(2);
    unsorted_det_coupling.emplace_back(ucsr_mat_t({n_unique[0],n_unique[1]},
                {0, 0}, counts_alpha, shared_allocator<ComplexType>{TGwfn.Node()}));
    unsorted_det_coupling.emplace_back(ucsr_mat_t({n_unique[1],n_unique[0]},
                {0, 0}, counts_beta, shared_allocator<ComplexType>{TGwfn.Node()}));

    if (TGwfn.Node().root())
    {
      int ni = 0;
      for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it, ++ni)
      {
        // sparse matrix
        unsorted_det_coupling[0].emplace(std::array<int, 2>{std::get<0>(*it),std::get<1>(*it)},
                                       ma::conj(std::get<2>(*it)));
        unsorted_det_coupling[1].emplace(std::array<int, 2>{std::get<1>(*it),std::get<0>(*it)},
                                       ma::conj(std::get<2>(*it)));
      }
    }
    TGwfn.Node().barrier();

    // ucsr -> csr	
    det_coupling_matrix.emplace_back( unsorted_det_coupling[0] );
    det_coupling_matrix.emplace_back( unsorted_det_coupling[1] );
    TGwfn.Node().barrier();

#if !defined(ENABLE_CUDA) && !defined(ENABLE_HIP)
    if (TGwfn.TG_local().size() > 1)
    {
      SlaterDetOperations SDetOp(SlaterDetOperations_shared<ComplexType>(NPOL * NMO, nup));
      if(mixed_precision) {
        auto HOps(getHamOps<true>(restart_file, walker_type, NMO, nup, ndown, PsiT,
		TGprop, TGwfn, h));
        TGwfn.Node().barrier();
        return Wavefunction(PHMSD<true>(AFinfo, pt, TGwfn, std::move(SDetOp), std::move(HOps),
                    std::move(acta2mo), std::move(actb2mo), std::move(abij), 
                    std::move(det_coupling_matrix),
                    std::move(PsiT), walker_type, NCE, targetNW));
      } else {
        auto HOps(getHamOps<false>(restart_file, walker_type, NMO, nup, ndown, PsiT,
		TGprop, TGwfn, h));
        TGwfn.Node().barrier();
        return Wavefunction(PHMSD<false>(AFinfo, pt, TGwfn, std::move(SDetOp), std::move(HOps),
                    std::move(acta2mo), std::move(actb2mo), std::move(abij), 
                    std::move(det_coupling_matrix),
                    std::move(PsiT), walker_type, NCE, targetNW));
      }
    } 
    else 
#endif
    {
      SlaterDetOperations SDetOp(
                SlaterDetOperations_serial<ComplexType, DeviceBufferManager>(NPOL * NMO, 
                        nup, DeviceBufferManager{})
                                );
      if(mixed_precision) {
        auto HOps(getHamOps<true>(restart_file, walker_type, NMO, nup, ndown, PsiT,
		TGprop, TGwfn, h));
        TGwfn.Node().barrier();
        return Wavefunction(PHMSD<true>(AFinfo, pt, TGwfn, std::move(SDetOp), std::move(HOps),
                    std::move(acta2mo), std::move(actb2mo), std::move(abij), 
                    std::move(det_coupling_matrix),
                    std::move(PsiT), walker_type, NCE, targetNW));
      } else {
        auto HOps(getHamOps<false>(restart_file, walker_type, NMO, nup, ndown, PsiT,
		TGprop, TGwfn, h));
        TGwfn.Node().barrier();
        return Wavefunction(PHMSD<false>(AFinfo, pt, TGwfn, std::move(SDetOp), std::move(HOps),
                    std::move(acta2mo), std::move(actb2mo), std::move(abij), 
                    std::move(det_coupling_matrix),
                    std::move(PsiT), walker_type, NCE, targetNW));
      }
    }
*/
  }
  else
  {
    utils::check(false," Error: Unknown wave-function wfn_type: {}", int(wfn_type));
    return Wavefunction{};
  }
  return Wavefunction{};
}

/*
 * Read Initial walker from file. Needs all mpi tasks, since it allocates on shared memory.
*/
void WavefunctionFactory::getInitialGuess(h5::group grp,
         std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
         std::string& name, int NMO, int nup, int ndown, WALKER_TYPES walker_type)
{
  int nspin = (walker_type == COLLINEAR) ? 2 : 1;
  int npol = (walker_type == NONCOLLINEAR) ? 2 : 1;
  nda::array<int,1> dims(5);
  nda::h5_read(grp,"dims",dims);
  WALKER_TYPES wtype(initWALKER_TYPES(dims[3]));
  auto guess = initial_guess.find(name);
  utils::check(guess == initial_guess.end(), 
               "Error: Problems adding new initial guess, already exists.");
  auto newg = initial_guess.insert(std::make_pair(name, memory::make_shared_array<HOST_MEMORY,ComplexType, 3>(mpi,{nspin, npol * NMO, nup})));
  utils::check(newg.second, " Error: Problems adding new initial guess. ");
  if(mpi->comm.root()) {  
    auto M = ((newg.first)->second)();
    M() = ComplexType(0.0, 0.0);
    auto Mup = M(0,nda::ellipsis{});
    utils::h5_read(grp,"Psi0_alpha",Mup);
    if (walker_type == COLLINEAR)
    {
      if (wtype == COLLINEAR)
      {
        auto Mdn = M(1,nda::range::all,nda::range(ndown));
        utils::h5_read(grp,"Psi0_beta",Mdn);
      }
      else if (wtype == CLOSED)
      {
        utils::check(nup == ndown, "Error: wfn_type:Closed with nup != ndown.");
        M(1,nda::ellipsis{}) = Mup();
      }
      else
        utils::check(false," Error: Unknown wtype. ");
    }
  }
  mpi->comm.barrier();
}


/*
void WavefunctionFactory::computeVariationalEnergyPHMSD(TaskGroup_& TG,
                                                        Hamiltonian& ham,
                                                        boost::multi::array_ref<int, 2>& occs,
                                                        std::vector<ComplexType>& coeff,
                                                        int ndets,
                                                        int nup,
                                                        int ndown,
                                                        int NMO,
                                                        bool recompute_ci)
{
  // CI coefficients can in general be complex and want to avoid two mpi communications so
  // keep everything complex even if Hamiltonian matrix elements are real.
  // Allocate H in Node's shared memory, but use as a raw array with proper synchronization
  int dim((recompute_ci ? ndets : 0));
  boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>> H({dim, dim}, TG.Node());
  boost::multi::array<ComplexType, 1> energy(iextensions<1u>{2});
  using std::fill_n;
  fill_n(H.origin(), H.num_elements(), ComplexType(0.0));           // this call synchronizes
  fill_n(energy.origin(), energy.num_elements(), ComplexType(0.0)); // this call synchronizes
  ComplexType enuc = ham.getNuclearCoulombEnergy();
  for (int idet = 0; idet < ndets; idet++)
  {
    // These should already be sorted.
    boost::multi::array_ref<int, 1> deti(occs[idet].origin(), {nup + ndown});
    ComplexType cidet = coeff[idet];
    for (int jdet = idet; jdet < ndets; jdet++)
    {
      // Compute <Di|H|Dj>
      if ((idet * ndets + jdet) % TG.Global().size() == TG.Global().rank())
      {
        if (idet == jdet)
        {
          ComplexType Hii(0.0);
          Hii = slaterCondon0(ham, deti, NMO) + enuc;
          energy[0] += ma::conj(cidet) * cidet * Hii;
          energy[1] += ma::conj(cidet) * cidet;
          if (recompute_ci)
            H[idet][idet] = Hii;
        }
        else
        {
          ComplexType Hij(0.0);
          boost::multi::array_ref<int, 1> detj(occs[jdet].origin(), {nup + ndown});
          ComplexType cjdet = coeff[jdet];
          int perm          = 1;
          std::vector<int> excit;
          int nexcit = getExcitation(deti, detj, excit, perm);
          if (nexcit == 1)
          {
            Hij = ComplexType(perm) * slaterCondon1(ham, excit, detj, NMO);
          }
          else if (nexcit == 2)
          {
            Hij = ComplexType(perm) * slaterCondon2(ham, excit, NMO);
          }
          energy[0] += ma::conj(cidet) * cjdet * Hij + ma::conj(cjdet) * cidet * ma::conj(Hij);
          if (recompute_ci)
          {
            H[idet][jdet] = Hij;
            H[jdet][idet] = ma::conj(Hij);
          }
        }
      }
    }
  }
  TG.Node().barrier();
  if (TG.Node().root() && recompute_ci)
    TG.Cores().all_reduce_in_place_n(raw_pointer_cast(H.origin()), H.num_elements(), std::plus<>());
  TG.Global().all_reduce_in_place_n(energy.origin(), 2, std::plus<>());
  app_log(1," - Variational energy of trial wavefunction: {}", energy[0] / energy[1]);
  if (recompute_ci)
  {
    app_log(1," - Diagonalizing CI matrix.");
    using RVector = Vector<RealType>;
    using CMatrix = Matrix<ComplexType>;
    // Want a "unique" solution for all cores/nodes.
    if (TG.Global().rank() == 0)
    {
      std::pair<RVector, CMatrix> Sol = ma::symEigSelect<RVector, CMatrix>(H, 1);
      app_log(1," - Updating CI coefficients. ");
      app_log(1," - Recomputed coefficient of first determinant: {}", Sol.second[0][0]);
      for (int idet = 0; idet < ndets; idet++)
      {
        ComplexType ci = Sol.second[0][idet];
        // Do we want this much output?
        //app_log() << idet << " old: " << coeff[idet] << " new: " << ci << "";
        coeff[idet] = ci;
      }
      app_log(1," - Recomputed variational energy of trial wavefunction: {}",Sol.first[0]);
    }
    TG.Global().broadcast_n(raw_pointer_cast(coeff.data()), coeff.size(), 0);
  }
}

/ **
 * Compute the excitation level between two determinants.
 * /
int WavefunctionFactory::getExcitation(boost::multi::array_ref<int, 1>& deti,
                                       boost::multi::array_ref<int, 1>& detj,
                                       std::vector<int>& excit,
                                       int& perm)
{
  std::vector<int> from_orb, to_orb;
  // Work out which orbitals are excited from / to.
  std::set_difference(detj.begin(), detj.end(), deti.begin(), deti.end(), std::inserter(from_orb, from_orb.begin()));
  std::set_difference(deti.begin(), deti.end(), detj.begin(), detj.end(), std::inserter(to_orb, to_orb.begin()));
  int nexcit = from_orb.size();
  if (nexcit <= 2)
  {
    for (int i = 0; i < from_orb.size(); i++)
      excit.push_back(from_orb[i]);
    for (int i = 0; i < to_orb.size(); i++)
      excit.push_back(to_orb[i]);
    int nperm = 0;
    int nmove = 0;
    for (auto o : from_orb)
    {
      auto it = std::find(detj.begin(), detj.end(), o);
      int loc = std::distance(detj.begin(), it);
      nperm += loc - nmove;
      nmove += 1;
    }
    nmove = 0;
    for (auto o : to_orb)
    {
      auto it = std::find(deti.begin(), deti.end(), o);
      int loc = std::distance(deti.begin(), it);
      nperm += loc - nmove;
      nmove += 1;
    }
    perm = nperm % 2 == 1 ? -1 : 1;
  }
  return nexcit;
}

ComplexType WavefunctionFactory::slaterCondon0([[maybe_unused]] Hamiltonian& ham, 
                                               [[maybe_unused]] boost::multi::array_ref<int, 1>& det, 
                                               [[maybe_unused]] int NMO)
{
  APP_ABORT("Error: slaterCondon0 Feature removed.");
/ *
  ComplexType one_body = ComplexType(0.0);
  ComplexType two_body = ComplexType(0.0);
  for (int i = 0; i < det.size(); i++)
  {
    int oi = det[i];
    one_body += ComplexType(ham.H(oi, oi));
    for (int j = i + 1; j < det.size(); j++)
    {
      int oj = det[j];
      two_body += ComplexType(ham.H(oi, oj, oi, oj)) - ComplexType(ham.H(oi, oj, oj, oi));
    }
  }
  return one_body + two_body;
* /
  return ComplexType(0.0);
}

ComplexType WavefunctionFactory::slaterCondon1([[maybe_unused]] Hamiltonian& ham,
                                               [[maybe_unused]] std::vector<int>& excit,
                                               [[maybe_unused]] boost::multi::array_ref<int, 1>& det,
                                               [[maybe_unused]] int NMO)
{
  APP_ABORT("Error: slaterCondon1 Feature removed.");
/ *
  int i              = excit[0];
  int a              = excit[1];
  ComplexType one_body = ComplexType(ham.H(i, a));
  ComplexType two_body = ComplexType(0.0);
  for (auto j : det)
  {
    two_body += ComplexType(ham.H(i, j, a, j)) - ComplexType(ham.H(i, j, j, a));
  }
  return one_body + two_body;
* /
  return ComplexType(0.0);
}

ComplexType WavefunctionFactory::slaterCondon2([[maybe_unused]] Hamiltonian& ham, 
                                               [[maybe_unused]] std::vector<int>& excit, 
                                               [[maybe_unused]] int NMO)
{
  APP_ABORT("Error: slaterCondon2 Feature removed.");
/ *
  int i = excit[0];
  int j = excit[1];
  int a = excit[2];
  int b = excit[3];
  return ComplexType(ham.H(i, j, a, b) - ham.H(i, j, b, a));
* /
  return ComplexType(0.0);
}

void WavefunctionFactory::build_PsiT_MO_phmsd(TaskGroup_& TG, 
      WALKER_TYPES walker_type, int NPOL, 
      int NMO, int nup, int ndown, int ndets_to_read, 
      std::vector<ComplexType>& coeffs, 
      std::vector<int>& occbuff, std::vector<PsiT_Matrix>& PsiT_MO)
{
  using Alloc = shared_allocator<ComplexType>;
  ComplexType one(1.0, 0.0);
  boost::multi::array_ref<int, 2> occs(raw_pointer_cast(occbuff.data()), {ndets_to_read, nup + ndown});
  PsiT_MO.clear();

  bool trivial_ref = true;
  for(int i=0; i<nup; i++)
    if(occs[0][i] != i) {
      trivial_ref = false;
      break;
    }
  for(int i=0; i<ndown; i++)
    if(occs[0][nup+i] != NMO+i) {
      trivial_ref = false;
      break;
    }

  auto sort_with_sign = [](int NE, int* Iwork) {
    RealType ci=1.0;
    for (int i = 0; i < NE; i++)
      for (int j = i + 1; j < NE; j++)
      {
        if (Iwork[j] < Iwork[i])
        {
          ci *= RealType(-1.0);
          std::swap(Iwork[i], Iwork[j]);
        }
      }
    return ci;
  };

  if(trivial_ref) {

    PsiT_MO.reserve(1);
    PsiT_MO.emplace_back(PsiT_Matrix({NPOL*NMO, NPOL*NMO},
                                     {0, 0}, 1, Alloc(TG.Node())));

    // makes sense to move the reordering of non-compact excitations to here!
    if (TG.Node().root())
      for (int k = 0; k < NPOL*NMO; k++)
        PsiT_MO.back().emplace_back({k, k}, one);

  } else {

    // reference determinant is non-trivial (occupy bottom nalpha/nbeta states...)
    // build non-trivial reference and redefine excitations with respect to this new reference...
    app_log(1," Found non-trivial reference determinant. Constructing appropriate reference state.");

    // if beta reference configuration has singly occupied states, 
    // you will need separate references
    bool separate_references = false;
    if(walker_type == COLLINEAR)
      for(int i=0; i<ndown; ++i) 
        if( *std::find(occbuff.begin(), occbuff.begin() + nup, occbuff[nup+i]-NMO) !=
          occbuff[nup+i]-NMO ) {
          separate_references = true;
          break;
        }

    if(separate_references) { // only if collinear and can't find a unique reference


      PsiT_MO.reserve(2);
      PsiT_MO.emplace_back(PsiT_Matrix({NMO, NMO},
                                       {0, 0}, 1, Alloc(TG.Node())));
      PsiT_MO.emplace_back(PsiT_Matrix({NMO, NMO},
                                       {0, 0}, 1, Alloc(TG.Node())));

      if (TG.Node().root()) {

        // each spin has its own reference
        int dN_[2] = {0,NMO};
        int E0_[2] = {0,nup};
        int E1_[2] = {nup,nup+ndown};
        for(int is=0; is<2; is++) {
          int dN = dN_[is]; 
          int E0 = E0_[is]; 
          int E1 = E1_[is]; 
          std::vector<int> m(NMO,-1);  
          std::vector<int> im(NMO,-1);
          int norbs=0;  // number of states found so far
          // doubly occupied first, and since we checked all beta are doubly occp
          for(int i=E0; i<E1; i++) { 
            im[occs[0][i]-dN] = norbs;
            m[norbs++] = occs[0][i]-dN;
          }
          // now add all remaining states
          for(int n=1; n<occs.size(0); ++n) { 
            for(int i=E0; i<E1; i++) 
              if(im[occs[n][i]-dN] < 0) {
                im[occs[n][i]-dN] = norbs;
                m[norbs++] = occs[n][i]-dN;
              } 
          }
          // now change occupation strings according to the generated map
          for(int n=0; n<occs.size(0); ++n) { 
            for(int i=E0; i<E1; i++) occs[n][i] = im[occs[n][i]-dN]+dN;    
            //std::sort(occs[n].origin()+E0,occs[n].origin()+E1);
            coeffs[n] *= sort_with_sign(E1-E0,occs[n].origin()+E0);
          }
          for (int k = 0; k < norbs; k++) 
            PsiT_MO[is].emplace_back({k, m[k]}, one);
        } // is
    
      } // Node().root()

    } else { // separate_references

      PsiT_MO.reserve(1);
      PsiT_MO.emplace_back(PsiT_Matrix({NPOL*NMO, NPOL*NMO},
                                       {0, 0}, 1, Alloc(TG.Node())));

      if (TG.Node().root()) {

        std::vector<int> m(NPOL*NMO,-1);  
        std::vector<int> im(NPOL*NMO,-1);
        int norbs=0;  // number of states found so far
        if(walker_type == NONCOLLINEAR) {
          for(int i=0; i<nup; i++) {
	    im[occs[0][i]] = norbs;
	    m[norbs++] = occs[0][i];
	  }
          // now add all remaining states
          for(int n=1; n<occs.size(0); ++n) {
            for(int i=0; i<nup; i++) 
              if(im[occs[n][i]] < 0) {
                im[occs[n][i]] = norbs;
                m[norbs++] = occs[n][i];
              }
          }
        } else {
          // doubly occupied first, and since we checked all beta are doubly occp
          for(int i=0; i<ndown; i++) { 
            im[occs[0][nup+i]-NMO] = norbs;
            m[norbs++] = occs[0][nup+i]-NMO;
          }
          // singly occupied now
          for(int i=0; i<nup; ++i) 
            if( *std::find(occbuff.begin()+nup, occbuff.begin()+nup+ndown, 
		occbuff[i]+NMO) != occbuff[i]+NMO ) { 
              im[occs[0][i]] = norbs;
              m[norbs++] = occs[0][i];
	    }
          // now add all remaining states
          for(int n=1; n<occs.size(0); ++n) { 
            for(int i=0; i<nup; i++) 
              if(im[occs[n][i]] < 0) {
                im[occs[n][i]] = norbs;
                m[norbs++] = occs[n][i];
              } 
            for(int i=nup; i<nup+ndown; i++)        
              if(im[occs[n][i]-NMO] < 0) {
                im[occs[n][i]-NMO] = norbs;
                m[norbs++] = occs[n][i]-NMO;
              }
          }
        }
        // now change occupation strings according to the generated map
        for(int n=0; n<occs.size(0); ++n) { 
          for(int i=0; i<nup; i++) occs[n][i] = im[occs[n][i]];    
          if(walker_type == COLLINEAR)  
            for(int i=nup; i<nup+ndown; i++) occs[n][i] = im[occs[n][i]-NMO]+NMO;    
          //std::sort(occs[n].origin(),occs[n].origin()+nup+ndown);
          coeffs[n] *= sort_with_sign(nup+ndown,occs[n].origin());
        }
        for (int k = 0; k < norbs; k++) 
          PsiT_MO.back().emplace_back({k, m[k]}, one);

      } // Node().root()

    } // separate_references

    if(TG.Node().size() > 1) TG.Node().broadcast_n(occs.origin(),occs.num_elements(),0);

  } // trivial_ref
  TG.Node().barrier();
}

//ComplexType WavefunctionFactory::contractOneBody(std::vector<int>& det, std::vector<int>& excit, boost::multi::array_ref<ComplexType,2>& HSPot, int NMO)
//{
//ComplexType oneBody = ComplexType(0.0);
//int spini, spina;
//if(excit.size()==0) {
//for(auto i : det) {
//int oi = decodeSpinOrbital(i, spini, NMO);
//oneBody += HSPot[oi][oi];
//}
//} else {
//int oi = decodeSpinOrbital(excit[0], spini, NMO);
//int oa = decodeSpinOrbital(excit[1], spina, NMO);
//oneBody = HSPot[oi][oa];
//}
//return oneBody;
//}
*/

// Instantiate templates

template Wavefunction WavefunctionFactory::fromHDF5<HOST_MEMORY>(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>>,ptree,WALKER_TYPES,Hamiltonian&,int);

#if defined(ENABLE_DEVICE)

template Wavefunction WavefunctionFactory::fromHDF5<DEVICE_MEMORY>(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>>,ptree,WALKER_TYPES,Hamiltonian&,int);

#endif

} // namespace afqmc
} // namespace sfqmc
