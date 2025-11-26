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

#include <cstdlib>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config.h"
#include "utilities/check.hpp"

#include "AFQMC/config.h"

#include "ModelHamOpsGenerator.h"
//#include "AFQMC/HamiltonianOperations/ModelComponents/ModelComponent.hpp"
//#include "AFQMC/HamiltonianOperations/ModelComponents/SparseEnergy.hpp"
//#include "AFQMC/HamiltonianOperations/ModelHamOps.hpp"

namespace sfqmc
{
namespace afqmc
{
template<MEMORY_SPACE MEM, bool REAL> HamiltonianOperations<MEM> 
ModelHamOpsGenerator::getHamiltonianOperations_impl(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
/*
  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;

  if(type == CLOSED)
    APP_ABORT(" Error in ModelHamOpsGenerator::getHamiltonianOperations: CLOSED walker types not allowed with Model Hamiltonians. "); 

  // make sure there is at least 1 one-body hamiltonian in the components
  bool one_body_term(false);

  std::string base_error("Error in ModelHamOpsGenerator::getHamiltonianOperations(): ");

  if (type == COLLINEAR)
    utils::check(PsiT.size() % 2 == 0, "");
  int npol = ((type == NONCOLLINEAR) ? 2 : 1);

  // generate trial wavefunctions in appropriate form
  // ModelHamOps expects a vector of PsiC(i,a) = (psiT(a,i)) (complex of trial wfn Slater Matrix)
  std::vector<PsiC_Mat_Type> PsiC;
  PsiC.reserve(PsiT.size());
  for(auto const& v: PsiT) {
    utils::check(v.size(1) == npol*NMO, "");
    PsiC.emplace_back( PsiC_Mat_Type{{v.size(1), v.size(0)}, TGwfn.Node()} );
    // hide this behind some templated routine, since PsiT_Matrix is hard wired
    // to be a sparse matrix, this can be done for now
    if( TGwfn.Node().root() )
      ma::Matrix2MAREF('T',v,PsiC.back());
  }

  // everyone opens to be able to use HDF2CSR
  hdf_archive dump(TGwfn.Node());
  if (!dump.open(fileName, H5F_ACC_RDONLY))
    APP_ABORT(base_error + "Error opening integral file in ModelHamiltonian. ");
  if (dump.push("Hamiltonian", false)<0)
    APP_ABORT(base_error + "Group Hamiltonian not found. ");

  std::vector<int> Idata(8);
  if (!dump.readEntry(Idata, "dims"))
    APP_ABORT(base_error + " Problems reading dims. ");

  ValueType E0;
  {
    std::vector<RealType> E_(2);
    if (!dump.readEntry(E_, "Energies"))
      APP_ABORT(base_error + " Problems reading Energies. ");
    E0 = E_[0] + E_[1];
  }

  if (dump.push("ModelHamiltonian", false)<0)
    APP_ABORT(base_error + "Group ModelHamiltonian not found. ");
  
  int num_components(0);
  if (!dump.readEntry(num_components, "number_of_components"))
    APP_ABORT(base_error + " Problems reading dims. ");
  int maximum_connections(12);
  dump.readEntry(maximum_connections, "maximum_connectivity");
  
  std::vector<ModelComponent<MP,REAL>> Hams;
  Hams.reserve(num_components);
  shm_csrMat hij(tp_ul_ul{0,0}, tp_ul_ul{0, 0}, 0, Alloc(TGwfn.Node()));

  // accumulating terms 
  / *
   * Map:
   *   collect_U  ( opposite spin from 0-M, same spin from M-2M )  
   *   0: continuous charge 
   *   1: continuous spin 
   *   2: discrete charge 
   *   3: discrete spin 
   *   collect_J
   *   0: continuous charge 
   *   1: continuous spin 
   *   2: empty-container, for call to addComponent with discrete 
   * /
  std::vector<shm_csrMat> collect_U;  
  std::vector<shm_csrMat> collect_J;
  collect_U.reserve(4);
  for(int i=0; i<4; i++) 
    collect_U.emplace_back( shm_csrMat(tp_ul_ul{2*NMO, NMO}, tp_ul_ul{0, 0}, 
                                         maximum_connections, Alloc(TGwfn.Node()))); 
  collect_J.reserve(3); 
  // [2] is a place-holder for the empty case
  for(int i=0; i<3; i++)
    collect_J.emplace_back( shm_csrMat(tp_ul_ul{NMO, NMO}, tp_ul_ul{0, 0}, 
                                          maximum_connections, Alloc(TGwfn.Node()))); 

  for(int n=0; n<num_components; n++)  {
    if (dump.push("ModelComponent_"+std::to_string(n), false)<0)
      APP_ABORT(base_error + "Group ModelComponent_" + std::to_string(n) + " not found. ");
    
    std::string model_type("dummy");
    if (!dump.readEntry(model_type,"model_type"))
      APP_ABORT(base_error + " Problems reading model_type. ");
    std::transform(model_type.begin(), model_type.end(), model_type.begin(), (int (*)(int))tolower);

    if( model_type == "one_body" ) 
    {
      if(one_body_term)
        APP_ABORT(base_error + " Multiple one_body components defined.");  
      one_body_term=true;

      if (dump.push("tij", false)<0)
        APP_ABORT(base_error + " Problems reading tij dataset in model_type: " + model_type);
      shm_csrMat tij (csr_hdf5::HDF2CSR<shm_csrMat, 
                                        shared_allocator<SPValueType>>(dump, TGwfn.Node() ));
      dump.pop();  // tij

      std::string stype;
      if (!dump.readEntry(stype,"spin_type"))
        APP_ABORT(base_error + " Problems reading spin_type. ");
      std::transform(stype.begin(), stype.end(), stype.begin(), (int (*)(int))tolower);

      // returns a sparse matrix with the 1-body hamiltonian consistent with type
      hij = spin_to_walker_type(type, stype, tij);
    }
    else if( model_type == "hubbard_u" ) 
    {
      if (dump.push("Uij", false)<0)
        APP_ABORT(base_error + " Problems reading Uij dataset in model_type: " + model_type);
      shm_csrMat Uij (csr_hdf5::HDF2CSR<shm_csrMat, 
                                        shared_allocator<SPValueType>>(dump, TGwfn.Node() ));
      dump.pop();  // Uij

      if( (Uij.size(0) != NMO and 
          Uij.size(0) != 2*NMO) or 
          Uij.size(1) != NMO )
        APP_ABORT(base_error + " Found Hubbard_U model with inconsistent dimensions. ");
      // for safety
      if( Uij.num_non_zero_elements() == 0 )
        APP_ABORT(base_error + " Found empty Hubbard_U model. "); 

      std::string hst_type; 
      if (!dump.readEntry(hst_type,"hst_type"))
        APP_ABORT(base_error + " Problems reading hst_type. ");
      std::transform(hst_type.begin(), hst_type.end(), hst_type.begin(), (int (*)(int))tolower);

      int where_(-1);
      if(hst_type == "continuous_charge") where_ = 0; 
      else if(hst_type == "continuous_spin") where_ = 1;
      else if(hst_type == "discrete_charge") where_ = 2; 
      else if(hst_type == "discrete_spin") where_ = 3;
      else
        APP_ABORT(base_error + " Unknown hst_type: " + hst_type);

      app_log(1, "Hamiltonian component {} is using Hubbard-Stratanovich transformation type {} or Hubbard U", n, hst_type);

      if(TGwfn.Node().root()) {
        // doing this "by hand" to impose condition i>j
        // opposite spin (i<=j)
        // MAM: ignoring lower diagonal terms seems to confuse users, consider reading everything
        //      and just transposing the terms here when you write them in collect_U/J
        for( int i=0; i<NMO; ++i) {
          auto vals = Uij.non_zero_values_data(i);
          auto cols = Uij.non_zero_indices2_data(i);
          auto nnz = Uij.num_non_zero_elements(i);
          while( (nnz--) > 0 ) {
            auto u_ = *(vals++);
            int j = *(cols++);
            if( std::abs(u_) < 1e-6 or i>j ) {
              app_warning("Ignoring opposite spin Uij, i>j: i:{}, j:{}, U:{}",i,j,u_);
              continue;
	    }
            collect_U[where_].add( {i, j}, u_ );
          }
        }
        // same spin (i<j)
        for( int i=NMO; i<Uij.size(0); ++i) {
          auto vals = Uij.non_zero_values_data(i);
          auto cols = Uij.non_zero_indices2_data(i);
          auto nnz = Uij.num_non_zero_elements(i);
          while( (nnz--) > 0 ) {
            auto u_ = *(vals++);
            int j = *(cols++);
            if( std::abs(u_) < 1e-6 or (i-NMO)>=j ) { 
              app_warning("Ignoring same spin Uij, i=>j: i:{}, j:{}, U:{}",i,j,u_);
              continue;
	    }
            collect_U[where_].add( {i, j}, u_ );
          }
        }
      }
      TGwfn.Node().barrier();
    }
    else if( model_type == "hubbard_j" )  
    { 
      if (dump.push("Jij", false)<0)
        APP_ABORT(base_error + " Problems reading Jij dataset in model_type: " + model_type);
      shm_csrMat Jij (csr_hdf5::HDF2CSR<shm_csrMat, 
                                        shared_allocator<SPValueType>>(dump, TGwfn.Node() ));
      dump.pop();  // Jij
      
      if( Jij.size(0) != NMO or
          Jij.size(1) != NMO ) 
        APP_ABORT(base_error + " Found Hubbard_J model with inconsistent dimensions. ");
      // for safety
      if( Jij.num_non_zero_elements() == 0 )
        APP_ABORT(base_error + " Found empty Hubbard_J model. ");
      
      std::string hst_type; 
      if (!dump.readEntry(hst_type,"hst_type"))
        APP_ABORT(base_error + " Problems reading hst_type. ");
      std::transform(hst_type.begin(), hst_type.end(), hst_type.begin(), (int (*)(int))tolower);
      
      int where_(-1);
      if(hst_type == "continuous_charge") where_ = 0; 
      else if(hst_type == "continuous_spin") where_ = 1; 
      else if(hst_type == "discrete_charge") { 
        APP_ABORT(base_error + " Discrete HS transformations not allowed with Hubbard_J");
      } else if(hst_type == "discrete_spin") {
        APP_ABORT(base_error + " Discrete HS transformations not allowed with Hubbard_J");
      } else 
        APP_ABORT(base_error + " Unknown hst_type: " + hst_type);
    
      app_log(1, "Hamiltonian component {} is using Hubbard-Stratanovich transformation type {} for Hubbard J", n, hst_type);

      if(TGwfn.Node().root()) { 
        // doing this "by hand" to impose condition i>j
        // opposite spin (i<=j)
        for( int i=0; i<NMO; ++i) {
          auto vals = Jij.non_zero_values_data(i);
          auto cols = Jij.non_zero_indices2_data(i);
          auto nnz = Jij.num_non_zero_elements(i);
          while( (nnz--) > 0 ) {
            auto v_ = *(vals++);
            int j = *(cols++); 
            if( std::abs(v_) < 1e-6 or i>=j ) continue;
            collect_J[where_].add( {i, j}, v_ );
          }
        }
      }
      TGwfn.Node().barrier();
    }
    else  
      APP_ABORT(base_error + " Unknown model type: " + model_type);

    dump.pop();  // ModelComponent
  }
  if(not one_body_term)
    APP_ABORT(base_error + " Missing one_body component in ModelHamiltonian.");  
  if(hij.num_non_zero_elements() == 0)
    APP_ABORT(base_error + " Something went wrong, empty one_body component.");  

  // combine all U/J matrices for energy evaluation. 
  shm_csrMat combined_U(tp_ul_ul{2*NMO,NMO}, tp_ul_ul{0, 0}, maximum_connections, Alloc(TGwfn.Node()));
  shm_csrMat combined_J(tp_ul_ul{NMO,NMO}, tp_ul_ul{0, 0}, maximum_connections, Alloc(TGwfn.Node()));
  if(TGwfn.Node().root()) {
    for( auto& v: collect_U ) 
      csr::accumulate(SPValueType(1.0), v, combined_U);
    for( auto& v: collect_J ) 
      csr::accumulate(SPValueType(1.0), v, combined_J);
  }  
  TGwfn.Node().barrier();
  // make compact, since some libraries require it!
  combined_U.remove_empty_spaces();  
  combined_J.remove_empty_spaces();  
  hij.remove_empty_spaces();

  // create energy evaluation model class
  SparseEnergy<MP,REAL> ET( make_SparseEnergy<MP,REAL>(TGwfn, type, hij, combined_U, combined_J, E0) );

  // Add Jij terms to Uij. 
  if(TGwfn.Node().root()) {
    for( int i=0; i<NMO; ++i) {
      auto vals = collect_J[0].non_zero_values_data(i);
      auto cols = collect_J[0].non_zero_indices2_data(i);
      auto nnz = collect_J[0].num_non_zero_elements(i);
      // terms go in same spin component, so shift row by NMO
      while( (nnz--) > 0 )
        collect_U[0].add( {i+NMO, *(cols++)}, *(vals++) );
    }
    for( int i=0; i<NMO; ++i) {
      auto vals = collect_J[1].non_zero_values_data(i);
      auto cols = collect_J[1].non_zero_indices2_data(i);
      auto nnz = collect_J[1].num_non_zero_elements(i);
      // terms go in same spin component, so shift row by NMO
      // spin decomposition gets a factor of -1, so add to charge terms too 
      while( (nnz--) > 0 )
        collect_U[0].add( {i+NMO, *(cols++)}, SPValueType(-1.0)*(*(vals++)) );
    }
  }

  // generate "present" one-body terms
  using map_t = std::unordered_map<size_t, int>;
  Vector<size_t> n2IJ( find_occupied_pairs(type, collect_U, collect_J) );
  map_t IJ2n;
  IJ2n.reserve(n2IJ.size()); 
  for(int n=0; n<n2IJ.size(); n++) 
    IJ2n.insert(std::make_pair(n2IJ[n], n));

  addComponent<MP,REAL>( TGwfn, type, ContinuousChargePropagator, collect_U[0], 
		collect_J[0], Hams, n2IJ, IJ2n);
  addComponent<MP,REAL>( TGwfn, type, ContinuousSpinPropagator  , collect_U[1], 
		collect_J[1], Hams, n2IJ, IJ2n);
  // note: collect_J[2] should be empty
  utils::check(collect_J[2].num_non_zero_elements() == 0, "");
  addComponent<MP,REAL>( TGwfn, type, DiscreteChargePropagator, collect_U[2], 
		collect_J[2], Hams, n2IJ, IJ2n);
  addComponent<MP,REAL>( TGwfn, type, DiscreteSpinPropagator, collect_U[3], 
		collect_J[2], Hams, n2IJ, IJ2n);

  if (TGwfn.Node().root()) {
    dump.pop();  // ModelHamiltonian 
    dump.pop();  // Hamiltonian
    dump.close();
  }
  TGwfn.Global().barrier();

  using devSpCMatrix  = Matrix<SPComplexType, device_allocator<SPComplexType>>;
  return HamiltonianOperations<MP>(ModelHamOps<MP,REAL,devSpCMatrix>(TGwfn, type, std::move(PsiC), 
                        std::move(ET), std::move(Hams), n2IJ, sparse_g_eval)); 
*/
  return HamiltonianOperations<MEM>{};
}

template<MEMORY_SPACE MEM> HamiltonianOperations<MEM> 
ModelHamOpsGenerator::getHamiltonianOperations(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  bool Real = false;
/*
  if(mpi->comm.root()) {

    if (!dump.open(fileName, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening integral file in ModelHamOpsGenerator. ");
    if (dump.push("Hamiltonian", false)<0)
      APP_ABORT(" Error in ModelHamOpsGenerator::getHamiltonianOperations(): Group Hamiltonian not found. ");
    if (dump.push("ModelHamiltonian", false)<0)
      APP_ABORT(" Error in ModelHamOpsGenerator::getHamiltonianOperations(): Group ModelHamiltonian not found. ");

    std::string base_error("Error in ModelHamOpsGenerator::getHamiltonianOperations(): ");
    std::vector<int> shape;
    int num_components(0);
    if (!dump.readEntry(num_components, "number_of_components"))
       APP_ABORT(base_error + " /Hamiltonian/ModelHamiltonian/number_of_components not found. ");

    for(int n=0; n<num_components; n++)  {

      if (dump.push("ModelComponent_"+std::to_string(n), false)<0)
        APP_ABORT(base_error + "Group ModelComponent_" + std::to_string(n) + " not found. ");

      std::string model_type("dummy");
      if (!dump.readEntry(model_type,"model_type"))
        APP_ABORT(base_error + " Problems reading model_type. ");
      std::transform(model_type.begin(), model_type.end(), model_type.begin(), (int (*)(int))tolower);

      if( model_type == "one_body" )
      {
        if (dump.push("tij", false)<0)
          APP_ABORT(base_error + " Problems reading tij dataset in model_type: " + model_type);
      }
      else if( model_type == "hubbard_u" )
      {
        if (dump.push("Uij", false)<0)
          APP_ABORT(base_error + " Problems reading Uij dataset in model_type: " + model_type);
      }
      else if( model_type == "hubbard_j" )
      {
        if (dump.push("Jij", false)<0)
          APP_ABORT(base_error + " Problems reading Jij dataset in model_type: " + model_type);
      }
      else
        APP_ABORT(base_error + " Unknown model type: " + model_type);

      shape.clear();
      if (!dump.getShape<RealType>("data_", shape))
        APP_ABORT(" Error in ModelHamOpsGenerator::getHamiltonianOperations(): getShape(data_) returned error. ");

      dump.pop();  // tij, Uij, Jij
      dump.pop();  // ModelComponent_X	

      if( n==0 ) {
        if( shape.size() == 2 ) Real = false;
        else if( shape.size() == 1 ) Real = true;
        else APP_ABORT(" Error in ModelHamOpsGenerator::getHamiltonianOperations(): Inconsistent data shape. ");
      } else {
        if( (shape.size() != 2 and shape.size() != 1) or
            (shape.size() == 2 and Real ) or
            (shape.size() == 1 and not Real) )
          APP_ABORT(" Error in ModelHamOpsGenerator::getHamiltonianOperations(): Inconsistent data types in ModelComponents. "); 
      }

    } // for(n)

    dump.pop();
    dump.pop();
    dump.close();
  }
*/
  mpi->comm.broadcast_n(&Real, 1, 0);
  if(Real)
    return getHamiltonianOperations_impl<MEM,true>(type, mpi, PsiT);
  else
    return getHamiltonianOperations_impl<MEM,false>(type, mpi, PsiT);
}   

template HamiltonianOperations<HOST_MEMORY> 
ModelHamOpsGenerator::getHamiltonianOperations<HOST_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<HOST_MEMORY>,2>const&);
#if defined(ENABLE_DEVICE)
template HamiltonianOperations<DEVICE_MEMORY>
ModelHamOpsGenerator::getHamiltonianOperations<DEVICE_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<DEVICE_MEMORY>,2>const&);
#endif

} // namespace afqmc
} // namespace sfqmc

