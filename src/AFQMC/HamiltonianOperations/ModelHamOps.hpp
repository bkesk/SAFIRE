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

#pragma once

#include <vector>
#include <type_traits>

#include "config.h"
#include "AFQMC/config.h"
#include "nda/nda.hpp"                                 
#include "nda/tensor.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check_strides.hpp"
#include "numerics/shared_array/shared_array.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/shared_array/shared_array.hpp"

#include "AFQMC/HamiltonianOperations/ModelComponents/SparseEnergy.hpp"
#include "AFQMC/HamiltonianOperations/ModelComponents/ModelComponent.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM, bool REAL>
class ModelHamOps
{
  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:
  static const HamiltonianTypes HamOpType = ModelHamiltonian; 
  HamiltonianTypes getHamType() const { return ModelHamiltonian; }

  ModelHamOps() {};

  ModelHamOps(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
              WALKER_TYPES type,
              int nel_up, int nel_dn,
              memory::shared_array<MEM,ComplexType,4>&& psi_,
              SparseEnergy<MEM,REAL>&& et_,
              std::vector<ModelComponent<MEM,REAL>>&& h_,
              nda::MemoryVector auto&& n2ij_,
              bool sparse_g_eval_ = true 
             )
      : mpi(_mpi),
        walker_type(type),
        nel{nel_up,nel_dn},
        PsiC(std::move(psi_)),
        ET(std::move(et_)),
        Hams(std::move(h_)),
        spvHS(0),
        n2IJ(std::move(n2ij_)),
        n2IJ_dev(n2IJ),
        nIJ_first_beta(n2IJ.extent(0)),  // default value for noncollinear
        n2IJ_vHS_dev(n2IJ_dev),          // default value for collinear
        sparse_G_eval(sparse_g_eval_)
  {
    num_ke_vectors=0;
    nCV=0;
    for(auto const& v: Hams) {
      field_ranges.emplace_back(nda::range(nCV,nCV+v.number_of_cholesky_vectors())); 
      ke_pos.emplace_back(nda::range(num_ke_vectors,num_ke_vectors+v.number_of_ke_vectors())); 
      num_ke_vectors += v.number_of_ke_vectors();
      nCV += v.number_of_cholesky_vectors();
    }

    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int M     = PsiC.extent(2);
    int NMO   = M/npol;
    if( walker_type == COLLINEAR ) {
      for(int n=0; n<n2IJ.size(); ++n) {
        if( n2IJ[n]/M >= NMO ) {
          nIJ_first_beta=n;
          break;
        }
      }
    }
 
    // setup n2IJ_vHS_dev, only needed for NONCOLLINEAR 
    // for COLLINEAR, n2IJ_vHS == n2IJ, so no need to update 
    if(walker_type == NONCOLLINEAR) {
      nda::array<int,1> n2IJ_vHS_h(n2IJ);
      for(auto& v : n2IJ_vHS_h) {
        int In = int(v/M);
        int Jn = int(v%M);
        v = In*NMO + (Jn%NMO);  
      }
      n2IJ_vHS_dev() = n2IJ_vHS_h(); 
    }
    mpi->comm.barrier();
  }

  ~ModelHamOps() {}

  ModelHamOps(const ModelHamOps& other) = default;
  ModelHamOps& operator=(const ModelHamOps& other) = default;
  ModelHamOps(ModelHamOps&& other)                 = default;
  ModelHamOps& operator=(ModelHamOps&& other) = default;

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    utils::check(vMF.size() == number_of_cholesky_vectors(), "Size mismatch"); 
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;

    nda::array<ComplexType, 3> H1(nspin, npol*NMO, npol*NMO);
    H1() = ComplexType(0.0);

    // accumulate terms from 1body
    ET.addOneBodyPropagatorMatrix(H1,dt);

    for(int i=0; i<Hams.size(); i++) 
      Hams[i].addOneBodyPropagatorMatrix(H1,dt, vMF(field_ranges[i]), n2IJ);

    // symmetrize
    for (int is = 0; is < nspin; is++) {
      for (int I = 0; I < npol * NMO; I++) {
        for (int J = I + 1; J < npol * NMO; J++) {
          // This is really cutoff dependent!!!
          if (std::abs(H1(is,I,J) - std::conj(H1(is,J,I))) * 2.0 > 1e-5)
          {
            app_warning(" WARNING in getOneBodyPropagatorMatrix. H1 is not hermitian. ");
            app_warning(" I:{}, J:{}, H[I,J]:{}, H[J,I]:{} ",I,J,H1(is,I,J),H1(is,J,I));
          }
          H1(is,I,J) = 0.5 * (H1(is,I,J) + std::conj(H1(is,J,I)));
          H1(is,J,I) = std::conj(H1(is,I,J));
        }
      }
    }

    return H1;
  }

  nda::array<int,1> getFieldTypes() const {
    int nvc = number_of_cholesky_vectors();
    nda::array<int,1> v(nvc);
    for(int i=0; i<Hams.size(); i++)
      Hams[i].getFieldTypes(v(field_ranges[i])); 
    return v;
  }
 
  void update_potentials(double dt, nda::MemoryVector auto const& nMF_, nda::MemoryVector auto&& vMF_, bool natural_shift)
  {
    // work with host copies
    auto nMF = nda::to_host(nMF_); 
    nda::array<ComplexType,1> vMF(vMF_.extent(0),ComplexType(0.0));
    std::unordered_map<long, int> IJ2n;
    if(IJ2n.size()==0) {
      IJ2n.reserve(n2IJ.size());
      for(int n=0; n<n2IJ.size(); n++)
        IJ2n.insert(std::make_pair(n2IJ[n], n));
    } 
    for(int i=0; i<Hams.size(); i++) 
      Hams[i].update(dt,nMF,n2IJ,IJ2n,vMF(field_ranges[i]),natural_shift);
    // copy result 
    vMF_() = vMF();
  }

  void energy(nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              int idet,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    using nda::range;
    auto all = range::all;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = G.extent(0);
    
    E() = ComplexType(0.0);
    utils::check(idet >= 0 and idet < PsiC.extent(0), "idet out of bounds:{} ndet:{}",idet, PsiC.extent(0));

    // ET.get_n2IJ() runs over [0,M^2) in COLLINEAR case 
    int nIJ(ET.get_n2IJ().extent(0));
    bool allocate_EJn (addEJ and walker_type==COLLINEAR);
    memory::buffered_array<MEM,ComplexType,3> EJn(nspin, (allocate_EJn?nwalk:1), (allocate_EJn?nIJ:1)); 
    EJn() = ComplexType(0.0);

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel[0]+nel[1],npol*NMO},G.data());
    
    // generate GIJ with custom mapping
    if( sparse_G_eval ) {
      for(int is=0; is<nspin; is++ ) {
        auto GIJ = getGIJ_for_energy(idet,is,G3d);
        ET.accumulate_energy(is, E, GIJ, EJn(is,all,all), addE1, addEJ, addEXX); 
      }
    } else {
      // generate full G
      auto ET_n2IJ = ET.get_n2IJ_dev();
      memory::buffered_array<MEM,ComplexType,2> GIJ(nIJ, nwalk);
      for(int is=0; is<nspin; is++ ) {
        auto Gfull = getGFull(idet,is,G3d);
        // B[n][:] = A[ I[n] ][:] 
        nda::copy_select(false, 1, ET_n2IJ, ComplexType(1.0), Gfull, ComplexType(0.0), GIJ);
        ET.accumulate_energy(is, E, GIJ, EJn(is,all,all), addE1, addEJ, addEXX); 
      }
    }

    // opposite spin EJ contribution
    if(addEJ and walker_type == COLLINEAR)
      nda::tensor::contract(ComplexType(1.0), EJn(0,all,all), "wi", EJn(1,all,all), "wi",
                            ComplexType(1.0), E(all,2), "w");
  }
/*
  template<class Mat, class MatB, class MatC,
	typename = std::enable_if_t<std::is_same_v<SPComplexType,typename std::decay_t<MatC>::element_type>>>
  void energy(SpinTypes spin_component,
              Mat&& E,
              MatB const& Gc,
              int nd,
              MatC&& EJn,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int ispin = ( spin_component==Alpha ? 0 : 1 );
    int NMO   = PsiC[0].size(0) / npol;
    int nel = PsiC[ nspin * nd + ispin].size(1);
    int nwalk = Gc.size(0);
    int nIJ(ET.get_n2IJ()->size());
    for (int n = 0; n < nwalk; n++)
      std::fill_n(E[n].origin(), 3, ComplexType(0.));    
    RUNTIME_CHECK(PsiC.size() >= nspin*nd + (nspin-1), "");
    if(addEJ and walker_type==COLLINEAR) {
      RUNTIME_CHECK(EJn.size(0) == nwalk, "");
      RUNTIME_CHECK(EJn.size(1) == nIJ, "");
    }

    // buffer allocators
    DeviceBufferManager device_buffer_manager{};
    
    StaticMatrix GIJ( {nIJ, nwalk}, SPComplexType(0.0),
      device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
    auto ET_n2IJ = ET.get_n2IJ_dev();

    // generate GIJ with custom mapping
    if( sparse_G_eval ) {

      auto&& G3D = Gc.rotated().partitioned(nel).unrotated();
      ma::getGIJ(G3D,GIJ,PsiC[ nspin * nd + ispin], *ET_n2IJ);

    } else {

      // generate full G
      StaticMatrix Gfull({npol * NMO * npol * NMO, nwalk},
             device_buffer_manager.get_generator().template get_allocator<SPComplexType>());
      getGFull(Gc,Gfull,PsiC[ nspin * nd + ispin]);

      // B[n][:] = A[ I[n] ][:] 
      ma::copy_select(Gfull, GIJ, *ET_n2IJ, false);

    }
    ET.accumulate_energy(ispin, E, GIJ, EJn, addE1, addEJ, addEXX); 

  }
*/

  template<class... Args>
  void fast_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: fast_energy not implemented in ModelHamOps. ");
  }

  template<class... Args>
  void ph_reference_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_reference_energy not implemented yet. ");
  }

  template<class... Args>
  void ph_excited_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_excited_energy not implemented yet. ");
  }

  auto vHS(nda::MemoryArrayOfRank<2> auto && X, double dt)
  {
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = X.extent(0);
    int nIJ = n2IJ.size();
    // sanity checks!
    utils::check(X.extent(1) == nCV, "Size mismatch");

    memory::buffered_array<MEM,ComplexType,2> vIJ(nwalk, nIJ);
    vIJ() = ComplexType(0.0);
    for(int i=0; i<Hams.size(); i++)
      Hams[i].vHS(X(nda::range::all,field_ranges[i]), vIJ, dt);

    memory::buffered_array<MEM,ComplexType,4> v(nwalk,nspin,npol*NMO,NMO);
    auto v2d = nda::reshape(v,std::array<long,2>{nwalk,nspin*npol*NMO*NMO});
    v() = ComplexType(0.0);
    // B[ I[n] ][:] += A[n][:] 
    nda::copy_select(true, 1, n2IJ_vHS_dev, ComplexType(1.0), vIJ, ComplexType(0.0), v2d);
    return v;
  }

  auto vHS_sparse(nda::MemoryArrayOfRank<2> auto const& X, double dt)
  {
    using nda::range;
    auto all = range::all;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = X.extent(0);
    int nIJ = n2IJ.size();
    // sanity checks!
    utils::check(X.extent(1) == nCV, "Size mismatch");

    if( spvHS.extent(0) != nspin or
        spvHS(0).extent(0) != nwalk*npol*NMO or spvHS(0).extent(1) != nwalk*npol*NMO or 
        spvHS(nspin-1).extent(0) != nwalk*npol*NMO or spvHS(nspin-1).extent(1) != nwalk*npol*NMO )
        make_csr_vHS(nwalk);
    utils::check(spvHS(0).extent(0) == nwalk*npol*NMO and spvHS(0).extent(1) == nwalk*npol*NMO, "Size mismatch");
    utils::check(spvHS(nspin-1).extent(0) == nwalk*npol*NMO and spvHS(nspin-1).extent(1) == nwalk*npol*NMO, "Size mismatch");
    if(walker_type == COLLINEAR) {
      utils::check(spvHS(0).capacity() == nwalk*nIJ_first_beta, "Size mismatch");
      utils::check(spvHS(1).capacity() == nwalk*(nIJ-nIJ_first_beta), "Size mismatch");
    } else if(walker_type == NONCOLLINEAR) {
      utils::check(spvHS(0).capacity() == nwalk*nIJ, "Size mismatch");
    }

    memory::buffered_array<MEM,ComplexType,2> vIJ(nwalk, nIJ);
    vIJ() = ComplexType(0.0);
    for(int i=0; i<Hams.size(); i++)
      Hams[i].vHS(X(nda::range::all,field_ranges[i]), vIJ, dt);

    auto vals_up = nda::reshape(spvHS(0).values(), std::array<long,2>{nwalk,nIJ_first_beta});
    vals_up() = vIJ(all,range(nIJ_first_beta));
    if(walker_type == COLLINEAR) {
      auto vals_dn = nda::reshape(spvHS(1).values(), std::array<long,2>{nwalk,nIJ-nIJ_first_beta});
      vals_dn() = vIJ(all,range(nIJ_first_beta,nIJ));
    }

    return spvHS(); 
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    auto all = nda::range::all;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = v.extent(0);
    int nci = PsiC.extent(0); 
    int nIJ = n2IJ.size(); 
    int nelec = nel[0]+nel[1];
    utils::check(v.extent(1) == nCV, "Size mismatch");

    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nelec,npol*NMO},G.data());
    memory::buffered_array<MEM,ComplexType,2> GIJ(nwalk, nIJ);
    GIJ() = ComplexType(0.0);

    if( nci == 1 ) {
      if( sparse_G_eval ) {
         getGIJ_for_vbias(G3d,GIJ);
      } else {
        // generate full G
        auto Gfull = getGFull(0,G3d);

        // B[:][n] = A[:][ I[n] ]
        nda::copy_select(false, 1, n2IJ_dev, ComplexType(1.0), Gfull, ComplexType(0.0), GIJ);
      }
      for(int i=0; i<Hams.size(); i++) 
        Hams[i].vbias(GIJ, v(all,field_ranges[i]), dt); 
    } else {
      // if nci > 1, we expect [...][nwalk]
      utils::check(G.shape() == std::array<long,2>{nwalk, nspin*npol*NMO*npol*NMO}, "Size mismatch");
      // B[:][n] = A[:][ I[n] ]
      nda::copy_select(false, 1, n2IJ_dev, ComplexType(1.0), G, ComplexType(0.0), GIJ);

      for(int i=0; i<Hams.size(); i++) 
        Hams[i].vbias(GIJ, v(all,field_ranges[i]), dt);
    }
  }

  template<class... Args> void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

  std::tuple<int,int> vHS_dims() const {
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    return std::make_tuple(nspin,npol);
  }
  int number_of_cholesky_vectors() const { return nCV; }
  int number_of_ke_vectors() const { return ET.get_n2IJ().extent(0); }

  bool fast_ph_energy() const { return false; }

  nda::array<ComplexType, 2> getHSPotentials()
  { return nda::array<ComplexType, 2>{}; }

private:

  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE;

  int nel[2];

  // (conjugated) orbital Matrix
  // This should be in node memory
  // PsiC(ndets,npin,i,a) = std::conj( PsiTrial(ndets,npin,i,a) )  
  memory::shared_array<MEM,ComplexType,4> PsiC;

  // One body Hamiltonian
  SparseEnergy<MEM,REAL> ET; 

  // list of ModelComponents
  // By convention, Hams[0] is a 1-body term.
  // All others (n>0) must be interacting terms. 
  std::vector<ModelComponent<MEM,REAL>> Hams;

  // csr_matrix matrix that will store vHS_sparse if needed
  nda::array<csrMat<ComplexType>,1> spvHS;

  // Vector with an ordered list of all IJ terms ( c^I cJ ) present in any term in the Hamiltonian,
  // n2IJ[n] = (I + si*NMO) * (npol * NMO) + J 
  // I/J are in range [0,NMO] for CLOSED/COLLINEAR and [0,2*NMO] for NONCOLLINEAR
  // only same spin blocks are present, consistent with the factorization of the coulomb 
  // e.g. no spin-orbit terms in the factorized pieces 
  nda::array<long,1> n2IJ;
  memory::array<MEM,long,1> n2IJ_dev;

  // first index of n2IJ of the spin=1 block. For NONCOLLINEAR, this should be equal to n2IJ.size() 
  int nIJ_first_beta;

  // n2IJ array, but with the index mapping consistent with the format of vHS
  // Notice that n2IJ is compatible with the Green's function indexing  
  memory::array<MEM,long,1> n2IJ_vHS_dev;

  std::vector<nda::range> field_ranges;
  std::vector<nda::range> ke_pos;

  // container for vHS in case vHS is requested in sparse form
  // sparse_vHS_t sparse_vHS;
  // sparse_vHS should be a csr_matrix in device memory. 

  int num_ke_vectors = 0;
  int nCV = 0;  

  bool sparse_G_eval = true;

  auto getGFull(int idet, int ispin, nda::MemoryArrayOfRank<3> auto const& Gc) 
  {
    using nda::range;
    auto all = range::all;
    int npol   = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO    = PsiC.extent(2) / npol;
    int nwalk  = Gc.extent(0);
    utils::check( Gc.shape() == std::array<long,3>{nwalk,nel[0]+nel[1],npol*NMO}, "Shape mismatch");
    memory::buffered_array<MEM,ComplexType,2> Gfull(nwalk, npol * NMO * npol * NMO);
    auto Gfull_3d = nda::reshape(Gfull, std::array<long,3>{nwalk, npol * NMO, npol * NMO});

    auto psi = PsiC()(idet,ispin,all,range(nel[ispin]));
    int n0 = (ispin == 0 ? 0 : nel[0]);
    nda::tensor::contract(psi,"ia",Gc(all,range(n0,n0+nel[ispin]),all),"waj",Gfull_3d(all,all,all),"wij");
    return Gfull;
  }

  auto getGFull(int idet, nda::MemoryArrayOfRank<3> auto const& Gc)
  {
    using nda::range;
    auto all = range::all;
    int npol   = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO    = PsiC.extent(2) / npol;
    int nwalk  = Gc.extent(0);
    utils::check( Gc.shape() == std::array<long,3>{nwalk,nel[0]+nel[1],npol*NMO}, "Shape mismatch");
    memory::buffered_array<MEM,ComplexType,2> Gfull(nwalk, nspin * npol * NMO * npol * NMO);
    auto Gfull_4d = nda::reshape(Gfull, std::array<long,4>{nwalk, nspin, npol * NMO, npol * NMO});

    auto psi = PsiC()(idet,0,all,range(nel[0]));
    int n0 = 0;
    nda::tensor::contract(psi,"ia",Gc(all,range(n0,n0+nel[0]),all),"waj",Gfull_4d(all,0,all,all),"wij");
    if( walker_type == COLLINEAR ) {
      auto psi_dn = PsiC()(idet,1,all,range(nel[1]));
      n0 = nel[0];
      nda::tensor::contract(psi_dn,"ia",Gc(all,range(n0,n0+nel[1]),all),"waj",Gfull_4d(all,1,all,all),"wij");
    }
    return Gfull;
  }

  auto getGIJ_for_energy(int idet, int ispin, nda::MemoryArrayOfRank<3> auto const& Gc)
  {
    using nda::range;
    auto all = range::all;
    int npol   = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int M      = PsiC.extent(2);
    int NMO    = PsiC.extent(2) / npol;
    int nwalk  = Gc.extent(0);
    utils::check( Gc.shape() == std::array<long,3>{nwalk,nel[0]+nel[1],npol*NMO}, "Shape mismatch");
    auto ET_n2IJ = ET.get_n2IJ_dev();
    int nIJ = ET_n2IJ.extent(0);
    memory::buffered_array<MEM,ComplexType,2> GIJ(nwalk, nIJ);

    auto psi = PsiC()(idet,ispin,all,range(nel[ispin]));
    int n0 = (ispin == 0 ? 0 : nel[0]);
    auto Gwaj = Gc(all,range(n0,n0+nel[ispin]),all);
 
    if constexpr (MEM==HOST_MEMORY) {
      //C[w][n] = sum_a psi[ I[n] ][a] Gwaj[w][a][ J[n] ]
      for(int n=0; n<nIJ; ++n) {
        int In = int(ET_n2IJ(n)/M);
        int Jn = int(ET_n2IJ(n)%M);
        nda::tensor::contract(psi(In,all),"a",Gwaj(all,all,Jn),"wa",GIJ(all,n),"w");
      }
    } else {
      utils::check(false, "finish");
      // dispatch with batched gemm interface
    }

    return GIJ;
  }

  void getGIJ_for_vbias(nda::MemoryArrayOfRank<3> auto const& Gc, nda::MemoryMatrix auto && GIJ)
  {     
    using nda::range;
    auto all = range::all;
    int npol   = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int M      = PsiC.extent(2);
    int NMO    = PsiC.extent(2) / npol;
    int nwalk  = Gc.extent(0);
    utils::check( Gc.shape() == std::array<long,3>{nwalk,nel[0]+nel[1],npol*NMO}, "Shape mismatch");
    int nIJ = n2IJ.extent(0);
    utils::check( GIJ.shape() == std::array<long,2>{nwalk,nIJ}, "Shape mismatch");
      
    if constexpr (MEM==HOST_MEMORY) {
      //C[w][n] = sum_a psi[ I[n] ][a] Gwaj[w][a][ J[n] ]
      {
        auto psi = PsiC()(0,0,all,range(nel[0]));
        auto Gwaj = Gc(all,range(nel[0]),all);
        for(int n=0; n<nIJ_first_beta; ++n) {
          int In = int(n2IJ(n)/M);
          int Jn = int(n2IJ(n)%M);
          nda::tensor::contract(psi(In,all),"a",Gwaj(all,all,Jn),"wa",GIJ(all,n),"w");
        }
      }
      if(walker_type == COLLINEAR) {
        auto psi = PsiC()(0,1,all,range(nel[1]));
        auto Gwaj = Gc(all,range(nel[0],nel[0]+nel[1]),all);
        for(int n=nIJ_first_beta; n<nIJ; ++n) {
          int In = int(n2IJ(n)/M) - NMO;
          int Jn = int(n2IJ(n)%M);
          nda::tensor::contract(psi(In,all),"a",Gwaj(all,all,Jn),"wa",GIJ(all,n),"w");
        }
      }
    } else {
      utils::check(false, "finish");
      // dispatch with batched gemv interface
    }
  }

  void make_csr_vHS(int nwalk) 
  {
    using nda::range;
    auto all = range::all;
    int nspin  = (walker_type == COLLINEAR) ? 2 : 1;
    int npol   = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int M      = PsiC.extent(2);
    int NMO    = PsiC.extent(2) / npol;
    int nIJ = n2IJ.extent(0);

    spvHS.resize(nspin);
    
    nda::array<int,1> cnt(nspin*M,0);
    for( auto IJ: n2IJ ) cnt( IJ/M )++;
 
    { // alpha
      nda::array<int,2> nnz(nwalk,M);
      for(int w=0; w<nwalk; w++)
        nnz(w,all) = cnt(range(M));

      math::sparse::csr_matrix<ComplexType, HOST_MEMORY, int, int>
          v_h({nwalk*M,nwalk*M},nda::flatten(nnz));

      for( auto IJ: n2IJ(range(nIJ_first_beta)) ) {
        int I(int(IJ/M)), J(int(IJ%M));
        for(int w=0; w<nwalk; w++)
          v_h.emplace_back( { w*M + I, w*M + J }, 0.0 );
      }
      spvHS(0) = std::move(v_h);
    }
    if(walker_type == COLLINEAR) {
      nda::array<int,2> nnz(nwalk,NMO);
      for(int w=0; w<nwalk; w++)
        nnz(w,all) = cnt(range(NMO,2*NMO));

      // need to setup in host 
      math::sparse::csr_matrix<ComplexType, HOST_MEMORY, int, int>
          v_h({nwalk*NMO,nwalk*NMO},nda::flatten(nnz));
       
      for( auto IJ: n2IJ(range(nIJ_first_beta,nIJ)) ) {
        int I(int(IJ/M)), J(int(IJ%M));
        for(int w=0; w<nwalk; w++)
          v_h.emplace_back( { w*NMO + (I-NMO), w*NMO + J }, 0.0 );
      }
      spvHS(1) = std::move(v_h);
    }

  }

};

} // namespace afqmc

} // namespace sfqmc

