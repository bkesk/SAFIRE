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

#undef NDEBUG

#include "catch2/catch.hpp"

#include "config.h"
#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "IO/app_loggers.h" 
#include "utilities/check.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "numerics/sparse/sparse.hpp"

#include <stdio.h>
#include <string>
#include <vector>
#include <complex>

#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"
#include "AFQMC/SlaterDeterminantOperations/propagate.hpp"
#include "AFQMC/SlaterDeterminantOperations/orthogonalize.hpp"

using std::complex;
using std::cout;
using std::endl;
using std::string;

namespace sfqmc
{
namespace afqmc
{

using namespace afqmc;
template<MEMORY_SPACE MEM>
void SDetOps()
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto all = nda::range::all;
  using nda::range;

  const int nwalk = 5;
  const int NMO = 4;
  const int NEL = 3;

  using Type        = ComplexType;
  using namespace std::complex_literals;
  using matrix       = memory::array<MEM,Type,2>;
  using matrix_view  = memory::array_view<MEM,Type,2>;
  using array       = memory::array<MEM,Type,3>;
  using array_view  = memory::array_view<MEM,Type,3>;
  using sfqmc::utils::ARRAY_EQUAL;

  const Type ov_base  = std::log(-7.62332599999999 + 22.20453200000000i);
  const Type ov2_base = std::log(-10.37150000000000 - 7.15750000000000i);
  nda::array<Type,1> ov(nwalk,ov_base);
  nda::array<Type,1> ov2(nwalk,ov2_base);

  // helper function, use with care
  auto copy_to_array = [](auto const& b, auto && A) { 
    for(int i=0; i<A.extent(0); i++) A(i,nda::ellipsis{}) = b();
  };

  // some arbitrary matrices 
  nda::array<Type,2> m_a = { 
                {0.90000 + 0.10000i, 0.40000 + 0.40000i, 1.40000 + 0.20000i, 0.40000 + 0.50000i},
                {2.40000 + 0.20000i, 1.00000 + 0.50000i, 1.60000 + 0.30000i, 0.20000 + 0.10000i},
                {3.00000 + 0.30000i, 1.20000 + 0.10000i, 3.60000 + 0.40000i, 0.10000 + 0.20000i}
                           };
  nda::array<Type,2> m_b = { 
                {1.90000 + 0.60000i, 1.40000 + 0.70000i, 0.40000 + 0.80000i}, 
                {1.40000 + 0.90000i, 0.20000 + 0.50000i, 2.20000 + 0.60000i}, 
                {0.40000 + 0.70000i, 2.60000 + 0.80000i, 0.60000 + 0.90000i}, 
                {1.10000 + 0.50000i, 0.30000 + 0.60000i, 0.90000 + 0.70000i} 
                           };

  matrix A(m_a);
  array B(nwalk,NMO,NEL);
  copy_to_array(m_b,B);

  matrix_view Aref(A());
  array_view Bref(B());

  // csr
  auto Acsr =math::sparse::to_csr<MEM,int,int>(A);

  // views
  auto A_ = A(range(0,2),range(0,3));
  auto B_ = B(all,range(0,3),range(0,2));

  /**** Overlaps ****/
  //SECTION("Overlaps")
  {
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));
    det_ops::Log_Overlap(A, B, ovlp);
    ARRAY_EQUAL(ovlp,ov);
    det_ops::Log_Overlap(Aref, B, ovlp);
    ARRAY_EQUAL(ovlp,ov);
    det_ops::Log_Overlap(A, Bref, ovlp);
    ARRAY_EQUAL(ovlp,ov);
    det_ops::Log_Overlap(Aref, Bref, ovlp);
    ARRAY_EQUAL(ovlp,ov);
  }

  //SECTION("range_overlap")
  {
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));
    det_ops::Log_Overlap(A_, B_, ovlp);
    ARRAY_EQUAL(ovlp,ov2);
    det_ops::Log_Overlap(A(range(0,2),range(0,3)), B_, ovlp);
    ARRAY_EQUAL(ovlp,ov2);
    det_ops::Log_Overlap(A_, B(all,range(0,3),range(0,2)), ovlp);
    ARRAY_EQUAL(ovlp,ov2);
  }

  {
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));
    det_ops::Log_Overlap(Acsr, B, ovlp);
    ARRAY_EQUAL(ovlp,ov);
    det_ops::Log_Overlap(Acsr, Bref, ovlp);
    ARRAY_EQUAL(ovlp,ov);
  } 

  /**** Density Matrices *****/
  nda::array<Type,2> v_ref    = {{1.17573619385025996 - 0.01580426445014660i,  -0.25295981756593167 + 0.28594469607401085i,
                  -0.07724502823533341 - 0.09687959052155870i, 0.30512858581808422 - 0.04506898328729603i},
                  {0.17912592806889663 + 0.08374315906672802i,  0.59381451118048767 + 0.13438888951771200i,
                  -0.02021475320201610 - 0.13737561982083193i, 0.32095003919745313 + 0.12832750636154097i},
                  {-0.04919549564646425 + 0.05402065222741825i, -0.00286990878355775 - 0.15806420733175885i,
                  1.05069081188635494 + 0.00793429988912356i,  -0.08048239150997794 + 0.09917405634760490i},
                  {-0.46434598219794548 - 0.09896890422706731i, 0.87746748807670427 - 0.53417787485950319i,
                  0.12162735438647077 + 0.31042401735800573i,  0.17975848308289613 - 0.12651892495668898i}};
  nda::array<Type,2> vc_ref   = {{-0.8721879971495297 + 0.6787593377585239i, 0.3244278932250768 - 1.8083537898275881i,
                   0.6130713546530860 + 0.0399736955598931i,  0.0132562806444336 - 0.2882495766584950i},
                   {0.8853626557603183 + 0.0978868569224204i,  -0.0598704127345155 - 0.0470889603064014i,
                   -0.7392693424168300 - 0.0715317395994149i, 0.3721269544963505 - 0.1797896522886788i},
                   {-0.0567190307022984 - 0.3114847157576828i, -0.1290126440468128 + 0.6815705660308808i,
                   0.3787012855335005 + 0.0039188686237135i,  -0.2005543456941538 + 0.2100886953142371i}};
  nda::array<Type,2> v_ref_2  = {{0.7361983496013835 - 0.0956505507662245i, 0.6467449689807925 + 0.2297471806893873i,
                    0.0189270005620390 - 0.1727975708935829i},
                    {0.2986893588843604 - 0.0099955730815030i, 0.2696068479051557 + 0.0292039710860386i,
                    0.0497734835066391 + 0.1783200397050796i},
                    {0.1020030246826092 + 0.0344707468383766i, -0.2500340021988402 - 0.0826863644855427i,
                    0.9941948024934623 + 0.0664465796801866i}};
  nda::array<Type,2> vc_ref_2 = {{-0.489369975192701 + 0.103038673040713i, -0.858850485405126 - 0.275734238124941i},
                     {1.219791948842170 + 0.118626922447301i,  0.486337033653898 - 0.098631569047656i},
                     {0.595497821653010 + 0.185288949671560i,  -0.455373025165330 - 0.129360996228044i}};

  array g_ref(nwalk,NMO,NMO);    copy_to_array(v_ref,g_ref);
  array gc_ref(nwalk,NEL,NMO);    copy_to_array(vc_ref,gc_ref);
  array g_ref_2(nwalk,3,3);    copy_to_array(v_ref_2,g_ref_2);
  array gc_ref_2(nwalk,2,3);    copy_to_array(vc_ref_2,gc_ref_2);

  //SECTION("density_matrices")
  {
    array G(nwalk, NMO, NMO);
    auto G_ = G(all,range(3),range(3));
    auto Gc_ = G(all,range(2),range(3));
    array Gc(nwalk, NEL, NMO);
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));

    det_ops::MixedDensityMatrix(A, B, G, ovlp, false);
    ARRAY_EQUAL(G,g_ref);
    ARRAY_EQUAL(ovlp,ov);
    det_ops::MixedDensityMatrix(A(), B(), G(), ovlp(), false);
    ARRAY_EQUAL(G,g_ref);
    ARRAY_EQUAL(ovlp,ov);

    det_ops::MixedDensityMatrix(A_, B_, G_, ovlp, false);
    ARRAY_EQUAL(G_,g_ref_2);

    det_ops::MixedDensityMatrix(A, B, Gc, ovlp, true);
    ARRAY_EQUAL(Gc,gc_ref);
    ARRAY_EQUAL(ovlp,ov);
    det_ops::MixedDensityMatrix(A(), B(), Gc(), ovlp(), true);
    ARRAY_EQUAL(Gc,gc_ref);
    ARRAY_EQUAL(ovlp,ov);

    det_ops::MixedDensityMatrix(A_, B_, Gc_, ovlp, true);
    ARRAY_EQUAL(Gc_,gc_ref_2);
  }

  {
    array G(nwalk, NMO, NMO);
    auto G_ = G(all,range(3),range(3));
    auto Gc_ = G(all,range(2),range(3));
    array Gc(nwalk, NEL, NMO);
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));

    det_ops::MixedDensityMatrix(Acsr, B, G, ovlp, false);
    ARRAY_EQUAL(G,g_ref);
    ARRAY_EQUAL(ovlp,ov);
    det_ops::MixedDensityMatrix(Acsr, B(), G(), ovlp(), false);
    ARRAY_EQUAL(G,g_ref);
    ARRAY_EQUAL(ovlp,ov);

    det_ops::MixedDensityMatrix(Acsr, B, Gc, ovlp, true);
    ARRAY_EQUAL(Gc,gc_ref);
    ARRAY_EQUAL(ovlp,ov);
    det_ops::MixedDensityMatrix(Acsr, B(), Gc(), ovlp(), true);
    ARRAY_EQUAL(Gc,gc_ref);
    ARRAY_EQUAL(ovlp,ov);
  }

  // Propagate
  {
    // not testing results, just checking things run. Get specific answers to test
    array SM(nwalk, NMO, NEL);
    matrix P(NMO, NMO);
    array V(nwalk, NMO, NMO);

    SM() = utils::make_random<Type>(nwalk,NMO,NEL);
    P() = utils::make_random<Type>(NMO,NMO);
    auto Vt = utils::make_random<Type>(nwalk,NMO,NMO);
    Vt() = Vt()*0.001;
    V() = Vt(); 
    auto Pcsr = math::sparse::to_csr<MEM,int,int>(P);

    matrix V2(nwalk*NMO, nwalk*NMO);
    V2() = Type(0);
    for(int i=0; i<nwalk; ++i)
      V2(nda::range(i*NMO,(i+1)*NMO),nda::range(i*NMO,(i+1)*NMO)) = utils::make_random<Type>(NMO,NMO); 
    auto Vcsr = math::sparse::to_csr<MEM,int,int>(V2);

    det_ops::Propagate(SM, P, V, 4, 'N');
    det_ops::Propagate(SM, P, V, 4, 'T');
    det_ops::Propagate(SM, P, V, 4, 'H');

    det_ops::Propagate(SM, Pcsr, V, 4, 'N');
    det_ops::Propagate(SM, Pcsr, V, 4, 'T');
    det_ops::Propagate(SM, Pcsr, V, 4, 'H');

    det_ops::Propagate(SM, P, Vcsr, 4, 'N');
    det_ops::Propagate(SM, P, Vcsr, 4, 'T');
    det_ops::Propagate(SM, P, Vcsr, 4, 'H');

    det_ops::Propagate(SM, Pcsr, Vcsr, 4, 'N');
    det_ops::Propagate(SM, Pcsr, Vcsr, 4, 'T');
    det_ops::Propagate(SM, Pcsr, Vcsr, 4, 'H');
  }

  // Propagate noncollinear with diagonal V
  { 
    // not testing results, just checking things run. Get specific answers to test
    array SM(nwalk, 2*NMO, NEL);
    matrix P(2*NMO, 2*NMO); 
    array V(nwalk, 2*NMO, NMO);
    
    SM() = utils::make_random<Type>(nwalk,2*NMO,NEL);
    P() = utils::make_random<Type>(2*NMO,2*NMO);
    auto Vt = utils::make_random<Type>(nwalk,2*NMO,NMO);
    Vt() = Vt()*0.001;
    V() = Vt(); 
    auto Pcsr = math::sparse::to_csr<MEM,int,int>(P);
  
    det_ops::Propagate_pol(2, SM, P, V, 4, 'N');
    det_ops::Propagate_pol(2, SM, P, V, 4, 'T');
    det_ops::Propagate_pol(2, SM, P, V, 4, 'H');
    
    det_ops::Propagate_pol(2, SM, Pcsr, V, 4, 'N');
    det_ops::Propagate_pol(2, SM, Pcsr, V, 4, 'T');
    det_ops::Propagate_pol(2, SM, Pcsr, V, 4, 'H');

    // now with separate spin up/down csr_matrix
  }

  // Orthogonalize
  //SECTION("orthogonalize")
  {
    array Q(B);
    memory::array<MEM,Type,1> ovlp(nwalk,Type(0.0));
    memory::array<MEM,Type,1> oref(nwalk,Type(0.0));
    det_ops::orthogonalize(Q);
    det_ops::Log_Overlap(Q, Q, ovlp);
    ARRAY_EQUAL(oref,ovlp);
  }
}


TEST_CASE("SDetOps", "[sdet_ops]")
{
  SDetOps<HOST_MEMORY>();
}

} // namespace afqmc
} // namespace sfqmc
