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

#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctype.h>

#include "AFQMC/config.h"

#include "nda/nda.hpp"
#include "numerics/sparse/sparse.hpp"

namespace sfqmc::afqmc
{

// assumes a block diagonal PsiT
template<MEMORY_SPACE MEM>
auto nocc_per_kpoint(WALKER_TYPES type, int nkpts, nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  int nspin = (type == COLLINEAR?2:1);
  int npol = (type == NONCOLLINEAR?2:1);
  int ndet = PsiT.extent(0);
  int NMO = PsiT(0,0).extent(1)/npol;
  int nbnd = NMO/nkpts;
  utils::check(NMO%nkpts==0, "NMO%nkpts != 0");
  utils::check(PsiT.extent(1) == nspin, "Shape mismatch"); 

  // compute based on first determinant, check all others match
  nda::array<int, 2> nocc_per_kp(nspin,nkpts);
  nocc_per_kp() = 0;
  // 1st det
  {
    for(int is=0; is<nspin; is++) { 
      auto vals = nda::to_host(PsiT(0,is).values());
      auto cols = nda::to_host(PsiT(0,is).columns());
      auto row_begin = nda::to_host(PsiT(0,is).row_begin());
      auto row_end = nda::to_host(PsiT(0,is).row_end());
      int kp = 0;
      int nel = PsiT(0,is).extent(0);
      for(int ia=0; ia<nel; ++ia) {
        auto k0 = (cols(row_begin(ia))/nbnd)%nkpts;  
        auto k1 = (cols(row_end(ia)-1)/nbnd)%nkpts;  
        // check 
        utils::check( (k0==k1) and ((k0==kp) or (k0==kp+1)), "Error: Wavefunction is not in block diagonal form in kpoint");
        kp=k0;
        nocc_per_kp(is,kp)++;
      } 
    }
  }
  for(int id=1; id<ndet; ++id) {
    nda::array<int, 2> nocc(nspin,nkpts);
    nocc() = 0;
    for(int is=0; is<nspin; is++) {
      auto vals = nda::to_host(PsiT(id,is).values());
      auto cols = nda::to_host(PsiT(id,is).columns());
      auto row_begin = nda::to_host(PsiT(id,is).row_begin());
      auto row_end = nda::to_host(PsiT(id,is).row_end());
      int kp = 0;
      int nel = PsiT(id,is).extent(0);
      for(int ia=0; ia<nel; ++ia) {
        auto k0 = (cols(row_begin(ia))/nbnd)%nkpts;
        auto k1 = (cols(row_end(ia)-1)/nbnd)%nkpts;
        // check 
        utils::check( (k0==k1) and ((k0==kp) or (k0==kp+1)), "Error: Wavefunction is not in block diagonal form in kpoint");
        kp=k0;
        nocc(is,kp)++;
      }
    }
    utils::check(nda::sum(nda::abs(nocc()-nocc_per_kp())) == 0, "Error: Occupation numbers for eack kpoint differ between determinants, idet:{}",id);
  }
  return nocc_per_kp;
}

} // namespace sfqmc::afqmc
