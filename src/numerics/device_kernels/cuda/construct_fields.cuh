
#pragma once

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"

namespace kernels::device
{

namespace detail
{
  template<typename V1, typename V2, typename V3, typename V4, typename V5>
  void construct_X_impl(bool zero, bool fp, double sqrtdt, double vbias_bound, V1 const& FieldTypes, 
     V2 const& vMF, V3& mf_factor, V3& hybrid_weight, V4 const& RN, V5& X);

}

  void construct_X(bool zero, bool fp, double sqrtdt, double vbias_bound, nda::MemoryVector auto const& FT, 
    nda::MemoryVector auto const& vMF, nda::MemoryVector auto&& MF,  
    nda::MemoryVector auto&& HW, nda::MemoryMatrix auto const& RN, nda::MemoryMatrix auto&& X) 
  {
    auto FT_b = to_basic_layout(FT());
    auto vMF_b = to_basic_layout(vMF());
    auto MF_b = to_basic_layout(MF());
    auto HW_b = to_basic_layout(HW());
    auto RN_b = to_basic_layout(RN());
    auto X_b = to_basic_layout(X());
    detail::construct_X_impl(zero,fp,sqrtdt,vbias_bound,FT_b,vMF_b,MF_b,HW_b,RN_b,X_b);
  }

} //kernels::device
