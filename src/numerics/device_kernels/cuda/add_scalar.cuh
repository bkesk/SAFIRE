#pragma once

#include "numerics/device_kernels/cuda/nda_aux.hpp"

namespace kernels::device
{

namespace detail
{

template<typename V1, typename V2>
void add_scalar_impl(V1& alpha, V2& a, int n);

}

template<nda::MemoryVector V1, nda::MemoryVector A>
void add_scalar(V1&& alpha, A&& a, int n){
    auto a_b = to_basic_layout(a());
    auto alpha_b = to_basic_layout(alpha());
    detail::add_scalar_impl(alpha_b,a_b,n);
}

}