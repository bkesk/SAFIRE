/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#ifndef NDA_KERNELS_AUX_HPP
#define NDA_KERNELS_AUX_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"

namespace kernels::device 
{ 

template<nda::MemoryArray Arr>
auto to_basic_layout(Arr && A)
requires( Arr::layout_t::static_extents_encoded == 0 and 
          nda::mem::have_device_compatible_addr_space<Arr> )
{
  constexpr int R = nda::get_rank<Arr>;
  using T = typename std::pointer_traits<decltype(A.data())>::element_type;
  static_assert( nda::mem::on_device<Arr> or nda::mem::on_unified<Arr>, 
                 "Only device/unified arrays allowed.");
  if constexpr ( Arr::is_stride_order_C() ) {
    using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<R>, nda::layout_prop_e::none>;
    if constexpr (nda::mem::on_device<Arr>) {
      nda::basic_array_view<T,R,basic_layout_t,'A',nda::default_accessor, nda::borrowed<nda::mem::Device>> Ab = A();
      return Ab;
    } else { 
      nda::basic_array_view<T,R,basic_layout_t,'A',nda::default_accessor, nda::borrowed<nda::mem::Unified>> Ab = A();
      return Ab;
    }
  } else if constexpr ( Arr::is_stride_order_Fortran() ) {
    using basic_layout_t = typename nda::basic_layout<0, nda::Fortran_stride_order<R>, nda::layout_prop_e::none>;
    if constexpr (nda::mem::on_device<Arr>) {
      nda::basic_array_view<T,R,basic_layout_t,'A',nda::default_accessor, nda::borrowed<nda::mem::Device>> Ab = A();
      return Ab;
    } else { 
      nda::basic_array_view<T,R,basic_layout_t,'A',nda::default_accessor, nda::borrowed<nda::mem::Unified>> Ab = A();
      return Ab;
    }
  } else {
    constexpr auto stride_order_encoded = Arr::layout_t::stride_order_encoded;
    using basic_layout_t = typename nda::basic_layout<0, stride_order_encoded, nda::layout_prop_e::none>;
    if constexpr (nda::mem::on_device<Arr>) {
      nda::basic_array_view<T,R,basic_layout_t,'A',nda::default_accessor, nda::borrowed<nda::mem::Device>> Ab = A();
      return Ab;
    } else { 
      nda::basic_array_view<T,R,basic_layout_t,'A',nda::default_accessor, nda::borrowed<nda::mem::Unified>> Ab = A();
      return Ab;
    }
  } 
}

} //kernels::device

#endif
