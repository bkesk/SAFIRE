#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/complex.h>
#include "numerics/device_kernels/cuda/cuda_aux.hpp"

#include "configuration.hpp"
//#include "numerics/device_kernels/cuda/add_scalar.cuh"

namespace kernels::device::detail
{

template<typename V>
struct add_scalar_func
{

  V alpha;

  __host__ __device__ explicit add_scalar_func(V _alpha) : alpha(_alpha){}

  __host__ __device__ V operator()(const V& x) const {return x + alpha;}
};

template<typename V1, typename V2>
void add_scalar_impl(V1& alpha, V2& a, int n){
  auto alpha_p = thrust::device_pointer_cast(kernels::device::complex_ptr_cast(alpha.data()));
  auto begin = thrust::device_pointer_cast(kernels::device::complex_ptr_cast(a.data()));
  auto end   = begin + n;
  thrust::transform(thrust::cuda::par,begin,end,begin,add_scalar_func<thrust::complex<double>>(*alpha_p));
}

using memory::device_array_view;
template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

template void add_scalar_impl(device_array_view<std::complex<double>,1,basic_layout_t<1>> &, device_array_view<std::complex<double>,1,basic_layout_t<1>> &, int);

}