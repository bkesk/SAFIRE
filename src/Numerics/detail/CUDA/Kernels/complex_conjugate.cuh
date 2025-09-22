#ifndef COMPLEX_CONJUGATE_KERNELS_HPP
#define COMPLEX_CONJUGATE_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{

template<typename T>
void complex_conjugate_impl(int N, int M, std::complex<T>* A, int lda, long stride, int nb);

} // namespace kernels

#endif
