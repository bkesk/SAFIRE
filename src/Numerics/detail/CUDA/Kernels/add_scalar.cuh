//////////////////////////////////////////////////////////////////////
// File created by:
// Miguel A. Morales
////////////////////////////////////////////////////////////////////////////////

#ifndef ADD_SCALAR_KERNELS_HPP
#define ADD_SCALAR_KERNELS_HPP

#include <complex>

namespace kernels
{

template<class T>
void add_scalar(int N, int M, T a, T* A ,int lda);

template<class T>
void add_scalar(int N, int M, std::complex<T> a, std::complex<T>* A ,int lda);

} // namespace kernels

#endif
