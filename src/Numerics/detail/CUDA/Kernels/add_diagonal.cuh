//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef ADD_DIAGONAL_KERNELS_HPP
#define ADD_DIAGONAL_KERNELS_HPP

#include <complex>

namespace kernels
{

template<typename T1, typename T2>
void add_diagonal(int N, int* y, std::complex<T1> const* w, int incx, 
                  std::complex<T2>* A, int lda, long Astride, int nb); 

} // namespace kernels

#endif
