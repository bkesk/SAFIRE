//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef COPY_SELECT_KERNELS_HPP
#define COPY_SELECT_KERNELS_HPP

#include <complex>

namespace kernels
{
template<typename T1, typename T2, typename T3, typename T4, typename Int>
void copy_select(int N, int M, std::complex<T1> alpha, std::complex<T2> const* A, int lda, 
                        long Astride, std::complex<T3> beta, std::complex<T4>* B, int ldb,
                        long Bstride, Int const* index, int nbatch, bool expand);
} // namespace kernels

#endif
