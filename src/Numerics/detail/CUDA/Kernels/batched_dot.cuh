//////////////////////////////////////////////////////////////////////
// File created by:
// Miguel A. Morales
////////////////////////////////////////////////////////////////////////////////

#ifndef BATCHED_DOT_KERNELS_HPP
#define BATCHED_DOT_KERNELS_HPP

#include <complex>

namespace kernels
{

// y[i * incy] = y[i * incy] + sum_j alpha * op(A)[i,j] * op(B)[i,j]
template<class T, class Q1, class Q2>
void strided_batched_dot(char TA, char TB, int N, int M,
                std::complex<T> const alpha, std::complex<Q1> const* A, int lda,
                std::complex<Q2> const* B, int ldb,
                std::complex<T> *y, int incy);

// C[b,i] = C[b,i] + sum_j alpha * op(A[b])[i,j] * op(B[b])[i,j]
template<class T, class Q1, class Q2>
void strided_batched_dot(char TA, char TB, int nbatch, int N, int M,
                std::complex<T> const alpha, std::complex<Q1> const* A, int lda, long Astride,
                std::complex<Q2> const* B, int ldb, long Bstride,
                std::complex<T> *C, int ldc, long Cstride);

} // namespace kernels

#endif
