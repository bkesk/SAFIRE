//////////////////////////////////////////////////////////////////////
// File created by:
// Miguel A. Morales
////////////////////////////////////////////////////////////////////////////////

#ifndef GETGIJ_KERNELS_HPP
#define GETGIJ_KERNELS_HPP

#include <complex>

namespace kernels
{

// C[n][w] = sum_a Aup[ I[n] ][a] B[w][a][ J[n] ]
// where I[n] = n2IJ[n]/M
//       J[n] = n2IJ[n]%M 
template<typename T, typename Q1, typename Q2>
void getGIJ_impl(int nw, int nIJ, int nspin, int M, int nel_a, int nel_b,
        std::complex<Q1> const* Aup, int ldau, std::complex<Q1> const* Adn, int ldad,
        std::complex<Q2> const* B, int ldb, long strideB, std::complex<T>* C, int ldc, size_t const* n2IJ);

} // namespace kernels

#endif
