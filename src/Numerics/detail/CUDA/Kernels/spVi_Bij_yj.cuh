//////////////////////////////////////////////////////////////////////
// File created by:
// Miguel A. Morales
////////////////////////////////////////////////////////////////////////////////

#ifndef SPVI_BIJ_YJ_KERNELS_HPP
#define SPVI_BIJ_YJ_KERNELS_HPP

#include <complex>

namespace kernels
{

template<typename I1, typename T1, typename T2, typename T3>
void spVi_Bij_yj(int nj, int nnz, I1 const* index, T1 const* values,
                 std::complex<T2> const* B, int ldb, std::complex<T3>* y, int incy);

template<typename I1, typename T1, typename T2, typename T3>
void spVi_Bij_yj(int nj, int nnz, I1 const* index, std::complex<T1> const* values,
                 std::complex<T2> const* B, int ldb, std::complex<T3>* y, int incy);

} // namespace kernels

#endif
