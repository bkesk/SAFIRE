//////////////////////////////////////////////////////////////////////
// File created by:
// Miguel A. Morales
////////////////////////////////////////////////////////////////////////////////

#ifndef NUMERICS_AGiwj_BGjwi_CG_CUDA_KERNEL_HPP
#define NUMERICS_AGiwj_BGjwi_CG_CUDA_KERNEL_HPP

#include <complex>
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{

// keeping this simple, assuming contiguous arrays 
template<class T1, class T2, class T3, class T4>
void AGiwj_BGjwi_CG_impl(int nbatch, int ng, int ni, int nwalk, int nj, 
        std::complex<T1> const* Xw, std::complex<T2> const* A, 
        std::complex<T3> const* B, std::complex<T4>* C);  


}

#endif
