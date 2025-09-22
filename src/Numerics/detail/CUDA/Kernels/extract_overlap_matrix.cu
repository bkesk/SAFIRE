#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{
// extract overlap matrix (M) for PHMSD
// M[i][p][q] = T[iexcit[i][p+nex]][iexcit[i][q]]
__global__ void kernel_extract_overlap_matrix(int nwalk, 
                                              int ndet,
                                              int nex,
                                              int const* iexcit,
                                              thrust::complex<double> const* T,
                                              int ldT,
                                              long Tstride,
                                              thrust::complex<double>* M,
                                              bool reverse_order)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = x/ndet;
  int idet = x%ndet;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z;
  if (iwalk < nwalk and idet < ndet and y < nex and z < nex)
  {
    int ip = iexcit[idet*2*nex + y + nex];
    int iq = iexcit[idet*2*nex + z];
    if(reverse_order)
      // M[idet][iwalk][ip][iq]
      M[ (idet*nwalk + iwalk)*nex*nex + y*nex + z] = T[iwalk*Tstride + ip*ldT + iq];
    else
      // M[iwalk][idet][ip][iq]
      M[ (iwalk*ndet + idet)*nex*nex + y*nex + z] = T[iwalk*Tstride + ip*ldT + iq];
  }
}

void extract_overlap_matrix(int nwalk,
                            int ndet,
                            int nex,
                            int const* iexcit,
                            std::complex<double> const* T,
                            int ldT,
                            long Tstride,
                            std::complex<double> *M,    
                            bool reverse_order)
{
  int xblock_dim = 8;  // ndet*nwalk blocking
  int yblock_dim = 8;  // nex blocking 
  if( nex==1 ) {
    xblock_dim = 64;
    yblock_dim = 1;
  } else if( nex == 2 ) {
    xblock_dim = 16;
    yblock_dim = 2;
  } else if( nex <= 4 ) { 
    yblock_dim = 4;
  }  
  int grid_dim_x = (ndet*nwalk + xblock_dim - 1) / xblock_dim;
  int grid_dim_y = (nex + yblock_dim - 1) / yblock_dim;
  dim3 grid_dim(grid_dim_x, grid_dim_y, grid_dim_y);
  dim3 block_dim(xblock_dim, yblock_dim, yblock_dim);
  kernel_extract_overlap_matrix<<<grid_dim, block_dim>>>(nwalk, ndet, nex, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 ldT, Tstride,
                                                 reinterpret_cast<thrust::complex<double>*>(M),
                                                 reverse_order);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

}
