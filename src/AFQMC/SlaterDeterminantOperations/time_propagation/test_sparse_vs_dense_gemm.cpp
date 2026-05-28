/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "catch_amalgamated.hpp"
#include "config.h"
#include "Utilities/AppAbort.hpp"

#include "hdf/hdf_archive.h"

#define MKL_INT int
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

#undef APP_ABORT
#define APP_ABORT(x) \
  {                  \
    std::cout << x;  \
    throw;           \
  }

#include <iostream>
#include <vector>
#include <complex>
#include <string>

#include "Utilities/Timer.hpp"
#include "Utilities/app_loggers.h"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Utilities/Utils.hpp"

#include "SparseMatrix/csr_matrix.hpp"
#include "Numerics/ma_operations.hpp"

#include "Memory/custom_pointers.hpp"
#include "Memory/buffer_managers.h"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

using boost::multi::array;
using std::vector;
template<std::ptrdiff_t D>
using iextensions = typename boost::multi::iextensions<D>;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{

namespace afqmc 
{

using stdcsrMat = ma::sparse::csr_matrix<ComplexType, int, int, std::allocator<ComplexType>>;
using devcsrMat = ma::sparse::csr_matrix<ComplexType, int, int, device_allocator<ComplexType>>;
using devMatrix = Matrix<ComplexType, device_allocator<ComplexType>>;
using dev3Tensor = Array<ComplexType, 3, device_allocator<ComplexType>>;

template<class Mat1, class Mat2, class Mat3,
        typename = typename std::enable_if_t<Mat2::dimensionality == 2>,
        typename = typename std::enable_if_t<std::decay_t<Mat3>::dimensionality == 2>
        >
double time_call(Mat1 const& A, Mat2 const& B, Mat3&& C)
{
  Watch time{};
  ma::product(A,B,C);
  time.reset();
  for(int i=0; i<6; i++)
    ma::product(A,B,C);
  return time.elapsed()/6.0;
}

template<class Mat1, class Mat2, class Mat3,
        typename = typename std::enable_if_t<Mat2::dimensionality == 3>,
        typename = typename std::enable_if_t<std::decay_t<Mat3>::dimensionality == 3>,
        typename = void
        >
double time_call(Mat1 const& A, Mat2 const& B, Mat3&& C)
{
  Watch time{};
  ma::productStridedBatched(A,B,C);
  time.reset();
  for(int i=0; i<6; i++)
    ma::productStridedBatched(A,B,C);
  return time.elapsed()/6.0;
}

/*
 * Compares the time taken to do each application of vHS to the Slater Matrix
 * using both sparse and dense vHS matrices in the hubbard model. 
 * Batched sparse is used and implemented by substituting the strided batched interface  
 * with a super vHS matrix in block-diagonal sparse form (each walker is a different block).
 * This leads to a single (non-batched) call to csrmm.
 */
void compare_sparse_dense_gemm([[maybe_unused]] boost::mpi3::communicator& world)
{

  std::vector<int> dims = {4, 8, 16, 20, 24, 32};
  for( auto Nx : dims ) 
  {

    int M = Nx*Nx;
    int N = M/2;

    stdcsrMat spV_h({M,M}, {0,0}, 1, std::allocator<ComplexType>{});
    for(int i=0; i<M; i++)
      spV_h[i][i] = ComplexType(1.0);
    Matrix<ComplexType> V_h( {M, M}, std::allocator<ComplexType>{} );
    for(int i=0; i<M; i++)
      V_h[i][i] = ComplexType(1.0);
    Matrix<ComplexType> B_h( {M, N}, ComplexType(0.0), std::allocator<ComplexType>{} );
    Matrix<ComplexType> C_h( {M, N}, ComplexType(0.0), std::allocator<ComplexType>{} );

    devcsrMat spV(spV_h);
    devMatrix V(V_h); 
    devMatrix B(B_h); 
    devMatrix C(C_h); 
  
    double serial_dense(time_call(V,B,C));  
    double serial_sparse(time_call(spV,B,C));  
    std::cout<<" Nx: " <<Nx <<"";
    std::cout<<" Dense  Single batch    " <<M <<" " <<serial_dense  <<" " <<1 <<"";
    std::cout<<" Sparse  Single batch   " <<M <<" " <<serial_sparse <<" " <<1 <<"";

    std::cout<<"*******************************************************************"
           <<" nbatch   td   td/(n*tserial)   ts   ts/(n*tserial)  ts/td  "
           <<"*******************************************************************";  


    for(int nb=2; nb<=64; nb*=2) {

      dev3Tensor Vb( {nb, M, M} );
      dev3Tensor Bb( {nb, M, N} );
      dev3Tensor Cb( {nb, M, N} );

      stdcsrMat spVb_h({nb*M,nb*M}, {0,0}, 1, std::allocator<ComplexType>{});
      for(int i=0; i<nb*M; i++)
        spVb_h[i][i] = ComplexType(1.0);
      devcsrMat spVb(spVb_h);

      auto Bb2( Bb.flatted() ); 
      auto Cb2( Cb.flatted() );
      double td(time_call(Vb,Bb,Cb));
      double ts(time_call(spVb,Bb2,Cb2));
      std::cout<<nb <<"   " <<td <<" " <<td/(nb*serial_dense) <<"  "
                            <<ts <<" " <<ts/(nb*serial_sparse) <<"  " <<ts/td  <<"";
    }  
  }
}

TEST_CASE("wfn_fac_distributed", "[wavefunction_factory]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  auto node = world.split_shared(world.rank());

  arch::INIT(node);
#else
  auto node   = world.split_shared(world.rank());
#endif
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);

  compare_sparse_dense_gemm(world); 
  release_memory_managers();
}

} // afqmc

} // sfqmc



