////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#include "catch_amalgamated.hpp"
#include "config.h"
#include "Utilities/AppAbort.hpp"

#include <vector>
#include <random>

#include "AFQMC/config.h"
#include "config.0.h"
#include "Numerics/ma_blas.hpp"
#include "Numerics/batched_operations.hpp"
#include "SparseMatrix/tests/matrix_helpers.h"
#if defined(ENABLE_CUDA)
#include "Numerics/detail/CUDA/blas_cuda_gpu_ptr.hpp"
#include "Numerics/device_kernels.hpp"
#elif defined(ENABLE_HIP)
#include "Numerics/detail/HIP/blas_hip_gpu_ptr.hpp"
#include "Numerics/device_kernels.hpp"
#endif
#include "Utilities/Timer.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "Numerics/ma_operations.hpp"
#include "Numerics/determinant.hpp"
#include "Numerics/ma_small_mat_ops.hpp"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"


using boost::multi::array;
using boost::multi::array_ref;
using boost::multi::iextensions;
using std::copy_n;

namespace sfqmc
{
using namespace afqmc;

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
template<typename T>
using Alloc = device::device_allocator<T>;
#else
template<typename T>
using Alloc = std::allocator<T>;
#endif
template<typename T>
using pointer = typename std::allocator_traits<Alloc<T>>::pointer;


template<typename T>
using Tensor1D = array<T, 1, Alloc<T>>;
template<typename T>
using Tensor2D = array<T, 2, Alloc<T>>;
template<typename T>
using Tensor3D = array<T, 3, Alloc<T>>;
template<typename T>
using Tensor4D = array<T, 4, Alloc<T>>;

/*
TEST_CASE("axpyBatched", "[Numerics][misc_kernels]")
{
  // Only implemented for complex
  Alloc<std::complex<double>> alloc{};
  Tensor2D<std::complex<double>> y({3, 4}, 1.0, alloc);
  Tensor2D<std::complex<double>> x({3, 4}, 1.0, alloc);
  Tensor1D<std::complex<double>> a(iextensions<1u>{3}, 2.0, alloc);
  std::vector<pointer<std::complex<double>>> x_batched, y_batched;
  for (int i = 0; i < x.size(0); i++)
  {
    x_batched.emplace_back(x[i].origin());
    y_batched.emplace_back(y[i].origin());
  }
  ma::axpyBatched(x.size(1), raw_pointer_cast(a.origin()), x_batched.data(), 1, 
							   y_batched.data(), 1, x_batched.size(), 
					      ma::select_backend<Tensor2D<std::complex<double>>>());
  // 1 + 2 = 3.
  Tensor2D<std::complex<double>> ref({3, 4}, 3.0, alloc);
  verify_approx(y, ref);
}
*/

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
// Not in dispatching routine yet, called directly from AFQMCBasePropagator.
TEST_CASE("construct_X", "[Numerics][misc_kernels]")
{
  Alloc<std::complex<double>> alloc{};
  int ncv                 = 11;
  int nsteps              = 2;
  int nwalk               = 3;
  bool fp                 = false;
  double sqrtdt           = 0.002;
  double vbound           = 40.0;
  std::complex<double> im = std::complex<double>(0.0, 1.0);
  Tensor1D<std::complex<double>> vmf(iextensions<1U>{ncv}, im, alloc);
  Tensor2D<std::complex<double>> vbias({ncv, nwalk}, 1.0, alloc);
  Tensor2D<std::complex<double>> hws({nsteps, nwalk}, -0.2, alloc);
  Tensor2D<std::complex<double>> mf({nsteps, nwalk}, 2.0, alloc);
  Tensor3D<std::complex<double>> x({ncv, nsteps, nwalk}, 0.1, alloc);
  using kernels::construct_X;
  construct_X(ncv, nsteps, nwalk, fp, sqrtdt, vbound, raw_pointer_cast(vmf.origin()), raw_pointer_cast(vbias.origin()),
              raw_pointer_cast(hws.origin()), raw_pointer_cast(mf.origin()), raw_pointer_cast(x.origin()));
  // captured from stdout.
  std::complex<double> ref_val = std::complex<double>(0.18, 0.08);
  Tensor3D<std::complex<double>> ref({ncv, nsteps, nwalk}, ref_val, alloc);
  verify_approx(ref, x);
}

// No cpu equivalent?
TEST_CASE("batched_dot", "[Numerics][misc_kernels]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  arch::INIT(node);
  Alloc<std::complex<double>> alloc{};
  std::complex<double> im = std::complex<double>(0.0, 1.0);
  int dim                 = 3;
  Tensor1D<std::complex<double>> y(iextensions<1U>{dim}, im, alloc);
  Tensor2D<std::complex<double>> A({dim, dim}, 1.0, alloc);
  Tensor2D<std::complex<double>> B({dim, dim}, -0.2, alloc);
  std::complex<double> alpha(2.0);
  std::complex<double> beta(-1.0);
  ma::dot('N','N', alpha, A, B, beta, y); 
  std::complex<double> ref_val(-1.2, -1.0);
  Tensor1D<std::complex<double>> ref(iextensions<1U>{dim}, ref_val, alloc);
  verify_approx(ref, y);
  y = Tensor1D<std::complex<double>>(iextensions<1U>{dim}, im, alloc);
  ma::dot('T','N', alpha, A, B, beta, y); 
  verify_approx(ref, y);
  y = Tensor1D<std::complex<double>>(iextensions<1U>{dim}, im, alloc);
  ma::dot('N','T', alpha, A, B, beta, y); 
  verify_approx(ref, y);
  y = Tensor1D<std::complex<double>>(iextensions<1U>{dim}, im, alloc);
  ma::dot('T','T', alpha, A, B, beta, y); 
  verify_approx(ref, y);
  y = Tensor1D<std::complex<double>>(iextensions<1U>{dim}, im, alloc);
  ma::dot('C','N', alpha, A, B, beta, y); 
  verify_approx(ref, y);
  y = Tensor1D<std::complex<double>>(iextensions<1U>{dim}, im, alloc);
  ma::dot('N','H', alpha, A, B, beta, y); 
  verify_approx(ref, y);
  y = Tensor1D<std::complex<double>>(iextensions<1U>{dim}, im, alloc);
  ma::dot('H','T', alpha, A, B, beta, y); 
  verify_approx(ref, y);
}

TEST_CASE("extract_overlap_matrix", "[Numerics][misc_kernels]")
{ 
  Alloc<ComplexType> alloc{};
  int ndet = 1000;
  int nmo = 100;
  std::cout<<" nex   Tfill_cpu    Tfill_gpu   -    Tfill_cpu/Tfill_gpu";
  for(int nex=1; nex<21; nex++) 
  {
    Tensor3D<ComplexType> M({ndet, nex, nex}, ComplexType(0.0), alloc);
    Tensor3D<ComplexType> T({1, nmo, nmo}, ComplexType(0.0), alloc);
    boost::multi::array<ComplexType, 3> M_({ndet, nex, nex}, ComplexType(0.0));
    boost::multi::array<ComplexType, 2> T_({nmo, nmo}, ComplexType(0.0));
    Tensor1D<int> excit(iextensions<1u>{ndet*2*nex}, 0, alloc);
    std::vector<int> tmpi(ndet*2*nex);
    { 
      std::vector<ComplexType> tmp(nmo*nmo);
      afqmc::fillRandomMatrix(tmp);
      copy_n(tmp.data(), tmp.size(), T.origin());
      copy_n(tmp.data(), tmp.size(), T_.origin());
      afqmc::fillRandomMatrix(tmpi, nmo);
      copy_n(tmpi.data(), tmpi.size(), excit.origin());
    }
    double tcpu = function_timer( [&] () {
      for (int i = 0; i < ndet; i++)
        for (int j = 0; j < nex; j++)
          for (int k = 0; k < nex; k++)
          { 
            M_[i][j][k] = T_[tmpi[i*2*nex+j+nex]][tmpi[i*2*nex+k]];
          }
      }
    );
    using kernels::extract_overlap_matrix;
    double tgpu = function_timer( [&] () {
      extract_overlap_matrix(1, ndet, nex, raw_pointer_cast(excit.origin()), 
                             raw_pointer_cast(T.origin()), T.stride(1), T.stride(0), 
			     raw_pointer_cast(M.origin())); } );
    { 
      boost::multi::array<ComplexType, 3> tmp({ndet,nex,nex}, 0.0);
      copy_n(M.origin(), M.num_elements(), tmp.origin());
      for (int i = 0; i < tmp.num_elements(); i++) {
        REQUIRE(std::real(*(tmp.origin()+i)) == Approx(std::real(*(M_.origin()+i))));
      }
    }
    std::cout<<nex <<" " <<tcpu <<" " <<tgpu <<"  -   " <<tcpu/tgpu <<"\n";
  }
}

TEST_CASE("phmsd_determinant", "[Numerics][misc_kernels]")
{ 
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  arch::INIT(node);
  using ma::strided_determinant_from_getrf;
  using kernels::phmsd_det;
  Alloc<ComplexType> alloc{};
  Alloc<int> ialloc{};
  int ndet = 1000;
  int nmo = 100;
  int nelec = 50;
  int nact = 100;
  std::cout<<"\n\n nex   nwalk   Tcpu    Tgpu1    Tgpu2    -    Tcpu/Tgpu2    Tgpu1/Tgpu2";
  // singles
  using Tptr = ComplexType*;
#ifdef NDEBUG
  for(int nex=1; nex<8; nex++) 
#else
  for(int nex=1; nex<4; nex++) 
#endif
  {
    std::cout<<" # excitations: " <<nex <<"";
    //for(int nwalk=1; nwalk<129; nwalk*=2) 
    for(int nwalk=1; nwalk<33; nwalk*=2) 
    {
      Tensor3D<ComplexType> T({nwalk, nmo, nmo}, ComplexType(0.0), alloc);
      Tensor1D<ComplexType> M(iextensions<1u>{nwalk*ndet*nex*nex}, ComplexType(0.0), alloc);
      Tensor2D<ComplexType> Ov1({ndet, nwalk}, ComplexType(0.0), alloc);
      Tensor2D<ComplexType> Ov2({ndet, nwalk}, ComplexType(0.0), alloc);
      Tensor1D<Tptr> Odev(iextensions<1u>{nwalk*ndet});
      Tensor1D<int> IWORK(iextensions<1u>{nwalk*ndet*(nex+1)});
      std::vector<Tptr> Ptrs;
      Ptrs.resize(nwalk*ndet);
      Ptrs[0] = raw_pointer_cast(M.origin());
      boost::multi::array<ComplexType, 3> T_({nwalk, nmo, nmo}, ComplexType(0.0));
      boost::multi::array<ComplexType, 2> Ov_({ndet, nwalk}, ComplexType(0.0));
      boost::multi::array<int, 2> orbs_({ndet,nelec});
      boost::multi::array<int, 2> iexcit_({ndet,2*nex});
      Tensor1D<int> iexcit(iextensions<1u>{ndet*2*nex}, ialloc);
      std::vector<int> tmpi(ndet*2*nex);
      Tensor1D<int> refc(iextensions<1u>{nelec}, ialloc);
      for(int i=0; i<nelec; i++) refc[i]=i;
      int cnt=0;
      {
        std::vector<ComplexType> tmp(nmo*nmo);
        afqmc::fillRandomMatrix(tmp);
        for(int iw=0; iw<nwalk; iw++) {
          copy_n(tmp.data(), T_[iw].num_elements(), T_[iw].origin());
          copy_n(tmp.data(), T[iw].num_elements(), T[iw].origin());
        }
        std::vector<int> tmpi(nex);
        std::mt19937 generator(0);
        {
          std::uniform_int_distribution<int> distribution(0, nelec-1);
          std::vector<int> iocc(nelec, 0);
          for(int nd=0; nd<ndet; nd++) {
            std::fill_n(iocc.begin(), nelec, 0);
            for(int i=0; i<nex; i++) {
              int v = distribution(generator);
              while( iocc[v] != 0 ) {
                v = distribution(generator);
                cnt++;
              }
              iocc[v]=1;
              iexcit_[nd][i] = v;
            }
          }
        }
        {
          std::uniform_int_distribution<int> distribution(0, nact-nelec-1);
          std::vector<int> iocc(nact-nelec, 0);
          for(int nd=0; nd<ndet; nd++) {
            std::fill_n(iocc.begin(), nact-nelec, 0);
            for(int i=0; i<nex; i++) {
              int v = distribution(generator);
              while( iocc[v] != 0 ) {
                v = distribution(generator);
                cnt++;
              }
              iocc[v]=1;
              iexcit_[nd][i+nex] = v+nelec;
            }
          }
        }
        copy_n(iexcit_.origin(), iexcit_.num_elements(), iexcit.origin());
        for(int nd=0; nd<ndet; nd++) {
          for(int i=0; i<nelec; i++)
            orbs_[nd][i] = i;
          for(int i=0; i<nex; i++)
            orbs_[nd][ iexcit_[nd][i] ] = iexcit_[nd][i+nex];
        }
      }
      double tcpu=1.0, tgpu1=1.0, tgpu2=1.0;
      Watch timer; 
      if(nex==1) {
        timer.reset();
        for (int iw = 0; iw < nwalk; iw++) {
          auto Tw( T_[iw] );
          for (int idet = 0; idet < ndet; idet++)
            Ov_[idet][iw] = Tw[iexcit_[idet][1]][iexcit_[idet][0]];
        }
        tcpu = timer.elapsed();
      } else if(nex==2) {
        timer.reset();
        for (int iw = 0; iw < nwalk; iw++) {
          auto Tw( T_[iw] );
          for (int idet = 0; idet < ndet; idet++) {
            auto ie(iexcit_[idet]);
            Ov_[idet][iw] = Tw[ie[2]][ie[0]] * Tw[ie[3]][ie[1]]
                           -Tw[ie[2]][ie[1]] * Tw[ie[3]][ie[0]]; 
          }
        }
        tcpu = timer.elapsed();
      } else if(nex==3) {
        timer.reset();
        for (int iw = 0; iw < nwalk; iw++) {
          auto Tw( T_[iw] );
          for (int idet = 0; idet < ndet; idet++) {
            auto ie(iexcit_[idet]);
            Ov_[idet][iw] = Tw[ie[3]][ie[0]] * (Tw[ie[4]][ie[1]] * Tw[ie[5]][ie[2]]
                                               -Tw[ie[4]][ie[2]] * Tw[ie[5]][ie[1]])
                           -Tw[ie[4]][ie[0]] * (Tw[ie[3]][ie[1]] * Tw[ie[5]][ie[2]]
                                               -Tw[ie[3]][ie[2]] * Tw[ie[5]][ie[1]])
                           +Tw[ie[5]][ie[0]] * (Tw[ie[3]][ie[1]] * Tw[ie[4]][ie[2]]
                                               -Tw[ie[3]][ie[2]] * Tw[ie[4]][ie[1]]);
          }
        }
        tcpu = timer.elapsed();
      }   
      // simulating phmsd_helpers::calculate_overlaps
      using kernels::extract_overlap_matrix;
      timer.reset();
      extract_overlap_matrix(nwalk, ndet, nex, raw_pointer_cast(iexcit.origin()), 
             raw_pointer_cast(T.origin()), T.stride(1), T.stride(0), raw_pointer_cast(M.origin()), true);
      for (int i = 1; i < ndet*nwalk; i++)
        Ptrs[i] = Ptrs[0] + i*nex*nex;
      arch::memcopy(raw_pointer_cast(Odev.origin()), Ptrs.data(), nwalk*ndet*sizeof(Tptr), arch::memcopyH2D);    
      qmc_cuda::cublas_check(cublas::cublas_getrfBatched(arch::global_cublas_handle, nex,
                raw_pointer_cast(Odev.origin()), nex, raw_pointer_cast(IWORK.origin()),
                raw_pointer_cast(IWORK.origin())+nwalk*ndet*nex, nwalk*ndet),
                "cublas_getrfBatched"); 
      // simulating case where Ovlps is contiguous
      strided_determinant_from_getrf(nex, M.origin(), nex, nex*nex, IWORK.origin(), nex,
               ComplexType(0.0), raw_pointer_cast(Ov1.origin()), 1, nwalk*ndet);  
      tgpu1 = timer.elapsed();
      if(nex<=3)
      { 
        boost::multi::array<ComplexType, 2> tmp(Ov1);
        for (int idet = 0; idet < ndet; idet++) {
          for (int iw = 0; iw < nwalk; iw++) {
            REQUIRE(std::abs(std::real(tmp[idet][iw])) == Approx(std::abs(std::real(Ov_[idet][iw]))));
            REQUIRE(std::abs(std::imag(tmp[idet][iw])) == Approx(std::abs(std::imag(Ov_[idet][iw]))));
          }
        }
      }
      timer.reset();
      phmsd_det(nwalk,ndet,nex,raw_pointer_cast(iexcit.origin()), raw_pointer_cast(T.origin()), 
                           T.stride(1), T.stride(0), raw_pointer_cast(Ov2.origin()), Ov2.stride(0)); 
      tgpu2 = timer.elapsed();
      {
        boost::multi::array<ComplexType, 2> tmp1(Ov1);
        boost::multi::array<ComplexType, 2> tmp2(Ov2);
        for (int i = 0; i < tmp1.num_elements(); i++) {
          REQUIRE(std::abs(std::real(*(tmp1.origin()+i))) == Approx(std::abs(std::real(*(tmp2.origin()+i)))));
          REQUIRE(std::abs(std::imag(*(tmp1.origin()+i))) == Approx(std::abs(std::imag(*(tmp2.origin()+i)))));
        }
      }
      std::cout<<"   " <<nwalk <<" "  <<tcpu <<" " <<tgpu1 <<" " <<tgpu2 <<"  -   " 
             <<tcpu/tgpu2 <<" " <<tgpu1/tgpu2 <<"";
    } // iwalk
    std::cout<<"";
  } // nex
}

TEST_CASE("phmsd_inverse", "[Numerics][misc_kernels]")
{ 
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  arch::INIT(node);
  using kernels::phmsd_inv;
  using kernels::extract_overlap_matrix;
  Alloc<ComplexType> alloc{};
  Alloc<int> ialloc{};
  int ndet = 1000;
  int nmo = 100;
  int nelec = 50;
  int nact = 100;
  std::cout<<"\n\n nex   nwalk   Tgpu1    Tgpu2    -    Tgpu1/Tgpu2";
  // singles
  using Tptr = ComplexType*;
  Tensor1D<int> refc(iextensions<1u>{nelec}, ialloc);
  for(int i=0; i<nelec; i++) 
    refc[i]=i;
#ifdef NDEBUG
  for(int nex=1; nex<6; nex++) 
#else
  for(int nex=1; nex<4; nex++) 
#endif
  {
    std::cout<<" # excitations: " <<nex <<"";
    boost::multi::array<ComplexType, 2> T_({nmo, nmo}, ComplexType(0.0));
    boost::multi::array<ComplexType, 3> M_({ndet, nex, nex}, ComplexType(0.0));
    boost::multi::array<ComplexType, 2> Ov_({ndet, 1}, ComplexType(0.0));
    boost::multi::array<int, 2> iexcit_({ndet,2*nex});
    Tensor1D<int> iexcit(iextensions<1u>{ndet*2*nex}, ialloc);
    {
      int cnt=0;
      std::vector<ComplexType> tmp(nmo*nmo);
      afqmc::fillRandomMatrix(tmp);
      copy_n(tmp.data(), T_.num_elements(), T_.origin());
      std::vector<int> tmpi(nex);
      std::mt19937 generator(0);
      {
        std::uniform_int_distribution<int> distribution(0, nelec-1);
        std::vector<int> iocc(nelec, 0);
        for(int nd=0; nd<ndet; nd++) {
          std::fill_n(iocc.begin(), nelec, 0);
          for(int i=0; i<nex; i++) {
            int v = distribution(generator);
            while( iocc[v] != 0 ) {
              v = distribution(generator);
              cnt++;
            }
            iocc[v]=1;
            iexcit_[nd][i] = v;
          }
        }
      }
      {
        std::uniform_int_distribution<int> distribution(0, nact-nelec-1);
        std::vector<int> iocc(nact-nelec, 0);
        for(int nd=0; nd<ndet; nd++) {
          std::fill_n(iocc.begin(), nact-nelec, 0);
          for(int i=0; i<nex; i++) {
            int v = distribution(generator);
            while( iocc[v] != 0 ) {
              v = distribution(generator);
              cnt++;
            }
            iocc[v]=1;
            iexcit_[nd][i+nex] = v+nelec;
          }
        }
      }
      copy_n(iexcit_.origin(), iexcit_.num_elements(), iexcit.origin());
    }
    //for(int nwalk=1; nwalk<129; nwalk*=2) 
    for(int nwalk=1; nwalk<33; nwalk*=2) 
    {
      Tensor3D<ComplexType> T({nwalk, nmo, nmo}, ComplexType(0.0), alloc);
      Tensor1D<ComplexType> M(iextensions<1u>{nwalk*ndet*nex*nex}, ComplexType(0.0), alloc);
      Tensor4D<ComplexType> Minv({nwalk,ndet,nex,nex}, ComplexType(0.0), alloc);
      Tensor1D<Tptr> Odev(iextensions<1u>{2*nwalk*ndet});
      Tensor1D<int> IWORK(iextensions<1u>{nwalk*ndet});
      std::vector<Tptr> Ptrs;
      Ptrs.resize(2*nwalk*ndet);
      Ptrs[0] = raw_pointer_cast(M.origin());
      Ptrs[nwalk*ndet] = raw_pointer_cast(Minv.origin());
      for(int iw=0; iw<nwalk; iw++) 
        copy_n(T_.origin(), T[iw].num_elements(), T[iw].origin());
      double tgpu1=1.0, tgpu2=1.0;  
      std::vector<int> IWORK_(nex+1);
      std::vector<ComplexType> WORK_(nex*nex);
      for (int idet = 0; idet < ndet; idet++) {
        auto ie(iexcit_[idet]);
        for(int ip=0; ip<nex; ip++) 
          for(int iq=0; iq<nex; iq++) 
            M_[idet][ip][iq] = T_[ie[ip + nex]][ie[iq]];
        Ov_[idet][0] = ma::invert<ComplexType>(M_[idet], IWORK_, WORK_, 0.0);
      }
      // simulating phmsd_helpers::calculate_overlaps
      Watch timer; 
      extract_overlap_matrix(nwalk, ndet, nex, raw_pointer_cast(iexcit.origin()), 
             raw_pointer_cast(T.origin()), T.stride(1), T.stride(0), raw_pointer_cast(M.origin()));
      for (int i = 1; i < ndet*nwalk; i++) {
        Ptrs[i] = Ptrs[0] + i*nex*nex;
        Ptrs[ndet*nwalk+i] = Ptrs[ndet*nwalk] + i*nex*nex;
      }
      arch::memcopy(raw_pointer_cast(Odev.origin()), Ptrs.data(), 2*nwalk*ndet*sizeof(Tptr), arch::memcopyH2D);    
      qmc_cuda::cublas_check(cublas::cublas_matinvBatched(arch::global_cublas_handle,
                nex, raw_pointer_cast(Odev.origin()), nex, raw_pointer_cast(Odev.origin())+nwalk*ndet, nex,
                raw_pointer_cast(IWORK.origin()), nwalk*ndet),
                "cublas_matinvBatched"); 
      tgpu1 = timer.elapsed();
      { 
        boost::multi::array<ComplexType, 4> tmp(Minv);
        for (int iw = 0; iw < nwalk; iw++) 
          for (int idet = 0; idet < ndet; idet++) 
            for (int p = 0; p < nex; p++) 
              for (int q = 0; q < nex; q++) {
                REQUIRE(std::abs(std::real(tmp[iw][idet][p][q])) == Approx(std::abs(std::real(M_[idet][p][q]))));
                REQUIRE(std::abs(std::imag(tmp[iw][idet][p][q])) == Approx(std::abs(std::imag(M_[idet][p][q]))));
              }
      }
      fill_n(Minv.origin(),Minv.num_elements(),ComplexType(0.0));
      if(nex < 6) {
        timer.reset();
        phmsd_inv(nwalk,ndet,nex,raw_pointer_cast(iexcit.origin()), raw_pointer_cast(T.origin()), 
                           T.stride(1), T.stride(0), raw_pointer_cast(Minv.origin()) ); 
        tgpu2 = timer.elapsed();
        boost::multi::array<ComplexType, 4> tmp(Minv);
        for (int iw = 0; iw < nwalk; iw++)
          for (int idet = 0; idet < ndet; idet++)
            for (int p = 0; p < nex; p++)
              for (int q = 0; q < nex; q++) {
                REQUIRE(std::abs(std::real(tmp[iw][idet][p][q])) == Approx(std::abs(std::real(M_[idet][p][q]))));
                REQUIRE(std::abs(std::imag(tmp[iw][idet][p][q])) == Approx(std::abs(std::imag(M_[idet][p][q]))));
              }
      }
      std::cout<<"   " <<nwalk <<" " <<tgpu1 <<" " <<tgpu2 <<"  -   " <<" " <<tgpu1/tgpu2 <<"";
    } // iwalk
    std::cout<<"";
  } // nex
}

TEST_CASE("construct_phmsd_R", "[Numerics][misc_kernels]")
{
  Alloc<ComplexType> alloc{};
  Alloc<int> ialloc{};
  int ndet = 2000;
  int nmo = 100;
  int nelec = 50;
  int nact = 100;
  int maxwalk=16;
  //for(int nex=1; nex<21; nex++) 
  for(int nex=1; nex<6; nex++) 
  {
    std::cout<<" # excitations: " <<nex <<std::endl;
    for(int nwalk=1; nwalk<=maxwalk; nwalk*=2) {
    //for(int nwalk=1; nwalk<2; nwalk++) {
      boost::multi::array<ComplexType, 3> T_({nwalk, nmo, nmo}, ComplexType(0.0));
      boost::multi::array<ComplexType, 4> I_({nwalk, ndet, nex, nex}, ComplexType(0.0));
      boost::multi::array<ComplexType, 3> R_({nwalk, nelec, nact}, ComplexType(0.0));
      boost::multi::array<ComplexType, 2> weights_({ndet, nwalk}, ComplexType(7.0));
      boost::multi::array<int, 2> orbs_({ndet,nelec});
      boost::multi::array<int, 2> iexcit_({ndet,2*nex});
      Tensor3D<ComplexType> T({nwalk, nmo, nmo}, ComplexType(0.0), alloc);
      Tensor4D<ComplexType> I({nwalk, ndet, nex, nex}, ComplexType(0.0), alloc);
      Tensor4D<ComplexType> Rbuff({nwalk, ndet, nex, nact}, ComplexType(0.0), alloc);
      Tensor3D<ComplexType> R({nwalk, nelec, nact}, ComplexType(0.0), alloc);
      Tensor2D<ComplexType> weights({ndet, nwalk}, ComplexType(7.0), alloc);
      Tensor1D<int> iexcit(iextensions<1u>{ndet*2*nex}, ialloc);
      Tensor1D<int> refc(iextensions<1u>{nelec}, ialloc);
      for(int i=0; i<nelec; i++) refc[i]=i;
      int cnt=0;  
      {
        std::vector<ComplexType> tmp( std::max(T[0].num_elements(), I[0].num_elements()) );
        afqmc::fillRandomMatrix(tmp);
        for(int iw=0; iw<nwalk; iw++) {
          copy_n(tmp.data(), T_[iw].num_elements(), T_[iw].origin());
          copy_n(tmp.data(), I_[iw].num_elements(), I_[iw].origin());
          copy_n(tmp.data(), T[iw].num_elements(), T[iw].origin());
          copy_n(tmp.data(), I[iw].num_elements(), I[iw].origin());
        }
        std::vector<int> tmpi(nex);
        std::mt19937 generator(0);
        {  
          std::uniform_int_distribution<int> distribution(0, nelec-1);
          std::vector<int> iocc(nelec, 0);
          for(int nd=0; nd<ndet; nd++) {
            std::fill_n(iocc.begin(), nelec, 0);
            for(int i=0; i<nex; i++) {
              int v = distribution(generator);
              while( iocc[v] != 0 ) {
                v = distribution(generator);
                cnt++;
              }
              iocc[v]=1;
              iexcit_[nd][i] = v;
            }
          }
        }  
        {
          std::uniform_int_distribution<int> distribution(0, nact-nelec-1);
          std::vector<int> iocc(nact-nelec, 0);
          for(int nd=0; nd<ndet; nd++) {
            std::fill_n(iocc.begin(), nact-nelec, 0);
            for(int i=0; i<nex; i++) {
              int v = distribution(generator);
              while( iocc[v] != 0 ) {
                v = distribution(generator);
                cnt++;
              }
              iocc[v]=1;
              iexcit_[nd][i+nex] = v+nelec;
            }
          }
        }
        copy_n(iexcit_.origin(), iexcit_.num_elements(), iexcit.origin());
        for(int nd=0; nd<ndet; nd++) {
          for(int i=0; i<nelec; i++)
            orbs_[nd][i] = i;
          for(int i=0; i<nex; i++)
            orbs_[nd][ iexcit_[nd][i] ] = iexcit_[nd][i+nex]; 
        }  
      }
      Watch timer;
      for(int iw=0; iw<nwalk; iw++) {
        auto Riw(R_[iw]);
        auto Tiw(T_[iw]);
        auto Iiw(I_[iw]);
        for (int nd = 0; nd < ndet; nd++) {
          auto o_(orbs_[nd]);
          auto e_(iexcit_[nd]);
          auto w_(weights_[nd][iw]);
          auto Q(Iiw[nd]);
          for (int i = 0; i < nelec; ++i)
            Riw[i][o_[i]] += w_;
          for (int p = 0; p < nex; ++p)
          {
            auto Rp = Riw[e_[p]];
            auto Ip = Q[p];
            for (int q = 0; q < nex; ++q)
            {
              auto Ipq = Ip[q];
              auto Tq  = Tiw[e_[q+nex]];
              for (int i = 0; i < nelec; ++i) {
                Rp[o_[i]] -= w_ * Ipq * Tq[i];
              }
              Rp[o_[e_[q]]] += w_ * Ipq;
            }
          }
        }
      }
      double tcpu(timer.elapsed());
      using kernels::construct_phmsd_R;
      timer.reset();
      construct_phmsd_R(nwalk, ndet, nex, nact, nelec, 
                    raw_pointer_cast(iexcit.origin()), raw_pointer_cast(refc.origin()),
                    raw_pointer_cast(T.origin()), T.stride(1), T.stride(0), 
                    raw_pointer_cast(I.origin()), raw_pointer_cast(Rbuff.origin()));
      double tgpu1(timer.elapsed());
      using kernels::reduce_phmsd_R;
      timer.reset();
      reduce_phmsd_R(nwalk, ndet, nex, nact, nelec, 
                 raw_pointer_cast(iexcit.origin()), raw_pointer_cast(refc.origin()),
                 raw_pointer_cast(weights.origin()), weights.stride(0),
                 raw_pointer_cast(Rbuff.origin()),
                 raw_pointer_cast(R.origin()));
      double tgpu2(timer.elapsed());
      {
        boost::multi::array<ComplexType, 3> tmp({nwalk, nelec,nact}, 0.0);
        copy_n(R.origin(), R.num_elements(), tmp.origin());
        for(int iw=0; iw<nwalk; iw++) {
          for (int iel = 0, i=0; iel < nelec; iel++) {
            for (int iv = 0; iv < nact; iv++, i++) {
              REQUIRE(std::real(tmp[iw][iel][iv]) == Approx(std::real(R_[iw][iel][iv])));
            }
          }
        }
      }
      std::cout<<"    " <<nwalk <<"     " <<tcpu <<" " <<tgpu1 <<" " <<tgpu2 <<"   -   " <<tcpu/(tgpu1+tgpu2) <<std::endl; 
    }  
  }
}

#endif

} // namespace sfqmc
