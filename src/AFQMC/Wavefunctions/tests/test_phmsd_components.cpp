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

#include <vector>
#include <random>

#include "AFQMC/config.h"
#include "Utilities/app_loggers.h"
#include "config.0.h"
#include "Numerics/ma_blas.hpp"
#include "SparseMatrix/tests/matrix_helpers.h"
#include "AFQMC/Utilities/test_utils.hpp"
#include "Numerics/ma_operations.hpp"
#include "Numerics/determinant.hpp"
#include "Numerics/ma_small_mat_ops.hpp"
#include "Numerics/batched_operations.hpp"
#include "AFQMC/Wavefunctions/detail/phmsd_impl.hpp"
#include "Utilities/Timer.hpp"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"


using boost::multi::array;
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

#if defined(ENABLE_DEVICE)
TEST_CASE("test_ph_excited_energy_real_dense_cholesky", "[Numerics][misc_kernels]")
{
  setup_loggers(true,2,0);
  Alloc<ComplexType> alloc{};
  Alloc<int> ialloc{};
// for timing/perf
//  int ndet = 500;
//  int nmo = 100;
//  int nelec = 50;
//  int nact = 100;
//  int nchol = 500;
// for unit testing 
  int ndet = 50;
  int nmo = 10;
  int nelec = 5;
  int nact = 10;
  int nchol = 50;
  int maxwalk=16;
  afqmc::HostBufferManager host_buffer(10uL * 1024uL * 1024uL);  
  afqmc::DeviceBufferManager device_buffer(10uL * 1024uL * 1024uL);  
  Tensor1D<int> refc(iextensions<1u>{nelec}, ialloc);
  boost::multi::array<int, 1> refc_(iextensions<1u>{nelec});
  for(int i=0; i<nelec; i++) refc[i]=i;
  for(int i=0; i<nelec; i++) refc_[i]=i;
  //for(int nex=5; nex<6; nex++) 
  for(int nex=1; nex<6; nex++) 
  {
//    std::cout<<" # excitations: " <<nex <<std::endl;
    boost::multi::array<ComplexType, 4> Tna({1, nelec, nchol, nact}, ComplexType(0.0));
    boost::multi::array<ComplexType, 4> Tan({1, nelec, nact, nchol}, ComplexType(0.0));
    boost::multi::array<ComplexType, 4> R_({1, ndet, nex, nact}, ComplexType(0.0));
    boost::multi::array<ComplexType, 2> wgt_({ndet, 1}, ComplexType(0.0));
    boost::multi::array<ComplexType, 3> KE_({ndet, 1, nchol}, ComplexType(0.0));
    boost::multi::array<int, 2> iexcit_({ndet,2*nex});
    Tensor1D<int> iexcit(iextensions<1u>{ndet*2*nex}, ialloc);
    {
      std::vector<ComplexType> tmp( std::max(Tna.num_elements(), R_.num_elements()) );
      afqmc::fillRandomMatrix(tmp);
      copy_n(tmp.data(), Tna.num_elements(), Tna.origin());
      copy_n(tmp.data(), R_.num_elements(), R_.origin());
      copy_n(tmp.data(), wgt_.num_elements(), wgt_.origin());
      for(int i=0; i<nelec; i++)
        for(int a=0; a<nact; a++)
          for(int n=0; n<nchol; n++)
            Tan[0][i][a][n] = Tna[0][i][n][a];
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
            }
            iocc[v]=1;
            iexcit_[nd][i+nex] = v+nelec;
          }
        }
      }
      copy_n(iexcit_.origin(), iexcit_.num_elements(), iexcit.origin());
    }
    Vector<ComplexType> EJ(iextensions<1u>{1}, 0.0);
    Vector<ComplexType> EX(iextensions<1u>{1}, 0.0);
    using ma::ph_excited_2body_energy_dense_cholesky_Tpan;
    Watch timer;
    ph_excited_2body_energy_dense_cholesky_Tpan(iexcit_.origin(), refc_.origin(),
          Tan, R_, wgt_, EX, EJ, KE_);
    double tcpu(timer.elapsed());
    double tgpu_1w(0.0);
    //for(int nwalk=1; nwalk<=72; nwalk+=71) {
    for(int nwalk=1; nwalk<=maxwalk; nwalk*=2) {
    //for(int nwalk=1; nwalk<2; nwalk++) {
      Tensor4D<ComplexType> T({nwalk,nelec,nchol,nact}, ComplexType(0.0), alloc);
      Tensor4D<ComplexType> R({nwalk,ndet,nex,nact}, ComplexType(0.0), alloc);
      Tensor2D<ComplexType> wgt({ndet,nwalk}, ComplexType(0.0), alloc);
      Tensor3D<ComplexType> KE({ndet,nwalk,nchol}, ComplexType(0.0), alloc);
      Tensor2D<ComplexType> E({nwalk,2}, ComplexType(0.0), alloc);
      for(int iw=0; iw<nwalk; iw++) {
        copy_n(Tna.origin(), T[iw].num_elements(), T[iw].origin());
        copy_n(R_.origin(), R[iw].num_elements(), R[iw].origin());
	wgt({0,ndet},iw) = wgt_({0,ndet},0);
      }
      timer.reset();
      ma::ph_excited_2body_energy_dense_cholesky_Tpna(iexcit.origin(), refc.origin(),
		T, R, wgt, E.rotated()[0], E.rotated()[1], KE);
      double tgpu(timer.elapsed());
      if(nwalk==1) 
        tgpu_1w=tgpu;
      {
        boost::multi::array<ComplexType, 2> tmp(E);
        boost::multi::array<ComplexType, 3> tmpK(KE);
        for(int iw=0; iw<nwalk; iw++) {
          REQUIRE(std::real(tmp[iw][0]) == Approx(std::real(EX[0])));
          REQUIRE(std::imag(tmp[iw][0]) == Approx(std::imag(EX[0])));
          REQUIRE(std::real(tmp[iw][1]) == Approx(std::real(EJ[0])));
          REQUIRE(std::imag(tmp[iw][1]) == Approx(std::imag(EJ[0])));
        }
        for(int id=0; id<ndet; id++) {
        for(int iw=0; iw<nwalk; iw++) {
        for(int n=0; n<nchol; n++) {
          REQUIRE(std::real(tmpK[id][iw][n]) == Approx(std::real(KE_[id][0][n])));
          REQUIRE(std::imag(tmpK[id][iw][n]) == Approx(std::imag(KE_[id][0][n])));
        }
        }
        }
      }
      ma::fill(E, ComplexType(0.0));
      ma::fill(KE.flatted(), ComplexType(0.0));
      timer.reset();
      for(int iw=0; iw<nwalk; iw++) {
        ma::ph_excited_2body_energy_dense_cholesky_Tpna(iexcit.origin(), refc.origin(),
		T.sliced(iw,iw+1), R.sliced(iw,iw+1), wgt(boost::multi::ALL, {iw,iw+1}), 
		E.sliced(iw,iw+1).rotated()[0], E.sliced(iw,iw+1).rotated()[1], 
		KE(boost::multi::ALL,{iw,iw+1},boost::multi::ALL));
      }
      double tgpu2(timer.elapsed());
      {
        boost::multi::array<ComplexType, 2> tmp(E);
        boost::multi::array<ComplexType, 3> tmpK(KE);
        for(int iw=0; iw<nwalk; iw++) {
          REQUIRE(std::real(tmp[iw][0]) == Approx(std::real(EX[0])));
          REQUIRE(std::imag(tmp[iw][0]) == Approx(std::imag(EX[0])));
          REQUIRE(std::real(tmp[iw][1]) == Approx(std::real(EJ[0])));
          REQUIRE(std::imag(tmp[iw][1]) == Approx(std::imag(EJ[0])));
        }
        for(int id=0; id<ndet; id++) {
        for(int iw=0; iw<nwalk; iw++) {
        for(int n=0; n<nchol; n++) {
          REQUIRE(std::real(tmpK[id][iw][n]) == Approx(std::real(KE_[id][0][n])));
          REQUIRE(std::imag(tmpK[id][iw][n]) == Approx(std::imag(KE_[id][0][n])));
        }
        }
        }
      }
//      std::cout<<"    " <<nwalk <<"     " <<tcpu*nwalk <<" " <<tgpu <<" " <<tgpu2 <<"   -   " <<tgpu/tgpu2 <<std::endl; 
    }  
  }
  host_buffer.release();
  device_buffer.release();
}
#else
TEST_CASE("test_ph_excited_energy_real_dense_cholesky", "[Numerics][misc_kernels]")
{
  setup_loggers(true,2,0);
  int ndet = 50;
  int nelec = 5;
  int nact = 10;
  int nchol = 50;
  // enough memry to avoid allocation
  afqmc::HostBufferManager host_buffer(10uL * 1024uL * 1024uL);  
  //for(int nex=1; nex<21; nex++) 
  for(int nex=1; nex<6; nex++) 
  {
//    std::cout<<" # excitations: " <<nex <<std::endl;
    boost::multi::array<ComplexType, 4> Tna({1, nelec, nchol, nact}, ComplexType(0.0));
    boost::multi::array<ComplexType, 4> Tan({1, nelec, nact, nchol}, ComplexType(0.0));
    boost::multi::array<ComplexType, 4> R_({1, ndet, nex, nact}, ComplexType(0.0));
    boost::multi::array<ComplexType, 2> wgt_({ndet, 1}, ComplexType(0.0));
    boost::multi::array<ComplexType, 3> KE_({ndet, 1, nchol}, ComplexType(0.0));
    boost::multi::array<int, 2> orbs_({ndet,nelec});
    boost::multi::array<int, 2> iexcit_({ndet,2*nex});
    boost::multi::array<int, 1> refc_(iextensions<1u>{nelec});
    for(int i=0; i<nelec; i++) refc_[i]=i;
    {
      std::vector<ComplexType> tmp( std::max(Tna.num_elements(), R_.num_elements()) );
      afqmc::fillRandomMatrix(tmp);
      copy_n(tmp.data(), Tna.num_elements(), Tna.origin());
      copy_n(tmp.data(), R_.num_elements(), R_.origin());
      copy_n(tmp.data(), wgt_.num_elements(), wgt_.origin());
      for(int i=0; i<nelec; i++)
        for(int a=0; a<nact; a++)
          for(int n=0; n<nchol; n++)
            Tan[0][i][a][n] = Tna[0][i][n][a];
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
            }
            iocc[v]=1;
            iexcit_[nd][i+nex] = v+nelec;
          }
        }
      }
    }
    Vector<ComplexType> EJ(iextensions<1u>{1}, 0.0);
    Vector<ComplexType> EX(iextensions<1u>{1}, 0.0);
    using ma::ph_excited_2body_energy_dense_cholesky_Tpna;
    using ma::ph_excited_2body_energy_dense_cholesky_Tpan;
    Watch timer;
    ph_excited_2body_energy_dense_cholesky_Tpan(iexcit_.origin(), refc_.origin(), 
	  Tan, R_, wgt_, EX, EJ, KE_);
    Vector<ComplexType> EJ2(iextensions<1u>{1}, 0.0);
    Vector<ComplexType> EX2(iextensions<1u>{1}, 0.0);
    timer.reset();
    ph_excited_2body_energy_dense_cholesky_Tpna(iexcit_.origin(), refc_.origin(), 
	  Tna, R_, wgt_, EX2, EJ2, KE_);
    REQUIRE(std::real(EX[0]) == Approx(std::real(EX2[0])));
    REQUIRE(std::imag(EX[0]) == Approx(std::imag(EX2[0])));
    REQUIRE(std::real(EJ[0]) == Approx(std::real(EJ2[0])));
    REQUIRE(std::imag(EJ[0]) == Approx(std::imag(EJ2[0])));
//    std::cout<<"    " <<tcpu1 <<" " <<tcpu2 <<std::endl; 
  }  
  host_buffer.release();
}
#endif

} // namespace sfqmc
