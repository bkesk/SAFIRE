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

#include "config.h"
#include "Utilities/AppAbort.hpp"

#include <vector>
#include <random>
#include <iomanip>

#include "Utilities/Timer.hpp"

#include "AFQMC/config.h"
#include "Memory/arch.hpp"
#include "Numerics/ma_blas.hpp"
#include "SparseMatrix/tests/matrix_helpers.h"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

// only meant to be compiled/run in DEVICE build
#if !defined(ENABLE_DEVICE)
#error
#endif

using namespace sfqmc;
using namespace afqmc;
using std::copy_n;

template<typename T>
using Alloc = device::device_allocator<T>;
template<typename T>
using pointer = typename std::allocator_traits<Alloc<T>>::pointer;

int main(int argc, char* argv[])
{
  //using T = double; //std::complex<double>;
  using T = std::complex<float>;
  T zero(0.0);
  T one(1.0);
  boost::mpi3::environment env(argc, argv);
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  setup_loggers(world.root(),2,0);
  arch::INIT(node);
  Watch timer;
  std::cout << "  N(MB)    tMulti     tKernel  ";
  { // D=1 tests
    int NMAX = 128*1024*1024; 
    std::cout<<"\n";
    std::cout << " - copy_n  D=1  HD \n";
    Vector<T, Alloc<T>> D({NMAX}, zero, Alloc<T>{});
    Vector<T, Alloc<T>> D2({NMAX}, zero, Alloc<T>{});
    Vector<T> H({NMAX}, zero);
    for(int N=2; N<=128; N*=2) {
      int N_(N*1024*1024);
      timer.reset(); timer.start(); 
      copy_n(H.origin(), N_, D.origin());
      timer.stop();
      double t1 = timer.elapsed();    
      timer.reset(); timer.start(); 
      arch::memcopy(raw_pointer_cast(D.origin()), H.origin(), N_ * sizeof(T));	
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2 
					   <<"    -     " <<t1/t2 <<"\n";  
    }
    std::cout<<"\n";
    std::cout << " - copy_n  D=1  DH \n";
    for(int N=2; N<=128; N*=2) {
      int N_(N*1024*1024);
      timer.reset(); timer.start();
      copy_n(D.origin(), N_, H.origin());
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      arch::memcopy(H.origin(), raw_pointer_cast(D.origin()), N_ * sizeof(T));
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2 
					   <<"    -     " <<t1/t2 <<"\n";  
    }
    std::cout<<"\n";
    std::cout << " - fill_n(zero)  D=1 \n";
    for(int N=2; N<=128; N*=2) {
      int N_(N*1024*1024);
      timer.reset(); timer.start();
      fill_n(D.origin(), N_, zero);
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      //kernels::fill_n(raw_pointer_cast(D.origin()), N_, zero);
      ma::fill(D.sliced(0,N_), zero);
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2 
					   <<"    -     " <<t1/t2 <<"\n";  
    }
    std::cout<<"\n";
    std::cout << " - fill_n('1')  D=1 \n";
    for(int N=2; N<=128; N*=2) {
      int N_(N*1024*1024);
      timer.reset(); timer.start();
      fill_n(D.origin(), N_, one);
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      kernels::fill_n(raw_pointer_cast(D.origin()), N_, one);
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
  }
  { // D=2 tests
    int NMAX = 32768/2;
    std::cout<<"\n";
    std::cout << " - contiguous D = H  D=2 \n";
    Matrix<T, Alloc<T>> D({NMAX, NMAX}, zero, Alloc<T>{});
    Matrix<T, Alloc<T>> D2({NMAX, NMAX}, zero, Alloc<T>{});
    Matrix<T> H({NMAX, NMAX}, zero);
    for(int N=1024; N<=NMAX; N*=2) {
      timer.reset(); timer.start();
      D.sliced(0,N) = H.sliced(0,N); 	
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      arch::memcopy(raw_pointer_cast(D.origin()), H.origin(), N*NMAX * sizeof(T));
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
    std::cout<<"\n";
    std::cout << " - contiguous H = D  D=2  DH \n";
    for(int N=1024; N<=NMAX; N*=2) {
      timer.reset(); timer.start();
      H.sliced(0,N) = D.sliced(0,N); 	
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      arch::memcopy(H.origin(), raw_pointer_cast(D.origin()), N*NMAX * sizeof(T));
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
    std::cout<<"\n";
    std::cout << " - contiguous D = D  D=2  DH \n";
    for(int N=1024; N<=NMAX; N*=2) {
      timer.reset(); timer.start();
      D2.sliced(0,N) = D.sliced(0,N);
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      arch::memcopy(raw_pointer_cast(D2.origin()), raw_pointer_cast(D.origin()), N*NMAX * sizeof(T));
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
    std::cout<<"\n";
    std::cout << " - D({0,N},{0,N}) = D({0,N},{0,N})  D=2 \n";
    for(int N=1024; N<=NMAX; N*=2) {
      timer.reset(); timer.start();
      D2({0,N},{0,N}) = D({0,N},{0,N});
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      arch::memcopy2D(raw_pointer_cast(D.origin()), sizeof(T) * NMAX,
                      raw_pointer_cast(D2.origin()), sizeof(T) * NMAX,
                      N*sizeof(T), N, arch::memcopyD2D);
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
    std::cout<<"\n";
    std::cout << " - D({0,N},{0,N}) = H({0,N},{0,N})  D=2 \n";
    for(int N=1024; N<=NMAX; N*=2) {
      timer.reset(); timer.start();
      D({0,N},{0,N}) = H({0,N},{0,N});
      double t1 = timer.elapsed();
      timer.stop();
      timer.reset(); timer.start();
      arch::memcopy2D(raw_pointer_cast(D.origin()), sizeof(T) * NMAX, 
		      H.origin(), sizeof(T) * NMAX,
                      N*sizeof(T), N, arch::memcopyH2D);
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
    std::cout<<"\n";
    std::cout << " - H({0,N},{0,N}) = D({0,N},{0,N})  D=2 \n";
    for(int N=1024; N<=NMAX; N*=2) {
      timer.reset(); timer.start();
      H({0,N},{0,N}) = D({0,N},{0,N});
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      arch::memcopy2D(H.origin(), sizeof(T) * NMAX,
		      raw_pointer_cast(D.origin()), sizeof(T) * NMAX, 
                      N*sizeof(T), N, arch::memcopyD2H);
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << N <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
    std::cout<<"\n";
    std::cout << " - H[0] = D[0] D=2\n";
    {
      timer.reset(); timer.start();
      H[0] = D[0];
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      copy_n(D.origin(), D.size(1), H.origin());
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << NMAX <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
    std::cout<<"\n";
    std::cout << " - D[0] = H[0] D=2\n";
    {
      timer.reset(); timer.start();
      D[0] = H[0];
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      copy_n(H.origin(), H.size(1), D.origin());
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << NMAX <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
    std::cout<<"\n";
    std::cout << " - H[0].sliced(0,NMAX) = D[0].sliced(0,NMAX) D=2\n";
    {
      timer.reset(); timer.start();
      H[0].sliced(0,NMAX/2) = D[0].sliced(0,NMAX/2);
      timer.stop();
      double t1 = timer.elapsed();
      timer.reset(); timer.start();
      copy_n(D.origin(), NMAX/2, H.origin());
      timer.stop();
      double t2 = timer.elapsed();
      std::cout<< std::setw(6) << NMAX/2 <<"  " << std::scientific <<t1 <<" " <<t2
                                           <<"    -     " <<t1/t2 <<"\n";
    }
  }
  return 0;
}
