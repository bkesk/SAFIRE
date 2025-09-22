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
#include "config.0.h"
#include "Numerics/ma_blas.hpp"
#include "Numerics/batched_operations.hpp"
#include "Numerics/ma_operations.hpp"
#include "SparseMatrix/tests/matrix_helpers.h"
#include "Memory/buffer_managers.h"
#include "Memory/arch.hpp"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

using namespace sfqmc;
using namespace afqmc;

using std::copy_n;

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
template<typename T>
using Alloc = device::device_allocator<T>;
#else
template<typename T>
using Alloc = std::allocator<T>;
#endif
template<typename T>
using pointer = typename std::allocator_traits<Alloc<T>>::pointer;


//template<typename T>
//using buffer_alloc_type = device_buffer_type<T>;

//using buffer_ialloc_type = device_buffer_type<int>;

template<typename T>
using Tensor1D = boost::multi::array<T, 1, Alloc<T>>;
template<typename T>
using Tensor2D = boost::multi::array<T, 2, Alloc<T>>;
template<typename T>
using Tensor3D = boost::multi::array<T, 3, Alloc<T>>;
template<typename T>
using Tensor5D = boost::multi::array<T, 5, Alloc<T>>;

template<typename T>
using Tensor5D_ref = boost::multi::array_ref<T, 5, pointer<T>>;
template<typename T>
using Tensor3D_ref = boost::multi::array_ref<T, 3, pointer<T>>;
template<typename T>
using Tensor2D_ref = boost::multi::array_ref<T, 2, pointer<T>>;
template<typename T>
using Tensor1D_ref = boost::multi::array_ref<T, 1, pointer<T>>;

template<typename T>
void fillRandomMatrix(std::vector<T>& vec)
{
  std::mt19937 generator(0);
  std::normal_distribution<T> distribution(0.0, 1.0);
  // avoid uninitialized warning
  T tmp = distribution(generator);
  for (int i = 0; i < vec.size(); i++)
  {
    T val  = distribution(generator);
    vec[i] = val;
  }
}

template<typename T>
void fillRandomMatrix(std::vector<std::complex<T>>& vec)
{
  std::mt19937 generator(0);
  std::normal_distribution<T> distribution(0.0, 1.0);
  T tmp = distribution(generator);
  for (int i = 0; i < vec.size(); i++)
  {
    T re   = distribution(generator);
    T im   = distribution(generator);
    vec[i] = std::complex<T>(re, im);
  }
}

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
template<class Allocator, class Buff>
void timeBatchedQR(std::ostream& out, Allocator& alloc, Buff& buffer, int nbatch, int m, int n)
{
  using T    = typename Allocator::value_type;
  int offset = 0;
  Tensor3D_ref<T> A(buffer.origin(), {nbatch, m, n});
  offset += A.num_elements();
  Tensor3D_ref<T> AT(buffer.origin() + offset, {nbatch, n, m});
  offset += AT.num_elements();
  Tensor2D_ref<T> T_(buffer.origin() + offset, {nbatch, m});
  offset += T_.num_elements();
  Tensor2D_ref<T> scl(buffer.origin() + offset, {nbatch, m});
  offset += T_.num_elements();
  int sz = ma::gqr_optimal_workspace_size(AT[0]);
  //std::cout << buffer.num_elements() << " " << 2*nbatch*m*n + 2*nbatch*m + nbatch*sz << " " << offset <<" " <<sz <<" " <<nbatch*sz << std::endl;
  Tensor2D<T> WORK({nbatch, sz}, Alloc<T>{});
  Alloc<int> ialloc{};
  std::vector<pointer<T>> Aarray;
  using std::copy_n;
  for (int i = 0; i < nbatch; i++)
  {
    Aarray.emplace_back(A[i].origin());
  }

  // Actual profile.
  Watch timer;
  timer.start();
  for (int i = 0; i < nbatch; i++)
    ma::transpose(A[i], AT[i]);
  timer.stop();
  double ttrans = timer.elapsed();
  timer.reset(); timer.start();
  ma::geqrf(AT, T_);
  timer.stop();
  double tgeqrf = timer.elapsed();
  timer.reset(); timer.start();
  for (int i = 0; i < nbatch; i++)
    ma::determinant_from_geqrf(n, AT[i].origin(), m, scl[i].origin(), T(0.0));
  timer.stop();
  double tdet = timer.elapsed();
  timer.reset(); timer.start();
  ma::gqr(AT, T_, WORK);
  timer.stop();
  double tgqr = timer.elapsed();
  out << "  " << std::setw(5) << nbatch << "   " << std::setw(5) << m << " " << std::setw(5) << n << " "
      << std::scientific << ttrans << " " << tgeqrf << " " << tdet << " " << tgqr << "\n";
}

template<class Allocator, class Buff>
void compareBatchedTF(std::ostream& out, Allocator& alloc, Buff& buffer, int nbatch, int m)
{ 
  using T    = typename Allocator::value_type;
  int offset = 0;
  Tensor3D_ref<T> a(buffer.origin(), {nbatch, m, m});
  Tensor3D_ref<T> b(buffer.origin()+a.num_elements(), {nbatch, m, m});
  Alloc<int> ialloc{};
  Tensor1D<int> IWORK(boost::multi::iextensions<1u>{nbatch * (m + 1)}, ialloc);
  std::vector<pointer<T>> A_array;
  A_array.reserve(nbatch);
  for (int i = 0; i < nbatch; i++)
  {
    A_array.emplace_back(a[i].origin());
  }
  Watch timer;
  ma::getrfBatched(m, A_array.data(), m, ma::pointer_dispatch(IWORK.origin()),
               ma::pointer_dispatch(IWORK.origin()) + nbatch * m, nbatch);
  double tgetrf = timer.elapsed();

  // from implementation of getrfBatched, measure overhead! 
  ma::copy(b.flatted().flatted(), a.flatted().flatted());
  timer.reset();
  // calling backend routine directly for now
  ma::getrfBatched_v2(m, A_array.data(), m, ma::pointer_dispatch(IWORK.origin()),
               	ma::pointer_dispatch(IWORK.origin()) + nbatch * m, nbatch); 
  double tgetrf_v2 = timer.elapsed();

  out << "  " <<std::setw(5) << nbatch << "   " << std::setw(5) << m << " "
      << std::scientific <<tgetrf <<"   " <<tgetrf_v2 <<"";
}
#endif

template<class Allocator, class Buff>
void timeQR(std::ostream& out, Allocator& alloc, Buff& buffer, int m)
{
  using T    = typename Allocator::value_type;
  int offset = 0;
  Tensor2D_ref<T> A(buffer.origin(), {m/2, m});
  offset += A.num_elements();
  Tensor1D_ref<T> TAU(buffer.origin() + offset, {m});
  offset += TAU.num_elements();
  int sz = ma::gqr_optimal_workspace_size(A);
  Tensor1D_ref<T> WORK(buffer.origin() + offset, boost::multi::iextensions<1u>{sz});
  Watch timer;
  ma::geqrf(A, TAU, WORK);
  double tgeqrf = timer.elapsed();
  timer.reset();
  ma::gqr(A, TAU, WORK);
  double tgqr = timer.elapsed();
  out << "  " << std::setw(5) << m << " " << std::setw(5) << m/2 << " " << std::scientific << tgeqrf << " "
      << " " << tgqr << "";
}

template<class Allocator, class Buff>
void timeExchangeKernel(std::ostream& out, Allocator& alloc, Buff& buffer, int nbatch, int nwalk, int nocc, int nchol)
{
  using T    = typename Allocator::value_type;
  int offset = 0;
  Tensor5D_ref<T> Twabn(buffer.origin(), {2 * nbatch, nwalk, nocc, nocc, nchol});
  offset += Twabn.num_elements();
  Tensor1D_ref<T> scal(buffer.origin() + offset, boost::multi::iextensions<1u>{nbatch});
  offset += scal.num_elements();
  Tensor1D_ref<T> result(buffer.origin() + offset, boost::multi::iextensions<1u>{nwalk});
  Watch timer;
  ma::Apwabn_Apwban_Bw(scal, Twabn, result); 
  double time = timer.elapsed();
  out << "    " << std::setw(5) << nbatch << " " << std::setw(5) << nwalk << " " << std::setw(5) << nocc << " "
      << std::setw(5) << nchol << "    " << std::scientific << time << "";
}

template<class Allocator, class Buff>
void timeBatchedGemm(std::ostream& out, Allocator& alloc, Buff& buffer, int nbatch, int m)
{
  using T    = typename Allocator::value_type;
  int offset = 0;
  Tensor2D_ref<T> a(buffer.origin(), {m, m});
  offset += a.num_elements();
  Tensor2D_ref<T> b(buffer.origin() + offset, {m, m});
  offset += b.num_elements();
  Tensor3D_ref<T> c(buffer.origin() + offset, {nbatch, m, m});
  //float scale = float(100.0);
  std::vector<pointer<T>> A_array;
  std::vector<pointer<T>> B_array;
  std::vector<pointer<T>> C_array;
  float alpha = 1.0;
  float beta  = 0.0;
  for (int i = 0; i < nbatch; i++)
  {
    A_array.emplace_back(a.origin());
    B_array.emplace_back(b.origin());
    C_array.emplace_back(c[i].origin());
  }
  Watch timer;
  ma::gemmBatched('N', 'N', m, m, m, alpha, A_array.data(), m, B_array.data(), m, beta, C_array.data(), m, nbatch,
			ma::select_backend<Tensor2D_ref<T>>());
  double tgemm = timer.elapsed();
  out << "  " << std::setw(6) << nbatch << " " << std::setw(5) << m << " " << std::scientific << tgemm << "";
}

template<class Allocator, class Buff>
void timeGemm(std::ostream& out, Allocator& alloc, Buff& buffer, int m, int n)
{
  using T    = typename Allocator::value_type;
  int offset = 0;
  Tensor2D_ref<T> a(buffer.origin(), {m, m});
  offset += a.num_elements();
  Tensor2D_ref<T> b(buffer.origin() + offset, {m, n});
  offset += b.num_elements();
  Tensor2D_ref<T> c(buffer.origin() + offset, {m, n});
  Watch timer;
  ma::product(a, b, c);
  double tproduct = timer.elapsed();
  out << "  " << std::setw(6) << m << " " << std::setw(5) << n << " " << std::scientific << tproduct << "";
}

template<class Allocator, class Buff>
void timeBatchedMatrixInverse(std::ostream& out, Allocator& alloc, Buff& buffer, int nbatch, int m)
{
  using T    = typename Allocator::value_type;
  int offset = 0;
  Tensor3D_ref<T> a(buffer.origin(), {nbatch, m, m});
  Tensor3D_ref<T> b(buffer.origin() + a.num_elements(), {nbatch, m, m});
  Alloc<int> ialloc{};
  Tensor1D<int> IWORK(boost::multi::iextensions<1u>{nbatch * (m + 1)}, ialloc);
  std::vector<pointer<T>> A_array, B_array;
  A_array.reserve(nbatch);
  B_array.reserve(nbatch);
  for (int i = 0; i < nbatch; i++)
  {
    A_array.emplace_back(a[i].origin());
    B_array.emplace_back(b[i].origin());
  }
  Watch timer;
  ma::getrfBatched(m, A_array.data(), m, ma::pointer_dispatch(IWORK.origin()),
               ma::pointer_dispatch(IWORK.origin()) + nbatch * m, nbatch);
  double tgetrf = timer.elapsed();
  timer.reset();
  ma::getriBatched(m, A_array.data(), m, ma::pointer_dispatch(IWORK.origin()), B_array.data(), m,
               ma::pointer_dispatch(IWORK.origin()) + nbatch * m, nbatch);
  double tgetri = timer.elapsed();
  out << "  " << std::setw(6) << nbatch << " " << std::setw(5) << m << " " << std::scientific << tgetrf << " " << tgetri
      << "";
}

template<class Allocator, class Buff>
void timeMatrixInverse(std::ostream& out, Allocator& alloc, Buff& buffer, int m)
{
  using T    = typename Allocator::value_type;
  int offset = 0;
  Tensor2D_ref<T> a(buffer.origin(), {m, m});
  Tensor1D_ref<T> WORK(buffer.origin() + a.num_elements(), boost::multi::iextensions<1u>{m * m});
  Alloc<int> ialloc{};
  Tensor1D<int> IWORK(boost::multi::iextensions<1u>{m + 1}, ialloc);
  Watch timer;
  ma::getrf(a, IWORK, WORK);
  double tgetrf = timer.elapsed();
  timer.reset();
  ma::getri(a, IWORK, WORK);
  double tgetri = timer.elapsed();
  out << "  " << std::setw(6) << m << " " << std::scientific << tgetrf << " " << tgetri << "";
}

int main(int argc, char* argv[])
{
  boost::mpi3::environment env(argc, argv);
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  setup_loggers(world.root(),2,0);
  afqmc::setup_memory_managers(node, 1024uL * 1024uL * 1024uL);
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  arch::INIT(node);
#endif
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  {
    std::ofstream out;
    out.open("time_batched_zqr.dat");
    std::cout << " - Batched zQR (nbatch, MxN)" << std::endl;
    out << " nbatch       M     N       ttrans       tgeqrf      tgetdet         tgqr";
    std::vector<int> batches  = {1, 5, 10, 20};
    std::vector<int> num_rows = {200, 400, 800};
    int max_batch             = batches[batches.size() - 1];
    int max_rows              = num_rows[num_rows.size() - 1];
    int size                  = (2 * max_batch * max_rows * (max_rows / 2.0) + 3 * max_batch * max_rows);
    Alloc<std::complex<double>> alloc{};
    Tensor1D<std::complex<double>> buffer(iextensions<1u>{size}, 1.0, alloc);
    for (auto nb : batches)
    {
      for (auto m : num_rows)
      {
        timeBatchedQR(out, alloc, buffer, nb, m, m / 2);
      }
    }
  }
#endif
  {
    std::ofstream out;
    out.open("time_zqr.dat");
    std::cout << " - zQR (MxM)" << std::endl;
    out << "      M     N      tzgeqrf        tzungqr";
    int size = 3 * 1000 * 1000;
    Alloc<std::complex<double>> alloc{};
    Tensor1D<std::complex<double>> buffer(iextensions<1u>{size}, 1.0, alloc);
    std::vector<int> dims = {100, 200, 500, 800, 1000};
    for (auto d : dims)
    {
      timeQR(out, alloc, buffer, d);
    }
  }
  {
    std::ofstream out;
    out.open("time_sgemm.dat");
    std::cout << " - sgemm (MxM)" << std::endl;
    out << "       M     M       tsgemm";
    int size = 3 * 8000 * 8000;
    Alloc<float> alloc{};
    Tensor1D<float> buffer(iextensions<1u>{size}, 1.0, alloc);
    std::vector<int> dims = {200, 500, 800, 1000, 2000, 3000, 4000, 8000};
    for (auto d : dims)
    {
      timeGemm(out, alloc, buffer, d, d);
    }
  }
  {
    std::ofstream out;
    out.open("time_batched_sgemm.dat");
    std::cout << " - batched sgemm (nbatch, MxM)" << std::endl;
    out << "  nbatch    M        tsgemm";
    Alloc<float> alloc{};
    std::vector<int> num_rows = {100, 200, 300, 400, 500, 600};
    std::vector<int> batches  = {128, 256, 512, 1024};
    int max_batch             = batches[batches.size() - 1];
    int max_rows              = num_rows[num_rows.size() - 1];
    int size                  = 3 * max_batch * max_rows * max_rows;
    Tensor1D<float> buffer(iextensions<1u>{size}, 1.0, alloc);
    for (auto nb : batches)
    {
      for (auto m : num_rows)
      {
        timeBatchedGemm(out, alloc, buffer, nb, m);
      }
    }
  }
  {
    std::ofstream out;
    out.open("time_exchange_kernel.dat");
    std::cout << " - exchange kernel (E[w] = sum_{abn} Twabn Twanb)" << std::endl;
    out << "   nbatch nwalk  nocc nchol tExchangeKernel";
    Alloc<std::complex<double>> alloc{};
    int nwalk                = 5;
    int nocc                 = 20;
    int nchol                = 270;
    std::vector<int> batches = {100, 200, 400, 800};
    int nbatch_max           = batches[batches.size() - 1];
    int size                 = 2 * nbatch_max * nwalk * nocc * nocc * nchol + nbatch_max + nwalk;
    Tensor1D<std::complex<double>> buffer(iextensions<1u>{size}, 1.0, alloc);
    for (auto b : batches)
    {
      timeExchangeKernel(out, alloc, buffer, b, nwalk, nocc, nchol);
    }
  }
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  {
    std::ofstream out;
    out.open("time_batched_matrix_inverse.dat");
    std::cout << " - batched matrix inverse (nbatch, MxM)" << std::endl;
    out << "  nbatch     M       tgetrf       tgetri";
    Alloc<std::complex<double>> alloc{};
    std::vector<int> batches  = {1, 5, 10, 20};
    std::vector<int> num_rows = {100, 110, 120, 200, 210, 300, 400, 500, 600, 700};
    int max_batch             = batches[batches.size() - 1];
    int max_rows              = num_rows[num_rows.size() - 1];
    int size                  = 2 * max_batch * max_rows * max_rows;
    Tensor1D<std::complex<double>> buffer(iextensions<1u>{size}, alloc);
    {
      std::vector<std::complex<double>> tmp(size);
      fillRandomMatrix(tmp);
      using std::copy_n;
      copy_n(tmp.data(), tmp.size(), buffer.origin());
    }
    for (auto b : batches)
    {
      for (auto m : num_rows)
      {
        timeBatchedMatrixInverse(out, alloc, buffer, b, m);
      }
    }
  }
  {
    std::ofstream out;
    out.open("compare_batched_tf.dat");
    std::cout << " - batched TF (nbatch, MxM)" << std::endl;
    out << "  nbatch     M       getrfBatched       getrf_withStreams";
    Alloc<std::complex<double>> alloc{};
    std::vector<int> batches  = {1, 10, 20, 50, 100}; 
    std::vector<int> num_rows = {64, 128, 256, 512, 768, 1024};
    int max_batch             = batches[batches.size() - 1];
    int max_rows              = num_rows[num_rows.size() - 1];
    int size                  = 2 * max_batch * max_rows * max_rows;
    Tensor1D<std::complex<double>> buffer(iextensions<1u>{size}, alloc);
    { 
      std::vector<std::complex<double>> tmp(size);
      fillRandomMatrix(tmp);
      using std::copy_n; 
      copy_n(tmp.data(), tmp.size(), buffer.origin());
    }
    for (auto b : batches)
    { 
      for (auto m : num_rows)
      { 
        compareBatchedTF(out, alloc, buffer, b, m);
      }
    }
  }
#endif
  {
    std::ofstream out;
    out.open("time_matrix_inverse.dat");
    std::cout << " - matrix inverse (nbatch, MxM)" << std::endl;
    out << "       M       tgetrf       tgetri";
    Alloc<std::complex<double>> alloc{};
    std::vector<int> num_rows = {100, 110, 120, 200, 210, 300, 400, 500, 600, 700, 800, 1000, 2000, 4000};
    int max_rows              = num_rows[num_rows.size() - 1];
    int size                  = 2 * max_rows * max_rows;
    Tensor1D<std::complex<double>> buffer(iextensions<1u>{size}, alloc);
    {
      std::vector<std::complex<double>> tmp(size);
      fillRandomMatrix(tmp);
      using std::copy_n;
      copy_n(tmp.data(), tmp.size(), buffer.origin());
    }
    for (auto m : num_rows)
    {
      timeMatrixInverse(out, alloc, buffer, m);
    }
  }
  afqmc::release_memory_managers();
}
