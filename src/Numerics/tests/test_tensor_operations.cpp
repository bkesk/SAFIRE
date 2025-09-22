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

#include "AFQMC/config.h"
#include "config.0.h"
#include "Numerics/ma_operations.hpp"
#include "Numerics/tensor_operations.hpp"
#include "SparseMatrix/tests/matrix_helpers.h"
#include "Utilities/Timer.hpp"
#include "Memory/device_rng.hpp"
#include "AFQMC/Utilities/test_utils.hpp"

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
template<typename T>
using Tensor5D = array<T, 5, Alloc<T>>;

template<typename T>
using Tensor2D_ref = array_ref<T, 2, pointer<T>>;
template<typename T>
using Tensor1D_ref = array_ref<T, 1, pointer<T>>;

template<typename T>
void create_data(std::vector<T>& buffer, T scale)
{
  T count = T(0);
  for (int i = 0; i < buffer.size(); i++)
  {
    buffer[i] = count;
    count += T(1) / scale;
  }
}

TEST_CASE("KaKjw_to_KKwaj", "[Numerics][tensor_operations]")
{
  Alloc<double> alloc{};
  int nwalk = 5;
  int occ_k = 5;
  int nmo_k = 10;
  int nk    = 3;
  std::vector<double> buffer(nk * occ_k * nk * nmo_k * nwalk);
  create_data(buffer, 1.0);
  Tensor5D<double> KaKjw({nk, occ_k, nk, nmo_k, nwalk}, alloc);
  Tensor5D<double> KKwaj({nk, nk, nwalk, occ_k, nmo_k}, 0.0, alloc);
  Tensor1D<int> nmo_pk  = {nmo_k, nmo_k, nmo_k};
  Tensor1D<int> nel_pk  = {occ_k, occ_k, occ_k};
  Tensor1D<int> nmo_pk0 = {0, nmo_k, 2 * nmo_k};
  Tensor1D<int> nel_pk0 = {0, occ_k, 2 * occ_k};
  copy_n(buffer.data(), buffer.size(), KaKjw.origin());
  // KaKjw = numpy.arange(nk*na*nk*nj*nw).reshape((nk,na,nk,nj,nw))
  // KKwaj = numpy.transpose(KaKjw, (0,2,1,3,4))
  ma::KaKjw_to_KKwaj(nwalk, nk, 1, nmo_k, nk * nmo_k, occ_k, nmo_pk.origin(), nmo_pk0.origin(), nel_pk.origin(),
                 nel_pk0.origin(), KaKjw.origin(), KKwaj.origin());
  // Reference values from python
  REQUIRE(KKwaj[1][2][3][4][4] == Approx(1473.0));
  REQUIRE(KKwaj[2][1][3][4][4] == Approx(2173.0));
  REQUIRE(KKwaj[2][1][4][3][4] == Approx(2024.0));
}

TEST_CASE("KaKjw_to_QKwaj", "[Numerics][tensor_operations]")
{
  Alloc<double> alloc{};
  int nwalk = 5;
  int occ_k = 4;
  int nmo_k = 8;
  int nk    = 4;
  std::vector<double> buffer(nk * occ_k * nk * nmo_k * nwalk);
  create_data(buffer, 1.0);
  Tensor5D<double> KaKjw({nk, occ_k, nk, nmo_k, nwalk}, alloc);
  Tensor5D<double> QKajw({nk, nk, occ_k, nmo_k, nwalk}, 0.0, alloc);
  Tensor1D<int> nmo_pk  = {nmo_k, nmo_k, nmo_k, nmo_k};
  Tensor1D<int> nel_pk  = {occ_k, occ_k, occ_k, occ_k};
  Tensor1D<int> nmo_pk0 = {0, nmo_k, 2 * nmo_k, 3 * nmo_k};
  Tensor1D<int> nel_pk0 = {0, occ_k, 2 * occ_k, 3 * occ_k};
  // From 221 k-point grid
  Tensor2D<int> qk_to_k2 = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
  copy_n(buffer.data(), buffer.size(), KaKjw.origin());
  ma::KaKjw_to_QKajw(nwalk, nk, 1, nmo_k, nk * nmo_k, occ_k, nmo_pk.origin(), nmo_pk0.origin(), nel_pk.origin(),
                 nel_pk0.origin(), qk_to_k2.origin(), KaKjw.origin(), QKajw.origin());
  // Just captured from output.
  REQUIRE(QKajw[1][2][3][4][4] == Approx(1904.0));
  REQUIRE(QKajw[2][1][3][4][4] == Approx(1264.0));
  REQUIRE(QKajw[2][1][1][3][4] == Approx(939.0));
  // Check structure consistency.
  for (int iq = 0; iq < nk; iq++)
  {
    for (int k1 = 0; k1 < nk; k1++)
    {
      int k2 = qk_to_k2[iq][k1];
      for (int ia = 0; ia < nel_pk[k1]; ia++)
      {
        for (int ij = 0; ij < nmo_pk[k2]; ij++)
        {
          for (int iw = 0; iw < nwalk; iw++)
          {
            REQUIRE(double(KaKjw[k1][ia][k2][ij][iw]) == double(QKajw[iq][k1][ia][ij][iw]));
          }
        }
      }
    }
  }
}

TEST_CASE("elementwise", "[Numerics][tensor_operations]")
{
  Alloc<ComplexType> alloc{};
  int nrow = 3;
  int ncol = 3;
  ComplexType one(1.0);
  Tensor2D<ComplexType> A({nrow, ncol}, alloc);
  Tensor1D<ComplexType> x(iextensions<1u>{ncol}, alloc);
  std::vector<ComplexType> buffer(nrow * ncol);
  create_data(buffer, ComplexType(1.0));
  copy_n(buffer.data(), buffer.size(), A.origin());
  copy_n(buffer.data(), x.size(), x.origin());
  Tensor2D<ComplexType> ref = {{0, 1, 2}, {4, 5, 6}, {8, 9, 10}};
  ma::elementwise(ma::TOp_PLUS, 0, one, x, A);
  verify_approx(ref, A);
  copy_n(buffer.data(), buffer.size(), A.origin());
  ma::elementwise(ma::TOp_PLUS, 1, one, x, A);
  // MAM: operator asignment from initializer lists is not working on Multi
  //      on some compilers( nvcc, intel19 )
  //      It does work on construction, so a temporary fix
  ref = Tensor2D<ComplexType>{{0, 2, 4}, {3, 5, 7}, {6, 8, 10}};
  verify_approx(ref, A);
  copy_n(buffer.data(), buffer.size(), A.origin());
  ma::elementwise(ma::TOp_MINUS, 0, one, x, A);
  ref = Tensor2D<ComplexType>{{0, 1, 2}, {2, 3, 4}, {4, 5, 6}};
  verify_approx(ref, A);
  copy_n(buffer.data(), buffer.size(), A.origin());
  ma::elementwise(ma::TOp_MINUS, 1, one, x, A);
  ref = Tensor2D<ComplexType>{{0, 0, 0}, {3, 3, 3}, {6, 6, 6}};
  verify_approx(ref, A);
  copy_n(buffer.data(), buffer.size(), A.origin());
  ma::elementwise(ma::TOp_MUL, 0, one, x, A);
  ref = Tensor2D<ComplexType>{{0, 0, 0}, {3, 4, 5}, {12, 14, 16}};
  verify_approx(ref, A);
  copy_n(buffer.data(), buffer.size(), A.origin());
  ma::elementwise(ma::TOp_MUL, 1, one, x, A);
  ref = Tensor2D<ComplexType>{{0, 1, 4}, {0, 4, 10}, {0, 7, 16}};
  verify_approx(ref, A);
  // Avoid dividing by zero
  copy_n(buffer.data() + 1, x.size(), x.origin());
  copy_n(buffer.data(), buffer.size(), A.origin());
  ma::elementwise(ma::TOp_DIV, 0, one, x, A);
  ref = Tensor2D<ComplexType>{{0, 1, 2}, {1.5, 2.0, 2.5}, {2, 2.333333333333, 2.666666666667}};
  verify_approx(ref, A);
  copy_n(buffer.data() + 1, x.size(), x.origin());
  copy_n(buffer.data(), buffer.size(), A.origin());
  ma::elementwise(ma::TOp_DIV, 1, one, x, A);
  ref = Tensor2D<ComplexType>{{0.0, 0.5, 0.666666666667}, {3.0, 2.0, 1.666666666667}, {6.0, 3.5, 2.666666666667}};
  verify_approx(ref, A);
}

TEST_CASE("transpose_wabn_to_wban", "[Numerics][tensor_operations]")
{
  Alloc<ComplexType> alloc{};
  int nwalk = 5;
  int na    = 7;
  int nb    = 5;
  int nchol = 11;
  std::vector<ComplexType> buffer(nwalk * na * nb * nchol);
  create_data(buffer, ComplexType(1.0));
  Tensor4D<ComplexType> Twabn({nwalk, na, nb, nchol}, alloc);
  Tensor4D<ComplexType> Twban({nwalk, nb, na, nchol}, 0.0, alloc);
  copy_n(buffer.data(), buffer.size(), Twabn.origin());
  // Twabn = numpy.arange(nw*na*nb*nc).reshape((nw,na,nb,nc))
  // Twban = numpy.transpose(Twabn, (0,2,1,3))
  ma::transpose_wabn_to_wban(Twabn, Twban);
  Tensor1D_ref<ComplexType> chunk(Twban.origin() + 10, iextensions<1u>{10});
  // From Twban.copy().ravel()[10:20].
  Tensor1D<ComplexType> ref = {10, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  verify_approx(chunk, ref);
}

TEST_CASE("vKKwij_tovwKiKj", "[Numerics][tensor_operations]")
{
  Alloc<ComplexType> alloc{};
  Alloc<int> ialloc{};
  int nwalk             = 5;
  int nmo_k             = 10;
  int nk                = 3;
  Tensor1D<int> nmo_pk  = {nmo_k, nmo_k, nmo_k};
  Tensor1D<int> nmo_pk0 = {0, nmo_k, 2 * nmo_k};
  // This routine is too complicated.
  Tensor2D<int> KKTransID({nk, nk}, 1, ialloc);
  Tensor3D<ComplexType> vKiKj({nwalk, nmo_k * nk, nmo_k * nk}, 0.0, alloc);
  Tensor3D<ComplexType> vKK({nk, nk, nwalk * nmo_k * nmo_k}, 0.0, alloc);
  std::vector<ComplexType> buffer(nk * nk * nwalk * nmo_k * nmo_k, 0.0);
  create_data(buffer, ComplexType(1.0));
  copy_n(buffer.data(), buffer.size(), vKK.origin());
  ma::vKKwij_to_vwKiKj(nwalk, nk, nmo_k, nk * nmo_k, KKTransID.origin(), nmo_pk.origin(), nmo_pk0.origin(), vKK.origin(),
                   vKiKj.origin()); 
  copy_n(vKiKj.origin(), vKiKj.num_elements(), buffer.data());
  // Just captured from output.
  REQUIRE(real(buffer[17]) == Approx(507.0));
  REQUIRE(real(buffer[0]) == Approx(0.0));
  REQUIRE(real(buffer[33]) == Approx(13.0));
}

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
// test/time the GPU kernel based on the CPU code.
TEST_CASE("AGiwj_BGjwi_CG", "[Numerics][tensor_operations]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  arch::INIT(node);

  using boost::multi::array;
//  using std::copy_n;
  Alloc<ComplexType> alloc{};
  int nbatch            = 10;
  int nwalk             = 16;
  int nmo               = 250;
  int ng                = 10;

//  std::cout<<" AGiwj_BGjwi_CG test. ";
  {
    Watch timer;
    Tensor5D<ComplexType> AGiwj({nbatch, ng, nmo, nwalk, nmo}, 0.0, alloc);
    Tensor5D<ComplexType> BGjwi({nbatch, ng, nmo, nwalk, nmo}, 0.0, alloc);
    Tensor1D<ComplexType> CG(iextensions<1u>{ng}, 0.0, alloc);
    Tensor1D<ComplexType> Xw(iextensions<1u>{nwalk}, 0.0, alloc);

    array<ComplexType, 5> AGh({nbatch, ng, nmo, nwalk, nmo}, ComplexType(0.0));
    array<ComplexType, 5> BGh({nbatch, ng, nmo, nwalk, nmo}, ComplexType(0.0));
    array<ComplexType, 1> CGh(iextensions<1u>{ng}, ComplexType(0.0));
    array<ComplexType, 1> Xwh(iextensions<1u>{nwalk}, ComplexType(0.0));
/*
    {
      std::vector<ComplexType> tmp( AGh[0].num_elements() ); 
      afqmc::fillRandomMatrix(tmp);
      for(int i=0; i<nbatch; i++) 
        copy_n(tmp.data(), tmp.size(), AGh[0].origin()); 
      afqmc::fillRandomMatrix(tmp);
      for(int i=0; i<nbatch; i++) 
        copy_n(tmp.data(), tmp.size(), BGh[0].origin()); 
      tmp.resize(nwalk);
      afqmc::fillRandomMatrix(tmp);
      copy_n(tmp.data(), nwalk, Xwh.origin());

      copy_n(AGh.origin(), AGh.num_elements(), AGiwj.origin());
      copy_n(BGh.origin(), BGh.num_elements(), BGjwi.origin());
      copy_n(Xwh.origin(), Xwh.num_elements(), Xw.origin());
    }
*/
    auto iseed = utils::make_seed(world);
    utils::DeviceRandomGenerator_t rng = utils::make_device_rng(iseed);
    kernels::sampleGaussianRNG(raw_pointer_cast(AGiwj.origin()), 
			       AGiwj.num_elements(), rng);
    kernels::sampleGaussianRNG(raw_pointer_cast(BGjwi.origin()), 
			       BGjwi.num_elements(),rng); 
    kernels::sampleGaussianRNG(raw_pointer_cast(Xw.origin()), 
			       Xw.num_elements(),rng); 

    copy_n(AGiwj.origin(), AGiwj.num_elements(), AGh.origin());
    copy_n(BGjwi.origin(), BGjwi.num_elements(), BGh.origin());
    copy_n(Xw.origin(), Xw.num_elements(), Xwh.origin());

    timer.reset();
    ma::AGiwj_BGjwi_CG(Xwh, AGh, BGh, CGh); 
    double tcpu = timer.elapsed();
    timer.reset();
    ma::AGiwj_BGjwi_CG(Xw, AGiwj, BGjwi, CG); 
    double tgpu = timer.elapsed();
    {
      array<ComplexType, 1> C_(CG);
      for(int i=0; i<ng; i++) {  
        REQUIRE(std::real(CGh[i]) == Approx(std::real(C_[i])));
        REQUIRE(std::imag(CGh[i]) == Approx(std::imag(C_[i])));
      }
    }
    std::cout<<"   " <<nwalk <<" " <<nmo <<" " <<tcpu <<" " <<tgpu <<""; 
  }
}
#endif

} // namespace sfqmc
