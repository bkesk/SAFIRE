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

#pragma once

#include<fstream>
#include<random>
#include<complex>
#include<vector>
#include<string>
#include<tuple>
#include<memory>
#include <filesystem>

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "nda/nda.hpp"
#include "utilities/mpi_context.h"
#include "AFQMC/config.h"
#include "IO/app_loggers.h"

namespace sfqmc {
namespace utils {

// mpi context for unit tests
namespace detail {
extern std::shared_ptr<mpi_context_t<boost::mpi3::communicator>> __unit_test_mpi_context__;
}

/* Path to unit test files included in the code base */
inline constexpr std::string unit_test_base() 
{
  //std::string pre = std::string(PROJECT_SOURCE_DIR_STR) + "/tests/unit_test_files/";
  std::string pre = std::string(PROJECT_SOURCE_DIR_STR) + "/utils/tests/functional/";
  return pre;
}

inline constexpr auto molecule_unit_tests_files(bool rhf, bool uhf, bool ghf, bool nomsd, bool phmsd) 
{
  std::vector< std::tuple<std::string, std::string, afqmc::WALKER_TYPES> > files;
  auto pre = unit_test_base() + "molecules/";
  if(nomsd) {
    if(rhf) {
      files.emplace_back( std::make_tuple(pre + "BH/afqmc_inputs/afqmc_H_rhf_closed.h5", 
                                          pre + "BH/afqmc_inputs/afqmc_rhf_nomsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE) );
    } 
    if(uhf) {
      files.emplace_back( std::make_tuple(pre + "BH/afqmc_inputs/afqmc_H_rhf_collinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_uhf_nomsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE) );
      files.emplace_back( std::make_tuple(pre + "BH/afqmc_inputs/afqmc_H_rhf_collinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_uhf_nomsd_init_rhf.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE) );
      files.emplace_back( std::make_tuple(pre + "Li/afqmc_inputs/hamil_closed.h5",
                                          pre + "Li/afqmc_inputs/rohf_nomsd_fullypolarized.h5",
                                          afqmc::FULLYPOLARIZED) );
    }
    if(ghf) {
      files.emplace_back( std::make_tuple(pre + "BH/afqmc_inputs/afqmc_H_rhf_noncollinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_ghf_nomsd.h5",
                                          afqmc::NONCOLLINEAR) );
      files.emplace_back( std::make_tuple(pre + "Pb/afqmc_inputs/afqmc_H_rhf_basis_noncollinear_sf.h5",
                                          pre + "Pb/afqmc_inputs/afqmc_ghf_sf_nomsd.h5",
                                          afqmc::NONCOLLINEAR) );
      // 🚧 Complex-value H1 in Q. Chem Hamiltonian got dropped! Add it back in after consulting with Miguel
      //files.emplace_back( std::make_tuple(pre + "Pb/afqmc_inputs/afqmc_H_rhf_basis_noncollinear_soc.h5",
      //                                    pre + "Pb/afqmc_inputs/afqmc_ghf_soc_nomsd.h5",
      //                                    afqmc::UNDEFINED_WALKER_TYPE) );
    }
  }
  if (phmsd) {
    if (uhf) {
      // edge case: leading det only
      files.emplace_back( std::make_tuple(pre + "BH/afqmc_inputs/afqmc_H_rhf_collinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_casci_uhf_1phmsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE) );
      files.emplace_back( std::make_tuple(pre + "BH/afqmc_inputs/afqmc_H_uhf_collinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_casci_uhf_phmsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE) );
      // may be redundant with above test: good for diversity of inputs
      files.emplace_back(std::make_tuple(pre + "N2/afqmc_inputs/cas_basis_hamil.h5",
                                        pre + "N2/afqmc_inputs/cas_wfn.h5",
                                        afqmc::UNDEFINED_WALKER_TYPE));
      }
  }
  return files;
}


inline constexpr auto lattice_unit_test_files(bool rhf, bool uhf, bool ghf, bool nomsd, bool phmsd) 
{
  std::vector< std::tuple<std::string, std::string, afqmc::WALKER_TYPES> > files;
  auto pre = unit_test_base() + "lattices/";
  if(nomsd) {
    if(rhf) {
      // Closed spin symmetry is not implemented - no tests expected to pass
    } 
    if(uhf) {
      files.emplace_back( std::make_tuple(pre + "square4x4/afqmc_inputs/afqmc_H_rhf_collinear.h5",
                                          pre + "square4x4/afqmc_inputs/afqmc_uhf_nomsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE) );
    }    
    if(ghf) {
      files.emplace_back( std::make_tuple(pre + "square4x4/afqmc_inputs/afqmc_H_rhf_noncollinear.h5",
                                          pre + "square4x4/afqmc_inputs/afqmc_ghf_nomsd.h5",
                                          afqmc::NONCOLLINEAR) );
    }
  }
  if (phmsd) {
    if (uhf) {
      files.emplace_back( std::make_tuple(pre + "square4x4/afqmc_inputs/afqmc_H_rhf_collinear.h5",
                                          pre + "square4x4/afqmc_inputs/afqmc_casci_uhf_phmsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE) );
    }
  }
  return files;
}


/* Checks if a file exists in the file system */
inline bool file_exists(const std::string& name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

inline std::shared_ptr<mpi_context_t<mpi3::communicator>>& make_unit_test_mpi_context()
{
  if(not detail::__unit_test_mpi_context__) {
    detail::__unit_test_mpi_context__ =
         std::make_shared<mpi_context_t<boost::mpi3::communicator>>(make_mpi_context());
  }
  return detail::__unit_test_mpi_context__;
}

template<typename T>
void VALUE_EQUAL(T const& A, T const& B, double m=1e-8, double eps=1e-8)
{
  REQUIRE_THAT(A,
               Catch::Matchers::WithinRel(B, T(eps)) ||
               Catch::Matchers::WithinAbs(B, T(m)));
}

template<typename T>
void VALUE_EQUAL(std::complex<T> const& A, std::complex<T> const& B, double m=1e-8, double eps=1e-8)
{
  REQUIRE_THAT(real(A),
               Catch::Matchers::WithinRel(real(B), T(eps)) ||
               Catch::Matchers::WithinAbs(real(B), T(m)));
  REQUIRE_THAT(imag(A),
               Catch::Matchers::WithinRel(imag(B), T(eps)) ||
               Catch::Matchers::WithinAbs(imag(B), T(m)));
}

template<typename T>
void VALUE_EQUAL(T const& A, std::complex<T> const& B, double m=1e-8, double eps=1e-8)
{
  REQUIRE_THAT(A,
               Catch::Matchers::WithinRel(real(B), T(eps)) ||
               Catch::Matchers::WithinAbs(real(B), T(m)));
  REQUIRE_THAT(imag(B),
               Catch::Matchers::WithinAbs(T(0.0), T(m)));
}


template<typename T>
void VALUE_EQUAL(std::complex<T> const& A, T const& B, double m=1e-8, double eps=1e-8)
{
  REQUIRE_THAT(real(A),
               Catch::Matchers::WithinRel(B, T(eps)) ||
               Catch::Matchers::WithinAbs(B, T(m)));
  REQUIRE_THAT(imag(A),
               Catch::Matchers::WithinAbs(T(0.0), T(m)));
}


template<nda::Array Arr1, nda::Array Arr2>
void ARRAY_EQUAL(Arr1&& A_, Arr2&& B_, double m=1e-8, double eps=1e-8)
{ 
  static_assert(nda::get_rank<std::decay_t<Arr1>> == 
	        nda::get_rank<std::decay_t<Arr2>>, "Rank mismatch.");
  REQUIRE(A_.size() == B_.size()); 
  auto A = nda::to_host(A_());
  auto B = nda::to_host(B_());
  auto itA = A.begin();
  auto itB = B.begin();
  auto itAend = A.end();
  auto itBend = B.end();
  for( ; itA != itAend; ++itA, ++itB ) { 
    check( itB != itBend , "Size mismatch.");
    VALUE_EQUAL( *itA, *itB, m, eps);
  }
}


template<nda::Array Arr>
void fillRandomArray(Arr&& A, double a = 0.0, double b = 1.0)
{
  using T = typename std::decay_t<Arr>::value_type;
  std::mt19937 generator(0);
  if constexpr (nda::is_complex_v<T>) {
    // for float, extract base type from T is complex
    std::uniform_real_distribution<double> distribution(a,b);
    for( auto& v: A )  { v  = T{distribution(generator),distribution(generator)}; }
  } else {
    std::uniform_real_distribution<T> distribution(T{a},T{b});
    for( auto& v: A )  { v  = distribution(generator); }
  }
}

template<typename T>
auto make_random(long N)
{ 
  if constexpr (nda::is_complex_v<T>) {
    auto a = nda::rand<remove_complex_t<T>>(2*N);
    nda::array<T,1> res(N);
    for(int i=0; i<N; i++) res(i) = T{a(2*i),a(2*i+1)};     
    return res;
  } else { 
    return nda::rand<T>(N);
  }
}

template<typename T>
auto make_random(long N1, long N2)
{
  if constexpr (nda::is_complex_v<T>) {
    auto a = nda::rand<remove_complex_t<T>>(N1,2*N2);
    nda::array<T,2> res(N1,N2);
    for(int i=0; i<N1; i++) 
      for(int j=0; j<N2; j++) 
        res(i,j) = T{a(i,2*j),a(i,2*j+1)};
    return res;
  } else {
    return nda::rand<T>(N1,N2);
  }
}

template<typename T>
auto make_random(long N1, long N2, long N3)
{
  if constexpr (nda::is_complex_v<T>) {
    auto a = nda::rand<remove_complex_t<T>>(N1,N2,2*N3);
    nda::array<T,3> res(N1,N2,N3);
    for(int i=0; i<N1; i++)
      for(int j=0; j<N2; j++)
        for(int k=0; k<N3; k++)
          res(i,j,k) = T{a(i,j,2*k),a(i,j,2*k+1)};
    return res;
  } else {
    return nda::rand<T>(N1,N2,N3);
  }
}

} // utils
} // sfqmc
