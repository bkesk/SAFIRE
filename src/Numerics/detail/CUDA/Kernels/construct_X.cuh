////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef CONSTRUCT_X_H
#define CONSTRUCT_X_H

#include <cassert>
#include <complex>

namespace kernels
{

void construct_X(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 std::complex<double> const* vMF,
                 std::complex<double> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 std::complex<double>* X);
void construct_X(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 std::complex<double> const* vMF,
                 std::complex<float> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 std::complex<float>* X);
void construct_X(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 std::complex<double> const* vMF,
                 std::complex<float> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 std::complex<double>* X);
void construct_X(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 std::complex<double> const* vMF,
                 std::complex<double> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 std::complex<float>* X);

void construct_X_model(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 int const* FieldTypes,
                 std::complex<double> const* vMF,
                 std::complex<double> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 double const* RNs,
                 std::complex<double>* X);
void construct_X_model(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 int const* FieldTypes,
                 std::complex<double> const* vMF,
                 std::complex<float> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 double const* RNs,
                 std::complex<float>* X);
void construct_X_model(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 int const* FieldTypes,
                 std::complex<double> const* vMF,
                 std::complex<float> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 double const* RNs,
                 std::complex<double>* X);
void construct_X_model(int nCV,
                 int nsteps,
                 int nwalk,
                 bool free_projection,
                 double sqrtdt,
                 double vbound,
                 int const* FieldTypes,
                 std::complex<double> const* vMF,
                 std::complex<double> const* vbias,
                 std::complex<double>* HW,
                 std::complex<double>* MF,
                 double const* RNs,
                 std::complex<float>* X);


} // namespace kernels

#endif
