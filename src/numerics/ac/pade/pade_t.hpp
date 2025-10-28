/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#ifndef COQUI_PADE_HPP
#define COQUI_PADE_HPP

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "nda/nda.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>

namespace analyt_cont {
  template<int prec=100>
  class pade_t {
  public:

    // Customized precision
    using mp_float = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<prec>, boost::multiprecision::et_off>;
    using mp_complex = std::complex<mp_float>;

  public:
    pade_t() = default;
    pade_t(pade_t const&) = default;
    pade_t(pade_t &&) = default;
    pade_t& operator=(pade_t const& other) = default;
    pade_t& operator=(pade_t && other) = default;

    ~pade_t() = default;

    /**
     * interpolate A(iw) to the full complex plane using pade interpolation
     * @param iw_mesh - [INPUT] imaginary frequency grid
     * @param A_iw    - [INPUT] function on the Matsuabara axis
     */
    template<nda::ArrayOfRank<1> iw_mesh_t, nda::ArrayOfRank<2> Array_iw_t>
    void init(iw_mesh_t &&iw_mesh, Array_iw_t &&A_iw) {
      _Nfit = A_iw.shape(0);
      _dim1 = A_iw.shape(1);

      _iw_fit = nda::array<mp_complex, 1>(_Nfit);
      _coeffs = nda::array<mp_complex, 2>(_dim1, _Nfit);
      nda::array<mp_complex, 1> Ad_iw(_Nfit);

      for (long w=0; w<_Nfit; ++w)
        _iw_fit(w) = mp_complex(mp_float(iw_mesh(w).real()), mp_float(iw_mesh(w).imag()));

      for (long d=0; d<_dim1; ++d) {
        for (long w=0; w<_Nfit; ++w)
          Ad_iw(w) = mp_complex(mp_float(A_iw(w, d).real()),
                                mp_float(A_iw(w, d).imag()));
        fit(Ad_iw, _iw_fit, _coeffs(d,nda::range::all));
      }
    }

    template<nda::ArrayOfRank<1> w_mesh_t>
    auto evaluate(w_mesh_t &&w_mesh) -> nda::array<ComplexType, 2> {
      long Nw = w_mesh.shape(0);
      nda::array<ComplexType, 2> A_wI(Nw, _dim1);
      for (long d1 = 0; d1 < _dim1; ++d1) {
        for (long w = 0; w < Nw; ++w) {
          // a(Nfit) * (w - z_fit(Nfit-1) )
          mp_complex w_hpc( mp_float(w_mesh(w).real()), mp_float(w_mesh(w).imag()) );
          // Xw = _coeffs(d1,_Nfit-1) * ( w_hpc - _iw_fit(_Nfit-2) );
          mp_complex Xw = _mul_sub(_coeffs(d1,_Nfit-1), w_hpc, _iw_fit(_Nfit-2));
          for (long f = 1; f < _Nfit-1; ++f) {
            long idx = _Nfit - f - 1;
            // Xw = _coeffs(d1,idx) * ( w_hpc - _iw_fit(idx-1) ) / ( (mp_complex)1.0 + Xw);
            Xw = _mul_sub_div_add(_coeffs(d1,idx), w_hpc, _iw_fit(idx-1), Xw);
          }
          // Xw = _coeffs(d1,0) / ( (mp_complex)1.0 + Xw);
          Xw = _div_add(_coeffs(d1,0), Xw);
          A_wI(w, d1) = ComplexType(Xw.real().template convert_to<RealType>(),
                                    Xw.imag().template convert_to<RealType>());
        }
      }
      return A_wI;
    }

    template<nda::ArrayOfRank<1> w_mesh_t>
    auto evaluate(w_mesh_t &&w_mesh, long d1) -> nda::array<ComplexType, 1> {
      long Nw = w_mesh.shape(0);
      nda::array<ComplexType, 1> A_w(Nw);
      for (long w = 0; w < Nw; ++w) {
        mp_complex w_hpc( mp_float(w_mesh(w).real()), mp_float(w_mesh(w).imag()) );

        // Xw = _coeffs(d1,_Nfit-1) * ( w_hpc - _iw_fit(_Nfit-2) );
        mp_complex Xw = _mul_sub(_coeffs(d1,_Nfit-1), w_hpc, _iw_fit(_Nfit-2));

        for (long f = 1; f < _Nfit-1; ++f) {
          long idx = _Nfit - f - 1;

          // Xw = _coeffs(d1,idx) * ( w_hpc - _iw_fit(idx-1) ) / ( (mp_complex)1.0 + Xw);
          Xw = _mul_sub_div_add(_coeffs(d1,idx), w_hpc, _iw_fit(idx-1), Xw);
        }

        // Xw = _coeffs(d1,0) / ( (mp_complex)1.0 + Xw);
        Xw = _div_add(_coeffs(d1,0), Xw);

        A_w(w) = ComplexType(Xw.real().template convert_to<RealType>(),
                             Xw.imag().template convert_to<RealType>());
      }
      return A_w;
    }

    ComplexType evaluate(ComplexType w, long d1) {
      mp_complex w_hpc( mp_float(w.real()), mp_float(w.imag()) );

      // Xw = _coeffs(d1,_Nfit-1) * ( w_hpc - _iw_fit(_Nfit-2) );
      mp_complex Xw = _mul_sub(_coeffs(d1,_Nfit-1), w_hpc, _iw_fit(_Nfit-2));

      for (long f = 1; f < _Nfit-1; ++f) {
        long idx = _Nfit-f-1;

        // Xw = _coeffs(d1,idx) * ( w_hpc - _iw_fit(idx-1) ) / ( (mp_complex)1.0 + Xw);
        Xw = _mul_sub_div_add(_coeffs(d1,idx), w_hpc, _iw_fit(idx-1), Xw);

      }

      // Xw = _coeffs(d1,0) / ( (mp_complex)1.0 + Xw);
      Xw = _div_add(_coeffs(d1,0), Xw);

      return ComplexType(Xw.real().template convert_to<RealType>(),
                         Xw.imag().template convert_to<RealType>());
    }

    void reset() {
      _iw_fit = nda::array<mp_complex, 1>();
      _coeffs = nda::array<mp_complex, 2>();
    }

  private:
    /**
     * N-point pade interpolation from Thiele's reciprocal difference method.
     * Ref: J. of Low Temp. Phys. 29, 179–192 (1977)
     * @param Ai
     * @param iw_fit
     * @return
     */
    template<nda::ArrayOfRank<1> Array_t, nda::ArrayOfRank<1> w_mesh_t>
    auto fit(Array_t &&Az, w_mesh_t &&z_fit) -> nda::array<ComplexType, 1> {
      long nfit = z_fit.shape(0);
      // g(i, j) = g_i(z_j)
      nda::array<ComplexType, 2> g(nfit, nfit);

      g(0, nda::range::all) = Az();
      for (long i = 1; i < nfit; ++i) {
        for (long j = i; j < nfit; ++j) {
          // g(i, j) = ( g(i-1,i-1) - g(i-1,j) ) / ( (z_fit(j) - z_fit(i-1)) * g(i-1, j) );
          g(i, j) = _sub_div_mul(g(i-1,i-1), g(i-1,j), z_fit(j), z_fit(i-1), g(i-1,j));
        }
      }
      // coefficients a(i) = g(i, i) = g_i(z_i)
      auto a = nda::make_regular(nda::diagonal(g));
      return a;
    }

    template<nda::ArrayOfRank<1> Az_t, nda::ArrayOfRank<1> w_mesh_t, nda::ArrayOfRank<1> coeff_t>
    void fit(Az_t &&Az, w_mesh_t &&z_fit, coeff_t &&coeffs) {
      // g(i, j) = g_i(z_j)
      nda::array<mp_complex, 2> g(_Nfit, _Nfit);
      g(0, nda::range::all) = Az();

      for (long i=1; i<_Nfit; ++i) {
        for (long j=i; j<_Nfit; ++j) {
          // g(i, j) = ( g(i-1,i-1) - g(i-1,j) ) / ( (z_fit(j) - z_fit(i-1)) * g(i-1, j) );
          g(i, j) = _sub_div_mul(g(i-1,i-1), g(i-1,j), z_fit(j), z_fit(i-1), g(i-1,j));
        }
      }
      coeffs() = nda::diagonal(g);
    }

    // helper functions to avoid std::complex<mp_float> arithmetics directly
    // Sadly, boost multiprecision number type is not fully compatible with std::complex<...> arithmetics
    // A * (B - C)
    inline mp_complex _mul_sub(const mp_complex& A, const mp_complex& B, const mp_complex& C) {
      mp_float br = B.real() - C.real();
      mp_float bi = B.imag() - C.imag();
      mp_float ar = A.real();
      mp_float ai = A.imag();

      return mp_complex{
          ar * br - ai * bi,
          ar * bi + ai * br
      };
    }
    // A * (B - C) / (1 + D)
    inline mp_complex _mul_sub_div_add(const mp_complex& A, const mp_complex& B, const mp_complex& C, const mp_complex& D) {
      // Numerator: A * (B - C)
      mp_float br = B.real() - C.real();
      mp_float bi = B.imag() - C.imag();
      mp_float ar = A.real();
      mp_float ai = A.imag();

      mp_float num_r = ar * br - ai * bi;
      mp_float num_i = ar * bi + ai * br;

      // Denominator: 1 + D
      mp_float dr = mp_float(1) + D.real();
      mp_float di = D.imag();
      mp_float denom = dr * dr + di * di;

      return mp_complex{
          (num_r * dr + num_i * di) / denom,
          (num_i * dr - num_r * di) / denom
      };
    }
    // A / (1 + B)
    inline mp_complex _div_add(const mp_complex& A, const mp_complex& B) {
      // Denominator: 1 + B
      mp_float br = mp_float(1) + B.real();
      mp_float bi = B.imag();
      mp_float denom = br * br + bi * bi;

      mp_float ar = A.real();
      mp_float ai = A.imag();

      return mp_complex{
          (ar * br + ai * bi) / denom,
          (ai * br - ar * bi) / denom
      };
    }

    mp_complex _sub_div_mul(const mp_complex& A,
                           const mp_complex& B,
                           const mp_complex& C,
                           const mp_complex& D,
                           const mp_complex& E) {
      // U = A - B
      mp_float ur = A.real() - B.real();
      mp_float ui = A.imag() - B.imag();

      // V = (C - D)
      mp_float vr = C.real() - D.real();
      mp_float vi = C.imag() - D.imag();

      // V * E
      mp_float er = E.real();
      mp_float ei = E.imag();

      // V * E = (vr * er - vi * ei) + i(vr * ei + vi * er)
      mp_float den_r = vr * er - vi * ei;
      mp_float den_i = vr * ei + vi * er;

      // |denominator|^2
      mp_float denom = den_r * den_r + den_i * den_i;

      // (U / (V * E))
      return mp_complex{
          (ur * den_r + ui * den_i) / denom,
          (ui * den_r - ur * den_i) / denom
      };
    }

  private:
    long _dim1 = 0;
    long _Nfit = 0;
    nda::array<mp_complex, 1> _iw_fit;
    nda::array<mp_complex, 2> _coeffs;
  };
} // analyt_cont


#endif //COQUI_PADE_HPP
