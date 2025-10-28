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


#ifndef COQUI_IAFT_HPP
#define COQUI_IAFT_HPP

#include <math.h>
#include <variant>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/variant_helpers.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/nda_functions.hpp"

#include "iaft_enum_e.hpp"
#include "ir/ir_driver.hpp"

namespace imag_axes_ft {

  // TODO:
  //   - Interface with dlrlib

  /**
   * A generic class for Fourier transform between imaginary axes for different types of grids on imaginary axes.
   * Grid information is provided from the grid driver (grid_var).
   * Still in the design stage.
   */
  class IAFT {

    template<int N>
    using shape_t = std::array<long,N>;

  public:
    IAFT(): grid_var{} { APP_ABORT(" imag_axes_ft::IAFT(): Empty state is not allowed. \n"); }
    IAFT(double beta, double wmax, source_e source, std::string prec = "high", bool print_meta_log = false) {
      if (source == dlr_source) {
        APP_ABORT(" imag_axes_ft::IAFT(): DLR interface is not ready yet. \n");
      } else if (source == ir_source) {
        grid_var = ir::IR(beta, wmax, prec, print_meta_log);
      } else {
        APP_ABORT(" imag_axes_ft::IAFT(): Invalid value of imag_axes_ft::source_e. \n");
      }
    }

    // ir interface
    explicit IAFT(const ir::IR& IR_ ): grid_var(IR_) {}
    explicit IAFT(ir::IR&& IR_): grid_var(std::move(IR_)) {}
    IAFT& operator=(const ir::IR& IR_) { grid_var = IR_; return *this; }
    IAFT& operator=(ir::IR&& IR_) { grid_var = std::move(IR_); return *this; }

    ~IAFT() {}

    /**
     * Matsubara frequency iw_n = i*n*pi/beta
     */
    inline ComplexType omega(long n) const {
      return ComplexType(0.0, n * M_PI / beta());
    }

    /**
     * Transformation from imaginary times to Matsubara frequencies
     * @param X_ti  - [INPUT] imaginary-time tensor
     * @param X_wi  - [OUTPUT] Matsubara frequency tensor
     * @param stats - [INPUT] statistics: fermi or boson
     */
    template<nda::MemoryArray ndaArray_A, nda::MemoryArray ndaArray_B>
    void tau_to_w(ndaArray_A &&X_ti, ndaArray_B &&X_wi, stats_e stats) const;

    template<nda::MemoryArray ndaArray_A, nda::MemoryArray ndaArray_B>
    void tau_to_w(ndaArray_A &&X_ti, ndaArray_B &&Xw_i, stats_e stats, size_t iw) const;
 
    /**
     * version for FT with particle-hole symmetry
     */
    template<nda::MemoryArray Array_tau_t, nda::MemoryArray Array_w_t>
    void tau_to_w_PHsym(Array_tau_t &&X_ti_pos, Array_w_t &&X_wi_pos) const;
 
    /**
     * Transformation from Matsubara frequencies to imaginary times
     * @param X_wi  - [INPUT] Matsubara frequency tensor
     * @param X_ti  - [OUTPUT] imaginary-time tensor
     * @param stats - [INPUT] statistics: fermi or boson
     */
    template<nda::MemoryArray ndaArray_A, nda::MemoryArray ndaArray_B>
    void w_to_tau(ndaArray_A &&X_wi, ndaArray_B &&X_ti, stats_e stats) const;
 
    /**
     * version for FT with particle-hole symmetry
     */
    template<nda::MemoryArray Array_w_t, nda::MemoryArray Array_tau_t>
    void w_to_tau_PHsym(Array_w_t &&X_wi_pos, Array_tau_t &&X_ti_pos) const;
 
    template<nda::MemoryArray ndaArray_A_t, nda::MemoryArray ndaArray_B_t>
    void w_to_tau(ndaArray_A_t &&X_wi, stats_e stats_A, ndaArray_B_t &&X_ti, stats_e stats_B) const;
 
    /**
     * Partial transformation from Matsubara frequencies to imaginary times
     * X(t, ...) = sum_{n} Ttw(t, wn) * X(wn, ...) = sum_{n} X(t, ..., n)
     * This function calculates X(t, i, n) for a given iwn
     * @param Xw_i   - [INPUT] Matsubara frequency tensor
     * @param X_ti   - [OUTPUT] imaginary-time tensor
     * @param stats  - [INPUT] statistics
     * @param iwn    - [INPUT] Matsubara frequency index
     */
    template<nda::MemoryArray ndaArray_A, nda::MemoryArray ndaArray_B>
    void w_to_tau_partial(ndaArray_A&& Xw_i, ndaArray_B&& X_ti, stats_e stats, size_t iwn) const;
 
    /**
     * Interpolation from tau_A grid to tau_B grid with different statistics
     * @param A_ti     - [INPUT] imaginary-time tensor on tau_A grid
     * @param A_stats  - [INPUT] statistics of tau_A grid
     * @param B_ti     - [OUTPUT] imaginary-time tensor on tau_B grid with the opposite statistics of A
     */
    template<nda::MemoryArray ndaArray_A, nda::MemoryArray ndaArray_B>
    void tau_to_tau(ndaArray_A &&A_ti, stats_e A_stats, ndaArray_B &&B_ti) const;
 
    template<nda::MemoryArray ndaArray_A, nda::MemoryArray ndaArray_B>
    void tau_to_tau(ndaArray_A &&A_ti, stats_e A_stats, ndaArray_B &&Bt_i, size_t it_B) const;
 
    /**
     * Interpolation from fermionic sparse sampling nodes to tau = beta^{-}.
     * This function is useful for Matsubara frequency summation: 1/beta \sum_{n} A(iwn) = -1.0 * A(tau=beta^{-})
     * @param A_ti     - [INPUT] imaginary-time tensor on fermionic tau grid
     * @param A_beta_i - [OUTPUT] imaginary-time tensor at tau = beta^{-}
     */
    template<nda::MemoryArray ndaArray_A, nda::MemoryArray ndaArray_B>
    void tau_to_beta(ndaArray_A &&A_ti, ndaArray_B &&A_beta_i) const;
 
    /**
     * Interpolation from fermionic sparse sampling nodes to tau = 0^{+}.
     * This function is needed to evaluate overlap matrix inverse as S = -(G(0^{+}) + G(\beta^{-}))
     * @param A_ti     - [INPUT] imaginary-time tensor on fermionic tau grid
     * @param A_zero_i - [OUTPUT] imaginary-time tensor at tau = zero^{+}
     */
    template<nda::MemoryArray ndaArray_A, nda::MemoryArray ndaArray_B>
    void tau_to_zero(ndaArray_A &&A_ti, ndaArray_B &&A_zero_i)  const;
 
    template<nda::MemoryArray Array_A, typename comm_t>
    void check_leakage(Array_A &&A_ti, stats_e stats, comm_t *comm, std::string A_name) const;
 
    template<nda::MemoryArray Array_A, typename comm_t>
    void check_leakage_PHsym(Array_A &&A_ti, stats_e stats, comm_t *comm, std::string A_name) const;
 
    template<typename A_t>
    void check_leakage(A_t &&A_ti, stats_e stats, std::string A_name, bool PHsym=false) const;
 
  private:
    std::variant<ir::IR> grid_var;

  public:
    source_e source() const {
      if (grid_var.index() == 0) {
        return ir_source;
      } else {
        utils::check(false, "IAFT::source(): This should not trigger");
        return ir_source;
      }
    }
    void metadata_log() const {
      std::visit( [&](auto&& v) { v.metadata_log(); }, grid_var);
    }
    std::string prec() const {
      return std::visit( [&](auto&& v) { return v.prec; }, grid_var);
    }
    int nt_f() const {
      return std::visit( [&](auto&& v) { return v.nt_f; }, grid_var);
    }
    int nt_b() const {
      return std::visit( [&](auto&& v) { return v.nt_f; }, grid_var);
    }
    int nw_f() const {
      return std::visit( [&](auto&& v) { return v.nw_f; }, grid_var);
    }
    int nw_b() const {
      return std::visit( [&](auto&& v) { return v.nw_b; }, grid_var);
    }
    double beta() const {
      return std::visit( [&](auto&& v) { return v.beta; }, grid_var);
    }
    double wmax() const {
      return std::visit( [&](auto&& v) { return v.wmax; }, grid_var);
    }
    double lambda() const {
      return std::visit( [&](auto&& v) { return v.lambda; }, grid_var);
    }
    decltype(auto) wn_mesh() const {
      return std::visit( [&](auto&& v) { return v.wn_mesh_f(); }, grid_var);
    }
    decltype(auto) wn_mesh_b() const {
      return std::visit( [&](auto&& v) { return v.wn_mesh_b(); }, grid_var);
    }
    decltype(auto) tau_mesh() const {
      return std::visit( [&](auto&& v) { return v.tau_mesh_f(); }, grid_var);
    }
    decltype(auto) tau_mesh_b() const {
      return std::visit( [&](auto&& v) { return v.tau_mesh_b(); }, grid_var);
    }
    decltype(auto) Ttw_ff() const {
      return std::visit( [&](auto&& v) { return v.Ttw_ff(); }, grid_var);
    }
    decltype(auto) Twt_ff() const {
      return std::visit( [&](auto&& v) { return v.Twt_ff(); }, grid_var);
    }
    decltype(auto) Ttw_bb() const {
      return std::visit( [&](auto&& v) { return v.Ttw_bb(); }, grid_var);
    }
    decltype(auto) Twt_bb() const {
      return std::visit( [&](auto&& v) { return v.Twt_bb(); }, grid_var);
    }
    decltype(auto) Ttt_bf() const {
      return std::visit( [&](auto&& v) { return v.Ttt_bf(); }, grid_var);
    }
    decltype(auto) Ttt_fb() const {
      return std::visit( [&](auto&& v) { return v.Ttt_fb(); }, grid_var);
    }
    decltype(auto) T_beta_t_ff() const {
      return std::visit( [&](auto&& v) { return v.T_beta_t_ff(); }, grid_var);
    }
    decltype(auto) T_zero_t_ff() const {
      return std::visit( [&](auto&& v) { return v.T_zero_t_ff(); }, grid_var);
    }
    decltype(auto) Tct_ff() const {
      return std::visit( [&](auto&& v) { return v.Tct_ff(); }, grid_var);
    }
    decltype(auto) Tct_bb() const {
      return std::visit( [&](auto&& v) { return v.Tct_bb(); }, grid_var);
    }
  };

  // include definition of template member functions
  #include"numerics/imag_axes_ft/IAFT.icc" 
} // imag_axes_ft


#endif //COQUI_IAFT_HPP
