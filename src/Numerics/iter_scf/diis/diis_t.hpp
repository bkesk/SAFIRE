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


#ifndef COQUI_DIIS_T_HPP
#define COQUI_DIIS_T_HPP

#include "configuration.hpp"
#include "utilities/check.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/iter_scf/iter_scf_type_e.hpp"

#include "numerics/iter_scf/diis/vspace.h"
#include "numerics/iter_scf/diis/vspace_fock_sigma.hpp"

#include "numerics/iter_scf/diis/state.h"
#include "numerics/iter_scf/diis/com_diis_residual.h"

#include "numerics/iter_scf/diis/diis_alg.hpp"
#include "numerics/iter_scf/damp/damp_t.hpp"

namespace iter_scf {
  /**
   * Simple class connecting an abstract DIIS with specific interface requirements
   * serving as a driver for DIIS algorithm
   */
  struct diis_t {
    using Array_4D = nda::array<ComplexType,4>;
    using Array_5D = nda::array<ComplexType,5>;

    static constexpr iter_alg_e iter_alg = DIIS;
    static constexpr iter_alg_e get_iter_alg() { return iter_alg; }

  public:
    diis_t() = default;
    diis_t(double mixing_, size_t max_subsp_size_, size_t diis_start_): 
        mixing(mixing_), max_subsp_size(max_subsp_size_), diis_start(diis_start_) {};

    diis_t(const diis_t& other) = default;
    diis_t(diis_t&& other) = default;
    diis_t& operator=(const diis_t& other) = default;
    diis_t& operator=(diis_t&& other) = default;

    ~diis_t(){}

    template<nda::MemoryArrayOfRank<4> F_t, nda::MemoryArrayOfRank<5> Sigma_t,
        nda::MemoryArrayOfRank<4> S_t, nda::MemoryArrayOfRank<4> H0_t>
    void initialize(F_t &&F, Sigma_t &&Sigma, double mu, S_t &&S, H0_t &&H0,
                    const imag_axes_ft::IAFT *FT, std::string mbpt_output_) {
        mbpt_output = mbpt_output_;
        FockSigma fs(F, Sigma, mu);
        state.initialize(fs);
        vsp.initialize("diis_vectors.h5");
        res_vsp.initialize("diis_residuals.h5");
        comFS_residual.initialize(&state, S, H0, FT, mbpt_output);
        d_alg.init(&state, &comFS_residual, max_subsp_size, true, &vsp, &res_vsp, fs);
        initialized = true;
    }

    template<nda::MemoryArray Array_H_t>
    double solve(Array_H_t &&H, std::string dataset, h5::group &grp, long iter) {
        (void) H; (void) dataset; (void) grp; (void) iter;
        APP_ABORT("This use case for DIIS is not ready yet");
        return 0.0; // to suppress compile warnings
    }
   

    template<nda::MemoryArray Array_4D_t, nda::MemoryArray Array_5D_t>
    std::array<double, 2> solve(Array_4D_t &&F, std::string dataset_F, Array_5D_t &&Sigma, std::string dataset_Sigma,
                 h5::group &scf_grp, long iter) {
      utils::check(initialized, "DIIS must be initialed before solving");
      // do damping and grow subspace
      if(vsp.size() == 1 || iter < diis_start) {
          damp_t damp(mixing);
          damp.metadata_log();

          // 1) grow DIIS subspace wo extrapolation
          FockSigma fs(F, Sigma, get_mu());
          d_alg.grow_xvsp = (vsp.size() <= 1);
          d_alg.extrap = false;
          d_alg.next_step(fs);

          // 2) return damped value
          return damp.solve(F, dataset_F, Sigma, dataset_Sigma, scf_grp, iter);
       }
       // DO DIIS
       else {
          d_alg.extrap = true; 
          d_alg.grow_xvsp = false;
          FockSigma fs(F, Sigma, get_mu());
          state.put(fs);
          int is_extrapolated = d_alg.next_step(fs);
          if(is_extrapolated != 0) {
              auto Fdiff = nda::make_regular(F - d_alg.get_x().get_fock());
              auto Sdiff = nda::make_regular(Sigma - d_alg.get_x().get_sigma());
              auto Fmax_iter = max_element(Fdiff.data(), Fdiff.data()+Fdiff.size(),
                                  [](auto a, auto b) { return std::abs(a) < std::abs(b); });
              auto Smax_iter = max_element(Sdiff.data(), Sdiff.data()+Sdiff.size(),
                                  [](auto a, auto b) { return std::abs(a) < std::abs(b); });
              F     = d_alg.get_x().get_fock();
              Sigma = d_alg.get_x().get_sigma();
              return std::array<double, 2>{std::abs(*Fmax_iter), std::abs(*Smax_iter)};
          }
          else { // No DIIS extrapolation has been applied
              damp_t damp(mixing);
              damp.metadata_log();
              return damp.solve(F, dataset_F, Sigma, dataset_Sigma, scf_grp, iter);
          }
       }
    }

    // TODO: update if other DIIS versions will be plugged in
    void metadata_log() const {
      app_log(2, "\nIterative algorithm for SCF");
      app_log(2, "-----------------------------");
      app_log(2, "  * algorithm: frequency-dependent commutator DIIS\n"
                 "               P. Pokhilko, C.-N. Yeh, D. Zgid. J. Chem. Phys., 2022, 156, 094101\n"
                 "               https://doi.org/10.1063/5.0082586");
      app_log(2, "  * DIIS parameters: ");
      app_log(2, "    mixing = {}", mixing);
      app_log(2, "    max_subsp_size = {}", max_subsp_size);
      app_log(2, "    diis_start = {}", diis_start);
      app_log(2, "    mbpt_output = {}\n", mbpt_output);
    }

  public:
    double mixing = 0.2;
    size_t max_subsp_size = 5;
    size_t diis_start = 5;
    
  private:
    VSpace<FockSigma> vsp;     // space of Fock-self-energy vectors
    VSpace<FockSigma> res_vsp; // space of residuals-commutators
    opt_state<FockSigma> state; // current position of the Fock-self-energy vector
    com_diis_residual comFS_residual; // residual needed for DIIS initialization

    diis_alg<FockSigma> d_alg; // instance of the DIIS algorithm

    std::string mbpt_output;
    bool initialized = false;

    // Read mu from the checkpoint file
    double get_mu() {
        long iter_from_file;
        std::string filename = mbpt_output + ".mbpt.h5";
        h5::file file(filename, 'r');
        h5::group grp(file);
        utils::check(grp.has_subgroup("scf"), "Simulation HDF5 file does not have an scf group");
        auto scf_grp = grp.open_group("scf");
        h5::h5_read(scf_grp, "final_iter", iter_from_file);
        auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter_from_file));
        double mu;
        h5::h5_read(iter_grp, "mu", mu);
        return mu;
    }


  };
} // iter_scf

#endif //COQUI_DIIS_T_HPP
