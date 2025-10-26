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


#ifndef COQUI_IAFT_UTILS_HPP
#define COQUI_IAFT_UTILS_HPP

#include "h5/h5.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"

namespace imag_axes_ft {
  /**
   * Reconstruct IAFT object from the metadata in bdft scf output
   * @return IAFT
   */
  inline decltype(auto) read_iaft(std::string scf_file, bool print_meta_log = true) {
    double beta;
    double wmax;
    std::string prec;
    std::string source;

    h5::file file(scf_file, 'r');
    h5::group grp(file);
    auto iaft_grp = grp.open_group("imaginary_fourier_transform");
    h5::h5_read(iaft_grp, "source", source);
    h5::h5_read(iaft_grp, "prec", prec);
    h5::h5_read(iaft_grp, "beta", beta);

    if (iaft_grp.has_dataset("wmax")) {
      h5::h5_read(iaft_grp, "wmax", wmax);
    } else {
      double lambda;
      h5::h5_read(iaft_grp, "lambda", lambda);
      wmax = lambda / beta;
    }
    return imag_axes_ft::IAFT(beta, wmax, imag_axes_ft::string_to_source_enum(source), prec, print_meta_log);
  }

} // imag_axes_ft



#endif //COQUI_IAFT_UTILS_HPP
