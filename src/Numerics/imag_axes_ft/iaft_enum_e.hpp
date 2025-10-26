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


#ifndef COQUI_IAFT_ENUM_E_HPP
#define COQUI_IAFT_ENUM_E_HPP

namespace imag_axes_ft {
  enum stats_e {
    fermi, boson
  };

  inline std::string stats_enum_to_string(int stats_enum) {
    switch(stats_enum) {
      case stats_e::fermi:
        return "fermi";
      case stats_e::boson:
        return "boson";
      default:
        return "not recognized...";
    }
  }

  inline stats_e string_to_stats_enum(std::string stats) {
    if (stats == "fermi") {
      return stats_e::fermi;
    } else if (stats == "boson") {
      return stats_e::boson;
    } else {
      utils::check(false, "Unrecognized stats: {}. Available options: fermi, boson", stats);
      return stats_e::fermi;
    }
  }

  enum source_e {
    dlr_source, ir_source
  };

  inline std::string source_enum_to_string(int source_enum) {
    switch (source_enum) {
      case source_e::dlr_source:
        return "dlr";
      case source_e::ir_source:
        return "ir";
      default:
        return "not recognized...";
    }
  }

  inline source_e string_to_source_enum(std::string iaft_source) {
    if (iaft_source == "dlr") {
      return source_e::dlr_source;
    } else if (iaft_source == "ir") {
      return source_e::ir_source;
    } else {
      utils::check(false, "Unrecognized IAFT source: {}. Available options: dlr, ir", iaft_source);
      return source_e::ir_source;
    }
  }
}

#endif //COQUI_IAFT_ENUM_E_HPP
