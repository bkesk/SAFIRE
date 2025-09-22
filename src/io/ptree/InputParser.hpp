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

#ifndef IO_INPUTPARSER_HPP
#define IO_INPUTPARSER_HPP
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "io/ptree/ptree_utilities.hpp" 
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "Utilities/app_loggers.h"

class InputParser
{
public:
  InputParser() = default; 
  ptree get_root() const {return pt;}
  InputParser(const InputParser& inp) : pt(inp.get_root()) {}
  InputParser(const ptree& pt0) : pt(pt0) {}

  void read(std::string filename)
  { 
    try {
      std::ifstream fp(filename);
      std::string extension = io::get_file_extension(filename);
      pt = parse(fp, extension);
      fp.close();
    } catch (std::exception const& e) {
      throw e; 
    }
  }

  ptree parse(std::basic_istream< typename ptree::key_type::value_type >& s, std::string extension)
  {  // call appropriate parser based on file extension
    if (extension == "json")
    { 
      boost::property_tree::read_json(s, pt);
    } else if (extension == "xml") {
      ptree pt0;
      boost::property_tree::read_xml(s, pt0);
      pt = io::convert_xml(pt0);
    } else {
      std::string msg = "unknown extension " + extension;
      std::cout << msg << std::endl;
      throw std::runtime_error(extension.c_str());
    }
    return pt;
  }

  ptree parse(std::string s, std::string extension)
  {
    std::stringstream ss;
    ss << s;
    return parse(ss, extension);
  }

private:
  ptree pt;
};

#endif
