/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <boost/property_tree/ptree.hpp>
#include <boost/optional.hpp>
#include "IO/app_loggers.h"

using boost::property_tree::ptree;

namespace io
{

inline ptree convert_xml(const ptree& pt0)
{
  ptree pt1;
  for(auto& it : pt0)
  {
    std::string cname = it.first;
    ptree child = it.second;
    if (cname == "<xmlattr>"){ // promote to child
      for(auto& it1 : child)
      {
        pt1.put(it1.first, it1.second.get_value<std::string>());
      }
    } else if (cname == "<xmlcomment>") { // ignore
    } else if (cname == "parameter") { // rename child by attribute "name"
      std::string pname = child.get<std::string>("<xmlattr>.name");
      std::string text = child.get_value<std::string>();
      pt1.put(pname, text);
    } else if (child.size() < 1) {
      std::string text = child.get_value<std::string>();
      pt1.put(cname, text);
    } else { // recurse
      ptree pt2 = convert_xml(child);
      pt1.add_child(cname, pt2);
    }
  }
  return pt1;
}

template<typename T>
inline bool check_exists(ptree const& pt, std::string path) 
{
  if( boost::optional<T> val = pt.get_optional<T>(path) ) return true;
  return false;
}

/**
 * @brief Compares the keys of one ptree object against a reference ptree and checks if there are unknown keys.
 *
 * @details checks each key in the "user ptree" against the keys in the "reference ptree". 
 *   If the key is not found in the reference ptree, it is printed to the screen. 
 *   If strict_keys is true, an exception is thrown. If extra_keys is provided, 
 *   these keys are ignored if they are found in the user ptree.
 *   Under the hood, constructs an unordered_set of keys for constant time lookup.
 * 
 * @param location_name Name of the location where the check is being performed.
 * @param ref_pt Reference ptree object containing the known keys.
 * @param user_pt User ptree object containing the keys to be checked.
 * @param extra_keys Set of extra keys to be ignored if found in the user ptree. (optional, default: empty set)
 */
inline bool compare_known_keys(std::string && location_name, ptree const& ref_pt, ptree const& user_pt,  std::unordered_set<std::string> const& extra_keys = {}) {
  std::string base_error = "in " + location_name + " : Unknown input parameter(s) provided. Check the input file.";
  bool found_unknown_keys = false;
  // convert to unordered_set in N time, N space for constant time lookup
  std::unordered_set<std::string> ref_keys = [&ref_pt](){
    std::unordered_set<std::string> keys;
    for (const auto& item : ref_pt) {
      keys.insert(item.first);
    }
    return keys;
  }();
  std::unordered_set<std::string> user_keys = [&user_pt](){
    std::unordered_set<std::string> keys;
    for (const auto& item : user_pt) {
      keys.insert(item.first);
    }
    return keys;
  }();
  for (const auto& item : user_keys)
  {
    if (ref_keys.count(item) == 0 && extra_keys.count(item) == 0)
    {
      found_unknown_keys = true;
      app_warning("Unknown key: {}", item);
    }
  }
  if (found_unknown_keys)
    app_warning(base_error);

  return found_unknown_keys;
}

/* -------------------------------- utilities ------------------------------- */
inline void str_rep(std::ostream &out, ptree const& pt, int indent=0) 
{
  for(auto& it : pt)
  {
    for (int ii=0; ii<indent; ii++) out << "  ";
    out << it.first << ": " << it.second.get_value<std::string>() << std::endl;
    str_rep(out, it.second, indent+1);
  }
}

inline void tolower(std::string& s)
{
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c){ return std::tolower(c); });
}

inline std::string get_file_extension(const std::string &s)
{ // oreilly c-cookbook/0596007612/ch10s14.html
  size_t i = s.rfind('.', s.length());
  if (i == std::string::npos) return "";
  std::string ext = s.substr(i+1, s.length() - i);
  tolower(ext);
  return ext;
}

template <typename dtype>
std::vector<dtype> str2vec(const std::string s)
{
  std::stringstream ss(s);
  dtype val;
  std::vector<dtype> vec;
  while (ss>>val) vec.push_back(val);
  return vec;
}
  
inline std::string to_string(ptree const& pt)
{  
  std::stringstream ss;
  str_rep(ss, pt);
  return ss.str();
}


/**
 * @brief Reads a vector of values from a JSON stream using a specified path.
 * 
 * This function parses JSON data from an input stream and extracts values from
 * a specified path into a vector. Each item found at the given path is converted
 * to type T and added to the output vector. If a conversion fails, a warning is
 * logged and the item is skipped.
 * 
 * @tparam T The type of elements to store in the vector
 * @tparam streamtype The type of the input stream (e.g., std::ifstream, std::stringstream)
 * @param input_stream The stream containing JSON data to read from
 * @param path The JSON path string specifying where to find the array/collection of values
 * @param vec The output vector to populate with parsed values
 * 
 * @throws boost::property_tree::json_parser_error If JSON parsing fails
 * @throws boost::property_tree::ptree_bad_path If the specified path doesn't exist
 * 
 * @note Values that cannot be converted to type T are skipped with a warning message
 */
template <typename T, typename streamtype>
void read_vector(streamtype& input_stream, std::string path, std::vector<T>& vec) {
    ptree pt;
    read_json(input_stream, pt);
    for (const auto& item : pt.get_child(path)) {
        try {
            vec.push_back(item.second.get_value<T>());
        } catch (const boost::property_tree::ptree_bad_data& e) {
          app_warning("Could not convert value to target type, skipping: {}", item.second.data());
        }
    }
}


/**
 * @brief Reads either a single value of type T or a std::vector<T> from a ptree.
 * 
 * This function examines the ptree structure to determine if the specified key
 * contains a single value or an array/collection of values. It uses the presence
 * of child nodes to make this determination:
 * - If the node has children, it's treated as an array and all child values are read
 * - If the node has no children or doesn't exist, it's treated as a single value
 * 
 * @tparam T The type of values to read (e.g., int, double, std::string)
 * @param pt The property tree to read from
 * @param key The key/path to look for in the property tree
 * @param default_value The default value to use if the key doesn't exist
 * @return std::vector<T> containing either the single value or all array values
 * 
 * @note For single values, the returned vector will contain exactly one element
 * @note If conversion fails for any array element, it will be skipped with a warning
 */
template<typename T>
std::vector<T> get_value_or_vector(const ptree& pt, const std::string& key, const T& default_value = T{})
{
    std::vector<T> result;
    
    auto node = pt.get_child_optional(key);
    if (node && !node->empty()) {
        // It's an array/vector - iterate through children
        for (const auto& item : *node) {
            try {
                result.push_back(item.second.get_value<T>());
            } catch (const boost::property_tree::ptree_bad_data& e) {
                std::cerr << "Warning: Could not convert array element to target type, skipping: " 
                          << item.second.data() << std::endl;
            }
        }
    } else {
        // It's a single value or doesn't exist
        T single_value = pt.get<T>(key, default_value);
        result.push_back(single_value);
    }
    
    return result;
}

} // namespace io

/**
 * @brief Removes the first occurrence of a key from the property tree if it exists.
 * 
 * This function searches for the specified key in the given property tree
 * and removes the first occurrence if found. If there are multiple children
 * with the same key (which is possible in ptree), only the first one is removed.
 * The function provides a safe way to remove keys without throwing exceptions
 * when the key doesn't exist.
 * 
 * @param pt Reference to the property tree to modify
 * @param key The key to search for and remove
 * @return true if the key was found and successfully removed, false otherwise
 */
inline bool remove_node_if_exists(ptree& pt, const std::string& key)
{
  if (pt.get_child_optional(key))
  {
    pt.erase(key);
    return true;
  } else {
    return false;
  }
}

inline std::ostream& operator<<(std::ostream &out, const ptree &pt)
{
  io::str_rep(out, pt);
  return out;
}

template<typename T, typename Factory>
T get_parameter(Factory const& F, std::string obj, std::string param, T def)
{
  //auto pt = F.get_input(obj);
  ptree pt = F.get_input(obj);
  return pt.get<T>(param,def);
}

