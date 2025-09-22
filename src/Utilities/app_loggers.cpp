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

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

int __app_debug_level__ = -10000; 
int __app_output_level__ = -10000; 

// currently using 2 separate loggers
// app_log: uses "std_console" with a clean format only on Global().root()
// app_warning/app_error/app_critical/app_debug: use "err_console" shared by everyone
// consider later using app_debug as a separate logger to file per mpi rank 

void setup_loggers(bool root, int output_level, int debug_level)
{
  if(root) {
    __app_output_level__ = output_level;
    auto l = spdlog::get("std_console");
    if(not l) {
      auto console = spdlog::stdout_color_mt("std_console");  
      spdlog::get("std_console")->set_pattern("%v");
    }
  } else {
    __app_output_level__ = -10000;
  }
  __app_debug_level__ = debug_level;
  auto l = spdlog::get("err_console");
  if(not l) {
    auto err_logger = spdlog::stdout_color_mt("err_console");   
    spdlog::get("err_console")->set_pattern("%^[%l]%$ %v");
  }
  if(debug_level > 0)
    spdlog::get("err_console")->set_level(spdlog::level::debug);
}

void set_debug_level(bool root, int debug_level)
{
  __app_debug_level__ = debug_level;
  if(debug_level > 0) {  
    auto l = spdlog::get("err_console");
    if(not l) {
      auto err_logger = spdlog::stdout_color_mt("err_console");
      spdlog::get("err_console")->set_pattern("%^[%l]%$ %v");
    }
    spdlog::get("err_console")->set_level(spdlog::level::debug);
  }
}

void set_output_level(bool root, int output_level)
{
  if(root) {
    // should check that logger exists! 
    __app_output_level__ = output_level;
    auto l = spdlog::get("std_console");
    if(not l) {
      auto console = spdlog::stdout_color_mt("std_console");
      spdlog::get("std_console")->set_pattern("%v");
    }
  } else
    __app_output_level__ = -10000;
}

