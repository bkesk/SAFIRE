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

#ifndef UTILITIES_APPABORT_HPP
#define UTILITIES_APPABORT_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <mpi.h>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

template<class... Args>
void APP_ABORT(Args&&... args)
{
  //open err_console and output message
  auto l = spdlog::get("err_console");
  if(not l) 
    auto err_logger = spdlog::stdout_color_mt("err_console");
  spdlog::get("err_console")->error("**********************************************");
  spdlog::get("err_console")->error("        APPLICATION ABORT: Fatal Error.");
  spdlog::get("err_console")->error("**********************************************");
  spdlog::get("err_console")->error(std::forward<Args>(args)...);
  spdlog::get("err_console")->error("**********************************************");
  spdlog::get("err_console")->flush(); 
  // Abort
  MPI_Abort(MPI_COMM_WORLD, 1);
}

#endif
