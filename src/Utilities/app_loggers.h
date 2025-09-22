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

#ifndef UTILITIES_APP_LOGGERS_HPP
#define UTILITIES_APP_LOGGERS_HPP

#include "spdlog/spdlog.h"
#include "Utilities/fmt_extensions.hpp"
#include "Utilities/AppAbort.hpp"

extern int __app_debug_level__; 
extern int __app_output_level__; 

// currently using 2 separate loggers
// app_log: uses "std_console" with a clean format only on Global().root()
// app_warning/app_error/app_critical/app_debug: use "err_console" shared by everyone
// consider later using app_debug as a separate logger to file per mpi rank 

void setup_loggers(bool root=true, int output_level=2, int debug_level=0);
void set_output_level(bool root, int output_level);
void set_debug_level(bool root, int debug_level);

template<class... Args>
void app_log(int level, Args&&... args)
{
  if(__app_output_level__ > 0 and level <= __app_output_level__) {
    auto l = spdlog::get("std_console");
    if(l) 
      l->info(std::forward<Args>(args)...);
    else
      APP_ABORT(" Error: app_log used uninitilaled."); 
  }
}

template<class... Args>
void app_warning(Args&&... args)
{ 
  if(__app_output_level__ > 0) { 
    auto l = spdlog::get("err_console");
    if(l)
      l->warn(std::forward<Args>(args)...);
    else
      APP_ABORT(" Error: app_warning used uninitilaled.");
  }
}

template<class... Args>
void app_error(Args&&... args)
{ 
  auto l = spdlog::get("err_console");
  if(l) { 
    l->error(std::forward<Args>(args)...);
    l->flush();
  } else
    APP_ABORT(" Error: app_error used uninitilaled.");
}

template<class... Args>
void app_debug(int level, Args&&... args)
{ 
  if(__app_debug_level__ > 0 and level <= __app_debug_level__) {
    auto l = spdlog::get("err_console");
    if(l)
      l->debug(std::forward<Args>(args)...);
    else
      APP_ABORT(" Error: app_debug used uninitilaled.");
  }
}

inline void app_log_flush() 
{
  if(__app_output_level__ > 0) {
    auto l = spdlog::get("std_console");
    if(l)
      l->flush();
    else
      APP_ABORT(" Error: app_warning used uninitilaled.");
  }
}

inline void app_error_flush() 
{
  auto l = spdlog::get("err_console");
  if(l)
    l->flush();
  else
    APP_ABORT(" Error: app_debug used uninitilaled.");
}

#endif
