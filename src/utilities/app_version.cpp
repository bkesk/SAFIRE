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

#include <iostream>
#include "config.h"

#include "qmcapp_version.h"
#include "app_version.h"

#define STR_EXPAND(x) #x
#define STR(x) STR_EXPAND(x)


void print_version(int verbosity){

  std::cout << "AF App version: " << AF_APP_VERSION << std::endl;
#ifdef AFAPP_RELEASE
  std::cout << "This is a release build" << std::endl;
#endif
  if (verbosity > 1){
    // Could include more if we find use
    std::cout << "This build has the following enabled:" << std::endl;
#ifdef HAVE_MKL
    std::cout << " - MKL:  1" << std::endl;
#else
    std::cout << " - MKL:  0" << std::endl;
#endif

#ifdef ENABLE_CUDA
    std::cout << " - CUDA: 1"<<  std::endl;
#else
    std::cout << " - CUDA: 0"<<  std::endl;
#endif

#ifdef HAVE_HIP
    std::cout << " - HIP:  1" << std::endl;
#else
    std::cout << " - HIP:  0" << std::endl;
#endif
  }

#ifdef AF_APP_GIT_BRANCH
  std::cout << "app git branch: " << AF_APP_GIT_BRANCH << std::endl;
  std::cout << "app git commit: " << AF_APP_GIT_HASH << std::endl;
  if (verbosity > 1){
    std::cout << "app git commit date: " << AF_APP_GIT_COMMIT_LAST_CHANGED << std::endl;
  }
#else
  std::cout << "app was not built with a git repository" << std::endl;
#endif
}
