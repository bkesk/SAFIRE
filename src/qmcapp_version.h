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

// Wrapper around the auto-generated Git repository revision
// information file (git-rev.h)
// If not building from a git repository, the git-rev.h file is empty

#ifndef AF_APP_VERSION_INCLUDE
#define AF_APP_VERSION_INCLUDE

#define STR_EXPAND(x) #x
#define STR(x) STR_EXPAND(x)

#ifdef IS_GIT_PROJECT
#include "git-rev.h"
#endif

#ifdef GIT_BRANCH_RAW
#define AF_APP_GIT_BRANCH STR(GIT_BRANCH_RAW)
#define AF_APP_GIT_HASH STR(GIT_HASH_RAW)
#define AF_APP_GIT_COMMIT_LAST_CHANGED STR(GIT_COMMIT_LAST_CHANGED_RAW)
#endif

#endif
