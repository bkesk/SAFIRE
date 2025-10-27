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

#ifndef HIP_INIT_H
#define HIP_INIT_H
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

namespace qmc_hip
{
void HIP_INIT(boost::mpi3::shared_communicator& node, unsigned long long int iseed = 911ULL);

}

#endif
