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

//
//

#ifndef AFQMC_AFQMCTIMER_H
#define AFQMC_AFQMCTIMER_H

#include "Utilities/Timer.hpp"

namespace sfqmc
{
namespace afqmc
{

// pre-defined timer labels
extern int block_timer;
extern int pseudo_energy_timer;
extern int energy_timer;
extern int vHS_timer;
extern int assemble_X_timer;
extern int vbias_timer;
extern int G_for_vbias_timer;
extern int propagate_timer;
extern int mixed_estimator_timer;
extern int back_propagate_timer;
extern int E_comm_overhead_timer;
extern int vHS_comm_overhead_timer;
extern int popcont_timer;
extern int ortho_timer;
extern int setup_timer;
extern int extra_timer;

extern TimerManager AFQMCTimer;

void setup_AFQMC_timer(); 

}
}

#endif
