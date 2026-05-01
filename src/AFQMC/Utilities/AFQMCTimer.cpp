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

#include "AFQMCTimer.h"

#include "utilities/Timer.hpp"

namespace sfqmc
{
namespace afqmc
{

/*
// pre-defined timer labels
int block_timer;
int pseudo_energy_timer;
int energy_timer;
int vHS_timer;
int assemble_X_timer;
int vbias_timer;
int G_for_vbias_timer;
int propagate_timer;
int mixed_estimator_timer;
int back_propagate_timer;
int E_comm_overhead_timer;
int vHS_comm_overhead_timer;
int popcont_timer;
int ortho_timer;
int setup_timer;
int extra_timer;
*/

TimerManager AFQMCTimer;

//void setup_AFQMC_timer() 
//{
int block_timer = AFQMCTimer.add("Block");
int pseudo_energy_timer = AFQMCTimer.add("PseudoEnergy");
int energy_timer = AFQMCTimer.add("Energy");
int vHS_timer = AFQMCTimer.add("vHS");
int assemble_X_timer = AFQMCTimer.add("X");
int vbias_timer = AFQMCTimer.add("vbias");
int G_for_vbias_timer = AFQMCTimer.add("G_for_vbias");
int propagate_timer = AFQMCTimer.add("Propagate");
int mixed_estimator_timer = AFQMCTimer.add("MixedEstimator");
int back_propagate_timer = AFQMCTimer.add("BackPropagate");
int E_comm_overhead_timer = AFQMCTimer.add("Energy_comm_overhead");
int vHS_comm_overhead_timer = AFQMCTimer.add("vHS_comm_overhead");
int popcont_timer = AFQMCTimer.add("population_control");
int ortho_timer = AFQMCTimer.add("walker_orthogonalization");
int setup_timer = AFQMCTimer.add("setup");
int extra_timer = AFQMCTimer.add("extra");
// AFQMCTimer.reset_all();
//}

} // namespace afqmc
} // namespace sfqmc

