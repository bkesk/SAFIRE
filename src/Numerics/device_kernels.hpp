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

#ifndef DEVICE_KERNELS_HPP
#define DEVICE_KERNELS_HPP


#if defined(ENABLE_CUDA)

#include "Numerics/detail/CUDA/Kernels/determinant.cuh"
#include "Numerics/detail/CUDA/Kernels/adotpby.cuh"
#include "Numerics/detail/CUDA/Kernels/fill_n.cuh"
#include "Numerics/detail/CUDA/Kernels/uninitialized_fill_n.cuh"
#include "Numerics/detail/CUDA/Kernels/uninitialized_copy_n.cuh"
#include "Numerics/detail/CUDA/Kernels/axty.cuh"
#include "Numerics/detail/CUDA/Kernels/adiagApy.cuh"
#include "Numerics/detail/CUDA/Kernels/sum.cuh"
#include "Numerics/detail/CUDA/Kernels/acAxpbB.cuh"
#include "Numerics/detail/CUDA/Kernels/print.cuh"
#include "Numerics/detail/CUDA/Kernels/setIdentity.cuh"
#include "Numerics/detail/CUDA/Kernels/zero_complex_part.cuh"
#include "Numerics/detail/CUDA/Kernels/copy_n_cast.cuh"
#include "Numerics/detail/CUDA/Kernels/inplace_cast.cuh"
#include "Numerics/detail/CUDA/Kernels/ajw_to_waj.cuh"
#include "Numerics/detail/CUDA/Kernels/vKKwij_to_vwKiKj.cuh"
#include "Numerics/detail/CUDA/Kernels/KaKjw_to_QKajw.cuh"
#include "Numerics/detail/CUDA/Kernels/vbias_from_v1.cuh"
#include "Numerics/detail/CUDA/Kernels/KaKjw_to_KKwaj.cuh"
#include "Numerics/detail/CUDA/Kernels/batched_dot_wabn_wban.cuh"
#include "Numerics/detail/CUDA/Kernels/batched_Tab_to_Klr.cuh"
#include "Numerics/detail/CUDA/Kernels/dot_wabn.cuh"
#include "Numerics/detail/CUDA/Kernels/Tab_to_Kl.cuh"
#include "Numerics/detail/CUDA/Kernels/sampleRNG.cuh"
#include "Numerics/detail/CUDA/Kernels/construct_X.cuh"
#include "Numerics/detail/CUDA/Kernels/reference_operations.cuh"
#include "Numerics/detail/CUDA/Kernels/term_by_term_matrix_vec.cuh"
#include "Numerics/detail/CUDA/Kernels/term_by_term_matrix_mat.cuh"
#include "Numerics/detail/CUDA/Kernels/axpyBatched.cuh"
#include "Numerics/detail/CUDA/Kernels/Auwn_Bun_Cuw.cuh"
#include "Numerics/detail/CUDA/Kernels/get_diagonal.cuh"
#include "Numerics/detail/CUDA/Kernels/copy_select.cuh"
#include "Numerics/detail/CUDA/Kernels/spVi_Bij_yj.cuh"
#include "Numerics/detail/CUDA/Kernels/batched_dot.cuh"
#include "Numerics/detail/CUDA/Kernels/getGIJ.cuh"
#include "Numerics/detail/CUDA/Kernels/extract_overlap_matrix.cuh"
#include "Numerics/detail/CUDA/Kernels/construct_phmsd_R.cuh"
#include "Numerics/detail/CUDA/Kernels/accumulate.cuh"
#include "Numerics/detail/CUDA/Kernels/add_diagonal.cuh"
#include "Numerics/detail/CUDA/Kernels/phmsd_determinants.cuh"
#include "Numerics/detail/CUDA/Kernels/phmsd_inverse.cuh"
#include "Numerics/detail/CUDA/Kernels/phmsd_energy.cuh"
#include "Numerics/detail/CUDA/Kernels/complex_conjugate.cuh"
#include "Numerics/detail/CUDA/Kernels/AGiwj_BGjwi_CG.cuh"

#elif defined(ENABLE_HIP)

#include "Numerics/detail/HIP/Kernels/determinant.hip.h"
#include "Numerics/detail/HIP/Kernels/adotpby.hip.h"
#include "Numerics/detail/HIP/Kernels/fill_n.hip.h"
#include "Numerics/detail/HIP/Kernels/uninitialized_fill_n.hip.h"
#include "Numerics/detail/HIP/Kernels/uninitialized_copy_n.hip.h"
#include "Numerics/detail/HIP/Kernels/axty.hip.h"
#include "Numerics/detail/HIP/Kernels/adiagApy.hip.h"
#include "Numerics/detail/HIP/Kernels/sum.hip.h"
#include "Numerics/detail/HIP/Kernels/acAxpbB.hip.h"
#include "Numerics/detail/HIP/Kernels/print.hip.h"
#include "Numerics/detail/HIP/Kernels/setIdentity.hip.h"
#include "Numerics/detail/HIP/Kernels/zero_complex_part.hip.h"
#include "Numerics/detail/HIP/Kernels/batchedDot.hip.h"
#include "Numerics/detail/HIP/Kernels/copy_n_cast.hip.h"
#include "Numerics/detail/HIP/Kernels/inplace_cast.hip.h"
#include "Numerics/detail/HIP/Kernels/ajw_to_waj.hip.h"
#include "Numerics/detail/HIP/Kernels/vKKwij_to_vwKiKj.hip.h"
#include "Numerics/detail/HIP/Kernels/KaKjw_to_QKajw.hip.h"
#include "Numerics/detail/HIP/Kernels/vbias_from_v1.hip.h"
#include "Numerics/detail/HIP/Kernels/KaKjw_to_KKwaj.hip.h"
#include "Numerics/detail/HIP/Kernels/batched_dot_wabn_wban.hip.h"
#include "Numerics/detail/HIP/Kernels/batched_Tab_to_Klr.hip.h"
#include "Numerics/detail/HIP/Kernels/dot_wabn.hip.h"
#include "Numerics/detail/HIP/Kernels/Tab_to_Kl.hip.h"
#include "Numerics/detail/HIP/Kernels/sampleGaussianRNG.hip.h"
#include "Numerics/detail/HIP/Kernels/construct_X.hip.h"
#include "Numerics/detail/HIP/Kernels/reference_operations.hip.h"
#include "Numerics/detail/HIP/Kernels/term_by_term_matrix_vec.hip.h"
#include "Numerics/detail/HIP/Kernels/axpyBatched.hip.h"
#include "Numerics/detail/HIP/Kernels/Auwn_Bun_Cuw.hip.h"
#include "Numerics/detail/HIP/Kernels/get_diagonal.hip.h"

#endif

#endif
