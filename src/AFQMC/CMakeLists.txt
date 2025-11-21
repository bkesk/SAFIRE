
#-------------------------------------------------------------------
# Sources
#-------------------------------------------------------------------
set (AFQMC_SRCS
  Utilities/AFQMCTimer.cpp 
  Utilities/readWfn.cpp
  Utilities/probit.cpp
  AFQMCFactory.cpp
  Drivers/DriverFactory.cpp
  Drivers/AFQMCDriver.cpp
#  Drivers/CSAFQMCDriver.cpp
  Propagators/AFQMCBasePropagator.cpp
#  Propagators/AFQMCModelPropagator.cpp
  Propagators/PropagatorFactory.cpp
  Wavefunctions/WavefunctionFactory.cpp
#  Hamiltonians/KPFactorizedHamiltonian.cpp
#  Hamiltonians/RealDenseHamiltonian.cpp
#  Hamiltonians/RealDenseHamiltonian_v2.cpp
#  Hamiltonians/ModelHamOpsGenerator.cpp
  Hamiltonians/HamiltonianFactory.cpp
  Hamiltonians/THCHamiltonian.cpp
  )

add_library(afqmc ${AFQMC_SRCS})
target_link_libraries(afqmc PUBLIC utils qmcio numerics)

if (BUILD_UNIT_TESTS)
  INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/extern/catch)
#  if (ENABLE_DEVICE)
#    set(AFQMC_UNIT_TEST_INPUTS 
#      # Removed FactorizedSparse test files: ham_chol_sc.h5 (3 pairs), ham_chol.h5 (1 pair)
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_chol_1e-5.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_thc_1e-6.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/ham_chol_uc.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/wfn_rhf.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5     	
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/hubb4x4U1.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/wfn_uhf.h5
#    )
#  else()
#    set(AFQMC_UNIT_TEST_INPUTS 
#      # Removed FactorizedSparse test files: ham_chol_sc.h5 (3 pairs), ham_chol.h5 (1 pair)
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_chol_1e-5.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_thc_1e-6.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/ham_chol_uc.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/wfn_rhf.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/hubb4x4U1.h5
#      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/wfn_uhf.h5
#    )
#  endif()
  SUBDIRS(SlaterDeterminantOperations/tests)
#  SUBDIRS(SlaterDeterminantOperations/time_propagation)
  SUBDIRS(Walkers/tests)
  SUBDIRS(Hamiltonians/tests)
  SUBDIRS(HamiltonianOperations/tests)
  SUBDIRS(Wavefunctions/tests)
  SUBDIRS(Propagators/tests)
  SUBDIRS(Drivers/tests)
  SUBDIRS(Estimators/tests)
endif()

