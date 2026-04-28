target_sources(safire_lib
  PRIVATE
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
    HamiltonianOperations/HamiltonianOperations.cpp
    Hamiltonians/KPFactorizedHamiltonian.cpp
    Hamiltonians/RealDenseHamiltonian.cpp
    Hamiltonians/ModelHamOpsGenerator.cpp
    Hamiltonians/HamiltonianFactory.cpp
    Hamiltonians/THCHamiltonian.cpp
    Hamiltonians/KPTHCHamiltonian.cpp
)

if (BUILD_UNIT_TESTS)
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
    set(AFQMC_UNIT_TEST_INPUTS 
      # Removed FactorizedSparse test files: ham_chol_sc.h5 (3 pairs), ham_chol.h5 (1 pair)
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_chol_1e-5.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_thc_1e-6.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/ham_chol_uc.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/wfn_rhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/hubb4x4U1.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/wfn_uhf.h5
    )
#  endif()
  add_subdirectory(SlaterDeterminantOperations/tests)
#  add_subdirectory(SlaterDeterminantOperations/time_propagation)
  add_subdirectory(Walkers/tests)
  add_subdirectory(Hamiltonians/tests)
  add_subdirectory(HamiltonianOperations/tests)
  add_subdirectory(Wavefunctions/tests)
  add_subdirectory(Propagators/tests)
  add_subdirectory(Drivers/tests)
  add_subdirectory(Estimators/tests)
endif()

