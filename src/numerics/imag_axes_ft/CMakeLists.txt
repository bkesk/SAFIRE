#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

# Install ir data directory to the install directory under 'share/ir/data'
set(DATA_DIR "${PROJECT_SOURCE_DIR}/src/numerics/imag_axes_ft/ir/data")
file(GLOB H5_FILES "${DATA_DIR}/*.h5")
install(FILES ${H5_FILES}
        DESTINATION data/ir)

if (BUILD_UNIT_TESTS)
  add_subdirectory(ir/tests)
  add_subdirectory(tests)
endif()
