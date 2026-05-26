#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

########
# Git setup
########

SET(GITREV_BARE_FILE git-rev.h)
SET(GITREV_FILE ${CMAKE_BINARY_DIR}/src/${GITREV_BARE_FILE})

IF (IS_GIT_PROJECT)
  # Pull the git revision information.
  add_custom_command(
    OUTPUT ${GITREV_FILE}
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/gitversion.sh
          --git-executable ${GIT_EXECUTABLE}
          --output ${GITREV_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ${CMAKE_SOURCE_DIR}/.git/HEAD
            ${CMAKE_SOURCE_DIR}/.git/index
    VERBATIM
  )
  add_custom_target(gitversion DEPENDS ${GITREV_FILE})

  add_dependencies(safire_lib gitversion)
ELSE()
  file(WRITE ${GITREV_FILE})
ENDIF()

target_sources(safire_lib
  PRIVATE
    app_version.cpp
    mpi_context.cpp
    parser.cpp
    freemem.cpp
)

if(BUILD_UNIT_TESTS)
  add_library(catch_main STATIC catch_main.cpp)
  target_link_libraries(catch_main PUBLIC safire_lib Catch2::Catch2)
  if(ENABLE_CPPTRACE)
    target_link_libraries(safire_lib PUBLIC cpptrace::cpptrace)
  endif()
endif(BUILD_UNIT_TESTS)

