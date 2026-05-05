#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

########
# Git setup
########

SET(GITREV_BARE_FILE git-rev.h)
SET(GITREV_BARE_TMP git-rev-tmp.h)
SET(GITREV_FILE ${CMAKE_BINARY_DIR}/src/${GITREV_BARE_FILE})
SET(GITREV_TMP ${CMAKE_BINARY_DIR}/src/${GITREV_BARE_TMP})

# Always clean up git-rev-tmp.h file when cmake is run
EXECUTE_PROCESS(
  COMMAND ${CMAKE_COMMAND} -E remove ${GITREV_TMP}
)

IF (IS_GIT_PROJECT)
  # Pull the git revision information.
  
  # target without OUTPUT is always out of date which is what we want.
  add_custom_target(gitversion
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/gitversion.sh
          --git-executable ${GIT_EXECUTABLE}
          --gitrev-tmp ${GITREV_TMP}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GITREV_TMP} ${GITREV_FILE}
    COMMAND ${CMAKE_COMMAND} -E remove ${GITREV_TMP}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
  )

  # Print some configure-time git info (useful for understanding what commits
  # are in particular build for the nightly CDash reports)

  EXECUTE_PROCESS(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
    OUTPUT_VARIABLE GIT_CONFIG_BRANCH
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  EXECUTE_PROCESS(
    COMMAND ${GIT_EXECUTABLE} describe --always --dirty --abbrev
    OUTPUT_VARIABLE GIT_CONFIG_COMMIT_HASH
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  MESSAGE("Git branch: ${GIT_CONFIG_BRANCH}")
  MESSAGE("Git commit hash: ${GIT_CONFIG_COMMIT_HASH}")
ELSE()
  file(WRITE ${GITREV_FILE})
ENDIF()


target_sources(safire_lib
  PRIVATE
    app_version.cpp
    mpi_context.cpp
    parser.cpp
    freemem.cpp
    ${GITREV_FILE}
)

if(BUILD_UNIT_TESTS)
  add_library(catch_main STATIC catch_main.cpp)
  target_link_libraries(catch_main PUBLIC safire_lib Catch2::Catch2 cpptrace::cpptrace)
endif(BUILD_UNIT_TESTS)

