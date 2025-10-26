
SET(SRCS_IO
  ptree/InputParser.hpp
  app_loggers.cpp
)

add_library(qmcio ${SRCS_IO}) 

#target_include_directories(qmcio PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
TARGET_LINK_LIBRARIES(qmcio PUBLIC tomlplusplus::tomlplusplus nda_c h5_c) 
target_link_libraries(qmcio PUBLIC $<BUILD_INTERFACE:${PROJECT_NAME}_warnings>)
IF(ENABLE_SPDLOG)
  TARGET_LINK_LIBRARIES(qmcio PUBLIC spdlog::spdlog)
ENDIF()
IF(ENABLE_CPPTRACE)
  TARGET_LINK_LIBRARIES(qmcio PUBLIC cpptrace::cpptrace)
ENDIF()
install(TARGETS qmcio
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

