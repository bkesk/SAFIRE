

subdirs(hdf)

add_library(qmcio INTERFACE)
target_include_directories(qmcio INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(qmcio INTERFACE io_hdf)
target_link_libraries(qmcio INTERFACE $<BUILD_INTERFACE:${PROJECT_NAME}_warnings>)
