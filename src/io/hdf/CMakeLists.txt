
#if(ENABLE_CUDA)
#  set_source_files_properties(hdf_archive.cpp PROPERTIES LANGUAGE CUDA)
#endif()

add_library(io_hdf hdf_archive.cpp)

target_link_libraries(io_hdf PUBLIC IO::HDF5)

