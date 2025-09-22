
if(ENABLE_CUDA)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(safire main.cpp)
target_link_libraries(safire utils afqmc) 

install(TARGETS safire
        RUNTIME DESTINATION bin)

