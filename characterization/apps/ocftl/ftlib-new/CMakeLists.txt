cmake_minimum_required(VERSION 3.10)

project(FTLIB-NEW)

set(src_files 
    ft.cpp
)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_STANDARD 11)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)

include_directories(
  ${MPI_INCLUDE_PATH}
)

set(EXEC_NAME ocftl)

add_library(${EXEC_NAME} STATIC ${src_files})

# PIC is necessary when building STATIC libraries in OpenMPI
set_property(TARGET ${EXEC_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Install common libraries under the lib destination folder.
install(TARGETS ${EXEC_NAME} ARCHIVE DESTINATION ${PROJECT_BINARY_DIR})

message(${PROJECT_BINARY_DIR})

target_link_libraries(${EXEC_NAME}
  ${MPI_LIBRARIES}}
  Threads::Threads)
  
if(MPI_COMPILE_FLAGS)
  set_target_properties(${EXEC_NAME} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
  set_target_properties(${EXEC_NAME} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
