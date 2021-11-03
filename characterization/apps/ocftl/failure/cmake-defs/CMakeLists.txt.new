cmake_minimum_required(VERSION 3.0)

project(FAILURE)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)

set(OCFTL_DIR "../ftlib-new")

include_directories(${OCFTL_DIR} ${MPI_INCLUDE_PATH})
link_directories(${OCFTL_DIR}/lib)

set(FAILURE_LIBS ocftl)

# Info about the executable
set(FAILURE_SOURCES failure.cpp)
set(FAILURE_EXEC failure)

add_executable(${FAILURE_EXEC} ${FAILURE_SOURCES})

# PIC is necessary when building STATIC libraries in OpenMPI
set_property(TARGET ${FAILURE_EXEC} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${FAILURE_EXEC} ${FAILURE_LIBS} ${MPI_LIBRARIES} Threads::Threads)

if(MPI_COMPILE_FLAGS)
  set_target_properties(${FAILURE_EXEC} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
  set_target_properties(${FAILURE_EXEC} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
