cmake_minimum_required(VERSION 3.0)

project(LIBONLY)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)

set(OCFTL_DIR "../ftlib-nng")
set(NNG_DIR "/apps/nng")

include_directories(${NNG_DIR}/include ${OCFTL_DIR} ${MPI_INCLUDE_PATH})
link_directories(${NNG_DIR}/lib ${OCFTL_DIR}/lib)

# PIC is necessary when building STATIC libraries in OpenMPI
set_property(TARGET ${EXEC_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

set(LIBONLY_LIBS ocftl nng)

# Info about the executable
set(LIBONLY_SOURCES libonly)
set(LIBONLY_EXEC libonly)

add_executable(${LIBONLY_EXEC} ${LIBONLY_SOURCES})
target_link_libraries(${LIBONLY_EXEC} ${LIBONLY_LIBS} ${MPI_LIBRARIES} Threads::Threads)

if(MPI_COMPILE_FLAGS)
  set_target_properties(${EXEC_NAME} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
  set_target_properties(${EXEC_NAME} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()