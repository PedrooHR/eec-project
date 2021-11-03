cmake_minimum_required(VERSION 3.10)

project("NNG-OCFTL")
set(src_files ft.cpp)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_STANDARD 11)

find_package(Threads REQUIRED)

set(NNG_DIR "/home/pedro.rosso/eec/nng")

include_directories(${NNG_DIR}/include)

link_directories(${NNG_DIR}/lib)

set(EXEC_NAME ocftl)

add_library(${EXEC_NAME} STATIC ${src_files})

# Install common libraries under the lib destination folder.
install(TARGETS ${EXEC_NAME} ARCHIVE DESTINATION ${PROJECT_BINARY_DIR})

target_link_libraries(${EXEC_NAME} nng Threads::Threads)

