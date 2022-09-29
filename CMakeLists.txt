cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(gemm LANGUAGES CXX CUDA)

find_package(BLAS REQUIRED)

find_package(PkgConfig)
pkg_check_modules(STARPU REQUIRED starpu-1.3)
if (STARPU_FOUND)
    include_directories (${STARPU_INCLUDE_DIRS})
    link_directories    (${STARPU_STATIC_LIBRARY_DIRS})
    link_libraries      (${STARPU_STATIC_LIBRARIES})
else (STARPU_FOUND)
    message(FATAL_ERROR "StarPU not found")
endif()

add_library(lib STATIC
    blas.cu blas.hpp
    gemm_func.cu gemm_func.hpp
    bzero_func.cu bzero_func.hpp
    accumulate_func.cu accumulate_func.hpp
    fill_func.cu fill_func.hpp
    cublas_perf.cu cublas_perf.hpp
    matrix.hpp)

add_executable(gemm gemm.cpp)
target_link_libraries(gemm PRIVATE lib -lcublas)
