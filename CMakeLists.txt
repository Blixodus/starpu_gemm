cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(gemm LANGUAGES CXX)

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

option(gemm_USE_CUDA "Build with CUDA" ON)
if(gemm_USE_CUDA)
    add_definitions(-DUSE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
else(gemm_USE_CUDA)
    set_source_files_properties(blas.cu gemm_func.cu bzero_func.cu accumulate_func.cu fill_func.cu cublas_perf.cu PROPERTIES LANGUAGE CXX)
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
target_link_libraries(gemm PRIVATE lib)

if(gemm_UDE_CUDA)
    target_link_libraries(gemm PRIVATE CUDA::cublas)
endif()
