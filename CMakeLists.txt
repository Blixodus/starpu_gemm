cmake_minimum_required (VERSION 3.3)
project(gemm) # LANGUAGES C CXX

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

## # Disable in source directory build
## set( CMAKE_DISABLE_IN_SOURCE_BUILD ON )
## 
## # use, i.e. don't skip the full RPATH for the build tree
## set( CMAKE_SKIP_BUILD_RPATH FALSE )
## 
## # when building, don't use the install RPATH already
## # (but later on when installing)
## set( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE )
## 
## option(BUILD_SHARED_LIBS
##   "Build shared libraries" ON)

# Check that we do no try to configure/build inside the source directory
# ----------------------------------------------------------------------
if( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
  message(FATAL_ERROR "
    This project is set such that you can not configure/build inside the source directory.
    Please:
     1) remove entirely this directory
     2) extract or clone a fresh revision of the project
     3) create a build directory and run cmake from this directory or
    run with (for exemple):
    mkdir build && cd build && cmake ..
    or
    cmake . -B./build
")
endif()

# Disable in source directory build
set( CMAKE_DISABLE_IN_SOURCE_BUILD ON )

# use, i.e. don't skip, the full RPATH for the build tree
set( CMAKE_SKIP_BUILD_RPATH FALSE )

# when building, don't use the install RPATH already
# (but later on when installing)
set( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE )

# Build shared libraries
set( BUILD_SHARED_LIBS ON )

# Check that the submodule exists and is up to date
# -------------------------------------------------
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()

if( NOT EXISTS "${PROJECT_SOURCE_DIR}/cmake_modules/morse_cmake/modules/MorseInit.cmake" )
  message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# Add extra cmake module path and initialize morse cmake modules
# This is mainly done to integrate the BLAS/LAPACK detection
# --------------------------------------------------------------
if (IS_DIRECTORY ${CMAKE_SOURCE_DIR}/cmake_modules/morse_cmake/modules)
  set( MORSE_CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules/morse_cmake/modules )
  list(APPEND CMAKE_MODULE_PATH ${MORSE_CMAKE_MODULE_PATH})
  include(MorseInit)
else()
  message(FATAL_ERROR "Submodule cmake_morse not initialized - run `git submodule update --init`")
endif()


# Define the compilation options of the project
# ---------------------------------------------

# Option to enable/disable the MPI compilation
option( ENABLE_MPI "Define if MPI support should be enabled" OFF )

# Option to enable/disable the StarPU package
option( ENABLE_STARPU "Define if StarPU support is enabled or not" ON )

# Option to enable/disable CUDA support
option( ENABLE_CUDA "Define if CUDA support is enabled or not" ON )

# Generate the config file that holds the define of the lib
# ---------------------------------------------------------
# configure_file (
#   "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
#   "${CMAKE_CURRENT_BINARY_DIR}/include/config.h")

# Find all required external packages
# -----------------------------------

# Search for MPI
if (ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

# Search for StarPU
if( ENABLE_STARPU )
  # Specify the minimum version
  set( STARPU_REQUIRED_VERSION "1.3" CACHE STRING "oldest STARPU version desired" )

  find_package( STARPU ${STARPU_REQUIRED_VERSION} REQUIRED )
endif()

# Search for OpenMP
find_package(OpenMP REQUIRED)

# Search for libm
find_package(M REQUIRED)

# Search for cblas in sequential
set( CBLAS_MT FALSE )
find_package(CBLAS REQUIRED)

# Search for lapacke in sequential
set( LAPACKE_MT FALSE )
find_package(LAPACKE REQUIRED)

# USE_CUDA
# --------
option(gemm_USE_CUDA "Build with CUDA" ON)
if(gemm_USE_CUDA)
  add_definitions(-DUSE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit)
endif()

# Add libraries
# -------------

set( gemmlib_srcs
  blas.cpp
  gemm_func.cpp
  bzero_func.cpp
  accumulate_func.cpp
  fill_func.cpp
  cublas_perf.cpp
)

add_library( gemmlib
  ${gemmlib_srcs}
)

target_link_libraries( gemmlib PUBLIC OpenMP::OpenMP_C )

if(ENABLE_MPI)
  target_link_libraries( gemmlib PUBLIC MPI::MPI_C )
endif()

if(ENABLE_STARPU)
  target_link_libraries( gemmlib PRIVATE MORSE::STARPU )
endif()

target_link_libraries( gemmlib PUBLIC MORSE::LAPACKE )
target_link_libraries( gemmlib PUBLIC MORSE::CBLAS   )
target_link_libraries( gemmlib PUBLIC MORSE::M       )

install(TARGETS gemmlib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

# Add executable

add_executable(gemm gemm.cpp)
target_link_libraries(gemm PRIVATE gemmlib)

if(gemm_UDE_CUDA)
  target_link_libraries(gemm PRIVATE CUDA::cublas)
endif()


if (ENABLE_MPI)
  target_link_libraries( gemm PUBLIC MPI::MPI_C )
endif (ENABLE_MPI)
if (ENABLE_STARPU)
  target_link_libraries(gemm PUBLIC MORSE::STARPU)
endif (ENABLE_STARPU)

install(TARGETS gemm RUNTIME DESTINATION bin)




