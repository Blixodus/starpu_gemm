#include <starpu.h>
#ifdef USE_CUDA
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#endif
#include <iostream>

#include "bzero_func.hpp"
#include "blas.hpp"

template <typename DataType>
void bzero_matrix_cpu(void * buffers[], void * cl_args) {
  //std::cout << "BZERO CPU" << std::endl;
  int m = STARPU_MATRIX_GET_NX(buffers[0]);
  int n = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType * mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  for(int j = 0; j < n; j++) {
    memset(&mat[ld*j], 0, m*sizeof(DataType));
  }
}

template void bzero_matrix_cpu<float>(void * buffers[], void * cl_args);
template void bzero_matrix_cpu<double>(void * buffers[], void * cl_args);

#ifdef USE_CUDA
template <typename DataType>
void bzero_matrix_cuda(void * buffers[], void * cl_args) {
  //std::cout << "BZERO CUDA" << std::endl;
  int m = STARPU_MATRIX_GET_NX(buffers[0]);
  int n = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType * mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  DataType alpha = 0, beta = 0;
  cublas<DataType>::geam(starpu_cublas_get_local_handle(), 'N', 'N', m, n, (DataType)0, mat, ld, (DataType)0, mat, ld, mat, ld);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

template void bzero_matrix_cuda<float>(void * buffers[], void * cl_args);
template void bzero_matrix_cuda<double>(void * buffers[], void * cl_args);
#endif
