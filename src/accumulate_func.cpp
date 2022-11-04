#include <starpu.h>
#ifdef USE_CUDA
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#endif
#include <iostream>

#include "accumulate_func.hpp"
#include "blas.hpp"

template <typename DataType>
void accumulate_matrix_cpu(void * buffers[], void * cl_args) {
  //std::cout << "ACCUM CPU" << std::endl;
  int m = STARPU_MATRIX_GET_NX(buffers[0]);
  int n = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld_dst = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType * dst = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  int ld_src = STARPU_MATRIX_GET_LD(buffers[1]);
  DataType * src = (DataType*)STARPU_MATRIX_GET_PTR(buffers[1]);
  for(int j = 0; j < n; j++) {
    for(int i = 0; i < m; i++) {
      dst[i + ld_dst * j] = dst[i + ld_dst * j] + src[i + ld_src * j];
    }
  }
}

template void accumulate_matrix_cpu<float>(void * buffers[], void * cl_args);
template void accumulate_matrix_cpu<double>(void * buffers[], void * cl_args);

#ifdef USE_CUDA
template <typename DataType>
void accumulate_matrix_cuda(void * buffers[], void * cl_args) {
  //std::cout << "ACCUM CUDA" << std::endl;
  int m = STARPU_MATRIX_GET_NX(buffers[0]);
  int n = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld_dst = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType * dst = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  int ld_src = STARPU_MATRIX_GET_LD(buffers[1]);
  DataType * src = (DataType*)STARPU_MATRIX_GET_PTR(buffers[1]);
  cublas<DataType>::geam(starpu_cublas_get_local_handle(), 'N', 'N', m, n, (DataType)1, dst, ld_dst, (DataType)1, src, ld_src, dst, ld_dst);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

template void accumulate_matrix_cuda<float>(void * buffers[], void * cl_args);
template void accumulate_matrix_cuda<double>(void * buffers[], void * cl_args);
#endif
