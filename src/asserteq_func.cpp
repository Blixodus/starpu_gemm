#include <starpu.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif
#include <iostream>
#include "asserteq_func.hpp"

template <typename DataType>
void asserteq_cpu_func(void * buffers[], void * cl_args) {
  //std::cerr << "ASSERTEQ CPU\n";
  DataType val;
  starpu_codelet_unpack_args(cl_args, &val);
  int rows = STARPU_MATRIX_GET_NX(buffers[0]);
  int cols = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType *mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  for(int i = 0; i < cols; i++) {
    for(int j = 0; j < rows; j++) {
      if(mat[i*ld + j] != val) { printf("Wrong ! at (%d, %d) found %f expected %f\n", i, j, mat[i*ld + j], val); };
    }
  }
}

template void asserteq_cpu_func<float>(void * buffers[], void * cl_args);
template void asserteq_cpu_func<double>(void * buffers[], void * cl_args);

#ifdef USE_CUDA
template <typename DataType>
__global__ void asserteq_kernel(DataType *mat, int rows, int cols, int ld, DataType val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < rows && j < cols) {
    if(mat[j*ld + i] != val) { printf("Wrong ! at (%d, %d) found %f expected %f\n", i, j, mat[i*ld + j], val); }
  }
}

template <typename DataType>
void asserteq_cuda_func(void * buffers[], void * cl_args) {
  //std::cerr << "ASSERTEQ CUDA\n";
  DataType val;
  starpu_codelet_unpack_args(cl_args, &val);
  int rows = STARPU_MATRIX_GET_NX(buffers[0]);
  int cols = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType *mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  dim3 threadsPerBlock(32,32);
  dim3 numBlocks((rows+threadsPerBlock.x-1)/threadsPerBlock.x, (cols+threadsPerBlock.y-1)/threadsPerBlock.y);
  asserteq_kernel<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(mat, rows, cols, ld, val);
}

template void asserteq_cuda_func<float>(void * buffers[], void * cl_args);
template void asserteq_cuda_func<double>(void * buffers[], void * cl_args);
#endif
