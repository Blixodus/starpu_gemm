#include <starpu.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif
#include <cstdio>
#include "print_func.hpp"

template <typename DataType>
void print_cpu_func(void * buffers[], void * cl_args) {
  int rows = STARPU_MATRIX_GET_NX(buffers[0]);
  int cols = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType *mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  for(int i = 0; i < cols; i++) {
    for(int j = 0; j < rows; j++) {
      printf("%d, %d, %d, %f\n", i, j, ld, mat[i*ld + j]);
    }
  }
}

template void print_cpu_func<float>(void * buffers[], void * cl_args);
template void print_cpu_func<double>(void * buffers[], void * cl_args);

#ifdef USE_CUDA
template <typename DataType>
__global__ void print_kernel(DataType *mat, int rows, int cols, int ld) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < rows && j < cols) {
    printf("%d, %d, %d, %f\n", j, i, ld, mat[j*ld + i]);
  }
}

template <typename DataType>
void print_cuda_func(void * buffers[], void * cl_args) {
  int rows = STARPU_MATRIX_GET_NX(buffers[0]);
  int cols = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType *mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  dim3 threadsPerBlock(32,32);
  dim3 numBlocks((rows+threadsPerBlock.x-1)/threadsPerBlock.x, (cols+threadsPerBlock.y-1)/threadsPerBlock.y);
  print_kernel<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(mat, rows, cols, ld);
}

template void print_cuda_func<float>(void * buffers[], void * cl_args);
template void print_cuda_func<double>(void * buffers[], void * cl_args);
#endif
