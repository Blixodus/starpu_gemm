#include <starpu.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif
#include <iostream>
#include "fill_func.hpp"

template <typename DataType>
void fill_cpu_func(void * buffers[], void * cl_args) {
  std::cerr << "FILL CPU\n";
  DataType e;
  starpu_codelet_unpack_args(cl_args, &e);
  int rows = STARPU_MATRIX_GET_NX(buffers[0]);
  int cols = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType *mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  for(int i = 0; i < cols; i++) {
    for(int j = 0; j < rows; j++) {
      mat[i*ld + j] = e;
    }
  }
}

template void fill_cpu_func<float>(void * buffers[], void * cl_args);
template void fill_cpu_func<double>(void * buffers[], void * cl_args);

#ifdef USE_CUDA
template <typename DataType>
__global__ void fill_kernel(DataType *mat, int rows, int cols, int ld, DataType val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < rows && j < cols) {
    mat[j*ld + i] = val;
  }
}

template <typename DataType>
void fill_cuda_func(void * buffers[], void * cl_args) {
  std::cerr << "FILL CUDA\n";
  DataType e;
  starpu_codelet_unpack_args(cl_args, &e);
  int rows = STARPU_MATRIX_GET_NX(buffers[0]);
  int cols = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType *mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  dim3 threadsPerBlock(32,32);
  dim3 numBlocks((rows+threadsPerBlock.x-1)/threadsPerBlock.x, (cols+threadsPerBlock.y-1)/threadsPerBlock.y);
  fill_kernel<<<numBlocks, threadsPerBlock>>>(mat, rows, cols, ld, e);
}

template void fill_cuda_func<float>(void * buffers[], void * cl_args);
template void fill_cuda_func<double>(void * buffers[], void * cl_args);
#endif
