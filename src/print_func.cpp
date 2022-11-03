#include <starpu.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif
#include <cstdio>
#include "print_func.hpp"

template <typename DataType>
void print_cpu_func(void * buffers[], void * cl_args) {
  char c;
  int row, col;
  size_t bs;
  starpu_codelet_unpack_args(cl_args, &c, &row, &col, &bs);
  int rows = STARPU_MATRIX_GET_NX(buffers[0]);
  int cols = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType *mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  for(int i = 0; i < cols; i++) {
    for(int j = 0; j < rows; j++) {
      printf("%c : %d, %d, %d, %f\n", c, i+row*bs, j+col*bs, ld, mat[i*ld + j]);
    }
  }
}

template void print_cpu_func<float>(void * buffers[], void * cl_args);
template void print_cpu_func<double>(void * buffers[], void * cl_args);

#ifdef USE_CUDA
template <typename DataType>
__global__ void print_kernel(DataType *mat, int rows, int cols, int ld, char c, int row, int col, int bs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < rows && j < cols) {
    printf("%c : %d, %d, %d, %f\n", c, j+col*bs, i+row*bs, ld, mat[j*ld + i]);
  }
}

template <typename DataType>
void print_cuda_func(void * buffers[], void * cl_args) {
  char c;
  int row, col;
  size_t bs;
  starpu_codelet_unpack_args(cl_args, &c, &row, &col, &bs);
  int rows = STARPU_MATRIX_GET_NX(buffers[0]);
  int cols = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType *mat = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  dim3 threadsPerBlock(32,32);
  dim3 numBlocks((rows+threadsPerBlock.x-1)/threadsPerBlock.x, (cols+threadsPerBlock.y-1)/threadsPerBlock.y);
  print_kernel<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(mat, rows, cols, ld, c, row, col, bs);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

template void print_cuda_func<float>(void * buffers[], void * cl_args);
template void print_cuda_func<double>(void * buffers[], void * cl_args);
#endif
