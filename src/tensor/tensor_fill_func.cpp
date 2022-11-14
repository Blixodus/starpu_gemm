#include <starpu.h>
#include <vector>
#include <cstdio>
#include "tensor_fill_func.hpp"
#include "algorithm.hpp"

template <typename DataType>
void tensor_fill_cpu_func(void *buffers[], void *cl_args) {
  DataType e;
  starpu_codelet_unpack_args(cl_args, &e);
  DataType *ten = (DataType*)STARPU_NDIM_GET_PTR(buffers[0]);
  unsigned int *dim_size = STARPU_NDIM_GET_NN(buffers[0]);
  unsigned int *ld = STARPU_NDIM_GET_LDN(buffers[0]);
  size_t ndim = STARPU_NDIM_GET_NDIM(buffers[0]);
  
  unsigned int cont_len;
  std::vector<std::vector<unsigned int>> lin_idx_vec(1);
  compute_contiguous(1, ndim, dim_size, &ld, cont_len, lin_idx_vec);
  // Update each contiguous part separately
  for(auto & lin_idx : lin_idx_vec[0]) {
    for(unsigned int i = 0; i < cont_len; i++) {
      ten[lin_idx + i] = e;
    }
  }
}

template <typename DataType>
__global__ void tensor_fill_kernel(DataType *start_idx, unsigned int len, DataType val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) {
    start_idx[i] = val;
  }
}

template <typename DataType>
void tensor_fill_cuda_func(void *buffers[], void *cl_args) {
  DataType e;
  starpu_codelet_unpack_args(cl_args, &e);
  DataType *ten = (DataType*)STARPU_NDIM_GET_PTR(buffers[0]);
  unsigned int *dim_size = STARPU_NDIM_GET_NN(buffers[0]);
  unsigned int *ld = STARPU_NDIM_GET_LDN(buffers[0]);
  size_t ndim = STARPU_NDIM_GET_NDIM(buffers[0]);
  for(int i = 0; i < ndim; i++) {
    printf("LD / DIM %d %d\n", ld[i], dim_size[i]);
  }
  
  unsigned int cont_len;
  std::vector<std::vector<unsigned int>> lin_idx_vec(1);
  compute_contiguous(1, ndim, dim_size, &ld, cont_len, lin_idx_vec);
  printf("FILL CUDA %p %ld %d %lf\n", ten, ten, cont_len, e);
  // Update each contiguous part separately
  for(auto & lin_idx : lin_idx_vec[0]) {
    dim3 threadsPerBlock(256);
    dim3 numBlocks((cont_len+threadsPerBlock.x-1)/threadsPerBlock.x);
    tensor_fill_kernel<<<numBlocks, threadsPerBlock>>>(&ten[lin_idx], cont_len, e);
    printf("lin_idx = %d, threads=%d, blocks=%d\n", lin_idx, threadsPerBlock.x, numBlocks.x);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
  }
}

template void tensor_fill_cpu_func<float>(void *buffers[], void *cl_args);
template void tensor_fill_cpu_func<double>(void *buffers[], void *cl_args);
template void tensor_fill_cuda_func<float>(void *buffers[], void *cl_args);
template void tensor_fill_cuda_func<double>(void *buffers[], void *cl_args);


template <typename DataType>
__global__ void tensor_print_kernel(DataType *start_idx, unsigned int len, int block_idx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) {
    printf("%d, %lf\n", block_idx+i, start_idx[i]);
  }
}

template <typename DataType>
void tensor_print_cuda_func(void *buffers[], void *cl_args) {
  printf("PRINT CUDA\n");
  int block_idx;
  starpu_codelet_unpack_args(cl_args, &block_idx);
  DataType *ten = (DataType*)STARPU_NDIM_GET_PTR(buffers[0]);
  unsigned int *dim_size = STARPU_NDIM_GET_NN(buffers[0]);
  unsigned int *ld = STARPU_NDIM_GET_LDN(buffers[0]);
  size_t ndim = STARPU_NDIM_GET_NDIM(buffers[0]);
  for(int i = 0; i < ndim; i++) {
    printf("LD / DIM %d %d\n", ld[i], dim_size[i]);
  }
  
  unsigned int cont_len;
  std::vector<std::vector<unsigned int>> lin_idx_vec(1);
  compute_contiguous(1, ndim, dim_size, &ld, cont_len, lin_idx_vec);
  printf("PRINT CUDA %p %ld %d\n", ten, ten, cont_len);
  // Update each contiguous part separately
  for(auto & lin_idx : lin_idx_vec[0]) {
    dim3 threadsPerBlock(256);
    dim3 numBlocks((cont_len+threadsPerBlock.x-1)/threadsPerBlock.x);
    tensor_print_kernel<<<numBlocks, threadsPerBlock>>>(&ten[lin_idx], cont_len, block_idx);
    printf("lin_idx = %d, threads=%d, blocks=%d\n", lin_idx, threadsPerBlock.x, numBlocks.x);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
  }
}

template void tensor_print_cuda_func<float>(void *buffers[], void *cl_args);
template void tensor_print_cuda_func<double>(void *buffers[], void *cl_args);