#include <vector>
#include <starpu.h>
#include "tensor_add_func.hpp"
#include "algorithm.hpp"

template <typename DataType>
void tensor_add_cpu_func(void *buffers[], void *cl_args) {
  DataType *A = (DataType*)STARPU_NDIM_GET_PTR(buffers[0]);
  DataType *B = (DataType*)STARPU_NDIM_GET_PTR(buffers[1]);
  DataType *C = (DataType*)STARPU_NDIM_GET_PTR(buffers[2]);
  unsigned int *dim_size = STARPU_NDIM_GET_NN(buffers[0]);
  unsigned int *ld_A = STARPU_NDIM_GET_LDN(buffers[0]);
  unsigned int *ld_B = STARPU_NDIM_GET_LDN(buffers[1]);
  unsigned int *ld_C = STARPU_NDIM_GET_LDN(buffers[2]);
  size_t ndim = STARPU_NDIM_GET_NDIM(buffers[0]);

  unsigned int cont_len = 1;
  std::vector<unsigned int*> ld = { ld_A, ld_B, ld_C };
  std::vector<std::vector<unsigned int>> lin_idx(3);
  compute_contiguous(3, ndim, dim_size, &ld[0], cont_len, lin_idx);
  // Update each common contiguous part separately
  for(unsigned int e = 0; e < lin_idx[0].size(); e++) {
    for(unsigned int i = 0; i < cont_len; i++) {
      C[lin_idx[2][e] + i] = A[lin_idx[0][e] + i] + B[lin_idx[1][e] + i];
    }
  }
}

template <typename DataType>
void tensor_add_cuda_func(void *buffers[], void *cl_args) {

}

template void tensor_add_cpu_func<float>(void *buffers[], void *cl_args);
template void tensor_add_cpu_func<double>(void *buffers[], void *cl_args);
template void tensor_add_cuda_func<float>(void *buffers[], void *cl_args);
template void tensor_add_cuda_func<double>(void *buffers[], void *cl_args);