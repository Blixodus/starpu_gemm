#include <starpu.h>
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"

extern "C" void bzero_matrix_cpu(void * buffers[], void * cl_args) {
  printf("BZERO CPU\n");
  int m = STARPU_MATRIX_GET_NX(buffers[0]);
  int n = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  float * mat = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
  for(int j = 0; j < n; j++) {
    memset(&mat[ld*j], 0, m*sizeof(float));
  }
}

extern "C" void bzero_matrix_cuda(void * buffers[], void * cl_args) {
  printf("BZERO CUDA\n");
  int m = STARPU_MATRIX_GET_NX(buffers[0]);
  int n = STARPU_MATRIX_GET_NY(buffers[0]);
  int ld = STARPU_MATRIX_GET_LD(buffers[0]);
  float * mat = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
  //for(int j = 0; j < n; j++) {
  //  cudaMemset(&mat[ld*j], 0, m*sizeof(float));
  //}
  float alpha = 0, beta = 0;
  cublasStatus_t stat = cublasSgeam(starpu_cublas_get_local_handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, mat, ld, &beta, mat, ld, mat, ld);
  if(stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS GEMM failed\n");
  }
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}