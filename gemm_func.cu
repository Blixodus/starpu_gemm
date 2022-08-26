#include <starpu.h>
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#include <exception>
#include <iostream>

#include "gemm_func.hpp"
#include "blas.hpp"

template <typename DataType>
void gemm_cpu_func(void * buffers[], void * cl_args) {
  char transA, transB;
  DataType alpha, beta;
  starpu_codelet_unpack_args(cl_args, &transA, &transB, &alpha, &beta);
  int m = (transA == 'N') ? STARPU_MATRIX_GET_NX(buffers[0]) : STARPU_MATRIX_GET_NY(buffers[0]);
  int n = (transB == 'N') ? STARPU_MATRIX_GET_NY(buffers[1]) : STARPU_MATRIX_GET_NX(buffers[1]);
  int k = (transA == 'N') ? STARPU_MATRIX_GET_NY(buffers[0]) : STARPU_MATRIX_GET_NX(buffers[0]);
  int ld_A = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType * A = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  int ld_B = STARPU_MATRIX_GET_LD(buffers[1]);
  DataType * B = (DataType*)STARPU_MATRIX_GET_PTR(buffers[1]);
  int ld_C = STARPU_MATRIX_GET_LD(buffers[2]);
  DataType * C = (DataType*)STARPU_MATRIX_GET_PTR(buffers[2]);
  gemm(transA, transB, m, n, k, alpha, A, ld_A, B, ld_B, beta, C, ld_C);
}

template <typename DataType>
void gemm_cuda_func(void * buffers[], void * cl_args) {
  char transA, transB;
  DataType alpha, beta;
  starpu_codelet_unpack_args(cl_args, &transA, &transB, &alpha, &beta);
  int m = (transA == 'N') ? STARPU_MATRIX_GET_NX(buffers[0]) : STARPU_MATRIX_GET_NY(buffers[0]);
  int n = (transB == 'N') ? STARPU_MATRIX_GET_NY(buffers[1]) : STARPU_MATRIX_GET_NX(buffers[1]);
  int k = (transA == 'N') ? STARPU_MATRIX_GET_NY(buffers[0]) : STARPU_MATRIX_GET_NX(buffers[0]);
  int ld_A = STARPU_MATRIX_GET_LD(buffers[0]);
  DataType * A = (DataType*)STARPU_MATRIX_GET_PTR(buffers[0]);
  int ld_B = STARPU_MATRIX_GET_LD(buffers[1]);
  DataType * B = (DataType*)STARPU_MATRIX_GET_PTR(buffers[1]);
  int ld_C = STARPU_MATRIX_GET_LD(buffers[2]);
  DataType * C = (DataType*)STARPU_MATRIX_GET_PTR(buffers[2]);
  cublasgemm(starpu_cublas_get_local_handle(), convertToCublas(transA), convertToCublas(transB), m, n, k, alpha, A, ld_A, B, ld_B, beta, C, ld_C);
}

template void gemm_cpu_func<float>(void *buffers[], void *cl_args);
template void gemm_cuda_func<float>(void *buffers[], void *cl_args);