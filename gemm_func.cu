#include <starpu.h>
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#include <exception>
#include "gemm_func.h"

extern "C" void sgemm_(char *transA, char *transB, int *m, int *n, int *k, float *alpha, float *A, int *lda, float *B, int *ldb, float *beta, float *C, int *ldc);
extern "C" void dgemm_(char *transA, char *transB, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);

template <typename DataType>
void gemm(char transA, char transB, int m, int n, int k, DataType alpha, DataType * A, int lda, DataType * B, int ldb, DataType beta, DataType * C, int ldc) {
  if constexpr(std::is_same_v<DataType, float>) {
    sgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }
  if constexpr(std::is_same_v<DataType, double>) {
    dgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }
}

void gemm_cpu_func(void * buffers[], void * cl_args) {
  char transA, transB;
  float alpha, beta;
  starpu_codelet_unpack_args(cl_args, &transA, &transB, &alpha, &beta);
  int m = (transA == 'N') ? STARPU_MATRIX_GET_NX(buffers[0]) : STARPU_MATRIX_GET_NY(buffers[0]);
  int n = (transB == 'N') ? STARPU_MATRIX_GET_NY(buffers[1]) : STARPU_MATRIX_GET_NX(buffers[1]);
  int k = (transA == 'N') ? STARPU_MATRIX_GET_NY(buffers[0]) : STARPU_MATRIX_GET_NX(buffers[0]);
  int ld_A = STARPU_MATRIX_GET_LD(buffers[0]);
  float * A = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
  int ld_B = STARPU_MATRIX_GET_LD(buffers[1]);
  float * B = (float*)STARPU_MATRIX_GET_PTR(buffers[1]);
  int ld_C = STARPU_MATRIX_GET_LD(buffers[2]);
  float * C = (float*)STARPU_MATRIX_GET_PTR(buffers[2]);
  gemm(transA, transB, m, n, k, alpha, A, ld_A, B, ld_B, beta, C, ld_C);
}

cublasOperation_t convertToCublas(char trans) {
  switch(trans) {
  case 'N':
    return CUBLAS_OP_N;
  case 'T':
    return CUBLAS_OP_T;
  case 'C':
    return CUBLAS_OP_C;
  default:
    throw std::exception();
  }
}

void gemm_cuda_func(void * buffers[], void * cl_args) {
  char transA, transB;
  float alpha, beta;
  starpu_codelet_unpack_args(cl_args, &transA, &transB, &alpha, &beta);
  int m = (transA == 'N') ? STARPU_MATRIX_GET_NX(buffers[0]) : STARPU_MATRIX_GET_NY(buffers[0]);
  int n = (transB == 'N') ? STARPU_MATRIX_GET_NY(buffers[1]) : STARPU_MATRIX_GET_NX(buffers[1]);
  int k = (transA == 'N') ? STARPU_MATRIX_GET_NY(buffers[0]) : STARPU_MATRIX_GET_NX(buffers[0]);
  int ld_A = STARPU_MATRIX_GET_LD(buffers[0]);
  float * A = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
  int ld_B = STARPU_MATRIX_GET_LD(buffers[1]);
  float * B = (float*)STARPU_MATRIX_GET_PTR(buffers[1]);
  int ld_C = STARPU_MATRIX_GET_LD(buffers[2]);
  float * C = (float*)STARPU_MATRIX_GET_PTR(buffers[2]);
  
  cublasStatus_t stat = cublasSgemm(starpu_cublas_get_local_handle(), convertToCublas(transA), convertToCublas(transB), m, n, k, &alpha, A, ld_A, B, ld_B, &beta, C, ld_C);
  if(stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS GEMM failed\n");
  }
}
