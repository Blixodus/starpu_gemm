#include <starpu.h>
#ifdef USE_CUDA
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#endif
#include <iostream>
#include <exception>

#include "blas.hpp"

extern "C" void sgemm_(char *transA, char *transB, int *m, int *n, int *k, float *alpha, float *A, int *lda, float *B, int *ldb, float *beta, float *C, int *ldc);
extern "C" void dgemm_(char *transA, char *transB, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);

template <typename DataType>
void blas<DataType>::gemm(char transA, char transB, int m, int n, int k, DataType alpha, DataType * A, int lda, DataType * B, int ldb, DataType beta, DataType * C, int ldc) {
  if constexpr(std::is_same_v<DataType, float>) {
    sgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }
  if constexpr(std::is_same_v<DataType, double>) {
    dgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }
}

template struct blas<float>;
template struct blas<double>;

#ifdef USE_CUDA
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

template <typename DataType>
void cublas<DataType>::gemm(cublasHandle_t handle, char transa, char transb, int m, int n, int k, const DataType alpha, const DataType *A, int lda, const DataType *B, int ldb, const DataType beta, DataType* C, int ldc) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<DataType, float>) {
    stat = cublasSgemm(handle, convertToCublas(transa), convertToCublas(transb), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }
  if constexpr(std::is_same_v<DataType, double>) {
    stat = cublasDgemm(handle, convertToCublas(transa), convertToCublas(transb), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }
  if(stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS GEMM failed" << std::endl;
  }
}

template <typename DataType>
void cublas<DataType>::geam(cublasHandle_t handle, char transa, char transb, int m, int n, const DataType alpha, const DataType *A, int lda, const DataType beta, const DataType *B, int ldb, DataType* C, int ldc) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<DataType, float>) {
    stat = cublasSgeam(handle, convertToCublas(transa), convertToCublas(transb), m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
  }
  if constexpr(std::is_same_v<DataType, double>) {
    stat = cublasDgeam(handle, convertToCublas(transa), convertToCublas(transb), m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
  }
  if(stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS GEAM failed" << std::endl;
  }
}

template struct cublas<float>;
template struct cublas<double>;
#endif
