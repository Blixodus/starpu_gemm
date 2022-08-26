#include <starpu.h>
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#include <iostream>
#include <exception>

#include "blas.hpp"

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

template <typename DataType>
void cublasgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const DataType alpha, const DataType *A, int lda, const DataType *B, int ldb, const DataType beta, DataType* C, int ldc) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<DataType, float>) {
    stat = cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }
  if constexpr(std::is_same_v<DataType, double>) {
    stat = cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }
  if(stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS GEMM failed" << std::endl;
  }
}

template <typename DataType>
void cublasgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const DataType alpha, const DataType *A, int lda, const DataType beta, const DataType *B, int ldb, DataType* C, int ldc) {
  cublasStatus_t stat;
  if constexpr(std::is_same_v<DataType, float>) {
    stat = cublasSgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
  }
  if constexpr(std::is_same_v<DataType, double>) {
    stat = cublasDgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
  }
  if(stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS GEAM failed" << std::endl;
  }
}

template void gemm<float>(char transA, char transB, int m, int n, int k, float alpha, float * A, int lda, float * B, int ldb, float beta, float * C, int ldc);
template void gemm<double>(char transA, char transB, int m, int n, int k, double alpha, double * A, int lda, double * B, int ldb, double beta, double * C, int ldc);
template void cublasgemm<float>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float alpha, const float *A, int lda, const float *B, int ldb, const float beta, float* C, int ldc);
template void cublasgemm<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double alpha, const double *A, int lda, const double *B, int ldb, const double beta, double* C, int ldc);
template void cublasgeam<float>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float alpha, const float *A, int lda, const float beta, const float *B, int ldb, float* C, int ldc);
template void cublasgeam<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double alpha, const double *A, int lda, const double beta, const double *B, int ldb, double* C, int ldc);