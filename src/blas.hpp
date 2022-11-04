#pragma once

template <typename DataType>
void gemm(char transA, char transB, int m, int n, int k, DataType alpha, DataType * A, int lda, DataType * B, int ldb, DataType beta, DataType * C, int ldc);

#ifdef USE_CUDA
cublasOperation_t convertToCublas(char trans);

template <typename DataType>
void cublasgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const DataType alpha, const DataType *A, int lda, const DataType *B, int ldb, const DataType beta, DataType* C, int ldc);

template <typename DataType>
void cublasgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const DataType alpha, const DataType *A, int lda, const DataType beta, const DataType *B, int ldb, DataType* C, int ldc);
#endif
