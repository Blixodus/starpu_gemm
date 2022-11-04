#pragma once

template <typename DataType>
struct blas {
  static void gemm(char transA, char transB, int m, int n, int k, DataType alpha, DataType * A, int lda, DataType * B, int ldb, DataType beta, DataType * C, int ldc);
};

#ifdef USE_CUDA
template <typename DataType>
struct cublas {
  static void gemm(cublasHandle_t handle, char transa, char transb, int m, int n, int k, const DataType alpha, const DataType *A, int lda, const DataType *B, int ldb, const DataType beta, DataType* C, int ldc);

  static void geam(cublasHandle_t handle, char transa, char transb, int m, int n, const DataType alpha, const DataType *A, int lda, const DataType beta, const DataType *B, int ldb, DataType* C, int ldc);
};
#endif
