#pragma once

template <typename DataType>
struct blas {
  static void gemm(char transA, char transB, uint32_t m, uint32_t n, uint32_t k, DataType alpha, DataType * A, uint32_t lda, DataType * B, uint32_t ldb, DataType beta, DataType * C, uint32_t ldc);
};

#ifdef USE_CUDA
template <typename DataType>
struct cublas {
  static void gemm(cublasHandle_t handle, char transa, char transb, uint32_t m, uint32_t n, uint32_t k, const DataType alpha, const DataType *A, uint32_t lda, const DataType *B, uint32_t ldb, const DataType beta, DataType* C, uint32_t ldc);

  static void geam(cublasHandle_t handle, char transa, char transb, uint32_t m, uint32_t n, const DataType alpha, const DataType *A, uint32_t lda, const DataType beta, const DataType *B, uint32_t ldb, DataType* C, uint32_t ldc);
};
#endif
