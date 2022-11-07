#pragma once

#include "helper.hpp"

template <typename DataType>
struct blas {
	static void gemm(
		char transA,
		char transB,
		u32 m,
		u32 n,
		u32 k,
		DataType alpha,
		DataType* A,
		u32 lda,
		DataType* B,
		u32 ldb,
		DataType beta,
		DataType* C,
		u32 ldc
	);
};

#ifdef USE_CUDA
template <typename DataType>
struct cublas {
	static void gemm(
		cublasHandle_t handle,
		char transa,
		char transb,
		u32 m,
		u32 n,
		u32 k,
		const DataType alpha,
		const DataType* A,
		u32 lda,
		const DataType* B,
		u32 ldb,
		const DataType beta,
		DataType* C,
		u32 ldc
	);

	static void geam(
		cublasHandle_t handle,
		char transa,
		char transb,
		u32 m,
		u32 n,
		const DataType alpha,
		const DataType* A,
		u32 lda,
		const DataType beta,
		const DataType* B,
		u32 ldb,
		DataType* C,
		u32 ldc
	);
};
#endif
