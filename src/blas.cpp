#include <starpu.h>
#ifdef USE_CUDA
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#endif
#include <exception>
#include <iostream>

#include "blas.hpp"

extern "C" void sgemm_(
	char* transA,
	char* transB,
	int* m,
	int* n,
	int* k,
	float* alpha,
	float* A,
	int* lda,
	float* B,
	int* ldb,
	float* beta,
	float* C,
	int* ldc
);
extern "C" void dgemm_(
	char* transA,
	char* transB,
	int* m,
	int* n,
	int* k,
	double* alpha,
	double* A,
	int* lda,
	double* B,
	int* ldb,
	double* beta,
	double* C,
	int* ldc
);

template <typename DataType>
void blas<DataType>::gemm(
	char transA,
	char transB,
	uint32_t m,
	uint32_t n,
	uint32_t k,
	DataType alpha,
	DataType* A,
	uint32_t lda,
	DataType* B,
	uint32_t ldb,
	DataType beta,
	DataType* C,
	uint32_t ldc
) {
	auto m_cast = static_cast<int>(m);
	auto n_cast = static_cast<int>(n);
	auto k_cast = static_cast<int>(k);

	auto lda_cast = static_cast<int>(lda);
	auto ldb_cast = static_cast<int>(ldb);
	auto ldc_cast = static_cast<int>(ldc);

	if constexpr (std::is_same_v<DataType, float>) {
		sgemm_(&transA, &transB, &m_cast, &n_cast, &k_cast, &alpha, A, &lda_cast, B, &ldb_cast, &beta, C, &ldc_cast);
	}

	if constexpr (std::is_same_v<DataType, double>) {
		dgemm_(&transA, &transB, &m_cast, &n_cast, &k_cast, &alpha, A, &lda_cast, B, &ldb_cast, &beta, C, &ldc_cast);
	}
}

template struct blas<float>;
template struct blas<double>;

#ifdef USE_CUDA
cublasOperation_t convertToCublas(char trans) {
	switch (trans) {
		case 'N': return CUBLAS_OP_N;
		case 'T': return CUBLAS_OP_T;
		case 'C': return CUBLAS_OP_C;
		default: throw std::exception();
	}
}

template <typename DataType>
void cublas<DataType>::gemm(
	cublasHandle_t handle,
	char transa,
	char transb,
	uint32_t m,
	uint32_t n,
	uint32_t k,
	const DataType alpha,
	const DataType* A,
	uint32_t lda,
	const DataType* B,
	uint32_t ldb,
	const DataType beta,
	DataType* C,
	uint32_t ldc
) {
	cublasStatus_t stat;

	auto m_cast = static_cast<int>(m);
	auto n_cast = static_cast<int>(n);
	auto k_cast = static_cast<int>(k);

	auto lda_cast = static_cast<int>(lda);
	auto ldb_cast = static_cast<int>(ldb);
	auto ldc_cast = static_cast<int>(ldc);

	if constexpr (std::is_same_v<DataType, float>) {
		stat = cublasSgemm(
			handle, convertToCublas(transa), convertToCublas(transb), m_cast, n_cast, k_cast, &alpha, A, lda_cast, B,
			ldb_cast, &beta, C, ldc_cast
		);
	}

	if constexpr (std::is_same_v<DataType, double>) {
		stat = cublasDgemm(
			handle, convertToCublas(transa), convertToCublas(transb), m_cast, n_cast, k_cast, &alpha, A, lda_cast, B,
			ldb_cast, &beta, C, ldc_cast
		);
	}

	if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "CUBLAS GEMM failed" << std::endl;
	}
}

template <typename DataType>
void cublas<DataType>::geam(
	cublasHandle_t handle,
	char transa,
	char transb,
	uint32_t m,
	uint32_t n,
	const DataType alpha,
	const DataType* A,
	uint32_t lda,
	const DataType beta,
	const DataType* B,
	uint32_t ldb,
	DataType* C,
	uint32_t ldc
) {
	cublasStatus_t stat;

	auto m_cast = static_cast<int>(m);
	auto n_cast = static_cast<int>(n);

	auto lda_cast = static_cast<int>(lda);
	auto ldb_cast = static_cast<int>(ldb);
	auto ldc_cast = static_cast<int>(ldc);

	if constexpr (std::is_same_v<DataType, float>) {
		stat = cublasSgeam(
			handle, convertToCublas(transa), convertToCublas(transb), m_cast, n_cast, &alpha, A, lda_cast, &beta, B,
			ldb_cast, C, ldc_cast
		);
	}

	if constexpr (std::is_same_v<DataType, double>) {
		stat = cublasDgeam(
			handle, convertToCublas(transa), convertToCublas(transb), m_cast, n_cast, &alpha, A, lda_cast, &beta, B,
			ldb_cast, C, ldc_cast
		);
	}

	if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "CUBLAS GEAM failed" << std::endl;
	}
}

template struct cublas<float>;
template struct cublas<double>;
#endif
