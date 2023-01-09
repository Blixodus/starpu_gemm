#include <starpu.h>
#ifdef USE_CUDA
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#endif

#include "blas.hpp"
#include "../util/lapackAPI.hpp"

template <typename DataType>
void blas<DataType>::gemm(
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
) {
	auto m_cast = checked_cast<int>(m);
	auto n_cast = checked_cast<int>(n);
	auto k_cast = checked_cast<int>(k);

	auto lda_cast = checked_cast<int>(lda);
	auto ldb_cast = checked_cast<int>(ldb);
	auto ldc_cast = checked_cast<int>(ldc);

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
template <typename DataType>
void cublas<DataType>::gemm(
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
) {
	cublasStatus_t stat;

	auto m_cast = checked_cast<int>(m);
	auto n_cast = checked_cast<int>(n);
	auto k_cast = checked_cast<int>(k);

	auto lda_cast = checked_cast<int>(lda);
	auto ldb_cast = checked_cast<int>(ldb);
	auto ldc_cast = checked_cast<int>(ldc);

	if constexpr (std::is_same_v<DataType, float>) {
		stat = cublasSgemm(
			handle, convertToCublas(transa), convertToCublas(transb), m_cast, n_cast, k_cast, &alpha, A, lda_cast, B,
			ldb_cast, &beta, C, ldc_cast
		);
	} else if constexpr (std::is_same_v<DataType, double>) {
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
) {
	cublasStatus_t stat;

	auto m_cast = checked_cast<int>(m);
	auto n_cast = checked_cast<int>(n);

	auto lda_cast = checked_cast<int>(lda);
	auto ldb_cast = checked_cast<int>(ldb);
	auto ldc_cast = checked_cast<int>(ldc);

	if constexpr (std::is_same_v<DataType, float>) {
		stat = cublasSgeam(
			handle, convertToCublas(transa), convertToCublas(transb), m_cast, n_cast, &alpha, A, lda_cast, &beta, B,
			ldb_cast, C, ldc_cast
		);
	} else if constexpr (std::is_same_v<DataType, double>) {
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
