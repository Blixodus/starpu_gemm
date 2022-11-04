#include <starpu.h>
#ifdef USE_CUDA
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#endif
#include <cassert>
#include <exception>
#include <iostream>

#include "blas.hpp"
#include "gemm_func.hpp"
#include "helper.hpp"

template <typename DataType>
void gemm_cpu_func(void* buffers[], void* cl_args) {
	char transA, transB;
	DataType alpha, beta;
	starpu_codelet_unpack_args(cl_args, &transA, &transB, &alpha, &beta);

	auto A = as_matrix<DataType>(buffers[0]);
	auto B = as_matrix<DataType>(buffers[1]);
	auto C = as_matrix<DataType>(buffers[2]);

	auto m = (transA == 'N') ? A.rows : A.cols;
	auto n = (transB == 'N') ? B.cols : B.rows;
	auto k = (transA == 'N') ? A.cols : A.rows;

	blas<DataType>::gemm(transA, transB, m, n, k, alpha, A.ptr, A.ld, B.ptr, B.ld, beta, C.ptr, C.ld);
}

template void gemm_cpu_func<float>(void* buffers[], void* cl_args);
template void gemm_cpu_func<double>(void* buffers[], void* cl_args);

#ifdef USE_CUDA
template <typename DataType>
void gemm_cuda_func(void* buffers[], void* cl_args) {
	char transA, transB;
	DataType alpha, beta;
	starpu_codelet_unpack_args(cl_args, &transA, &transB, &alpha, &beta);

	auto A = as_matrix<DataType>(buffers[0]);
	auto B = as_matrix<DataType>(buffers[1]);
	auto C = as_matrix<DataType>(buffers[2]);

	auto m = (transA == 'N') ? A.rows : A.cols;
	auto n = (transB == 'N') ? B.cols : B.rows;
	auto k = (transA == 'N') ? A.cols : A.rows;

	cublas<DataType>::gemm(
		starpu_cublas_get_local_handle(), transA, transB, m, n, k, alpha, A.ptr, A.ld, B.ptr, B.ld, beta, C.ptr, C.ld
	);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

template void gemm_cuda_func<float>(void* buffers[], void* cl_args);
template void gemm_cuda_func<double>(void* buffers[], void* cl_args);
#endif
