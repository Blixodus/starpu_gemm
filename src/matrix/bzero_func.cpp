#include <starpu.h>
#ifdef USE_CUDA
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#endif
#include <iostream>

#include "blas.hpp"
#include "bzero_func.hpp"
#include "../util/helper.hpp"

template <typename DataType>
void bzero_matrix_cpu(void* buffers[], void* cl_args) {
	// std::cout << "BZERO CPU" << std::endl;
	auto M = as_matrix<DataType>(buffers[0]);

	for (u32 j = 0; j < M.cols; j++) {
		memset(&M.ptr[M.ld * j], 0, M.rows * sizeof(DataType));
	}
}

template void bzero_matrix_cpu<float>(void* buffers[], void* cl_args);
template void bzero_matrix_cpu<double>(void* buffers[], void* cl_args);

#ifdef USE_CUDA
template <typename DataType>
void bzero_matrix_cuda(void* buffers[], void* cl_args) {
	// TODO: replace with cudaMemset2D
	auto M = as_matrix<DataType>(buffers[0]);

	DataType alpha{0}, beta{0};

	cublas<DataType>::geam(
		starpu_cublas_get_local_handle(), 'N', 'N', M.rows, M.cols, alpha, M.ptr, M.ld, beta, M.ptr, M.ld, M.ptr, M.ld
	);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

template void bzero_matrix_cuda<float>(void* buffers[], void* cl_args);
template void bzero_matrix_cuda<double>(void* buffers[], void* cl_args);
#endif
