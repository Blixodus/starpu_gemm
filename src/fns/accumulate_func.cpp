#include <starpu.h>
#ifdef USE_CUDA
#include <starpu_cublas_v2.h>
#include "cublas_v2.h"
#endif
#include <iostream>

#include "accumulate_func.hpp"
#include "blas.hpp"
#include "../util/helper.hpp"

template <typename DataType>
void accumulate_matrix_cpu(void* buffers[], void* cl_args) {
	auto dst = as_matrix<DataType>(buffers[0]);
	auto src = as_matrix<DataType>(buffers[1]);

	for (u32 j = 0; j < dst.cols; j++) {
		for (u32 i = 0; i < dst.rows; i++) {
			dst.ptr[i + dst.ld * j] = dst.ptr[i + dst.ld * j] + src.ptr[i + src.ld * j];
		}
	}
}

template void accumulate_matrix_cpu<float>(void* buffers[], void* cl_args);
template void accumulate_matrix_cpu<double>(void* buffers[], void* cl_args);

#ifdef USE_CUDA
template <typename DataType>
void accumulate_matrix_cuda(void* buffers[], void* cl_args) {
	auto dst = as_matrix<DataType>(buffers[0]);
	auto src = as_matrix<DataType>(buffers[1]);

	DataType alpha{1}, beta{1};

	cublas<DataType>::geam(
		starpu_cublas_get_local_handle(), 'N', 'N', dst.rows, dst.cols, alpha, dst.ptr, dst.ld, beta, src.ptr, src.ld,
		dst.ptr, dst.ld
	);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

template void accumulate_matrix_cuda<float>(void* buffers[], void* cl_args);
template void accumulate_matrix_cuda<double>(void* buffers[], void* cl_args);
#endif
