#include "algorithm.hpp"

#include <starpu.h>
#include <vector>
#include <cstdio>

#include "tensor_fill_func.hpp"
#include "../util/helper.hpp"


template <typename DataType>
void tensor_fill_cpu_func(void *buffers[], void *cl_args) {
	DataType e;
	starpu_codelet_unpack_args(cl_args, &e);

	auto T = as_tensor<DataType>(buffers[0]);

	//printf("FILL CPU %p\n", T.ptr);

	std::vector<std::vector<u32>> lin_idx_vec(1);
	u32 cont_len = compute_contiguous(1, T.ndim, T.nn, &T.ldn, lin_idx_vec);

	// Update each contiguous part separately
	for(auto& lin_idx : lin_idx_vec[0]) {
		for(u32 i = 0; i < cont_len; i++) {
			T.ptr[lin_idx + i] = e;
		}
	}
}

template <typename DataType>
__global__ void tensor_fill_kernel(DataType *start_idx, u32 len, DataType val) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < len) {
		start_idx[i] = val;
	}
}

template <typename DataType>
void tensor_fill_cuda_func(void *buffers[], void *cl_args) {
	DataType e;
	starpu_codelet_unpack_args(cl_args, &e);

	auto T = as_tensor<DataType>(buffers[0]);

	//printf("FILL CUDA %p\n", T.ptr);

	for(size_t i = 0; i < T.ndim; i++) {
		//printf("LD / DIM %d %d\n", T.ldn[i], T.nn[i]);
	}
	
	std::vector<std::vector<u32>> lin_idx_vec(1);
	u32 cont_len = compute_contiguous(1, T.ndim, T.nn, &T.ldn, lin_idx_vec);

	// Update each contiguous part separately
	for(auto& lin_idx : lin_idx_vec[0]) {
		dim3 threadsPerBlock(256);
		dim3 numBlocks(ceilDiv(cont_len, threadsPerBlock.x));

		tensor_fill_kernel<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(&T.ptr[lin_idx], cont_len, e);

		//printf("lin_idx = %d, threads=%d, blocks=%d\n", lin_idx, threadsPerBlock.x, numBlocks.x);
		cudaError_t status = cudaGetLastError();

		if (status != cudaSuccess) {
			STARPU_CUDA_REPORT_ERROR(status);
		}
	}
}

template void tensor_fill_cpu_func<float>(void *buffers[], void *cl_args);
template void tensor_fill_cpu_func<double>(void *buffers[], void *cl_args);
template void tensor_fill_cuda_func<float>(void *buffers[], void *cl_args);
template void tensor_fill_cuda_func<double>(void *buffers[], void *cl_args);
