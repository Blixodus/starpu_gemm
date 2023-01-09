#include "algorithm.hpp"

#include <starpu.h>
#include <vector>
#include <functional>
#include <cmath>

#include "tensor_asserteq_func.hpp"
#include "../util/helper.hpp"

template <typename DataType>
void tensor_asserteq_cpu_func(void *buffers[], void *cl_args) {
	DataType val;
	starpu_codelet_unpack_args(cl_args, &val);

	auto T = as_tensor<DataType>(buffers[0]);
	auto [cont_len, lin_idx_vec] = compute_contiguous(1, T.ndim, T.nn, &T.ldn);

	// Check each contiguous part separately
	for(auto& lin_idx : lin_idx_vec[0]) {
		for(u32 i = 0; i < cont_len; i++) {
			if(fabs(T.ptr[lin_idx + i] - val) > 1e-6) {
        printf("Wrong found %f expected %f !\n", T.ptr[lin_idx + i], val);
      }
		}
	}
}

template void tensor_asserteq_cpu_func<float>(void *buffers[], void *cl_args);
template void tensor_asserteq_cpu_func<double>(void *buffers[], void *cl_args);

#ifdef USE_CUDA
template <typename DataType>
__global__ void tensor_asserteq_kernel(DataType *start_idx, u32 len, DataType val) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < len) {
		if(fabs(start_idx[i] - val) > 1e-6) {
      printf("Wrong found %f expected %f !\n", start_idx[i], val);
    }
	}
}

template <typename DataType>
void tensor_asserteq_cuda_func(void *buffers[], void *cl_args) {
	DataType val;
	starpu_codelet_unpack_args(cl_args, &val);

	auto T = as_tensor<DataType>(buffers[0]);
	auto [cont_len, lin_idx_vec] = compute_contiguous(1, T.ndim, T.nn, &T.ldn);

	// Update each contiguous part separately
	for(auto& lin_idx : lin_idx_vec[0]) {
		dim3 threadsPerBlock(256);
		dim3 numBlocks(ceilDiv(cont_len, threadsPerBlock.x));

		tensor_asserteq_kernel<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(&T.ptr[lin_idx], cont_len, val);

		cudaError_t status = cudaGetLastError();
		if (status != cudaSuccess) {
			STARPU_CUDA_REPORT_ERROR(status);
		}
	}
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

template void tensor_asserteq_cuda_func<float>(void *buffers[], void *cl_args);
template void tensor_asserteq_cuda_func<double>(void *buffers[], void *cl_args);
#endif
