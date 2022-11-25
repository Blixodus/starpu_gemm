#include <starpu.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif
#include <iostream>
#include "asserteq_func.hpp"
#include "../util/helper.hpp"

#include <cmath>

template <typename DataType>
void asserteq_cpu_func(void* buffers[], void* cl_args) {
	DataType val;
	starpu_codelet_unpack_args(cl_args, &val);

	auto M = as_matrix<DataType>(buffers[0]);

	for (u32 i = 0; i < M.cols; i++) {
		for (u32 j = 0; j < M.rows; j++) {
			if (fabs(M.ptr[i * M.ld + j] - val) > 1e-6) {
				printf("Wrong at (%d, %d) found %f expected %f !\n", i, j, M.ptr[i * M.ld + j], val);
			};
		}
	}
}

template void asserteq_cpu_func<float>(void* buffers[], void* cl_args);
template void asserteq_cpu_func<double>(void* buffers[], void* cl_args);

#ifdef USE_CUDA
template <typename DataType>
__global__ void asserteq_kernel(DataType* mat, u32 rows, u32 cols, u32 ld, DataType val) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < rows) && (j < cols)) {
		if (mat[j * ld + i] != val) {
			printf("Wrong at (%d, %d) found %f expected %f !\n", i, j, mat[i * ld + j], val);
		}
	}
}

template <typename DataType>
void asserteq_cuda_func(void* buffers[], void* cl_args) {
	DataType val;
	starpu_codelet_unpack_args(cl_args, &val);

	auto M = as_matrix<DataType>(buffers[0]);

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceilDiv(M.rows, threadsPerBlock.x), ceilDiv(M.cols, threadsPerBlock.y));
	asserteq_kernel<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(M.ptr, M.rows, M.cols, M.ld, val);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) { printf("[CUDA ERROR] %s\n", cudaGetErrorString(error)); }
}

template void asserteq_cuda_func<float>(void* buffers[], void* cl_args);
template void asserteq_cuda_func<double>(void* buffers[], void* cl_args);
#endif
