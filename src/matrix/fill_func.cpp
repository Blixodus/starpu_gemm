#include <starpu.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif
#include "fill_func.hpp"
#include "../util/helper.hpp"


template <typename DataType>
void fill_cpu_func(void* buffers[], void* cl_args) {
	DataType e;
	starpu_codelet_unpack_args(cl_args, &e);

	auto M = as_matrix<DataType>(buffers[0]);

	for (u32 i = 0; i < M.cols; i++) {
		for (u32 j = 0; j < M.rows; j++) {
			M.ptr[i * M.ld + j] = e;
		}
	}
}

template void fill_cpu_func<float>(void* buffers[], void* cl_args);
template void fill_cpu_func<double>(void* buffers[], void* cl_args);

#ifdef USE_CUDA
template <typename DataType>
__global__ void fill_kernel(DataType* mat, u32 rows, u32 cols, u32 ld, DataType val) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < rows) && (j < cols)) {
		mat[j * ld + i] = val;
	}
}

template <typename DataType>
void fill_cuda_func(void* buffers[], void* cl_args) {
	DataType e;
	starpu_codelet_unpack_args(cl_args, &e);

	auto M = as_matrix<DataType>(buffers[0]);

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceilDiv(M.rows, threadsPerBlock.x), ceilDiv(M.cols, threadsPerBlock.y));

	fill_kernel<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(M.ptr, M.rows, M.cols, M.ld, e);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) { printf("[CUDA ERROR] %s\n", cudaGetErrorString(error)); }
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

template void fill_cuda_func<float>(void* buffers[], void* cl_args);
template void fill_cuda_func<double>(void* buffers[], void* cl_args);
#endif
