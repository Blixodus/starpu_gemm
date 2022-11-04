#include <starpu.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif
#include <cstdio>
#include "helper.hpp"
#include "print_func.hpp"

template <typename DataType>
void print_cpu_func(void* buffers[], void* cl_args) {
	char c;
	uint32_t row, col, bs;
	starpu_codelet_unpack_args(cl_args, &c, &row, &col, &bs);

	auto M = as_matrix<DataType>(buffers[0]);

	for (uint32_t i = 0; i < M.cols; i++) {
		for (uint32_t j = 0; j < M.rows; j++) {
			printf("%c : %u, %u, %u, %f\n", c, i + row * bs, j + col * bs, M.ld, M.ptr[i * M.ld + j]);
		}
	}
}

template void print_cpu_func<float>(void* buffers[], void* cl_args);
template void print_cpu_func<double>(void* buffers[], void* cl_args);

#ifdef USE_CUDA
template <typename DataType>
__global__ void print_kernel(
	DataType* mat,
	uint32_t rows,
	uint32_t cols,
	uint32_t ld,
	char c,
	uint32_t row,
	uint32_t col,
	uint32_t bs
) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i < rows) && (j < cols)) {
		printf("%c : %d, %d, %d, %f\n", c, j + col * bs, i + row * bs, ld, mat[j * ld + i]);
	}
}

template <typename DataType>
void print_cuda_func(void* buffers[], void* cl_args) {
	char c;
	uint32_t row, col, bs;
	starpu_codelet_unpack_args(cl_args, &c, &row, &col, &bs);

	auto M = as_matrix<DataType>(buffers[0]);

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceilDiv(M.rows, threadsPerBlock.x), ceilDiv(M.cols, threadsPerBlock.y));
	print_kernel<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(
		M.ptr, M.rows, M.cols, M.ld, c, row, col, bs
	);
}

template void print_cuda_func<float>(void* buffers[], void* cl_args);
template void print_cuda_func<double>(void* buffers[], void* cl_args);
#endif
