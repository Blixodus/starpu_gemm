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
	u32 row, col, bs;
	starpu_codelet_unpack_args(cl_args, &c, &row, &col, &bs);

	auto M = as_matrix<DataType>(buffers[0]);

	for (u32 i = 0; i < M.cols; i++) {
		for (u32 j = 0; j < M.rows; j++) {
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
	u32 rows,
	u32 cols,
	u32 ld,
	char c,
	u32 row,
	u32 col,
	u32 bs
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
	u32 row, col, bs;
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
