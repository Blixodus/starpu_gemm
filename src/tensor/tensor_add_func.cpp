#include "tensor_add_func.hpp"

#include <vector>
#include <starpu.h>

#include "algorithm.hpp"
#include "../util/helper.hpp"


template <typename DataType>
void tensor_add_cpu_func(void *buffers[], void *cl_args) {
	auto A = as_tensor<DataType>(buffers[0]);
	auto B = as_tensor<DataType>(buffers[1]);
	auto C = as_tensor<DataType>(buffers[2]);

	std::vector<u32*> ld = { A.ldn, B.ldn, C.ldn };
	auto [cont_len, lin_idx] = compute_contiguous(3, A.ndim, A.nn, &ld[0]);
	
	// Update each common contiguous part separately
	for(u32 e = 0; e < lin_idx[0].size(); e++) {
		for(u32 i = 0; i < cont_len; i++) {
			C.ptr[lin_idx[2][e] + i] = A.ptr[lin_idx[0][e] + i] + B.ptr[lin_idx[1][e] + i];
		}
	}
}

template void tensor_add_cpu_func<float>(void *buffers[], void *cl_args);
template void tensor_add_cpu_func<double>(void *buffers[], void *cl_args);

#ifdef USE_CUDA
template <typename DataType>
__global__ void tensor_add_kernel(DataType *A_ptr, DataType *B_ptr, DataType *C_ptr, u32 len) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < len) {
	  C_ptr[i] = A_ptr[i] + B_ptr[i];
	}
}

template <typename DataType>
void tensor_add_cuda_func(void *buffers[], void *cl_args) {
	auto A = as_tensor<DataType>(buffers[0]);
	auto B = as_tensor<DataType>(buffers[1]);
	auto C = as_tensor<DataType>(buffers[2]);

	std::vector<u32*> ld = { A.ldn, B.ldn, C.ldn };
	auto [cont_len, lin_idx] = compute_contiguous(3, A.ndim, A.nn, &ld[0]);
	
	// Update each common contiguous part separately
	for(u32 e = 0; e < lin_idx[0].size(); e++) {
		dim3 threadsPerBlock(256);
		dim3 numBlocks(ceilDiv(cont_len, threadsPerBlock.x));

		tensor_add_kernel<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(&A.ptr[lin_idx[0][e]], &B.ptr[lin_idx[1][e]], &C.ptr[lin_idx[2][e]], cont_len);

		cudaError_t status = cudaGetLastError();

		if (status != cudaSuccess) {
			STARPU_CUDA_REPORT_ERROR(status);
		}
	}
}

template void tensor_add_cuda_func<float>(void *buffers[], void *cl_args);
template void tensor_add_cuda_func<double>(void *buffers[], void *cl_args);
#endif
