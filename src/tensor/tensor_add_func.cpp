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
	std::vector<std::vector<u32>> lin_idx(3);

	u32 cont_len = compute_contiguous(3, A.ndim, A.nn, &ld[0], lin_idx);
	
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
void tensor_add_cuda_func(void *buffers[], void *cl_args) {

}

template void tensor_add_cuda_func<float>(void *buffers[], void *cl_args);
template void tensor_add_cuda_func<double>(void *buffers[], void *cl_args);
#endif
