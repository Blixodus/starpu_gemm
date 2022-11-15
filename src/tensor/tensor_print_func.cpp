#include "tensor_print_func.hpp"

#include <starpu.h>
#include <cstdio>

#include "../util/helper.hpp"


template <typename DataType>
void tensor_print_cpu_func(void* buffers[], void* cl_args) {
	char tag;
	starpu_codelet_unpack_args(cl_args, &tag);

	auto T = as_tensor<DataType>(buffers[0]);

	// for each dimension of T
	for (size_t i = 0; i < T.ndim; ++i) {
		
	}
}

template void tensor_print_cpu_func<float>(void* buffers[], void* cl_args);
template void tensor_print_cpu_func<double>(void* buffers[], void* cl_args);
