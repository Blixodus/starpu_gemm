#ifndef TENSOR_ADD_FUNC_HPP
#define TENSOR_ADD_FUNC_HPP

template <typename DataType>
void tensor_add_cpu_func(void *buffers[], void *cl_args);
template <typename DataType>
void tensor_add_cuda_func(void *buffers[], void *cl_args);

#endif
