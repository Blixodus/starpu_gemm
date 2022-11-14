#ifndef TENSOR_FILL_FUNC_HPP
#define TENSOR_FILL_FUNC_HPP

template <typename DataType>
void tensor_fill_cpu_func(void *buffers[], void *cl_args);
template <typename DataType>
void tensor_fill_cuda_func(void *buffers[], void *cl_args);
template <typename DataType>
void tensor_print_cuda_func(void *buffers[], void *cl_args);

#endif
