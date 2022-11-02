#ifndef ASSERTEQ_FUNC_HPP
#define ASSERTEQ_FUNC_HPP

template <typename DataType>
void asserteq_cpu_func(void * buffers[], void * cl_args);

template <typename DataType>
void asserteq_cuda_func(void * buffers[], void * cl_args);

#endif
