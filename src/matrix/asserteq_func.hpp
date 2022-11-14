#pragma once

template <typename DataType>
void asserteq_cpu_func(void* buffers[], void* cl_args);

template <typename DataType>
void asserteq_cuda_func(void* buffers[], void* cl_args);
