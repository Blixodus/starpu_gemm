#pragma once

template <typename DataType>
void tensor_fill_cpu_func(void *buffers[], void *cl_args);

template <typename DataType>
void tensor_fill_cuda_func(void *buffers[], void *cl_args);
