#pragma once

template <typename DataType>
void fill_cpu_func(void * buffers[], void * cl_args);

template <typename DataType>
void fill_cuda_func(void * buffers[], void * cl_args);
