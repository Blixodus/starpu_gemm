#pragma once

template <typename DataType>
void print_cpu_func(void * buffers[], void * cl_args);

template <typename DataType>
void print_cuda_func(void * buffers[], void * cl_args);
