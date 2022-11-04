#pragma once

template <typename DataType>
void gemm_cpu_func(void* buffers[], void* cl_args);
template <typename DataType>
void gemm_cuda_func(void* buffers[], void* cl_args);
