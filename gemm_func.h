#ifndef GEMM_FUNC_H
#define GEMM_FUNC_H

void gemm_cpu_func(void * buffers[], void * cl_args);
void gemm_cuda_func(void * buffers[], void * cl_args);

#endif
