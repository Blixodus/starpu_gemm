#ifndef BZERO_FUNC_H
#define BZERO_FUNC_H

template <typename DataType>
void bzero_matrix_cpu(void * buffers[], void * cl_args);

template <typename DataType>
void bzero_matrix_cuda(void * buffers[], void * cl_args);

#endif
