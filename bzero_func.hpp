#ifndef BZERO_FUNC_H
#define BZERO_FUNC_H

void bzero_matrix_cpu(void * buffers[], void * cl_args);
void bzero_matrix_cuda(void * buffers[], void * cl_args);

#endif
