#ifndef ACCUMULATE_FUNC_H
#define ACCUMULATE_FUNC_H

void accumulate_matrix_cpu(void * buffers[], void * cl_args);
void accumulate_matrix_cuda(void * buffers[], void * cl_args);

#endif
