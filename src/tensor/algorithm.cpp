#include <vector>
#include <cstdio>
#include "algorithm.hpp"

void compute_contiguous(size_t ntensor, size_t ndim, unsigned int *dim_size, unsigned int **ld, unsigned int &cont_len, std::vector<std::vector<unsigned int>> &lin_idx_vec) {
  cont_len = 1;
  size_t cont_dim;
  // Compute common contiguous length for tensors
  for(cont_dim = 0; cont_dim < ndim; cont_dim++) {
    bool non_cont = false;
    for(int i = 0; i < ntensor; i++) {
      non_cont |= (cont_len != ld[i][cont_dim]);
    }
    if(non_cont) { break; }
    cont_len *= dim_size[cont_dim];
  }
  printf("cont_len=%d\n", cont_len);
  unsigned int noncont_elems = 1;
  // Compute the number of non-contiguous starting indices
  for(size_t dim = cont_dim; dim < ndim; dim++) {
    noncont_elems *= dim_size[dim];
  }
  std::vector<unsigned int> elem(ndim, 0);
  for(int i = 0; i < ntensor; i++) {
    lin_idx_vec[i] = std::vector<unsigned int>(noncont_elems, 0);
  }
  // Compute non-contiguous starting indices
  for(unsigned int e = 0; e < noncont_elems; e++) {
    for(size_t dim = cont_dim; dim < ndim; dim++) {
      for(int i = 0; i < ntensor; i++) {
        lin_idx_vec[i][e] += elem[dim] * ld[i][dim];
      }
    }
    for(size_t dim = cont_dim; dim < ndim; dim++) {
      elem[dim] = (elem[dim] < dim_size[dim]-1) ? elem[dim]+1 : 0;
      if(elem[dim]) { break; }
    }
  }
}
