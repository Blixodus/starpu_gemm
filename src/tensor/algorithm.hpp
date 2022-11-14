#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

/**
 * Compute longest common contiguous part for 1 or more tensors, as well as the starting indices for each contiguous segment
 */
void compute_contiguous(size_t ntensor, size_t ndim, unsigned int *dim_size, unsigned int **ld, unsigned int &cont_len, std::vector<std::vector<unsigned int>> &lin_idx_vec);
  
#endif
