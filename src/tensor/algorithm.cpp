#include "algorithm.hpp"

#include <vector>
#include <cstdio>
#include <numeric>


/**
 * Compute common contiguous length for tensors
*/
u32 compute_contiguous(size_t ntensor, size_t ndim, u32 *dim_size, u32 **ld, std::vector<std::vector<u32>> &lin_idx_vec) {
	u32 cont_len = 1;
	size_t cont_dim;

	// for each dimension
	for(cont_dim = 0; cont_dim < ndim; ++cont_dim) {
		// for each tensor, if the LD is different than the expected 
		for(size_t i = 0; i < ntensor; ++i) {
			if (cont_len != ld[i][cont_dim]) {
				goto cont_len_over;
			}
		}

		cont_len *= dim_size[cont_dim];
	}

	// we finished computing the maximum contiguous length
 cont_len_over:

	// Compute the number of non-contiguous starting indices
	u32 noncont_elems = std::accumulate(dim_size+cont_dim, dim_size+ndim, 1, std::multiplies<u32>());

	for(auto & lin_idx : lin_idx_vec) {
		lin_idx = std::vector<u32>(noncont_elems, 0);
	}

	// Compute non-contiguous starting indices
  size_t e = 0;
	for(auto & elem : DimIter(dim_size, ndim, cont_dim)) {
		for(size_t dim = cont_dim; dim < ndim; dim++) {
			for(size_t i = 0; i < ntensor; i++) {
				lin_idx_vec[i][e] += elem[dim] * ld[i][dim];
			}
		}
    e++;
	}
  
	return cont_len;
}
