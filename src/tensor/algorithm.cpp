#include "algorithm.hpp"

#include <vector>
#include <cstdio>


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

	// printf("cont_len=%d\n", cont_len);

	u32 noncont_elems = 1;

	// Compute the number of non-contiguous starting indices
	for(size_t dim = cont_dim; dim < ndim; ++dim) {
		noncont_elems *= dim_size[dim];
	}

	std::vector<u32> elem(ndim, 0);

	for(size_t i = 0; i < ntensor; i++) {
		lin_idx_vec[i] = std::vector<u32>(noncont_elems, 0);
	}

	// Compute non-contiguous starting indices
	for(u32 e = 0; e < noncont_elems; e++) {
		for(size_t dim = cont_dim; dim < ndim; dim++) {
			for(size_t i = 0; i < ntensor; i++) {
				lin_idx_vec[i][e] += elem[dim] * ld[i][dim];
			}
		}

		for(size_t dim = cont_dim; dim < ndim; dim++) {
			elem[dim] = (elem[dim] < dim_size[dim] - 1) ? elem[dim] + 1 : 0;

			if(elem[dim]) {
				break;
			}
		}
	}

	return cont_len;
}
