#include "algorithm.hpp"

#include <vector>
#include <cstdio>


void compute_contiguous(size_t ntensor, size_t ndim, u32 *dim_size, u32 **ld, u32 &cont_len, std::vector<std::vector<u32>> &lin_idx_vec) {
	cont_len = 1;
	size_t cont_dim;

	// Compute common contiguous length for tensors
	for(cont_dim = 0; cont_dim < ndim; cont_dim++) {
		bool non_cont = false;

		for(size_t i = 0; i < ntensor; i++) {
			non_cont |= (cont_len != ld[i][cont_dim]);

			if (non_cont) {
				break;
			}
		}

		cont_len *= dim_size[cont_dim];
	}

	printf("cont_len=%d\n", cont_len);

	u32 noncont_elems = 1;

	// Compute the number of non-contiguous starting indices
	for(size_t dim = cont_dim; dim < ndim; dim++) {
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
}
