#include "algorithm.hpp"

#include <numeric>
#include <tuple>
#include <vector>
#include <functional>

/**
 * Compute common contiguous length for tensors
 */
std::tuple<u32, std::vector<std::vector<u32>>> compute_contiguous(size_t ntensor, size_t ndim, u32* dim_size, u32** ld) {
	u32 cont_len = 1;
	size_t cont_dim;

	// for each dimension
	for (cont_dim = 0; cont_dim < ndim; ++cont_dim) {
		// for each tensor, if the LD is different than the expected
		for (size_t i = 0; i < ntensor; ++i) {
			if (cont_len != ld[i][cont_dim]) {
				goto cont_len_over;
			}
		}

		cont_len *= dim_size[cont_dim];
	}

	// we finished computing the maximum contiguous length
cont_len_over:

	// Compute the number of non-contiguous starting indices
	// u32 noncont_elems = static_cast<u32>(std::accumulate(dim_size+cont_dim, dim_size+ndim, 1,
	// std::multiplies<u32>()));
	std::vector<std::vector<u32>> lin_idx_vec{ntensor, std::vector<u32>{}};

	// Compute non-contiguous starting indices
	for (auto& elem : DimIter{dim_size, ndim, cont_dim}) {
		for (size_t i = 0; i < ntensor; i++) {
			u32 curr_lin_idx = 0;
			for (size_t dim = cont_dim; dim < ndim; dim++) {
				curr_lin_idx += elem[dim] * ld[i][dim];
			}

			lin_idx_vec[i].push_back(curr_lin_idx);
		}
	}

	return {cont_len, lin_idx_vec};
}
