#pragma once

#include "../util/helper.hpp"

/**
 * Compute longest common contiguous part for 1 or more tensors, as well as the starting indices for each contiguous segment
 */
u32 compute_contiguous(size_t ntensor, size_t ndim, u32 *dim_size, u32 **ld, std::vector<std::vector<u32>> &lin_idx_vec);
  