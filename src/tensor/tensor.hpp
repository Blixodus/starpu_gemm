#pragma once

#include "../util/helper.hpp"

#include <array>


template <typename DataType>
starpu_codelet make_tensor_add_cl() {
	static struct starpu_perfmodel model = {
		.type = STARPU_REGRESSION_BASED,
		.symbol = __PRETTY_FUNCTION__
	};

	return {
		.cpu_funcs = { tensor_add_cpu_func<DataType> },
	#ifdef USE_CUDA
		.cuda_funcs = { tensor_add_cuda_func<DataType> },
		.cuda_flags = { STARPU_CUDA_ASYNC },
	#endif
		.nbuffers = 3,
		.modes = { STARPU_R, STARPU_R, STARPU_RW },
		.model = &model,
	};
}

template <typename DataType>
static auto tensor_add_cl = make_tensor_add_cl<DataType>();

template <typename DataType>
starpu_codelet make_tensor_fill_cl() {
	static struct starpu_perfmodel model = {
		.type = STARPU_REGRESSION_BASED,
		.symbol = __PRETTY_FUNCTION__
	};

	return {
		.cpu_funcs = { tensor_fill_cpu_func<DataType> },
	#ifdef USE_CUDA
		.cuda_funcs = { tensor_fill_cuda_func<DataType> },
		.cuda_flags = { STARPU_CUDA_ASYNC },
	#endif
		.nbuffers = 1,
		.modes = { STARPU_W },
		.model = &model,
	};
}

template <typename DataType>
static auto tensor_fill_cl = make_tensor_fill_cl<DataType>();

template <typename DataType>
struct TensorData {
	std::vector<u32> dim_blocks;
	std::vector<starpu_data_handle_t> data_handle;

	TensorData(std::vector<u32> dims, u32 block_size) : dim_blocks(dims) {
		for(auto& elem : dim_blocks) {
			elem = ceilDiv(elem, block_size);
		}

		// Length of last block in each dimension
		std::vector<u32> dim_final(dim_blocks.size());

		for(size_t i = 0; i < dim_blocks.size(); i++) {
			dim_final[i] = (dims[i] % block_size) ? (dims[i] % block_size) : block_size;
		}

		// Create data handles for each block in the tensor
		data_handle = std::vector<starpu_data_handle_t>(
			unchecked_cast<typename decltype(data_handle)::size_type>(
				std::accumulate(dim_blocks.begin(), dim_blocks.end(), 1, std::multiplies<u32>())
			)
		);
		
		std::vector<u32> idx(dims.size(), 0);

		for(size_t i = 0; i < data_handle.size(); i++) {
			auto& handle = get(idx);

			std::cout << "Creating handle " << handle << " with dims: " << VecPrinter(idx) << std::endl;

			std::vector<u32> ld(dims.size(), 1);
			std::vector<u32> dim_size(dims.size());

			for(size_t d = 0; d < dims.size(); d++) {
				dim_size[d] = (idx[d] == dim_blocks[d]-1) ? dim_final[d] : block_size;
			}

			for(size_t d = 1; d < dims.size(); d++) {
				ld[d] = dim_size[d-1] * ld[d-1];
			}

			starpu_ndim_data_register(&handle, -1, 0, &ld[0], &dim_size[0], dims.size(), sizeof(DataType));
			
			for(size_t d = 0; d < dims.size(); d++) {
				idx[d] = (idx[d] >= dim_blocks[d] - 1) ? 0 : (idx[d] + 1);

				if(idx[d]) {
					break;
				}
			}
		}
    std::cout << VecPrinter(data_handle) << std::endl;
	}

	~TensorData() {
		for (auto& handle : data_handle) {
			starpu_data_unregister(handle);
		}
	}

	starpu_data_handle_t& get(std::vector<u32>& idx) {
		u32 ld = 1;
		u32 lin_idx = 0;

		for(size_t i = 0; i < dim_blocks.size(); i++) {
			lin_idx += idx[i] * ld;
			ld *= dim_blocks[i];
		}

		return data_handle[lin_idx];
	}
};

template <typename DataType>
struct Tensor {
	u32 block_size;
	std::vector<u32> dim_size;
	TensorData<DataType> data_handle;

	Tensor(std::vector<u32> dims, u32 bs) : block_size(bs), dim_size(dims), data_handle(dim_size, block_size) { }

	void fill(DataType e) {
		// std::cout << "Fill on handles: " << VecPrinter(data_handle.data_handle) << std::endl;

		for(auto& block_handle : data_handle.data_handle) {
      std::cout << "Task inserted (fill) : " << block_handle << std::endl;

			int err = starpu_task_insert(&tensor_fill_cl<DataType>,
											STARPU_VALUE, &e, sizeof(e),
											STARPU_W, block_handle,
											NULL);
			if(err) {
				throw std::exception();
			}

			starpu_task_wait_for_all();
		}
	}
	
	Tensor<DataType> operator+(Tensor<DataType>& other) {
		Tensor<DataType> result(dim_size, block_size);
		add(*this, other, result);
		return result;
	}

	/**
	 * Do operation C = A + B with tasks
	 **/
	static void add(Tensor<DataType>& A, Tensor<DataType>& B, Tensor<DataType>& C) {
		assert(A.dim_size == B.dim_size && B.dim_size == C.dim_size);
		assert(A.block_size == B.block_size && B.block_size == C.block_size);
		const auto &ndim = A.dim_size.size();
		const auto &blocks = A.data_handle.dim_blocks;
		const auto nb_blocks = std::accumulate(blocks.begin(), blocks.end(), 1, std::multiplies<u32>());
		
		std::vector<u32> curr_block(ndim, 0);
		for(int i = 0; i < nb_blocks; i++) {
			// Create task for current block
			auto block_handle_A = A.data_handle.get(curr_block);
		  auto block_handle_B = B.data_handle.get(curr_block);
		  auto block_handle_C = C.data_handle.get(curr_block);
      std::cout << "Task inserted (add) : " << block_handle_A << " " << block_handle_B << " " << block_handle_C << std::endl;
			int err = starpu_task_insert(&tensor_add_cl<DataType>,
																	 STARPU_R, block_handle_A,
																	 STARPU_R, block_handle_B,
																	 STARPU_RW, block_handle_C,
																	 0);
			if(err) { throw std::exception(); }
			// Move to next block
			for(int dim = 0; dim < ndim; dim++) {
				curr_block[dim] = (curr_block[dim] < blocks[dim]-1) ? curr_block[dim]+1 : 0;
				if(curr_block[dim]) { break; }
			}
		}
    
    starpu_task_wait_for_all();
	}
};
