#pragma once

#include "../util/helper.hpp"

#include "tensor_add_func.hpp"
#include "tensor_fill_func.hpp"
#include "tensor_print_func.hpp"
#include "tensor_asserteq_func.hpp"

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
starpu_codelet make_tensor_print_cl() {
	static struct starpu_perfmodel model = {
		.type = STARPU_REGRESSION_BASED,
		.symbol = __PRETTY_FUNCTION__
	};

	return {
		.cpu_funcs = { tensor_print_cpu_func<DataType> },
		.nbuffers = 1,
		.modes = { STARPU_R },
		.model = &model,
	};
}

template <typename DataType>
static auto tensor_print_cl = make_tensor_print_cl<DataType>();

template <typename DataType>
starpu_codelet make_tensor_asserteq_cl() {
  return {
		.cpu_funcs = { tensor_asserteq_cpu_func<DataType> },
    #ifdef USE_CUDA
		.cuda_funcs = { tensor_asserteq_cuda_func<DataType> },
    #endif
		.nbuffers = 1,
		.modes = { STARPU_R },
	};
}

template <typename DataType>
static auto tensor_asserteq_cl = make_tensor_asserteq_cl<DataType>();

static int tensor_mpi_tag = 0;
template <typename DataType>
struct TensorData {
	std::vector<u32> dim_blocks;
	std::vector<u32> ld;
	std::vector<starpu_data_handle_t> data_handles;

	TensorData(std::vector<u32> dims, u32 block_size) : dim_blocks(dims), ld(dims.size(), 1) {
		int rank, size;

		starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
		starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
    
		for(auto& elem : dim_blocks) {
			elem = ceilDiv(elem, block_size);
		}
    compute_ld();

		// Length of last block in each dimension
		std::vector<u32> dim_final(dim_blocks.size());

		for(size_t i = 0; i < dim_blocks.size(); i++) {
			dim_final[i] = (dims[i] % block_size) ? (dims[i] % block_size) : block_size;
		}

		// Create data handles for each block in the tensor
		data_handles = std::vector<starpu_data_handle_t>(
			unchecked_cast<typename decltype(data_handles)::size_type>(
				std::accumulate(dim_blocks.begin(), dim_blocks.end(), 1, std::multiplies<u32>())
			)
		);
		
		std::vector<u32> idx(dims.size(), 0);

		for(size_t i = 0; i < data_handles.size(); i++) {
			auto& handle = get(idx);

			std::vector<u32> ld(dims.size(), 1);
			std::vector<u32> dim_size(dims.size());

			for(size_t d = 0; d < dims.size(); d++) {
				dim_size[d] = (idx[d] == dim_blocks[d]-1) ? dim_final[d] : block_size;
			}

			for(size_t d = 1; d < dims.size(); d++) {
				ld[d] = dim_size[d-1] * ld[d-1];
			}

			starpu_ndim_data_register(&handle, -1, 0, &ld[0], &dim_size[0], dims.size(), sizeof(DataType));
      starpu_mpi_data_register(handle, tensor_mpi_tag++, static_cast<int>(linearize_idx(idx)) % size);
			
			for(size_t d = 0; d < dims.size(); d++) {
				idx[d] = (idx[d] >= dim_blocks[d] - 1) ? 0 : (idx[d] + 1);

				if(idx[d]) {
					break;
				}
			}
		}
	}

	~TensorData() {
		for (auto& handle : data_handles) {
			starpu_data_unregister(handle);
		}
	}

	starpu_data_handle_t& get(std::vector<u32>& idx) {
		return data_handles[linearize_idx(idx)];
	}
  
private:
  void compute_ld() {
    for(size_t i = 1; i < dim_blocks.size(); i++) {
      ld[i] = dim_blocks[i-1] * ld[i-1];
		}
  }

  u32 linearize_idx(std::vector<u32>& idx) {
		u32 lin_idx = 0;
		for(size_t i = 0; i < dim_blocks.size(); i++) {
			lin_idx += idx[i] * ld[i];
		}
    return lin_idx;
  }
};

template <typename DataType>
struct Tensor {
	u32 block_size;
	std::vector<u32> dim_size;
	TensorData<DataType> data_handle;

	Tensor(std::vector<u32> dims, u32 bs) : block_size(bs), dim_size(dims), data_handle(dim_size, block_size) { }

	void fill(DataType e) {
		for(auto& block_handle : data_handle.data_handles) {
      int err = starpu_mpi_task_insert(MPI_COMM_WORLD, &tensor_fill_cl<DataType>,
                                       STARPU_VALUE, &e, sizeof(e),
                                       STARPU_W, block_handle,
                                       NULL);
			if(err) {
				throw std::exception();
			}
      std::cout << "Task inserted [fill] (handle=" << block_handle << ")" << std::endl;
		}
	}

	void assertEq(DataType e) {
		for(auto& block_handle : data_handle.data_handles) {
      int err = starpu_mpi_task_insert(MPI_COMM_WORLD, &tensor_asserteq_cl<DataType>,
                                       STARPU_VALUE, &e, sizeof(e),
                                       STARPU_R, block_handle,
                                       NULL);
			if(err) {
				throw std::exception();
			}
      std::cout << "Task inserted [assert equal] (handle=" << block_handle << ")" << std::endl;
		}
	}

	void print(char tag) {
		for (auto& block_handle : data_handle.data_handles) {
			int err = starpu_mpi_task_insert(MPI_COMM_WORLD, &tensor_print_cl<DataType>,
                                       STARPU_VALUE, &tag, sizeof(tag),
                                       STARPU_R, block_handle,
                                       NULL);
			if(err) {
				throw std::exception();
			}
      std::cout << "Task inserted [print] (handle=" << block_handle << ")" << std::endl;
		}
	}

	/**
	 * Do operation C = A + B with tasks
	 **/
	static void add(Tensor<DataType>& A, Tensor<DataType>& B, Tensor<DataType>& C) {
		assert(A.dim_size == B.dim_size && B.dim_size == C.dim_size);
		assert(A.block_size == B.block_size && B.block_size == C.block_size);
		for(auto & curr_block : DimIter(A.data_handle.dim_blocks)) {
			// Create task for current block
			auto block_handle_A = A.data_handle.get(curr_block);
			auto block_handle_B = B.data_handle.get(curr_block);
			auto block_handle_C = C.data_handle.get(curr_block);
			int err = starpu_mpi_task_insert(MPI_COMM_WORLD, &tensor_add_cl<DataType>,
                                       STARPU_R, block_handle_A,
                                       STARPU_R, block_handle_B,
                                       STARPU_RW, block_handle_C,
                                       0);
			if(err) { throw std::exception(); }
			std::cout << "Task inserted [add] (handle_A=" << block_handle_A << ", handle_B=" << block_handle_B << ", handle_C=" << block_handle_C << ")" << std::endl;
		}
	}
};
