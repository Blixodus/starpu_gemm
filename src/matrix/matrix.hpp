#pragma once

#include <vector>
#include "starpu_mpi.h"

#include "../util/helper.hpp"

#include "print_func.hpp"
#include "gemm_func.hpp"
#include "bzero_func.hpp"
#include "accumulate_func.hpp"
#include "fill_func.hpp"
#include "asserteq_func.hpp"

static constexpr int enable_cpu = 1;
static constexpr int enable_gpu = 1;

static int can_execute(unsigned workerid, struct starpu_task* task, unsigned nimpl) {
	if (starpu_worker_get_type(checked_cast<int>(workerid)) == STARPU_CPU_WORKER) {
		return enable_cpu;
	}

	return enable_gpu;
}

template <typename DataType>
starpu_codelet make_gemm_cl() {
	static struct starpu_perfmodel model = {
		.type = STARPU_REGRESSION_BASED,
		.symbol = __PRETTY_FUNCTION__
	};

	return {
		.can_execute = can_execute,
		.cpu_funcs = { gemm_cpu_func<DataType> },
#ifdef USE_CUDA
		.cuda_funcs = { gemm_cuda_func<DataType> },
		.cuda_flags = { STARPU_CUDA_ASYNC },
#endif
		.nbuffers = 3,
#if ENABLE_REDUX != 0
		.modes = {STARPU_R, STARPU_R, STARPU_RW | STARPU_COMMUTE},
#else
		.modes = {STARPU_R, STARPU_R, STARPU_RW},
#endif
		.model = &model,
    .name = "gemm",
	};
}

template <typename DataType>
static auto gemm_cl = make_gemm_cl<DataType>();

template <typename DataType>
starpu_codelet make_bzero_matrix_cl() {
	static struct starpu_perfmodel model = {
		.type = STARPU_REGRESSION_BASED,
		.symbol = __PRETTY_FUNCTION__
	};

	return {
		.can_execute = can_execute,
		.cpu_funcs = { bzero_matrix_cpu<DataType> },
#ifdef USE_CUDA
		.cuda_funcs = { bzero_matrix_cuda<DataType> },
#endif
		.nbuffers = 1,
		.modes = { STARPU_W },
		.model = &model,
    .name = "bzero",
	};
}

template <typename DataType>
static auto bzero_matrix_cl = make_bzero_matrix_cl<DataType>();

template <typename DataType>
starpu_codelet make_accumulate_matrix_cl() {
	static struct starpu_perfmodel model = {
		.type = STARPU_REGRESSION_BASED,
		.symbol = __PRETTY_FUNCTION__
	};

	return {
		.can_execute = can_execute,
		.cpu_funcs = { accumulate_matrix_cpu<DataType> },
#ifdef USE_CUDA
		.cuda_funcs = { accumulate_matrix_cuda<DataType> },
#endif
		.nbuffers = 2,
		.modes = { STARPU_RW | STARPU_COMMUTE, STARPU_R },
		.model = &model,
    .name = "accumulate",
	};
}

template <typename DataType>
static auto accumulate_matrix_cl = make_accumulate_matrix_cl<DataType>();

template <typename DataType>
starpu_codelet make_fill_cl() {
	static struct starpu_perfmodel model = {
		.type = STARPU_REGRESSION_BASED,
		.symbol = __PRETTY_FUNCTION__
	};

	return {
		.can_execute = can_execute,
		.cpu_funcs = { fill_cpu_func<DataType> },
#ifdef USE_CUDA
		.cuda_funcs = { fill_cuda_func<DataType> },
		.cuda_flags = { STARPU_CUDA_ASYNC },
#endif
		.nbuffers = 1,
		.modes = { STARPU_W },
		.model = &model,
    .name = "fill",
	};
}

template <typename DataType>
static auto fill_cl = make_fill_cl<DataType>();

template <typename DataType>
starpu_codelet make_print_cl() {
	return {
		.can_execute = can_execute,
		.cpu_funcs = {print_cpu_func<DataType>},
#ifdef USE_CUDA
		.cuda_funcs = {print_cuda_func<DataType>},
		.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
		.nbuffers = 1,
		.modes = {STARPU_R},
    .name = "print",
	};
}

template <typename DataType>
static auto print_cl = make_print_cl<DataType>();

template <typename DataType>
starpu_codelet make_asserteq_cl() {
	return {
		.can_execute = can_execute,
		.cpu_funcs = {asserteq_cpu_func<DataType>},
#ifdef USE_CUDA
		.cuda_funcs = {asserteq_cuda_func<DataType>},
		.cuda_flags = { STARPU_CUDA_ASYNC },
#endif
		.nbuffers = 1,
		.modes = {STARPU_R},
    .name = "asserteq",
	};
}

template <typename DataType>
static auto asserteq_cl = make_asserteq_cl<DataType>();

static int matrix_mpi_tag = 0;
template <typename DataType>
struct MatrixData {
	u32 row_blocks, col_blocks;
	std::vector<starpu_data_handle_t> data_handle;

	MatrixData(u32 rows, u32 cols, u32 block_size)
		: row_blocks(ceilDiv(rows, block_size)),
		  col_blocks(ceilDiv(cols, block_size)),
		  data_handle(row_blocks * col_blocks) {
		int rank, size;

		starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
		starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

		auto row_final = (rows % block_size) ? (rows % block_size) : block_size;
    auto col_final = (cols % block_size) ? (cols % block_size) : block_size;

		for (u32 i = 0; i < row_blocks; i++) {
			for (u32 j = 0; j < col_blocks; j++) {
				auto& handle = get(i, j);

				auto rows_block = (i == row_blocks - 1) ? row_final : block_size;
				auto cols_block = (j == col_blocks - 1) ? col_final : block_size;

				starpu_matrix_data_register(&handle, -1, 0, rows_block, rows_block, cols_block, sizeof(DataType));
				starpu_mpi_data_register(handle, matrix_mpi_tag++, static_cast<int>(i + j) % size);
			}
		}
	}

	~MatrixData() {
		for (auto& handle : data_handle) {
			starpu_data_unregister(handle);
		}
	}

	starpu_data_handle_t& get(u32 i, u32 j) {
		return data_handle[i + j * row_blocks];
	}
};

template <typename DataType>
struct Matrix {
	u32 rows, cols;
	u32 block_size;
	MatrixData<DataType> data_handle;

	Matrix() = default;

	Matrix(u32 rows_, u32 cols_, u32 block_size_)
		: rows(rows_), cols(cols_), block_size(block_size_), data_handle(rows, cols, block_size){};

	void fill(DataType e) {
		for (u32 i = 0; i < data_handle.row_blocks; i++) {
			for (u32 j = 0; j < data_handle.col_blocks; j++) {
				auto handle = data_handle.get(i, j);

				int err = starpu_mpi_task_insert(
					MPI_COMM_WORLD, &fill_cl<DataType>,
					STARPU_VALUE, &e, sizeof(e),
					STARPU_W, handle,
					NULL
				);

				if (err) {
					throw std::exception();
				}
			}
		}
	}

	void print(char c) {
		for (u32 i = 0; i < data_handle.row_blocks; i++) {
			for (u32 j = 0; j < data_handle.col_blocks; j++) {
				auto handle = data_handle.get(i, j);

				int err = starpu_mpi_task_insert(
					MPI_COMM_WORLD, &print_cl<DataType>,
					STARPU_VALUE, &c, sizeof(c),
					STARPU_VALUE, &i, sizeof(i),
					STARPU_VALUE, &j, sizeof(j),
					STARPU_VALUE, &block_size, sizeof(block_size),
					STARPU_R, handle,
					NULL
				);

				if (err) {
					throw std::exception();
				}
			}
		}
	}

	void assertEq(const DataType val) {
		for (u32 i = 0; i < data_handle.row_blocks; i++) {
			for (u32 j = 0; j < data_handle.col_blocks; j++) {
				auto handle = data_handle.get(i, j);

				int err = starpu_mpi_task_insert(
					MPI_COMM_WORLD, &asserteq_cl<DataType>,
					STARPU_VALUE, &val, sizeof(val),
					STARPU_R, handle,
					NULL
				);

				if (err) {
					throw std::exception();
				}
			}
		}
	}

	static void gemm(
		char transA,
		char transB,
		DataType alpha,
		Matrix<DataType>& A,
		Matrix<DataType>& B,
		DataType beta,
		Matrix<DataType>& C
	) {
		assert(A.rows == C.rows);
		assert(B.cols == C.cols);
		assert(A.cols == B.rows);
		assert(A.block_size == B.block_size && B.block_size == C.block_size);

	#if ENABLE_REDUX != 0
		for (u32 i = 0; i < C.data_handle.row_blocks; i++) {
			for (u32 j = 0; j < C.data_handle.col_blocks; j++) {
				starpu_data_set_reduction_methods(C.data_handle.get(i, j), &accumulate_matrix_cl<DataType>, &bzero_matrix_cl<DataType>);
			}
		}
	#endif

		for (u32 i = 0; i < C.data_handle.row_blocks; i++) {
			for (u32 j = 0; j < C.data_handle.col_blocks; j++) {
        auto C_sub_handle = C.data_handle.get(i, j);
				for (u32 k = 0; k < A.data_handle.col_blocks; k++) {
					auto A_sub_handle = A.data_handle.get(i, k);
					auto B_sub_handle = B.data_handle.get(k, j);

					auto err = starpu_mpi_task_insert(
						MPI_COMM_WORLD, &gemm_cl<DataType>,
						STARPU_VALUE, &transA, sizeof(transA),
						STARPU_VALUE, &transB, sizeof(transB),
						STARPU_VALUE, &alpha, sizeof(alpha),
						STARPU_VALUE, &beta, sizeof(beta),
						STARPU_R, A_sub_handle,
            STARPU_R, B_sub_handle,
	#if ENABLE_REDUX != 0
						STARPU_MPI_REDUX, C_sub_handle,
	#else
						STARPU_RW, C_sub_handle,
	#endif
						STARPU_FLOPS, double(2L * starpu_matrix_get_nx(C_sub_handle) * starpu_matrix_get_ny(C_sub_handle) * starpu_matrix_get_ny(A_sub_handle)),
						NULL
					);

					if (err) { throw std::exception(); }
				}
#if ENABLE_REDUX != 0
        auto err = starpu_mpi_redux_data(MPI_COMM_WORLD, C_sub_handle);
        if (err) { throw std::exception(); }
#endif
			}
		}
	}
};
