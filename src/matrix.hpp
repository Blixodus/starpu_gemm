#pragma once

#include "print_func.hpp"
#include "helper.hpp"

static int enable_cpu = ENABLE;
static int enable_gpu = ENABLE;
static int can_execute(unsigned workerid, struct starpu_task* task, unsigned nimpl) {
	if (starpu_worker_get_type(safe_cast<int>(workerid)) == STARPU_CPU_WORKER) {
		return enable_cpu;
	}

	return enable_gpu;
}

static struct starpu_perfmodel gemm_perf_model_float = {
	.type = STARPU_REGRESSION_BASED,
	.symbol = "gemm_perf_model_float"};

static struct starpu_perfmodel gemm_perf_model_double = {
	.type = STARPU_REGRESSION_BASED,
	.symbol = "gemm_perf_model_double"};

template <typename DataType>
starpu_codelet gemm_cl = {
	.can_execute = can_execute,
	.cpu_funcs = {gemm_cpu_func<DataType>},
#ifdef USE_CUDA
	// .cuda_funcs = { gemm_cuda_func<DataType> },
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.nbuffers = 3,
#if ENABLE_REDUX != 0
	.modes = {STARPU_R, STARPU_R, starpu_data_access_mode(STARPU_RW | STARPU_COMMUTE)},
#else
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
#endif
	.model = (std::is_same_v<DataType, float>) ? &gemm_perf_model_float : &gemm_perf_model_double,
};

template <typename DataType>
starpu_codelet bzero_matrix_cl = {
	.can_execute = can_execute,
	.cpu_funcs = {bzero_matrix_cpu<DataType>},
#ifdef USE_CUDAd
	.cuda_funcs = {bzero_matrix_cuda<DataType>},
#endif
	.nbuffers = 1,
	.modes = {STARPU_W},
};

static struct starpu_perfmodel accumulate_perf_model_float = {
	.type = STARPU_REGRESSION_BASED,
	.symbol = "accumulate_perf_model_float"};

static struct starpu_perfmodel accumulate_perf_model_double = {
	.type = STARPU_REGRESSION_BASED,
	.symbol = "accumulate_perf_model_double"};

template <typename DataType>
starpu_codelet accumulate_matrix_cl = {
	.can_execute = can_execute,
	.cpu_funcs = {accumulate_matrix_cpu<DataType>},
#ifdef USE_CUDA
	.cuda_funcs = {accumulate_matrix_cuda<DataType>},
#endif
	.nbuffers = 2,
	.modes = {starpu_data_access_mode(STARPU_RW | STARPU_COMMUTE), STARPU_R},
	.model = (std::is_same_v<DataType, float>) ? &accumulate_perf_model_float
											   : &accumulate_perf_model_double,
};

static struct starpu_perfmodel fill_perf_model_float = {
	.type = STARPU_REGRESSION_BASED,
	.symbol = "fill_perf_model_float"};

static struct starpu_perfmodel fill_perf_model_double = {
	.type = STARPU_REGRESSION_BASED,
	.symbol = "fill_perf_model_double"};

template <typename DataType>
starpu_codelet fill_cl = {
	.cpu_funcs = {fill_cpu_func<DataType>},
#ifdef USE_CUDA
//.cuda_funcs = { fill_cuda_func<DataType> },
//.cuda_flags = { STARPU_CUDA_ASYNC },
#endif
	.nbuffers = 1,
	.modes = {STARPU_W},
	.model = (std::is_same_v<DataType, float>) ? &fill_perf_model_float : &fill_perf_model_double,
};

template <typename DataType>
starpu_codelet print_cl = {
	.cpu_funcs = {print_cpu_func<DataType>},
#ifdef USE_CUDA
	.cuda_funcs = {print_cuda_func<DataType>},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.nbuffers = 1,
	.modes = {STARPU_R},
};

template <typename DataType>
starpu_codelet asserteq_cl = {
	.cpu_funcs = {asserteq_cpu_func<DataType>},
#ifdef USE_CUDA
	.cuda_funcs = {asserteq_cuda_func<DataType>},
//.cuda_flags = { STARPU_CUDA_ASYNC },
#endif
	.nbuffers = 1,
	.modes = {STARPU_R},
};

static int mpi_tag = 0;

template <typename Int>
inline Int ceil_div(Int a, Int b) {
	return (a + b - 1) / b;
}

template <typename DataType>
struct MatrixData {
	u32 row_blocks, col_blocks;
	std::vector<starpu_data_handle_t> data_handle;

	MatrixData(u32 rows, u32 cols, u32 block_size)
		: row_blocks(ceil_div(rows, block_size)),
		  col_blocks(ceil_div(cols, block_size)),
		  data_handle(row_blocks * col_blocks) {
		int rank, size;

		starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
		starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

		auto row_final = (rows % block_size) ? rows % block_size : block_size;
		auto col_final = (cols % block_size) ? cols % block_size : block_size;

		for (u32 i = 0; i < row_blocks; i++) {
			for (u32 j = 0; j < col_blocks; j++) {
				auto& handle = get(i, j);

				auto rows_block = (i == row_blocks - 1) ? row_final : block_size;
				auto cols_block = (j == col_blocks - 1) ? col_final : block_size;

				starpu_matrix_data_register(
					&handle, -1, 0, rows_block, rows_block, cols_block, sizeof(DataType)
				);
				starpu_mpi_data_register(handle, mpi_tag++, static_cast<int>(i + j) % size);
				// std::cout << rank << " " << (i+j)%size << " " << handle << " " << i << " " << j
				// << " " << rows_block << " " << cols_block << std::endl;
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
					MPI_COMM_WORLD, &fill_cl<DataType>, STARPU_VALUE, &e, sizeof(e), STARPU_W,
					handle, NULL
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
					MPI_COMM_WORLD, &print_cl<DataType>, STARPU_VALUE, &c, sizeof(c), STARPU_VALUE,
					&i, sizeof(i), STARPU_VALUE, &j, sizeof(j), STARPU_VALUE, &block_size,
					sizeof(block_size), STARPU_R, handle, NULL
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
					MPI_COMM_WORLD, &asserteq_cl<DataType>, STARPU_VALUE, &val, sizeof(val),
					STARPU_R, handle, NULL
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
				for (u32 k = 0; k < A.data_handle.col_blocks; k++) {
					auto A_sub_handle = A.data_handle.get(i, k);
					auto B_sub_handle = B.data_handle.get(k, j);
					auto C_sub_handle = C.data_handle.get(i, j);

					int err = starpu_mpi_task_insert(
						MPI_COMM_WORLD, &gemm_cl<DataType>, STARPU_VALUE, &transA, sizeof(transA),
						STARPU_VALUE, &transB, sizeof(transB), STARPU_VALUE, &alpha, sizeof(alpha),
						STARPU_VALUE, &beta, sizeof(beta), STARPU_R, A_sub_handle, STARPU_R,
						B_sub_handle,
#if ENABLE_REDUX != 0
						STARPU_MPI_REDUX, C_sub_handle,
#else
						STARPU_RW, C_sub_handle,
#endif
						STARPU_FLOPS,
						double(
							2L * starpu_matrix_get_nx(C_sub_handle) *
							starpu_matrix_get_ny(C_sub_handle) * starpu_matrix_get_ny(A_sub_handle)
						),
						0
					);
					if (err) {
						throw std::exception();
					}
				}
			}
		}
	}
};
