#ifndef MATRIX_HPP
#define MATRIX_HPP

static int enable_cpu = ENABLE;
static int enable_gpu = ENABLE;
static int can_execute(unsigned workerid, struct starpu_task * task, unsigned nimpl) {
  if(starpu_worker_get_type(workerid) == STARPU_CPU_WORKER) return enable_cpu;
  return enable_gpu;
}

static struct starpu_perfmodel gemm_perf_model_float = {
    .type = STARPU_REGRESSION_BASED,
    .symbol = "gemm_perf_model_float"
};

static struct starpu_perfmodel gemm_perf_model_double = {
    .type = STARPU_REGRESSION_BASED,
    .symbol = "gemm_perf_model_double"
};

template <typename DataType>
starpu_codelet gemm_cl = {
  .can_execute = can_execute,
  .cpu_funcs = { gemm_cpu_func<DataType> },
#ifdef USE_CUDA
  .cuda_funcs = { gemm_cuda_func<DataType> },
  .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
  .nbuffers = 3,
#if ENABLE_REDUX != 0
  .modes = { STARPU_R, STARPU_R, starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE) },
#else
  .modes = { STARPU_R, STARPU_R, STARPU_RW },
#endif
  .model = (std::is_same_v<DataType, float>)?&gemm_perf_model_float:&gemm_perf_model_double,
};

template <typename DataType>
starpu_codelet bzero_matrix_cl = {
  .can_execute = can_execute,
  .cpu_funcs = { bzero_matrix_cpu<DataType> },
#ifdef USE_CUDA
  .cuda_funcs = { bzero_matrix_cuda<DataType> },
#endif
  .nbuffers = 1,
  .modes = { STARPU_W },
};

static struct starpu_perfmodel accumulate_perf_model_float =
{
    .type = STARPU_REGRESSION_BASED,
    .symbol = "accumulate_perf_model_float"
};

static struct starpu_perfmodel accumulate_perf_model_double =
{
    .type = STARPU_REGRESSION_BASED,
    .symbol = "accumulate_perf_model_double"
};

template <typename DataType>
starpu_codelet accumulate_matrix_cl = {
  .can_execute = can_execute,
  .cpu_funcs = { accumulate_matrix_cpu<DataType> },
#ifdef USE_CUDA
  .cuda_funcs = { accumulate_matrix_cuda<DataType> },
#endif
  .nbuffers = 2,
  .modes = { starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), STARPU_R },
  .model = (std::is_same_v<DataType, float>)?&accumulate_perf_model_float:&accumulate_perf_model_double,
};

static struct starpu_perfmodel fill_perf_model_float =
{
    .type = STARPU_REGRESSION_BASED,
    .symbol = "fill_perf_model_float"
};

static struct starpu_perfmodel fill_perf_model_double =
{
    .type = STARPU_REGRESSION_BASED,
    .symbol = "fill_perf_model_double"
};

template <typename DataType>
starpu_codelet fill_cl = {
  .cpu_funcs = { fill_cpu_func<DataType> },
#ifdef USE_CUDA
  .cuda_funcs = { fill_cuda_func<DataType> },
  .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
  .nbuffers = 1,
  .modes = { STARPU_W },
  .model = (std::is_same_v<DataType, float>)?&fill_perf_model_float:&fill_perf_model_double,
};

static int mpi_tag = 0;

template <typename DataType, size_t rows, size_t cols>
struct MatrixData {
  starpu_data_handle_t data_handle[rows * cols];

  MatrixData(size_t mat_rows, size_t mat_cols) {
    int rank, size;
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
    for(size_t i = 0; i < rows * cols; i++) {
      starpu_matrix_data_register(&data_handle[i], -1, 0, mat_rows, mat_rows, mat_cols, sizeof(DataType));
      starpu_mpi_data_register(data_handle[i], mpi_tag++, i%size);
    }
  }

  ~MatrixData() {
    for(size_t i = 0; i < rows * cols; i++) {
      starpu_data_unregister(data_handle[i]);
    }
  }

  starpu_data_handle_t get(int i, int j) {
    return data_handle[i + j * rows];
  }
};

template <typename DataType, size_t row_blocks, size_t col_blocks>
struct Matrix {
  size_t rows, cols;
  size_t row_b = row_blocks, col_b = col_blocks;
  MatrixData<DataType, row_blocks, col_blocks> data_handle;

  Matrix() = default;
  
  Matrix(size_t rows_, size_t cols_) : data_handle(rows_, cols_) { };

  //DataType& operator()(size_t row, size_t col) { return elem(row, col); }
  
  //void fill_random() {
  //  for(int i = 0; i < rows * cols; i++) { data[i] = (DataType)std::rand()/RAND_MAX; }
  //}

  void fill(DataType e) {
    for(int i = 0; i < row_blocks; i++) {
      for(int j = 0; j < col_blocks; j++) {
        starpu_data_handle_t handle = data_handle.get(i, j);
        int err = starpu_mpi_task_insert(MPI_COMM_WORLD, &fill_cl<DataType>,
                                         STARPU_VALUE, &e, sizeof(e),
                                         STARPU_W, handle,
                                         0);
        if(err) { throw std::exception(); }
      }
    }
  }

  /*
  void assertEq(DataType e) {
    unpartition();
    size_t nx = starpu_matrix_get_nx(data_handle);
    size_t ny = starpu_matrix_get_ny(data_handle);
    size_t ld = starpu_matrix_get_local_ld(data_handle);
    DataType *mat = (DataType*)starpu_matrix_get_local_ptr(data_handle);
    for(size_t i = 0; i < nx; i++) {
      for(size_t j = 0; j < ny; j++) {
        if(mat[i + j*ld] != e) {
          std::cout << mat[i + j*ld] << " " << e << std::endl;
          goto end;
        }
      }
    }
  end:
    return;
  }
  
  void print() {
    unpartition();
    size_t nx = starpu_matrix_get_nx(data_handle);
    size_t ny = starpu_matrix_get_ny(data_handle);
    size_t ld = starpu_matrix_get_local_ld(data_handle);
    DataType *mat = (DataType*)starpu_matrix_get_local_ptr(data_handle);
    for(size_t i = 0; i < nx; i++) {
      for(size_t j = 0; j < ny; j++) {
        printf("%5.2f ", mat[i + j*ld]);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  */
  
  static void gemm(char transA, char transB, DataType alpha, Matrix<DataType, row_blocks, col_blocks>& A, Matrix<DataType, row_blocks, col_blocks>& B, DataType beta, Matrix<DataType, row_blocks, col_blocks>& C) {
    assert(A.rows == C.rows);
    assert(B.cols == C.cols);
    assert(A.cols == B.rows);
    assert(A.row_b == C.row_b);
    assert(B.col_b == C.col_b);
    assert(A.col_b == B.row_b);
    
    for(int i = 0; i < C.row_b; i++) {
      for(int j = 0; j < C.col_b; j++) {
        for(int k = 0; k < A.col_b; k++) {
          starpu_data_handle_t A_sub_handle = A.data_handle.get(i, k);
          starpu_data_handle_t B_sub_handle = B.data_handle.get(k, j);
          starpu_data_handle_t C_sub_handle = C.data_handle.get(i, j);
          int err = starpu_mpi_task_insert(MPI_COMM_WORLD, &gemm_cl<DataType>,
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
                                           0);
          if(err) { throw std::exception(); }
        }
      }
    }
  }

//private:
  //DataType& elem(size_t row, size_t col) { return data[row + col * rows]; }
};

#endif
