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

template <typename DataType>
struct Matrix {
  size_t rows, cols;
  unsigned int row_blocks, col_blocks;
  starpu_data_handle_t data_handle;
  bool isPartitioned = false;

  Matrix() = default;
  
  Matrix(size_t rows_, size_t cols_) : rows(rows_), cols(cols_), row_blocks(1), col_blocks(1) {
    starpu_matrix_data_register(&data_handle, -1, 0, rows, rows, cols, sizeof(DataType));
  };
  
  Matrix(size_t rows_, size_t cols_, size_t block_size) : rows(rows_), cols(cols_), row_blocks(1), col_blocks(1) {
    starpu_matrix_data_register(&data_handle, -1, 0, rows, rows, cols, sizeof(DataType));
    partition(block_size);
  };

  ~Matrix() {
    // Need to unpartition before unregistering ?
    if(isPartitioned) { unpartition(); }
    starpu_data_unregister(data_handle);
  }

  void partition(size_t block_size) {
#if ENABLE_REDUX != 0
    starpu_data_set_reduction_methods(data_handle, &accumulate_matrix_cl<DataType>, &bzero_matrix_cl<DataType>);
#endif
    
    if(isPartitioned) { unpartition(); }
    
    row_blocks = (rows + block_size - 1)/block_size;
    col_blocks = (cols + block_size - 1)/block_size;
    
    starpu_data_filter row_partition {
      .filter_func = starpu_matrix_filter_block,
      .nchildren = row_blocks
    };
  
    starpu_data_filter col_partition {
      .filter_func = starpu_matrix_filter_vertical_block,
      .nchildren = col_blocks
    };
    
    starpu_data_map_filters(data_handle, 2, &row_partition, &col_partition);
    isPartitioned = true;
  }

  void unpartition() {
    starpu_data_unpartition(data_handle, STARPU_MAIN_RAM);
    row_blocks = 1;
    col_blocks = 1;
    isPartitioned = false;
  }

  //DataType& operator()(size_t row, size_t col) { return elem(row, col); }
  
  //void fill_random() {
  //  for(int i = 0; i < rows * cols; i++) { data[i] = (DataType)std::rand()/RAND_MAX; }
  //}

  void fill(DataType e) {
    for(int i = 0; i < row_blocks; i++) {
      for(int j = 0; j < col_blocks; j++) {
        starpu_data_handle_t sub_handle = starpu_data_get_sub_data(data_handle, 2, i, j);
        int err = starpu_task_insert(&fill_cl<DataType>,
                                     STARPU_VALUE, &e, sizeof(e),
                                     STARPU_W, sub_handle,
                                     0);
        if(err) { throw std::exception(); }
      }
    }
  }
  
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
  
  static void gemm(char transA, char transB, DataType alpha, Matrix<DataType>& A, Matrix<DataType>& B, DataType beta, Matrix<DataType>& C) {
    assert(A.rows == C.rows);
    assert(B.cols == C.cols);
    assert(A.cols == B.rows);
    assert(A.row_blocks == C.row_blocks);
    assert(B.col_blocks == C.col_blocks);
    assert(A.col_blocks == B.row_blocks);
    
    for(int i = 0; i < C.row_blocks; i++) {
      for(int j = 0; j < C.col_blocks; j++) {
        for(int k = 0; k < A.col_blocks; k++) {
          starpu_data_handle_t A_sub_handle = starpu_data_get_sub_data(A.data_handle, 2, i, k);
          starpu_data_handle_t B_sub_handle = starpu_data_get_sub_data(B.data_handle, 2, k, j);
          starpu_data_handle_t C_sub_handle = starpu_data_get_sub_data(C.data_handle, 2, i, j);
          int err = starpu_task_insert(&gemm_cl<DataType>,
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
