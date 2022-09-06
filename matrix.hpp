#ifndef MATRIX_HPP
#define MATRIX_HPP

template <typename DataType>
struct Matrix {
  DataType * data;
  size_t rows, cols;
  starpu_data_handle_t data_handle;

  /* CODELETS */
  static int can_execute(unsigned workerid, struct starpu_task * task, unsigned nimpl) {
    if(starpu_worker_get_type(workerid) == STARPU_CPU_WORKER) return enable_cpu;
    return enable_gpu;
  }

  static struct starpu_perfmodel gemm_perf_model =
    {
      .type = STARPU_HISTORY_BASED,
      .symbol = "gemm_perf_model"
    };

  starpu_codelet gemm_cl = {
    .can_execute = can_execute,
    .cpu_funcs = { gemm_cpu_func<DataType> },
    .cuda_funcs = { gemm_cuda_func<dataType> },
    .cuda_flags = { STARPU_CUDA_ASYNC },
    .nbuffers = 3,
#if ENABLE_REDUX != 0 && TWODIM != 0
    .modes = { STARPU_R, STARPU_R, STARPU_REDUX },
#else
    .modes = { STARPU_R, STARPU_R, STARPU_RW },
#endif
    .model = &gemm_perf_model,
  };

  starpu_codelet bzero_matrix_cl = {
    .can_execute = can_execute,
    .cpu_funcs = { bzero_matrix_cpu<DataType> },
    .cuda_funcs = { bzero_matrix_cuda<DataType> },
    //.cpu_funcs_name = { "bzero_matrix_cpu" },
    .nbuffers = 1,
    .modes = { STARPU_W },
  };

  static struct starpu_perfmodel accumulate_perf_model =
    {
      .type = STARPU_HISTORY_BASED,
      .symbol = "accumulate_perf_model"
    };

  starpu_codelet accumulate_matrix_cl = {
    .can_execute = can_execute,
    .cpu_funcs = { accumulate_matrix_cpu<DataType> },
    .cuda_funcs = { accumulate_matrix_cuda<DataType> },
    //.cpu_funcs_name = { "accumulate_matrix_cpu" },
    .nbuffers = 2,
    .modes = { starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), STARPU_R },
    .model = &accumulate_perf_model,
  };
  

  Matrix() = default;
  
  Matrix(size_t rows_, size_t cols_) : rows(rows_), cols(cols_) {
    int err = starpu_malloc((void**)&data, rows * cols * sizeof(DataType));
    if(err) { printf("MALLOC FAILED\n"); }
  };

  ~Matrix() {
    int err = starpu_free_noflag(data, rows * cols * sizeof(DataType));
    if(err) { printf("FREE FAILED\n"); }
  }

  DataType& operator()(size_t row, size_t col) { return elem(row, col); }

  Matrix<DataType> operator/(Matrix<DataType>& other) {
    Matrix<DataType> result(rows, cols);
    for(int i = 0; i < cols; i++) {
      for(int j = 0; j < rows; j++) {
        result(j, i) = (*this)(j, i)/other(j, i);
      }
    }
    return result;
  }
  
  void fill_random() {
    for(int i = 0; i < rows * cols; i++) { data[i] = (DataType)std::rand()/RAND_MAX; }
  }

  void fill(DataType e) {
    for(int i = 0; i < rows * cols; i++) { data[i] = e; }
  }

  void assertEq(DataType e) {
    for(int i = 0; i < rows * cols; i++) { if(data[i] != e) { std::cout << data[i] << " " << e << std::endl; break; } }
  }

  void print_sci() {
    for(size_t i = 0; i < rows; i++) {
      for(size_t j = 0; j < cols; j++) {
        std::cout << std::scientific << elem(i, j) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl << std::defaultfloat;
  }

  void print() {
    for(size_t i = 0; i < rows; i++) {
      for(size_t j = 0; j < cols; j++) {
        printf("%5.2f ", elem(i, j));
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  void data_register() {
    starpu_matrix_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t)&data[0], rows, rows, cols, sizeof(data[0]));
  }

  void data_unregister() {
    starpu_data_unregister(data_handle);
  }

  static void gemm(char transA, char transB, DataType alpha, Matrix<DataType>& A, Matrix<DataType>& B, DataType beta, Matrix<DataType>& C, int block_size);

private:
  DataType& elem(size_t row, size_t col) { return data[row + col * rows]; }
};

#endif
