#include <starpu.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "cublas_perf.hpp"
#include "gemm_func.hpp"
#include "bzero_func.hpp"
#include "accumulate_func.hpp"
#include "fill_func.hpp"

#define ENABLE 1
#define DISABLE 0
#define ENABLE_REDUX 0
#define TWODIM 1

static int enable_cpu = ENABLE;
static int enable_gpu = ENABLE;
static int can_execute(unsigned workerid, struct starpu_task * task, unsigned nimpl) {
  if(starpu_worker_get_type(workerid) == STARPU_CPU_WORKER) return enable_cpu;
  return enable_gpu;
}

static struct starpu_perfmodel gemm_perf_model_float =
{
    .type = STARPU_REGRESSION_BASED,
    .symbol = "gemm_perf_model_float"
};

static struct starpu_perfmodel gemm_perf_model_double =
{
    .type = STARPU_REGRESSION_BASED,
    .symbol = "gemm_perf_model_double"
};

template <typename DataType>
starpu_codelet gemm_cl = {
  .can_execute = can_execute,
  .cpu_funcs = { gemm_cpu_func<DataType> },
  .cuda_funcs = { gemm_cuda_func<DataType> },
  .cuda_flags = { STARPU_CUDA_ASYNC },
  .nbuffers = 3,
  //#if ENABLE_REDUX != 0 && TWODIM != 0
  //.modes = { STARPU_R, STARPU_R, STARPU_REDUX },
  //#else
  .modes = { STARPU_R, STARPU_R, STARPU_RW },
  //#endif
  .model = (std::is_same_v<DataType, float>)?&gemm_perf_model_float:&gemm_perf_model_double,
};

template <typename DataType>
starpu_codelet bzero_matrix_cl = {
  .can_execute = can_execute,
  .cpu_funcs = { bzero_matrix_cpu<DataType> },
  .cuda_funcs = { bzero_matrix_cuda<DataType> },
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
  .cuda_funcs = { accumulate_matrix_cuda<DataType> },
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
  .cuda_funcs = { fill_cuda_func<DataType> },
  .cuda_flags = { STARPU_CUDA_ASYNC },
  .nbuffers = 1,
  .modes = { STARPU_W },
  .model = (std::is_same_v<DataType, float>)?&fill_perf_model_float:&fill_perf_model_double,
};

template <typename DataType>
struct Matrix {
  size_t rows, cols;
  unsigned int row_blocks, col_blocks;
  starpu_data_handle_t data_handle;

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
    if(row_blocks > 1 || col_blocks > 1) { unpartition(); }
    starpu_data_unregister(data_handle);
  }

  void partition(size_t block_size) {
    if(row_blocks > 1 || col_blocks > 1) { unpartition(); }
    
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
  }

  void unpartition() {
    starpu_data_unpartition(data_handle, STARPU_MAIN_RAM);
    row_blocks = 1;
    col_blocks = 1;
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
                                       STARPU_RW, C_sub_handle,
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

void test(int m, int n, int k, int block_size, std::ofstream& resultFile) {
  std::cerr << "2D=" << TWODIM << " Reduction=" << ENABLE_REDUX << " CPU=" << enable_cpu << " GPU=" << enable_gpu << " M=" << m << " N=" << n << " K=" << k << " BS=" << block_size << std::endl;
  
  Matrix<float> A(m, k, block_size), B(k, n, block_size), C(m, n, block_size);
  
  A.fill(1);
  B.fill(1);
  C.fill(0);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  Matrix<float>::gemm('N', 'N', 1.0f, A, B, 1.0f, C);
  starpu_task_wait_for_all();
  
  std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
  std::cerr << "StarPU -- Time : " << time.count() << "s\n";
  std::cerr << "StarPU -- Performance : " << 2L * m * n * k / time.count() / 1e12 << "Tflop/s" << std::endl;
  
  resultFile << enable_cpu << ";" << enable_gpu << ";" << m << ";" << n << ";" << k << ";" << block_size << ";" << 2L * m * n * k / time.count() / 1e12 << std::endl;

  C.assertEq(k);
}

int main(int argc, char ** argv) {
  if(argc != 6) {
    std::cerr << "Usage : " << argv[0] << " [exp] [k_min] [k_max] [bs_min] [bs_max]" << std::endl;
    return 1;
  }
  const int exp = atoi(argv[1]);
  const int k_min = atoi(argv[2]);
  const int k_max = std::min(atoi(argv[3]), exp);
  const int b_min = atoi(argv[4]);
  const int b_max = std::min(atoi(argv[5]), exp);
  const int m = 1<<exp;
  const int n = 1<<exp;
  
  std::ofstream resultFile;
  char buffer[50];
  sprintf(buffer, "results/results_%s_%s_%s_%d_%d.csv", std::getenv("STARPU_SCHED"), TWODIM?"2D":"1D", ENABLE_REDUX?"RED":"NONRED", m, n);
  std::cerr << "Printing results in " << buffer << std::endl;
  resultFile.open(buffer);
  resultFile << "CPU;GPU;M;N;K;BLOCK;TFLOPS" << std::endl;
  
  for(int k_exp = k_min; k_exp <= k_max; k_exp++) {
    const int k = 1<<k_exp;
    cublas_perf_test(m, n, k, true, resultFile);
  }
  
  int err = starpu_init(NULL);
  if(err) { throw std::exception(); }
  starpu_cublas_init();

  //int w_scope = starpu_perf_knob_scope_name_to_id("per_worker");
  //int w_enable_id = starpu_perf_knob_name_to_id((starpu_perf_knob_scope)w_scope, "starpu.worker.w_enable_worker_knob");
  //int32_t val = starpu_perf_knob_get_per_worker_int32_value(w_enable_id, 5);
  //starpu_perf_knob_set_per_worker_int32_value(w_enable_id, 5, 0);
  
  //enable_cpu = DISABLE;
  for(int b_exp = b_min; b_exp <= b_max; b_exp++) {
    const int block_size = 1<<b_exp;
    for(int k_exp = k_min; k_exp <= k_max; k_exp++) {
      const int k = 1<<k_exp;
      test(m, n, k, block_size, resultFile);
    }
  }
  
  /*
  enable_gpu = DISABLE;
  for(int b_exp = b_min; b_exp <= b_max; b_exp++) {
    const int block_size = 1<<b_exp;
    for(int k_exp = k_min; k_exp <= k_max; k_exp++) {
      const int k = 1<<k_exp;
      test(m, n, k, block_size, resultFile);
    }
  }
  
  enable_cpu = DISABLE;
  enable_gpu = ENABLE;
  for(int b_exp = b_min; b_exp <= b_max; b_exp++) {
    const int block_size = 1<<b_exp;
    for(int k_exp = k_min; k_exp <= k_max; k_exp++) {
      const int k = 1<<k_exp;
      test(m, n, k, block_size, resultFile);
    }
  }
  */
  starpu_cublas_shutdown();
  starpu_shutdown();

  resultFile.close();
  std::cout << buffer << std::endl;
  return 0;
}
