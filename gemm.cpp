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
    .type = STARPU_HISTORY_BASED,
    .symbol = "gemm_perf_model_float"
};

static struct starpu_perfmodel gemm_perf_model_double =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "gemm_perf_model_double"
};

template <typename DataType>
starpu_codelet gemm_cl = {
  .can_execute = can_execute,
  .cpu_funcs = { gemm_cpu_func<DataType> },
  .cuda_funcs = { gemm_cuda_func<DataType> },
  .cuda_flags = { STARPU_CUDA_ASYNC },
  .nbuffers = 3,
#if ENABLE_REDUX != 0 && TWODIM != 0
  .modes = { STARPU_R, STARPU_R, STARPU_REDUX },
#else
  .modes = { STARPU_R, STARPU_R, STARPU_RW },
#endif
  .model = (std::is_same_v<DataType, float>)?&gemm_perf_model_float:&gemm_perf_model_double,
};

template <typename DataType>
starpu_codelet bzero_matrix_cl = {
  .cpu_funcs = { bzero_matrix_cpu<DataType> },
  .cuda_funcs = { bzero_matrix_cuda<DataType> },
  .nbuffers = 1,
  .modes = { STARPU_W },
};

static struct starpu_perfmodel accumulate_perf_model_float =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "accumulate_perf_model_float"
};

static struct starpu_perfmodel accumulate_perf_model_double =
{
    .type = STARPU_HISTORY_BASED,
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

template <typename DataType>
struct Matrix {
  DataType * data;
  size_t rows, cols;
  starpu_data_handle_t data_handle;

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

  static void gemm(char transA, char transB, DataType alpha, Matrix<DataType>& A, Matrix<DataType>& B, DataType beta, Matrix<DataType>& C, int block_size) {
    unsigned int m_blocks = (A.rows + block_size - 1)/block_size;
    unsigned int k_blocks = (A.cols + block_size - 1)/block_size;
    unsigned int n_blocks = (B.cols + block_size - 1)/block_size;
    
    starpu_data_filter m_partition {
      .filter_func = starpu_matrix_filter_block,
      .nchildren = m_blocks
    };

    starpu_data_filter k_partition_vertical {
      .filter_func = starpu_matrix_filter_vertical_block,
      .nchildren = k_blocks
    };

    starpu_data_filter k_partition_horizontal {
      .filter_func = starpu_matrix_filter_block,
      .nchildren = k_blocks
    };
  
    starpu_data_filter n_partition {
      .filter_func = starpu_matrix_filter_vertical_block,
      .nchildren = n_blocks
    };

#if ENABLE_REDUX != 0 && TWODIM != 0
    starpu_data_set_reduction_methods(C.data_handle, &accumulate_matrix_cl<DataType>, &bzero_matrix_cl<DataType>);
#endif
    
#if TWODIM != 0
    starpu_data_map_filters(A.data_handle, 2, &m_partition, &k_partition_vertical);
    starpu_data_map_filters(B.data_handle, 2, &k_partition_horizontal, &n_partition);
#else
    starpu_data_partition(A.data_handle, &m_partition);
    starpu_data_partition(B.data_handle, &n_partition);
#endif
    starpu_data_map_filters(C.data_handle, 2, &m_partition, &n_partition);

#if TWODIM != 0
    for(int i = 0; i < m_blocks; i++) {
      for(int j = 0; j < n_blocks; j++) {
        for(int k = 0; k < k_blocks; k++) {
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
                                       STARPU_REDUX, C_sub_handle,
#else
                                       STARPU_RW, C_sub_handle,
#endif
                                       0);
          if(err) { throw std::exception(); }
        }
      }
    }
#else
    for(int i = 0; i < m_blocks; i++) {
      for(int j = 0; j < n_blocks; j++) {
        starpu_data_handle_t A_sub_handle = starpu_data_get_sub_data(A.data_handle, 1, i);
        starpu_data_handle_t B_sub_handle = starpu_data_get_sub_data(B.data_handle, 1, j);
        starpu_data_handle_t C_sub_handle = starpu_data_get_sub_data(C.data_handle, 2, i, j);
        int err = starpu_task_insert(&gemm_cl<DataType>,
                                     STARPU_VALUE, &transA, sizeof(transA),
                                     STARPU_VALUE, &transB, sizeof(transB),
                                     STARPU_VALUE, &alpha, sizeof(alpha),
                                     STARPU_VALUE, &beta, sizeof(beta),
                                     STARPU_R, A_sub_handle,
                                     STARPU_R, B_sub_handle,
                                     STARPU_RW, C_sub_handle,
                                     0);
        if(err) { throw std::exception(); }
      }
    }
#endif
    
    starpu_data_unpartition(A.data_handle, STARPU_MAIN_RAM);
    starpu_data_unpartition(B.data_handle, STARPU_MAIN_RAM);
    starpu_data_unpartition(C.data_handle, STARPU_MAIN_RAM);
  }

private:
  DataType& elem(size_t row, size_t col) { return data[row + col * rows]; }
};

void test(int m, int n, int k, int block_size, std::ofstream& resultFile) {
  std::cerr << "2D=" << TWODIM << " Reduction=" << ENABLE_REDUX << " CPU=" << enable_cpu << " GPU=" << enable_gpu << " M=" << m << " N=" << n << " K=" << k << " BS=" << block_size << std::endl;
  
  Matrix<double> A(m, k), B(k, n), C(m, n);
  
  A.fill(1);
  B.fill(1);
  C.fill(0);
  
  A.data_register();
  B.data_register();
  C.data_register();
  
  auto start = std::chrono::high_resolution_clock::now();
  
  Matrix<double>::gemm('N', 'N', 1.0f, A, B, 1.0f, C, block_size);
  
  std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
  std::cerr << "StarPU -- Time : " << time.count() << "s\n";
  std::cerr << "StarPU -- Performance : " << 2L * m * n * k / time.count() / 1e12 << "Tflop/s" << std::endl;
  
  resultFile << enable_cpu << ";" << enable_gpu << ";" << m << ";" << n << ";" << k << ";" << block_size << ";" << 2L * m * n * k / time.count() / 1e12 << std::endl;
  
  A.data_unregister();
  B.data_unregister();
  C.data_unregister();

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
  
  for(int b_exp = b_min; b_exp <= b_max; b_exp++) {
    const int block_size = 1<<b_exp;
    for(int k_exp = k_min; k_exp <= k_max; k_exp++) {
      const int k = 1<<k_exp;
      test(m, n, k, block_size, resultFile);
    }
  }
  
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
  
  starpu_cublas_shutdown();
  starpu_shutdown();

  resultFile.close();
  std::cout << buffer << std::endl;
  return 0;
}
