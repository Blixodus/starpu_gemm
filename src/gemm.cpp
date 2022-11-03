#include <starpu.h>
#include <starpu_mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <fstream>

#define ENABLE 1
#define DISABLE 0
#define TWODIM 1
#if defined (HAVE_STARPU_MPI_REDUX)
#define ENABLE_REDUX 1
#else
#warning "ENABLE_REDUX 0"
#define ENABLE_REDUX 0
#endif

#include "cublas_perf.hpp"
#include "gemm_func.hpp"
#include "bzero_func.hpp"
#include "accumulate_func.hpp"
#include "fill_func.hpp"
#include "asserteq_func.hpp"
#include "print_func.hpp"
#include "matrix.hpp"

void test_gemm(int m, int n, int k, int block_size, std::ofstream& resultFile) {
  std::cerr << "2D=" << TWODIM << " Reduction=" << ENABLE_REDUX << " CPU=" << enable_cpu << " GPU=" << enable_gpu << " M=" << m << " N=" << n << " K=" << k << " BS=" << block_size << std::endl;
  
  Matrix<float> A(m, k, block_size), B(k, n, block_size), C(m, n, block_size);
  
  A.fill(1);
  B.fill(1);
  C.fill(0);

  /*
  A.print('A');
  starpu_mpi_barrier(MPI_COMM_WORLD);
  starpu_mpi_wait_for_all(MPI_COMM_WORLD);
  sleep(2);
  B.print('B');
  starpu_mpi_barrier(MPI_COMM_WORLD);
  starpu_mpi_wait_for_all(MPI_COMM_WORLD);
  sleep(2);
  C.print('C');
  starpu_mpi_barrier(MPI_COMM_WORLD);
  starpu_mpi_wait_for_all(MPI_COMM_WORLD);
  sleep(2);
  */
  
  auto start = std::chrono::high_resolution_clock::now();
  
  Matrix<float>::gemm('N', 'N', 1.0f, A, B, 1.0f, C);
  starpu_mpi_barrier(MPI_COMM_WORLD);
  starpu_mpi_wait_for_all(MPI_COMM_WORLD);
  
  std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
  std::cerr << "StarPU -- Time : " << time.count() << "s\n";
  std::cerr << "StarPU -- Performance : " << 2L * m * n * k / time.count() / 1e12 << "Tflop/s" << std::endl;
  
  //resultFile << enable_cpu << ";" << enable_gpu << ";" << m << ";" << n << ";" << k << ";" << block_size << ";" << 2L * m * n * k / time.count() / 1e12 << std::endl;

  C.assertEq(k);
  /*
  starpu_mpi_barrier(MPI_COMM_WORLD);
  starpu_mpi_wait_for_all(MPI_COMM_WORLD);
  sleep(2);
  C.print('C');
  starpu_mpi_barrier(MPI_COMM_WORLD);
  starpu_mpi_wait_for_all(MPI_COMM_WORLD);
  */
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
  /*
  char buffer[50];
  sprintf(buffer, "results/results_%s_%s_%s_%d_%d.csv", std::getenv("STARPU_SCHED"), TWODIM?"2D":"1D", ENABLE_REDUX?"RED":"NONRED", m, n);
  std::cerr << "Printing results in " << buffer << std::endl;
  resultFile.open(buffer);
  resultFile << "CPU;GPU;M;N;K;BLOCK;TFLOPS" << std::endl;
  
#ifdef USE_CUDA
  for(int k_exp = k_min; k_exp <= k_max; k_exp++) {
    const int k = 1<<k_exp;
    cublas_perf_test(m, n, k, true, resultFile);
  }
#endif
  */

  int err = starpu_init(NULL);
  if(err) { throw std::exception(); }
  err = starpu_mpi_init(&argc, &argv, 1);
  if(err) { throw std::exception(); }
#ifdef USE_CUDA
  starpu_cublas_init();
#endif
  //test_gemm(5, 5, 5, 2, resultFile);
  
  for(int b_exp = b_min; b_exp <= b_max; b_exp++) {
    const int block_size = 1<<b_exp;
    for(int k_exp = k_min; k_exp <= k_max; k_exp++) {
      const int k = 1<<k_exp;
      test_gemm(m, n, k, block_size, resultFile);
    }
  }
  
#ifdef USE_CUDA
  starpu_cublas_shutdown();
#endif
  starpu_mpi_shutdown();
  starpu_shutdown();

  //resultFile.close();
  //std::cout << buffer << std::endl;
  return 0;
}
