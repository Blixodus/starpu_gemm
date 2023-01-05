#include <starpu.h>
#include <starpu_mpi.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <string_view>

#define TWODIM 1
#if defined(HAVE_STARPU_MPI_REDUX)
#define ENABLE_REDUX 0
#else
#warning "ENABLE_REDUX 0"
#define ENABLE_REDUX 0
#endif

#include "matrix/matrix.hpp"

void test_gemm(u32 m, u32 n, u32 k, u32 block_size, std::ofstream& resultFile) {
	std::cerr << "2D=" << TWODIM << " Reduction=" << ENABLE_REDUX << " CPU=" << enable_cpu << " GPU=" << enable_gpu << " M=" << m << " N=" << n << " K=" << k << " BS=" << block_size << std::endl;

	Matrix<float> A(m, k, block_size), B(k, n, block_size), C(m, n, block_size);

	A.fill(1);
	B.fill(1);
	C.fill(0);

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	auto start = std::chrono::high_resolution_clock::now();

	Matrix<float>::gemm('N', 'N', 1.0f, A, B, 1.0f, C);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
	std::cerr << "StarPU -- Time : " << time.count() << "s\n";
	std::cerr << "StarPU -- Performance : " << 2.0 * m * n * k / time.count() / 1e12 << "Tflop/s" << std::endl;

	// resultFile << enable_cpu << ";" << enable_gpu << ";" << m << ";" << n << ";" << k << ";" <<
	// block_size << ";" << 2L * m * n * k / time.count() / 1e12 << std::endl;

	C.assertEq(static_cast<float>(k));
}

char* getArg(int argc, char** argv, std::string_view arg) {
  char** begin = argv;
  char** end = argv + argc;
  char** itr = std::find(begin, end, arg);
  if(itr != end && itr++ != end) {
    return *itr;
  }
  return 0;
}

bool hasArg(int argc, char** argv, std::string_view arg) {
  char** begin = argv;
  char** end = argv+argc;
  return (std::find(begin, end, arg) != end);
}

void printHelp() {
  std::cout << "Parameters for gemm:" << std::endl
            << "  -m     --  Set the size of M (as log)" << std::endl
            << "  -n     --  Set the size of N (as log)" << std::endl
            << "  -k     --  Set the size of K (as log)" << std::endl
            << "  -b     --  Set the block size (as log)" << std::endl
            << "  -g     --  Set the number of GPUs used for task execution" << std::endl
            << "  --perf-test  --  Test CUBLAS performace on given sizes" << std::endl;
}

void parseArgs(int argc, char** argv, u32& m, u32& n, u32& k_min, u32& k_max, u32& b_min, u32& b_max, starpu_conf& conf) {
  if(hasArg(argc, argv, "-h")) { printHelp(); exit(0); }
  m = 1<<((hasArg(argc, argv, "-m")) ? stoui(getArg(argc, argv, "-m")) : 10);
  n = 1<<((hasArg(argc, argv, "-n")) ? stoui(getArg(argc, argv, "-n")) : 10);
  k_min = k_max = ((hasArg(argc, argv, "-k")) ? stoui(getArg(argc, argv, "-k")) : 10);
  b_min = b_max = ((hasArg(argc, argv, "-b")) ? stoui(getArg(argc, argv, "-b")) : 10);
}

int main(int argc, char** argv) {
	u32  m, n, k_min, k_max, b_min, b_max;
  starpu_conf conf;
  starpu_conf_init(&conf);
  parseArgs(argc, argv, m, n, k_min, k_max, b_min, b_max, conf);

	std::ofstream resultFile;
	/* 
	char buffer[50];
	sprintf(buffer, "results/results_%s_%s_%s_%d_%d.csv", std::getenv("STARPU_SCHED"),
  TWODIM?"2D":"1D", ENABLE_REDUX?"RED":"NONRED", m, n); std::cerr << "Printing results in " <<
  buffer << std::endl; resultFile.open(buffer); resultFile << "CPU;GPU;M;N;K;BLOCK;TFLOPS" <<
  std::endl;
  */

#ifdef USE_CUDA
  if(hasArg(argc, argv, "--perf-test")) {
    for(int k_exp = k_min; k_exp <= k_max; k_exp++) {
      const int k = 1<<k_exp;
      cublas_perf_test(m, n, k, true, resultFile);
    }
  }
#endif

	int err = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, &conf);
	if (err) {
		throw std::exception();
	}

#ifdef USE_CUDA
	starpu_cublas_init();
#endif

	for (u32 b_exp = b_min; b_exp <= b_max; b_exp++) {
		const u32 block_size = 1 << b_exp;
		for (u32 k_exp = k_min; k_exp <= k_max; k_exp++) {
			const u32 k = 1 << k_exp;
			test_gemm(m, n, k, block_size, resultFile);
		}
	}

#ifdef USE_CUDA
	starpu_cublas_shutdown();
#endif
	starpu_mpi_shutdown();

	// resultFile.close();
	return 0;
}
