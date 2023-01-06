#include <starpu.h>
#include <starpu_mpi.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "fmt/core.h"
#include "matrix/matrix.hpp"
#include "util/helper.hpp"
#include "util/argparse.hpp"

#define TWODIM 1
#if defined(HAVE_STARPU_MPI_REDUX)
#define ENABLE_REDUX 1
#else
#warning "ENABLE_REDUX 0"
#define ENABLE_REDUX 0
#endif

void test_gemm(u32 m, u32 n, u32 k, u32 block_size, bool quiet) {
	if (!quiet) {
		fmt::print(
			"2D={} Reduction={} CPU={} GPU={} M={} N={} K={} BS={}\n", TWODIM, ENABLE_REDUX, enable_cpu, enable_gpu, m, n,
			k, block_size
		);
	}

	Matrix<double> A(m, k, block_size), B(k, n, block_size), C(m, n, block_size);

	A.fill(1);
	B.fill(3);
	C.fill(0);

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	auto start = std::chrono::high_resolution_clock::now();

	Matrix<double>::gemm('N', 'N', 1.0f, A, B, 1.0f, C);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
	auto flops = 2.0 * m * n * k / time.count() / 1e12;
	
	if (quiet) {
		fmt::print("{},{},{},{},{:.3f}\n", m, n, k, block_size, flops * 1000);
	} else {
		fmt::print("StarPU -- Time : {}s\n", time.count());
		fmt::print("StarPU -- Performance : {:.3f}Tflop/s\n", 2.0 * m * n * k / time.count() / 1e12);
	}

	C.assertEq(static_cast<float>(k * 3));
}

void printHelp() {
	std::cout << "Parameters for gemm:\n"
			  << "  -m     --  Set the size of M (as log)\n"
			  << "  -n     --  Set the size of N (as log)\n"
			  << "  -k     --  Set the size of K (as log)\n"
			  << "  -b     --  Set the block size (as log)\n"
			  << "  -g     --  Set the number of GPUs used for task execution\n"
			  << "  --perf-test  --  Test CUBLAS performace on given sizes" << std::endl;
}

void parseArgs(
	int argc,
	char** argv,
	u32& m,
	u32& n,
	u32& k,
	u32& b,
	bool& quiet
) {
	if (hasArg(argc, argv, "-h")) {
		printHelp();
		exit(0);
	}

	m = 1 << (hasArg(argc, argv, "-m") ? stoui(getArg(argc, argv, "-m")) : 10);
	n = 1 << (hasArg(argc, argv, "-n") ? stoui(getArg(argc, argv, "-n")) : 10);
	k = 1 << (hasArg(argc, argv, "-k") ? stoui(getArg(argc, argv, "-k")) : 10);
	b = 1 << (hasArg(argc, argv, "-b") ? stoui(getArg(argc, argv, "-b")) : 10);
    quiet = hasArg(argc, argv, "-q") ? true : false;
}

int main(int argc, char** argv) {
	u32 m, n, k, b;
	bool quiet;
	parseArgs(argc, argv, m, n, k, b, quiet);

	if (starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL)) {
		throw std::exception();
	}

#ifdef USE_CUDA
	starpu_cublas_init();
#endif

	test_gemm(m, n, k, b, quiet);

#ifdef USE_CUDA
	starpu_cublas_shutdown();
#endif
	starpu_mpi_shutdown();

	return 0;
}
