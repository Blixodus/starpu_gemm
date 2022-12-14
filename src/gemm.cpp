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
#include <fmt/core.h>

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

template <typename DataType>
void test_gemm(u32 m, u32 n, u32 k, u32 block_size, bool quiet) {
	if (!quiet) {
		fmt::print(
			"2D={} Reduction={} CPU={} GPU={} M={} N={} K={} BS={} DT={}\n", TWODIM, ENABLE_REDUX, enable_cpu, enable_gpu, m, n,
			k, block_size, type_name<DataType>()
		);
	}

	Matrix<DataType> A(m, k, block_size), B(k, n, block_size), C(m, n, block_size);

	A.fill(1);
	B.fill(3);
	C.fill(0);

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	auto start = std::chrono::high_resolution_clock::now();

	Matrix<DataType>::gemm('N', 'N', 1.0f, A, B, 1.0f, C);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
	auto flops = 2.0 * m * n * k / time.count() / 1e12;
	
	if (quiet) {
		fmt::print("{},{},{:.6f},{},{:.3f},{:.3f}\n", m, 0, time.count(), 0, flops * 1000, flops * 1000);
	} else {
		fmt::print("StarPU -- Time : {}s\n", time.count());
		fmt::print("StarPU -- Performance : {:.3f}Tflop/s\n", 2.0 * m * n * k / time.count() / 1e12);
	}

	C.assertEq(static_cast<DataType>(k * 3));
}

void printHelp() {
	std::cout << "Parameters for gemm:\n"
			  << "  -m        		 --  Set the size of M (as log)\n"
			  << "  -n        		 --  Set the size of N (as log)\n"
			  << "  -k        		 --  Set the size of K (as log)\n"
			  << "  -b        		 --  Set the block size (as log)\n"
			  << "  -g        		 --  Set the number of GPUs used for task execution\n"
			  << "  -t <s/d>  		 --  Set the data type (double/single)" << std::endl;
}

void parseArgs(
	int argc,
	char** argv,
	u32& m,
	u32& n,
	u32& k,
	u32& b,
	char& type,
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
	type = (hasArg(argc, argv, "-t") ? (char) tolower(getArg(argc, argv, "-t")[0]) : 's');
    quiet = hasArg(argc, argv, "-q") ? true : false;
}

int main(int argc, char** argv) {
	u32 m, n, k, b;
	bool quiet;
	char type;
	parseArgs(argc, argv, m, n, k, b, type, quiet);

	if (starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL)) {
		throw std::runtime_error("Unable to init starpu_mpi");
	}

#ifdef USE_CUDA
	starpu_cublas_init();
#endif

	switch (type) {
		case 's':
			test_gemm<f32>(m, n, k, b, quiet);
		break;

		case 'd':
			test_gemm<f64>(m, n, k, b, quiet);
		break;

		default:
			fmt::print("Invalid data type: {}, must be s or d\n", type);
			return 1;
	}

#ifdef USE_CUDA
	starpu_cublas_shutdown();
#endif
	starpu_mpi_shutdown();

	return 0;
}
