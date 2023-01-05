#include <starpu.h>
#include <starpu_mpi.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <fmt/core.h>
#include "cublas_v2.h"

#include "matrix/matrix.hpp"
#include "util/argparse.hpp"
#include "ppmatrix/ppmatrix.hpp"

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
	bool& quiet
) {
	if (hasArg(argc, argv, "-h")) {
		printHelp();
		exit(0);
	}

	m = 1 << (hasArg(argc, argv, "-m") ? stoui(getArg(argc, argv, "-m")) : 10);
	n = 1 << (hasArg(argc, argv, "-n") ? stoui(getArg(argc, argv, "-n")) : 10);
	k = 1 << (hasArg(argc, argv, "-k") ? stoui(getArg(argc, argv, "-k")) : 10);
    quiet = hasArg(argc, argv, "-q") ? true : false;
}

void test_gemm(cublasHandle_t handle, u32 m, u32 n, u32 k, bool quiet) {
    if (!quiet) {
        fmt::print("[gemm] M={} N={} K={}\n", m, n, k);
    }

    PPMatrix<f64> A(m, k), B(k, n), C(m, n);

    A.fill(1);
    B.fill(3);
    C.fill(0);

    auto start = std::chrono::high_resolution_clock::now();

    auto perf = PPMatrix<f64>::gemm(handle, 'N', 'N', 1.0f, A, B, 0.0f, C);

    cudaDeviceSynchronize();
    
    std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
    auto flops = 2.0 * m * n * k / time.count() / 1e12;

    if (quiet) {
        fmt::print("{},{},{},{},{:.3f}\n", m, perf.h2d.count(), perf.compute.count(), perf.d2h.count(), flops);
    } else {
        fmt::print("[mono] -- Time : {}s\n", time.count());
        fmt::print("[mono] -- Performance : {:.3f}Tflop/s\n", flops);
    }

    C.assertEq(static_cast<f64>(k * 3));
}

int main(int argc, char** argv) {
    u32 m, n, k;
	bool quiet;
	parseArgs(argc, argv, m, n, k, quiet);

    cublasHandle_t handle;
    cublasCreate(&handle);

    test_gemm(handle, m, n, k, quiet);
    cublasDestroy(handle);
}
