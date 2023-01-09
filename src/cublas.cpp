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
			  << "  -m           --  Set the size of M (as log)\n"
			  << "  -n           --  Set the size of N (as log)\n"
			  << "  -k           --  Set the size of K (as log)\n"
			  << "  -b           --  Set the block size (as log)\n"
			  << "  -g           --  Set the number of GPUs used for task execution\n"
			  << "  -t <s/d>     --  Set the data type (double/single)\n"
			  << "  --perf-test  --  Test CUBLAS performace on given sizes" << std::endl;
}

void parseArgs(
	int argc,
	char** argv,
	u32& m,
	u32& n,
	u32& k,
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
	type = (hasArg(argc, argv, "-t") ? (char) tolower(getArg(argc, argv, "-t")[0]) : 's');
    quiet = hasArg(argc, argv, "-q") ? true : false;
}

template<typename DataType>
void test_gemm(cublasHandle_t handle, u32 m, u32 n, u32 k, bool quiet) {
    if (!quiet) {
        fmt::print("[gemm] M={} N={} K={}\n", m, n, k);
    }

    PPMatrix<DataType> A(m, k), B(k, n), C(m, n);

    A.fill(1);
    B.fill(3);
    C.fill(0);

    auto perf = PPMatrix<DataType>::gemm(handle, 'N', 'N', 1.0f, A, B, 0.0f, C);
    
    std::chrono::duration<double> time = perf.h2d + perf.compute + perf.d2h;
    auto flops = 2.0 * m * n * k / time.count() / 1e12;

    if (quiet) {
        fmt::print("{},{},{},{},{:.3f}\n", m, perf.h2d.count(), perf.compute.count(), perf.d2h.count(), flops * 1000);
    } else {
        fmt::print("[mono] -- Time : {}s\n", time.count());
        fmt::print("[mono] -- Performance : {:.3f}Tflop/s\n", flops);
    }

    C.assertEq(static_cast<DataType>(k * 3));
}

int main(int argc, char** argv) {
    u32 m, n, k;
	bool quiet;
    char type;
	parseArgs(argc, argv, m, n, k, type, quiet);

    cublasHandle_t handle;
    cublasCreate(&handle);

    switch (type) {
        case 's':
            test_gemm<f32>(handle, m, n, k, quiet);
        break;

        case 'd':
            test_gemm<f64>(handle, m, n, k, quiet);
        break;

        default:
            fmt::print("Invalid data type: {}, must be s or d\n", type);
            exit(1);
    }
    
    cublasDestroy(handle);
}
