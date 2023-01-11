#include <chrono>
#include <iostream>
#include <fmt/core.h>

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
			  << "  --run-checks" << std::endl;
}

void parseArgs(
	int argc,
	char** argv,
	u32& m,
	u32& n,
	u32& k,
	char& type,
    bool& run_checks,
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
    run_checks = hasArg(argc, argv, "--run-checks") ? true : false;
    quiet = hasArg(argc, argv, "-q") ? true : false;
}

template<typename DataType>
void test_gemm_chk(cublasHandle_t handle, u32 m, u32 n, u32 k, bool quiet) {
    fmt::print("[gemm_chk] M={} N={} K={}\n", m, n, k);

    PPMatrix<DataType> A(m, k), B(k, n), C(m, n), D(m, n), T(m, n);

    A.rndFill();
    B.rndFill();
    C.fill(0);
    T.fill(0);
    D.fill(0);

    PPMatrix<DataType>::gemm(handle, 'N', 'N', 1.0f, A, B, 0.0f, C);
    
    fmt::print("computing openblas truth source...\n");
    PPMatrix<DataType>::blasGemm('N', 'N', 1.0f, A, B, 0.0f, T);

    fmt::print("checking...\n");
    PPMatrix<DataType>::sub(T, C, D);
    
    auto diffNorm = D.norm('F');
    auto truthNorm = T.norm('F');

    fmt::print("relative error = {}\n", diffNorm / truthNorm);
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
    auto compute_flops = 2.0 * m * n * k / perf.compute.count() / 1e12;

    if (quiet) {
        fmt::print("{},{:.6f},{:.6f},{:.6f},{:.3f},{:.3f}\n", m, perf.h2d.count(), perf.compute.count(), perf.d2h.count(), compute_flops * 1000, flops * 1000);
    } else {
        fmt::print("[gemm] -- Time : {}s\n", time.count());
        fmt::print("[gemm] -- Performance : {:.3f}Tflop/s\n", flops);
    }

    C.assertEq(static_cast<DataType>(k * 3));
}

int main(int argc, char** argv) {
    do_hello();

    u32 m, n, k;
	bool quiet, run_checks;
    char type;
	parseArgs(argc, argv, m, n, k, type, run_checks, quiet);

    if (run_checks && quiet) {
        fmt::print("Cannot run checks in quiet mode\n");
        return 1;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    if (run_checks) {
        switch (type) {
            case 's':
                test_gemm_chk<f32>(handle, m, n, k, quiet);
            break;

            case 'd':
                test_gemm_chk<f64>(handle, m, n, k, quiet);
            break;

            default:
                fmt::print("Invalid data type: {}, must be s or d\n", type);
                exit(1);
        }
    } else {
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
    }
    
    cublasDestroy(handle);
}
