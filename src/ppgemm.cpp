#include <starpu.h>
#include <starpu_mpi.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "fmt/core.h"
#include "cublas_v2.h"

#include "matrix/matrix.hpp"
#include "ppmatrix/ppmatrix.hpp"
#include "util/argparse.hpp"


#if defined(HAVE_STARPU_MPI_REDUX)
    #define ENABLE_REDUX 1
#else
    #warning "ENABLE_REDUX 0"
    #define ENABLE_REDUX 0
#endif

void printHelp() {
	std::cout << "Parameters for ppgemm:\n"
			  << "  -m               --  Set the size of M (as log)\n"
			  << "  -n               --  Set the size of N (as log)\n"
			  << "  -k               --  Set the size of K (as log)\n"
			  << "  -b               --  Set the block size (as log)\n"
			  << "  -t               --  Enable tiled mode"
			  << "  -q               --  Quiet mode"
			  << "  --run-checks     --  run advanced checks" << std::endl;
}

void parseArgs(
	int argc,
	char** argv,
	u32& m,
	u32& n,
	u32& k,
	u32& b,
    bool& tiled,
    bool& quiet,
    bool& run_checks
) {
	if (hasArg(argc, argv, "-h")) {
		printHelp();
		exit(0);
	}

	m = 1 << (hasArg(argc, argv, "-m") ? stoui(getArg(argc, argv, "-m")) : 10);
	n = 1 << (hasArg(argc, argv, "-n") ? stoui(getArg(argc, argv, "-n")) : 10);
	k = 1 << (hasArg(argc, argv, "-k") ? stoui(getArg(argc, argv, "-k")) : 10);
	b = 1 << (hasArg(argc, argv, "-b") ? stoui(getArg(argc, argv, "-b")) : 10);
    tiled = hasArg(argc, argv, "-t") ? true : false;
    quiet = hasArg(argc, argv, "-q") ? true : false;
    run_checks = hasArg(argc, argv, "--run-checks") ? true : false;
}

void test_ppgem_extchk(cublasHandle_t handle, u32 m, u32 n, u32 k, bool quiet) {
    fmt::print("[mono] M={} N={} K={}\n", m, n, k);

    PPMatrix<f64> A(m, k), B(k, n), C(m, n), T(m, n), D(m, n), OB(m, n);

    fmt::print("random fill...\n");
    A.rndFill();
    B.rndFill();
    C.fill(0);
    T.fill(0);
    D.fill(0);

    fmt::print("ppgemm...\n");
    PPMatrix<f64>::ppgemm(handle, 'N', 'N', 1.0f, A, B, 0.0f, C);

    fmt::print("computing blas truth source...\n");
    cublasSetStream(handle, 0);
    PPMatrix<f64>::gemm(handle, 'N', 'N', 1.0f, A, B, 0.0f, T);


    PPMatrix<f64>::blasGemm('N', 'N', 1.0f, A, B, 0.0f, OB);


    fmt::print("checking...\n");
    PPMatrix<f64>::sub(T, C, D);
    
    auto diffNorm = PPMatrix<f64>::norm('F', D);
    auto truthNorm = PPMatrix<f64>::norm('F', T);


    fmt::print(" error = {}\n", diffNorm / truthNorm);
}

void test_ppgemm_mono(cublasHandle_t handle, u32 m, u32 n, u32 k, bool quiet) {
	if (!quiet) {
        fmt::print("[mono] M={} N={} K={}\n", m, n, k);
    }

    PPMatrix<f64> A(m, k), B(k, n), C(m, n);

    A.fill(1);
    B.fill(3);

    auto perf = PPMatrix<f64>::ppgemm(handle, 'N', 'N', 1.0f, A, B, 0.0f, C);

    cudaDeviceSynchronize();
    std::chrono::duration<double> time = perf.d2h + perf.compute + perf.h2d;

    auto flops = 2.0 * m * n * k / time.count() / 1e12;

    if (quiet) {
        fmt::print("{},{},{},{},{:.3f}\n", m, perf.h2d.count(), perf.compute.count(), perf.d2h.count(), flops * 1000);
    } else {
        fmt::print("[mono] -- Time : {}s\n", time.count());
        fmt::print("[mono] -- Performance : {:.3f}Tflop/s\n", flops);
    }

    C.assertEq(static_cast<f64>(k * 3));
}

void test_ppgemm_tiled(u32 m, u32 n, u32 k, u32 block_size, bool quiet) {
	if (!quiet) {
        fmt::print("[tiled] Redux={} CPU={} GPU={} M={} N={} K={} BS={}\n", ENABLE_REDUX, enable_cpu, enable_gpu, m, n, k, block_size);
    }

    Matrix<f64> A(m, k, block_size), B(k, n, block_size), C(m, n, block_size);

	A.fill(1);
	B.fill(1);
	C.fill(0);

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	auto start = std::chrono::high_resolution_clock::now();

	Matrix<f64>::gemm('N', 'N', 1.0f, A, B, 0.0f, C);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
    auto flops = 2.0 * m * n * k / time.count() / 1e12;

    if (quiet) {
        fmt::print("{},{},{},{},{:.3f}\n", m, n, k, block_size, flops * 1000);
    } else {
        fmt::print("[tiled] -- Time : {}s\n", time.count());
        fmt::print("[tiled] -- Performance : {:.3f}Tflop/s\n", flops);
    }

	C.assertEq(static_cast<f64>(k));
}

int main(int argc, char** argv) {
    u32 m, n, k, b;
    bool tiled, quiet, run_checks;
    parseArgs(argc, argv, m, n, k, b, tiled, quiet, run_checks);

    if (run_checks&& (quiet || tiled)) {
        fmt::print("Cannot run checks in quiet or tiled mode\n");
        return 1;
    }

    if (tiled) {
        // init starpu
        if(starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL)) {
            throw std::exception();
        }

        #ifdef USE_CUDA
            starpu_cublas_init();
        #endif

        test_ppgemm_tiled(m, n, k, b, quiet);

        #ifdef USE_CUDA
            starpu_cublas_shutdown();
        #endif

        // shutdown starpu
        starpu_mpi_shutdown();
    } else {
        cublasHandle_t handle;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            fmt::print("cublasCreate failed: {}\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }

        if (run_checks) {
            test_ppgem_extchk(handle, m, n, k, quiet);
        } else {
            test_ppgemm_mono(handle, m, n, k, quiet);
        }
    }
}
