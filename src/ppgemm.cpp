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

#include "matrix/matrix.hpp"
#include "ppmatrix/ppmatrix.hpp"
#include "cublas_v2.h"

#if defined(HAVE_STARPU_MPI_REDUX)
    #define ENABLE_REDUX 1
#else
    #warning "ENABLE_REDUX 0"
    #define ENABLE_REDUX 0
#endif


void test_ppgemm_mono(cublasHandle_t handle, u32 m, u32 n, u32 k) {
	fmt::print("[mono] M={} N={} K={}\n", m, n, k);

    PPMatrix<f64> A(m, k), B(k, n), C(m, n);

    A.fill(1);
    B.fill(3);
    C.fill(0);

    auto start = std::chrono::high_resolution_clock::now();

    PPMatrix<f64>::ppgemm(handle, 'N', 'N', 1.0f, A, B, 1.0f, C);

    cudaDeviceSynchronize();
    std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;

    auto flops = 2.0 * m * n * k / time.count() / 1e12;

    fmt::print("[mono] -- Time : {}s\n", time.count());
    fmt::print("[mono] -- Performance : {:.3f}Tflop/s\n", flops);

    fmt::print(stderr, "{},{:.3f}\n", m, flops);

    C.assertEq(static_cast<f64>(k * 3));
}

void test_ppgemm_tiled(u32 m, u32 n, u32 k, u32 block_size) {
	fmt::print("[tiled] Redux={} CPU={} GPU={} M={} N={} K={} BS={}\n", ENABLE_REDUX, enable_cpu, enable_gpu, m, n, k, block_size);

    Matrix<f64> A(m, k, block_size), B(k, n, block_size), C(m, n, block_size);

	A.fill(1);
	B.fill(1);
	C.fill(0);

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	auto start = std::chrono::high_resolution_clock::now();

	Matrix<f64>::gemm('N', 'N', 1.0f, A, B, 0.0f, C);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
	fmt::print("[tiled] -- Time : {}s\n", time.count());
	fmt::print("[tiled] -- Performance : {:.3f}Tflop/s\n", 2.0 * m * n * k / time.count() / 1e12);

	C.assertEq(static_cast<f64>(k));
}

int main(int argc, char** argv) {
    if (argc != 7) {
		fmt::print("Usage: {} -m <m> -n <n> -k <k> -t <true/false>\n", argv[0]);
		return 1;
	}
	
	const u32 exp = stoui(argv[1]);
	const u32 k_min = stoui(argv[2]);
	const u32 k_max = std::min(stoui(argv[3]), exp);
	const u32 b_min = stoui(argv[4]);
	const u32 b_max = std::min(stoui(argv[5]), exp);
	const u32 m = 1 << exp;
	const u32 n = 1 << exp;

    const bool tiled = (strcmp(argv[6], "true") == 0);

    if (tiled) {
        fmt::print("init\n");
        // init starpu
        if(starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL)) {
            throw std::exception();
        }

        #ifdef USE_CUDA
            starpu_cublas_init();
        #endif

        for (u32 b_exp = b_min; b_exp <= b_max; b_exp++) {
            const u32 block_size = 1 << b_exp;
		    for (u32 k_exp = k_min; k_exp <= k_max; k_exp++) {
			    const u32 k = 1 << k_exp;
                test_ppgemm_tiled(m, n, k, block_size);
            }
        }

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

        for (u32 k_exp = k_min; k_exp <= k_max; k_exp++) {
            const u32 k = 1 << k_exp;
            test_ppgemm_mono(handle, m, n, k);
        }
    }
}
