#pragma once

#include <vector>
#include <memory>
#include "../util/helper.hpp"
#include "cublas_v2.h"
#include "fmt/core.h"

template <typename DataType>
struct PPMatrix;

PerfRecord ppgemm_f32(
    cublasHandle_t handle,
    char transA,
    char transB,
    f32 alpha,
    PPMatrix<f32>& A,
    PPMatrix<f32>& B,
    f32 beta,
    PPMatrix<f32>& C
);

PerfRecord ppgemm_f64(
    cublasHandle_t handle,
    char transA,
    char transB,
    f64 alpha,
    PPMatrix<f64>& A,
    PPMatrix<f64>& B,
    f64 beta,
    PPMatrix<f64>& C
);

template <typename DataType>
struct PPMatrix {
    u32 rows, cols, ld;
    DataType* ptr;

    PPMatrix(u32 rows_, u32 cols_) 
        : rows(rows_), cols(cols_), ld(rows_), ptr(new DataType[rows * cols])
    { }

    PPMatrix(u32 rows_, u32 cols_, u32 ld_, DataType* ptr_)
        : rows(rows_), cols(cols_), ld(ld_), ptr(ptr_)
    { }

    ~PPMatrix() {
        delete[] ptr;
    }

    void fill(DataType e) {
        for (u32 i = 0; i < cols; ++i) {
            for (u32 j = 0; j < rows; ++j) {
                ptr[i * ld + j] = e;
            }
        }
    }

    void assertEq(DataType e) {
        for (u32 i = 0; i < cols; ++i) {
            for (u32 j = 0; j < rows; ++j) {
                if (std::abs(ptr[i * ld + j] - e) > 1e-6) {
                    fmt::print("Assertion failed at ({}, {}): {} != {}\n", i, j, ptr[i * ld + j], e);
                }
            }
        }
    }

    static PerfRecord gemm(
        cublasHandle_t handle,
        char transA,
        char transB,
        DataType alpha,
        PPMatrix<DataType>& A,
        PPMatrix<DataType>& B,
        DataType beta,
        PPMatrix<DataType>& C
    ) {
        DataType *dA, *dB, *dC;

        bool use_beta = !is_literal_zero(beta);

        int m = static_cast<int>((transA == 'N') ? A.rows : A.cols);
        int n = static_cast<int>((transB == 'N') ? B.cols : B.rows);
        int k = static_cast<int>((transA == 'N') ? A.cols : A.rows);

        HANDLE_ERR(cudaMalloc(&dA, A.rows * A.cols * sizeof(DataType)));
        cudaMalloc(&dB, B.rows * B.cols * sizeof(DataType));
        cudaMalloc(&dC, C.rows * C.cols * sizeof(DataType));

        cudaHostRegister(A.ptr, A.rows * A.cols * sizeof(DataType), cudaHostRegisterDefault);
        cudaHostRegister(B.ptr, B.rows * B.cols * sizeof(DataType), cudaHostRegisterDefault);

        if (use_beta) {
            cudaHostRegister(C.ptr, C.rows * C.cols * sizeof(DataType), cudaHostRegisterDefault);
        }

        auto start = std::chrono::high_resolution_clock::now();

        cudaMemcpy(dA, A.ptr, A.rows * A.cols * sizeof(DataType), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B.ptr, B.rows * B.cols * sizeof(DataType), cudaMemcpyHostToDevice);

        if (use_beta) {
            cudaMemcpy(dC, C.ptr, C.rows * C.cols * sizeof(DataType), cudaMemcpyHostToDevice);
        }

        cudaDeviceSynchronize();

        auto h2dDone = std::chrono::high_resolution_clock::now();

        if constexpr (std::is_same_v<DataType, f32>) {
            cublasSgemm(
                handle,
                convertToCublas(transA), convertToCublas(transB),
                m, n, k,
                &alpha,
                dA, A.rows,
                dB, B.rows,
                &beta,
                dC, C.rows
            );
        } else {
            static_assert(std::is_same_v<DataType, f64>, "Unsupported data type (only f32 and f64 are supported).");
            cublasDgemm(
                handle,
                convertToCublas(transA), convertToCublas(transB),
                m, n, k,
                &alpha,
                dA, checked_cast<int>(A.rows),
                dB, checked_cast<int>(B.rows),
                &beta,
                dC, checked_cast<int>(C.rows)
            );
        }

        cudaDeviceSynchronize();

        auto computeDone = std::chrono::high_resolution_clock::now();

        cudaMemcpy(C.ptr, dC, C.rows * C.cols * sizeof(DataType), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        auto d2hDone = std::chrono::high_resolution_clock::now();

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        cudaHostUnregister(A.ptr);
        cudaHostUnregister(B.ptr);

        if (use_beta) {
            cudaHostUnregister(C.ptr);
        }

        return PerfRecord{ h2dDone - start, computeDone - h2dDone, d2hDone - computeDone };
    }

    static PerfRecord ppgemm(
        cublasHandle_t handle,
        char transA,
        char transB,
        DataType alpha,
        PPMatrix<DataType>& A,
        PPMatrix<DataType>& B,
        DataType beta,
        PPMatrix<DataType>& C
    ) {
        if constexpr (std::is_same_v<DataType, f32>) {
            return ppgemm_f32(handle, transA, transB, alpha, A, B, beta, C);
        } else {
            static_assert(std::is_same_v<DataType, f64>, "Unsupported data type (only f32 and f64 are supported).");
            return ppgemm_f64(handle, transA, transB, alpha, A, B, beta, C);
        }
    }
};
