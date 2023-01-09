#pragma once

#include <fmt/core.h>
#include "../util/helper.hpp"
#include "cublas_v2.h"


template <typename DataType>
struct PPMatrix;

PerfRecord ppgemm_f32(
    cublasHandle_t handle,
    char transA,
    char transB,
    f32 alpha,
    const PPMatrix<f32>& A,
    const PPMatrix<f32>& B,
    f32 beta,
    PPMatrix<f32>& C
);

PerfRecord ppgemm_f64(
    cublasHandle_t handle,
    char transA,
    char transB,
    f64 alpha,
    const PPMatrix<f64>& A,
    const PPMatrix<f64>& B,
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

    void fill(DataType e);
    void rndFill();
    void assertEq(DataType e);

    static PerfRecord gemm(
        cublasHandle_t handle,
        char transA,
        char transB,
        DataType alpha,
        const PPMatrix<DataType>& A,
        const PPMatrix<DataType>& B,
        DataType beta,
        PPMatrix<DataType>& C
    );

    static void blasGemm(
        char transA,
        char transB,
        DataType alpha,
        const PPMatrix<DataType>& A,
        const PPMatrix<DataType>& B,
        DataType beta,
        PPMatrix<DataType>& C
    );

    static PerfRecord ppgemm(
        cublasHandle_t handle,
        char transA,
        char transB,
        DataType alpha,
        const PPMatrix<DataType>& A,
        const PPMatrix<DataType>& B,
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

    static void sub(
        const PPMatrix<DataType>& A,
        const PPMatrix<DataType>& B,
        PPMatrix<DataType>& C
    );

    DataType norm(char norm) const;
    DataType norm2() const;
};
