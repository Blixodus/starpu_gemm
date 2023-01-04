#pragma once

#include <vector>
#include <memory>
#include "../util/helper.hpp"
#include "cublas_v2.h"
#include "fmt/core.h"

template <typename DataType>
struct PPMatrix;

__host__ void ppgemm_f32(
    cublasHandle_t handle,
    char transA,
    char transB,
    f32 alpha,
    PPMatrix<f32>& A,
    PPMatrix<f32>& B,
    f32 beta,
    PPMatrix<f32>& C
);

__host__ void ppgemm_f64(
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

    static void ppgemm(
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
            ppgemm_f32(handle, transA, transB, alpha, A, B, beta, C);
        } else if constexpr (std::is_same_v<DataType, f64>) {
            ppgemm_f64(handle, transA, transB, alpha, A, B, beta, C);
        }
    }
};
