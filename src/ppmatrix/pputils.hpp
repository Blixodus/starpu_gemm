#pragma once

#include "../util/helper.hpp"

static void de_f32f16_flat(float* __restrict in, f16* __restrict a, f16* __restrict b, f16* __restrict c, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float src = in[i];
        f16 ai = __float2half(src);
        float delta = __half2float(ai) - src;
        f16 bi = __float2half(delta);
        f16 ci = __float2half(__half2float(bi) - delta);

        a[i] = ai;
        b[i] = bi;
        c[i] = ci;
    }
}

static void de_f32F16(float* __restrict in, f16* __restrict a, f16* __restrict b, f16* __restrict c, size_t rows, size_t cols, size_t ld) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            const auto idx = i * ld + j;

            float src = in[idx];
            f16 ai = __float2half(src);
            float delta = __half2float(ai) - src;
            f16 bi = __float2half(delta);
            f16 ci = __float2half(__half2float(bi) - delta);

            a[idx] = ai;
            b[idx] = bi;
            c[idx] = ci;
        }
    }
}

static void re_f32f16_flat(f16* __restrict a, f16* __restrict b, f16* __restrict c, float* __restrict out, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float ai = __half2float(a[i]);
        float bi = __half2float(b[i]);
        float ci = __half2float(c[i]);

        out[i] = ai + bi + ci;
    }
}

static void de_f64f32_flat(const double* __restrict in, float* __restrict hi, float* __restrict lo, size_t size) {
    for (size_t i = 0; i < size; i++) {
        double src = in[i];
        float ai = static_cast<float>(src);
        float bi = static_cast<float>(static_cast<double>(ai) - src);

        hi[i] = ai;
        lo[i] = bi;
    }
}

static void de_f64f32(const double* __restrict in, float* __restrict hi, float* __restrict lo, size_t rows, size_t cols, size_t ld) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double src = in[i * ld + j];
            float ai = static_cast<float>(src);
            float bi = static_cast<float>(static_cast<double>(ai) - src);

            hi[i * ld + j] = ai;
            lo[i * ld + j] = bi;
        }
    }
}

static void re_f64f32_flat(const float* __restrict hi, float* __restrict lo, double* __restrict out, size_t size) {
    for (size_t i = 0; i < size; i++) {
        double ai = static_cast<double>(hi[i]);
        double bi = static_cast<double>(lo[i]);

        out[i] = ai + bi;
    }
}

static void re_f64f32(const float* __restrict hi, float* __restrict lo, double* __restrict out, size_t rows, size_t cols, size_t ld) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double ai = static_cast<double>(hi[i * rows + j]);
            double bi = static_cast<double>(lo[i * rows + j]);

            out[i * ld + j] = ai + bi;
        }
    }
}

__global__ void f32tof64_flat(f32* __restrict src, f64* __restrict dst, u32 size) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = static_cast<f64>(src[idx]);
    }
}

__global__ void extractf32high_flat(f64* __restrict src, f32* __restrict dst, u32 size) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = static_cast<f32>(src[idx]);
    }
}

__global__ void extractf32low_flat(f64* __restrict src, f32* __restrict dst, u32 size) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = static_cast<float>(src[idx] - static_cast<double>(static_cast<float>(src[idx])));
    }
}
