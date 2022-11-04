#pragma once

#include <string>

#include "starpu.h"

#define STARPU_MATRIX_LD(x) STARPU_MATRIX_GET_LD((x))
#define STARPU_MATRIX_ROWS(x) STARPU_MATRIX_GET_NX((x))
#define STARPU_MATRIX_COLS(x) STARPU_MATRIX_GET_NY((x))

template <typename T>
struct MatrixInfo {
    uint32_t ld, rows, cols;
    T* ptr;

    inline constexpr T at(uint32_t row, uint32_t col) const noexcept {
        return ptr[row + col * ld];
    }
};

template <typename T>
constexpr MatrixInfo<T> as_matrix(void* ptr) {
    auto iface = reinterpret_cast<starpu_matrix_interface*>(ptr);

    return {
        .ld = iface->ld,
        .rows = iface->nx,
        .cols = iface->ny,
        .ptr = reinterpret_cast<T*>(iface->ptr)
    };
}

template <typename T>
inline constexpr T ceilDiv(T a, T b) noexcept {
    return (a + b - 1) / b;
}

template <typename Res, typename Base>
inline constexpr Res safe_cast(Base val) {
    return static_cast<Res>(val);
}

inline uint32_t stoui(const char* str) {
    return safe_cast<uint32_t>(std::stoul(str));
}
