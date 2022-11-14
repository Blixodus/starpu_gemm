#pragma once

#include <string>
#include <type_traits>
#include "starpu.h"

// newtypes

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

#define STARPU_MATRIX_LD(x) STARPU_MATRIX_GET_LD((x))
#define STARPU_MATRIX_ROWS(x) STARPU_MATRIX_GET_NX((x))
#define STARPU_MATRIX_COLS(x) STARPU_MATRIX_GET_NY((x))

inline starpu_data_access_mode operator|(starpu_data_access_mode a, starpu_data_access_mode b) {
	using T = std::underlying_type_t<starpu_data_access_mode>;
	return static_cast<starpu_data_access_mode>(static_cast<T>(a) | static_cast<T>(b));
}

template <typename T>
struct MatrixInfo {
	u32 ld, rows, cols;
	T* ptr;

	inline constexpr T at(u32 row, u32 col) const noexcept {
		return ptr[row + col * ld];
	}
};

template <typename T>
constexpr MatrixInfo<T> as_matrix(void* ptr) {
	auto iface = reinterpret_cast<starpu_matrix_interface*>(ptr);

	return {.ld = iface->ld, .rows = iface->nx, .cols = iface->ny, .ptr = reinterpret_cast<T*>(iface->ptr)};
}

template <typename T>
inline constexpr T ceilDiv(T a, T b) noexcept {
	return (a + b - 1) / b;
}

template <typename Res, typename Base>
inline constexpr Res safe_cast(Base val) {
	return static_cast<Res>(val);
}

inline u32 stoui(const char* str) {
	return safe_cast<u32>(std::stoul(str));
}
