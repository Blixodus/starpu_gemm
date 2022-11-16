#pragma once

#include <string>
#include <array>
#include <type_traits>
#include <iostream>
#include "starpu.h"

#include "make_array.hpp"
#include "dim_iter.hpp"

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

	return {
		.ld = iface->ld,
		.rows = iface->nx,
		.cols = iface->ny,
		.ptr = reinterpret_cast<T*>(iface->ptr)
	};
}

template <typename T>
struct TensorInfo {
	T* ptr;
	u32* nn;
	u32* ldn;
	size_t ndim;
};

template <typename T>
constexpr TensorInfo<T> as_tensor(void* ptr) {
	auto iface = reinterpret_cast<starpu_ndim_interface*>(ptr);

	return {
		.ptr = reinterpret_cast<T*>(iface->ptr),
		.nn = iface->nn,
		.ldn = iface->ldn,
		.ndim = iface->ndim
	};
}

template <typename T>
inline constexpr T ceilDiv(T a, T b) noexcept {
	return (a + b - 1) / b;
}

template <typename Res, typename Base>
inline constexpr Res checked_cast(Base val) {
	return static_cast<Res>(val);
}

/**
 * Cast which is not runtime-checked by default
*/
template <typename Res, typename Base>
inline constexpr Res unchecked_cast(Base val) {
	return static_cast<Res>(val);
}

inline u32 stoui(const char* str) {
	return checked_cast<u32>(std::stoul(str));
}

template <typename V, typename... T>
inline constexpr std::array<V, sizeof...(T)> array(T&&... t) {
	return {{ std::forward<V>(t)... }};
}

template <typename P>
class VecPrinter {
	public:
		VecPrinter(const std::vector<P>& vec): vec(vec) {}

		friend std::ostream& operator << (std::ostream& os, const VecPrinter& vp) {
			os << "[";

			if (vp.vec.size() > 0) {
				os << vp.vec[0];

				for (size_t i = 1; i < vp.vec.size(); i++) {
					os << ", ";
					os << vp.vec[i];
				}
			}

			os << "]";
			return os;
		}

	private:
		const std::vector<P>& vec;
};
