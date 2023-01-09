#pragma once

#include <string>
#include <array>
#include <type_traits>
#include <iostream>
#include <chrono>
#include <fmt/core.h>

#include "starpu.h"

#include "make_array.hpp"
#include "dim_iter.hpp"

#ifdef USE_CUDA
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#endif

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

#ifdef USE_CUDA
using f16 = __half;
using bf16 = __nv_bfloat16;
#endif

#define STARPU_MATRIX_LD(x) STARPU_MATRIX_GET_LD((x))
#define STARPU_MATRIX_ROWS(x) STARPU_MATRIX_GET_NX((x))
#define STARPU_MATRIX_COLS(x) STARPU_MATRIX_GET_NY((x))

inline constexpr starpu_data_access_mode operator|(starpu_data_access_mode a, starpu_data_access_mode b) noexcept {
	using T = std::underlying_type_t<starpu_data_access_mode>;
	return static_cast<starpu_data_access_mode>(static_cast<T>(a) | static_cast<T>(b));
}

#ifdef USE_CUDA
#include "cublas_v2.h"

inline cublasOperation_t convertToCublas(char trans) {
	switch (trans) {
		case 'N': return CUBLAS_OP_N;
		case 'T': return CUBLAS_OP_T;
		case 'C': return CUBLAS_OP_C;
		default: throw std::exception();
	}
}
#endif

template <typename T>
struct MatrixInfo {
	u32 ld, rows, cols;
	T* ptr;

	inline constexpr T at(u32 row, u32 col) const noexcept {
		return ptr[row + col * ld];
	}
};

template <typename T>
constexpr MatrixInfo<T> as_matrix(void* ptr) noexcept {
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
constexpr TensorInfo<T> as_tensor(void* ptr) noexcept {
	auto iface = reinterpret_cast<starpu_ndim_interface*>(ptr);

	return {
		.ptr = reinterpret_cast<T*>(iface->ptr),
		.nn = iface->nn,
		.ldn = iface->ldn,
		.ndim = iface->ndim
	};
}

template <typename T>
constexpr T ceilDiv(T a, T b) noexcept {
	return (a + b - 1) / b;
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

class PerfRecord {
	using duration = std::chrono::duration<double>;

	public:
		duration h2d, compute, d2h;

		PerfRecord(): h2d(0), compute(0), d2h(0) { }

		PerfRecord(duration h2d, duration compute, duration d2h):
			h2d(h2d), compute(compute), d2h(d2h)
		{ }
};

template<typename T, typename U>
struct is_transmutable_into
    : std::integral_constant<bool,
        (std::alignment_of_v<T> == std::alignment_of_v<U>) &&
        (sizeof(T) == sizeof(U)) &&
        std::is_trivially_copyable_v<T> &&
        std::is_trivially_copyable_v<U>
    >
{ };

template<typename T, typename U>
constexpr bool is_transmutable_into_v = is_transmutable_into<T, U>::value;

template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
struct make_integral {
	using type =
		std::conditional_t<is_transmutable_into_v<T, char>, char,
		std::conditional_t<is_transmutable_into_v<T, short>, short, 
		std::conditional_t<is_transmutable_into_v<T, int>, int,
		std::conditional_t<is_transmutable_into_v<T, long>, long,
		std::conditional_t<is_transmutable_into_v<T, long long>, long long,
		void>>>>>;
};

template<typename T>
using make_integral_t = typename make_integral<T>::type;

template<typename T>
constexpr bool is_literal_zero(T val) noexcept {
	union {
		T base;
		make_integral_t<T> repr;
	} sh { val };

	return sh.repr == 0;
}

template <typename T>
struct is_valid_cast_target : std::integral_constant<bool,
		std::is_nothrow_constructible_v<T> && std::is_trivially_constructible_v<T>
	>
{ };

template <typename T>
constexpr bool is_valid_cast_target_v = is_valid_cast_target<T>::value;

template <typename Res, typename Base, std::enable_if_t<is_valid_cast_target_v<Res>, bool> = true>
constexpr Res checked_cast(Base val) noexcept(noexcept(static_cast<Res>(val))) {
	return static_cast<Res>(val);
}

/**
 * Cast which is not runtime-checked by default
*/
template <typename Res, typename Base, std::enable_if_t<is_valid_cast_target_v<Res>, bool> = true>
constexpr Res unchecked_cast(Base val) noexcept(noexcept(static_cast<Res>(val))) {
	return static_cast<Res>(val);
}

static inline u32 stoui(const char* str) {
	return checked_cast<u32>(std::stoul(str));
}

template <typename V, typename... T>
constexpr std::array<V, sizeof...(T)> array(T&&... t) {
	return {{ std::forward<V>(t)... }};
}

#ifdef USE_CUDA

#define HANDLE_ERR(val) handle_err((val), __LINE__)

inline void handle_err(cudaError_t val, int line) {
    if (__builtin_expect(val != cudaSuccess, 0)) {
        fmt::print("CUDA error at line {}: {}\n", line, cudaGetErrorString(cudaGetLastError()));
        throw std::exception();
    }
}

inline void handle_err(cublasStatus_t status, int line) {
	if (__builtin_expect(status != CUBLAS_STATUS_SUCCESS, 0)) {
		fmt::print("CUBLAS error at line {}: {}\n", line, status);
		throw std::exception();
	}
}

#endif

template <typename DataType>
constexpr std::string_view type_name() noexcept {
	if constexpr (std::is_same_v<DataType, float>) {
		return "single";
	} else if constexpr (std::is_same_v<DataType, double>) {
		return "double";
	} else {
		return "unknown";
	}
}
