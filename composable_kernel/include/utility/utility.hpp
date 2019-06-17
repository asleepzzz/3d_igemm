#ifndef CK_UTILITY_HPP
#define CK_UTILITY_HPP

#include <type_traits>
#include "config.hpp"

namespace ck {

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

template <class X, class Y>
using is_same = std::is_same<X, Y>;

namespace math {

template <class T, T s>
struct scales
{
    __host__ __device__ constexpr T operator()(T a) const { return s * a; }
};

template <class T>
struct plus
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a + b; }
};

template <class T>
struct minus
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a - b; }
};

template <class T>
struct multiplies
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a * b; }
};

template <class T>
struct integer_divide_ceiler
{
    __host__ __device__ constexpr T operator()(T a, T b) const
    {
        static_assert(is_same<T, index_t>{} || is_same<T, int>{}, "wrong type");

        return (a + b - 1) / b;
    }
};

template <class T>
__host__ __device__ constexpr T integer_divide_ceil(T a, T b)
{
    static_assert(is_same<T, index_t>{} || is_same<T, int>{}, "wrong type");

    return (a + b - 1) / b;
}

template <class T>
__host__ __device__ constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <class T, class... Ts>
__host__ __device__ constexpr T max(T x, Ts... xs)
{
    static_assert(sizeof...(xs) > 0, "not enough argument");

    auto y = max(xs...);

    static_assert(is_same<decltype(y), T>{}, "not the same type");

    return x > y ? x : y;
}

template <class T>
__host__ __device__ constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <class T, class... Ts>
__host__ __device__ constexpr T min(T x, Ts... xs)
{
    static_assert(sizeof...(xs) > 0, "not enough argument");

    auto y = min(xs...);

    static_assert(is_same<decltype(y), T>{}, "not the same type");

    return x < y ? x : y;
}

// this is WRONG
// TODO: implement least common multiple properly, instead of calling max()
template <class T, class... Ts>
__host__ __device__ constexpr T lcm(T x, Ts... xs)
{
    return max(x, xs...);
}

} // namespace math
} // namspace ck

#endif