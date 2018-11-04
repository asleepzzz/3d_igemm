#pragma once
#include "helper_cuda.h"

template <class T, T N>
struct Constant
{
    const T mValue = N;
};

template <unsigned I>
using Index = Constant<unsigned, I>;

template <unsigned... Is>
struct Sequence
{
    static constexpr unsigned nDim = sizeof...(Is);

    const unsigned mData[nDim] = {Is...};

    template <unsigned I>
    __host__ __device__ constexpr unsigned Get(Index<I>) const
    {
        return mData[I];
    }
};

#if 0
template<class F, class T, T... Is>
void for_each(F f, std::integer_sequence<T, Is...>)
{
    f(Is)...;
}

template<class F, class T, T N>
void for_n_time(F f, Constant<T, N>)
{
    for_each(f, std::make_integer_sequence<T, N>{});    
}
#endif

template <class Lengths, class Strides>
struct ConstantTensorDescriptor
{
    static constexpr unsigned nDim = Lengths::nDim;
    using NDimConstant             = Index<nDim>;

    __host__ __device__ constexpr ConstantTensorDescriptor()
    {
        static_assert(Lengths::nDim == Strides::nDim, "nDim not consistent");
    }

    __host__ __device__ constexpr unsigned GetDimension() const { return nDim; }

    __host__ __device__ constexpr Lengths GetLengths() const { return Lengths{}; }

    __host__ __device__ constexpr Strides GetStrides() const { return Strides{}; }

    template <unsigned I>
    __host__ __device__ constexpr unsigned GetLength(Index<I>) const
    {
        return Lengths{}.Get(Index<I>{});
    }

    template <unsigned I>
    __host__ __device__ constexpr unsigned GetStride(Index<I>) const
    {
        return Strides{}.Get(Index<I>{});
    }

#if 0
    template <class... Is>
    __host__ __device__ unsigned Get1dIndex(Is... is) const
    {
        static_assert(nDim == sizeof...(Is), "nDim not consistent");
        const unsigned iss[nDim] = {static_cast<unsigned>(is)...};
        unsigned idx             = 0;
        for_n_time([&](auto iDim) { idx += iss[iDim] * GetStride<iDim>(); }, NDimConstant{});
        return idx;
    }
#elif 1
    // this is ugly, only for 4d
    __host__ __device__ unsigned Get1dIndex(unsigned n, unsigned c, unsigned h, unsigned w) const
    {
        constexpr auto I0 = Index<0>{};
        constexpr auto I1 = Index<1>{};
        constexpr auto I2 = Index<2>{};
        constexpr auto I3 = Index<3>{};

        static_assert(nDim == 4, "nDim not consistent");
        return n * GetStride(I0) + c * GetStride(I1) + h * GetStride(I2) + w * GetStride(I3);
    }
#endif
};

// this is ugly, only for 4d
template <unsigned N, unsigned C, unsigned H, unsigned W>
__host__ __device__ constexpr auto calculate_default_strides(Sequence<N, C, H, W>)
{
    return Sequence<C * H * W, H * W, W, 1>{};
}

template <class Lengths>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor(Lengths)
{
    using Strides = decltype(calculate_default_strides(Lengths{}));
    return ConstantTensorDescriptor<Lengths, Strides>{};
}

template <class Lengths, class Strides>
__host__ __device__ constexpr auto make_ConstantTensorDescriptor(Lengths, Strides)
{
    return ConstantTensorDescriptor<Lengths, Strides>{};
}

// this is ugly, only for 4d
template <class InDesc, class WeiDesc>
__host__ __device__ constexpr auto get_output_4d_tensor_descriptor(InDesc, WeiDesc)
{
    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};

    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    static_assert(in_desc.GetDimension() == 4, "input nDim is not 4");
    static_assert(wei_desc.GetDimension() == 4, "weight nDim is not 4");
    static_assert(in_desc.GetLength(I1) == wei_desc.GetLength(I1),
                  "input & weight dimension not consistent");

    constexpr auto N  = in_desc.GetLength(I0);
    constexpr auto HI = in_desc.GetLength(I2);
    constexpr auto WI = in_desc.GetLength(I3);

    constexpr auto K = wei_desc.GetLength(I0);
    constexpr auto S = wei_desc.GetLength(I2);
    constexpr auto R = wei_desc.GetLength(I3);

    constexpr auto HO = HI - S + 1;
    constexpr auto WO = WI - R + 1;

    return make_ConstantTensorDescriptor(Sequence<N, K, HO, WO>{});
}

// this is ugly, only for 4d
template <class TDesc>
__host__ __device__ void print_ConstantTensorDescriptor(TDesc)
{
    constexpr auto desc = TDesc{};

    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    static_assert(desc.GetDimension() == 4, "dim is not 4");

    printf("dim %u, lengths {%u %u %u %u}, strides {%u %u %u %u}\n",
           desc.GetDimension(),
           desc.GetLength(I0),
           desc.GetLength(I1),
           desc.GetLength(I2),
           desc.GetLength(I3),
           desc.GetStride(I0),
           desc.GetStride(I1),
           desc.GetStride(I2),
           desc.GetStride(I3));
}