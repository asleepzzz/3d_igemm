#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "convolution_common.hpp"
#include "gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer.hpp"
using namespace ck;
void kevin_test()
{
constexpr index_t N=4;
constexpr index_t C=1;
constexpr index_t D=4;
constexpr index_t H=4;
constexpr index_t W=4;

    using LeftPads  = Sequence<0, 0,0>;
    using RightPads = Sequence<0, 0,0>;

auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C,D, H, W>{});
ostream_ConstantTensorDescriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");

    constexpr auto in_nchw_desc_native =
        make_native_tensor_descriptor(in_nchw_desc.GetLengths(), in_nchw_desc.GetStrides());

 constexpr auto in_n_c_hi_wi_global_desc  = decltype(in_nchw_desc_native){};
//nothing change ,only add pads
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(
                PassThrough<N>{}, PassThrough<C>{}, Pad<Sequence<D,H, W>, LeftPads, RightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3,4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3,4>{}));

        constexpr index_t Dip = in_n_c_hip_wip_global_desc.GetLengths()[2];
        constexpr index_t Hip = in_n_c_hip_wip_global_desc.GetLengths()[3];
        constexpr index_t Wip = in_n_c_hip_wip_global_desc.GetLengths()[4];
printf("=======dip %d hip %d wip %d ============\n",Dip,Hip,Wip);
constexpr index_t N0=1;
constexpr index_t N1=2;
constexpr index_t N2=2;
constexpr index_t T=3;
constexpr index_t R=2;
constexpr index_t S=2;
constexpr index_t Ho=3;
constexpr index_t Wo=3;
constexpr index_t Do=2;
        constexpr auto in_n0_n1_n2_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(UnMerge<Sequence<N0, N1, N2>>{},
                       PassThrough<C>{},
                       Embed<Dip, Sequence<T, Do>, Sequence<1, 1, 0>>{},
                       Embed<Hip, Sequence<R, Ho>, Sequence<1, 1, 0>>{},
                       Embed<Wip, Sequence<S, Wo>, Sequence<1, 1, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{},Sequence<4>{}),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}, Sequence< 6,7>{},Sequence< 8,9>{}));
    for (int i=0;i<10;i++)
    {
        printf("=======after conv dilation stride size %d %d============\n",i,in_n0_n1_n2_c_y_ho_x_wo_global_desc.GetLengths()[i]);
    }
printf("======offset %d=============\n",in_n0_n1_n2_c_y_ho_x_wo_global_desc.CalculateOffset(make_multi_index(0,0,0,0,2,1,0,0,1,2)));
}


