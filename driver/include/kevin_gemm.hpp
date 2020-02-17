#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "convolution_common.hpp"
#include "gridwise_kevin.hpp"

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
constexpr index_t N2=4;
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


        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

test_reorder();
//    constexpr auto kevin_test_re = reorder_tensor_descriptor_given_upper2lower(
//            unfold_tensor_descriptor(in_n0_n1_n2_c_y_ho_x_wo_global_desc, I1, I3), Sequence<1, 0>{});
//    for (int i=0;i<10;i++)
//    {
//        printf("======kevin_test_re %d %d============\n",i,kevin_test_re.GetLengths()[i]);
//    }

        constexpr auto in_e_n1_b_n2_global_desc = transform_tensor_descriptor(
            in_n0_n1_n2_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, T,R, S>>{},
                       PassThrough<N1>{},
                       Merge<Sequence<N0, Do,Ho, Wo>>{},
                       PassThrough<N2>{}),
            make_tuple(Sequence<3, 4, 6,8>{}, Sequence<1>{}, Sequence<0, 5, 7,9>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));



//E:CRS K=K B=NOO/N1N2


constexpr index_t BlockSize = 256;
    constexpr index_t BPerBlock = 16;
    constexpr index_t KPerBlock = 128;
    constexpr index_t EPerBlock = 16;

constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;
constexpr index_t InBlockCopySrcDataPerRead_B   = 1;


using InBlockCopySubLengths_E_N1_B_N2      = Sequence<4, 1, 1, 2>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<4, 2, 16, 2>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

        constexpr auto in_e_n1_b_n2_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<EPerBlock, N1, BPerBlock, N2>{}, Number<InBlockCopyDstDataPerWrite_N2>{});

//        constexpr index_t KBlockWork = 3;//K / KPerBlock;
//        constexpr index_t BBlockWork = 2;//B / BPerBlock;

//        constexpr auto block_work_desc =
//            make_cluster_descriptor(Sequence<KBlockWork, BBlockWork>{});

//        const auto block_work_id = block_work_desc.CalculateClusterIndex(5);
//printf("========block id %d %d===========\n",block_work_id[0],block_work_id[1]);



//        constexpr auto thread_cluster_desc =
//            make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

        // input tensor blockwise copy
/*
        auto blockwise_in_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(in_e_n1_b_n2_global_desc),
                                               decltype(in_e_n1_b_n2_block_desc),
                                               decltype(in_e_n1_b_n2_block_desc.GetLengths()),
                                               InBlockCopySubLengths_E_N1_B_N2,
                                               InBlockCopyClusterLengths_E_N1_B_N2,
                                               InBlockCopyThreadClusterArrangeOrder,
                                               InBlockCopySrcAccessOrder,
                                               InBlockCopyDstAccessOrder,
                                               2,
                                               3,
                                               InBlockCopySrcDataPerRead_B,
                                               InBlockCopyDstDataPerWrite_N2,
                                               AddressSpace::global,
                                               AddressSpace::vgpr,
                                               AddressSpace::lds,
                                               InMemoryDataOperation::none>(
                {0, 0, b_block_data_on_global, 0}, {0, 0, 0, 0});
*/
}


