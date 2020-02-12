#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "print_array.hpp"
#include "print_sequence.hpp"
#include "device.hpp"
#include "tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "kevin_gemm.hpp"
//kevin add 
#include "tensor_descriptor.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;
constexpr index_t N=1;
constexpr index_t C=1;
constexpr index_t H=4;
constexpr index_t W=4;
auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C, H, W>{});
//ostream_ConstantTensorDescriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");

in_nchw_desc.GetLengths();
kevin_test();
//change to native
//    constexpr auto in_nchw_desc_native =
//        make_native_tensor_descriptor(in_nchw_desc.GetLengths(), in_nchw_desc.GetStrides());
/*

        constexpr auto in_n0_n1_n2_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(UnMerge<Sequence<N0, N1, N2>>{},
                       PassThrough<C>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}, Sequence<6, 7>{}));
*/

/*
        constexpr auto in_e_n1_b_n2_global_desc = transform_tensor_descriptor(
            in_n0_n1_n2_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{},
                       PassThrough<N1>{},
                       Merge<Sequence<N0, Ho, Wo>>{},
                       PassThrough<N2>{}),
            make_tuple(Sequence<3, 4, 6>{}, Sequence<1>{}, Sequence<0, 5, 7>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
*/

/*
using ThreadwiseLoad =
ThreadwiseGenericTensorSliceCopy_v4r2
<>
*/
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

//Float*p_in_global;
//__shared__ Float p_in_block_double[2 * in_block_space];

//blockwise_in_copy.Run(p_in_global, p_in_block_double);
    return 0;
}


