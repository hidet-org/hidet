import bench
import hidet

#
# workload size in bert (b x m x n x k):
# attention:
#   self attention:
#    [batch_size] X [seq_length] X [hidden_size] X [hidden_size]   (4 times)
#    [batch_size * num_heads] X [seq_length] X [seq_length] X [hidden_size / num_heads]   (attention_scores)
#   self output:
#    [batch_size] X [seq_length] X [hidden_size] X [hidden_size]
# feed forward:
#  [batch_size] X [seq_length] X [intermediate_size] X [hidden_size]
#  [batch_size] X [seq_length] X [hidden_size] X [intermediate_size]


def main_old():
    with hidet.utils.CacheDir('./outs/cache'):
        for executor in [
            # '--exec trt',
            # '--exec ort --ort_provider cuda',
            # '--exec trt --precision f16 --trt_fp16',
            '--exec hidet --precision f32 --reduce_precision f32 --mma simt --hidet_space 2',
            # '--exec hidet --precision f16 --reduce_precision f16 --mma wmma --hidet_space 2',
            # '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 1 --parallel_k disabled',
            # '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 1 --parallel_k default',
            # '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 1 --parallel_k 6',
            # '--exec manual --precision f16 --reduce_precision f16 --manual_config default',
            # '--exec manual --precision f16 --reduce_precision f16 --manual_config cp_async',
            # '--exec manual --precision f16 --reduce_precision f16 --manual_config cp_async_multi_stage',
            # '--exec manual --precision f16 --reduce_precision f16 --manual_config ldmatrix',
            # '--exec manual --precision f16 --reduce_precision f16 --manual_config cp_async_ldmatrix',
            # '--exec manual --precision f16 --reduce_precision f16 --manual_config cp_async_ldmatrix_opt',
            # '--exec manual --precision f16 --reduce_precision f16 --manual_config all',
            # '--exec manual --manual_config none --disable-graph-cache',
            # '--exec hidet --precision f16 --reduce_precision f16 --mma mma_custom --hidet_space 2 --parallel_k disabled',
            # '--exec ort --ort_provider trt'
            # '--exec ansor --tvm_trial 800',
            # '--exec autotvm --tvm_trial 1000',
            '--exec ort --ort_provider cuda',
            # '--exec ort --ort_provider trt'
            # '--exec ansor --tvm_trial 800',
            # '--exec autotvm --tvm_trial 1000',
        ]:
            for bs in [
                '--bs 1',
                # '--bs 2',
                # '--bs 4',
                # '--bs 8',
                # '--bs 16'
            ]:
                """
n, c, h, w, kx, ky, sx, sy = out_shape 1 32 112 112 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 96 56 56 kernel 3 3 stride 2 2
n, c, h, w, kx, ky, sx, sy = out_shape 1 144 56 56 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 144 28 28 kernel 3 3 stride 2 2
n, c, h, w, kx, ky, sx, sy = out_shape 1 192 28 28 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 192 28 28 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 192 14 14 kernel 3 3 stride 2 2
n, c, h, w, kx, ky, sx, sy = out_shape 1 384 14 14 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 384 14 14 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 384 14 14 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 384 14 14 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 576 14 14 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 576 14 14 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 576 7 7 kernel 3 3 stride 2 2
n, c, h, w, kx, ky, sx, sy = out_shape 1 960 7 7 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 960 7 7 kernel 3 3 stride 1 1
n, c, h, w, kx, ky, sx, sy = out_shape 1 960 7 7 kernel 3 3 stride 1 1

'--model op_dwc_1_32_112_112_1_3',
'--model op_dwc_1_96_56_56_2_3',
'--model op_dwc_1_144_56_56_1_3',
'--model op_dwc_1_144_28_28_2_3',
'--model op_dwc_1_192_28_28_1_3',
'--model op_dwc_1_192_28_28_1_3',
'--model op_dwc_1_192_14_14_2_3',
'--model op_dwc_1_384_14_14_1_3',
'--model op_dwc_1_384_14_14_1_3',
'--model op_dwc_1_384_14_14_1_3',
'--model op_dwc_1_384_14_14_1_3',
'--model op_dwc_1_576_14_14_1_3',
'--model op_dwc_1_576_14_14_1_3',
'--model op_dwc_1_576_7_7_2_3',
'--model op_dwc_1_960_7_7_1_3',
'--model op_dwc_1_960_7_7_1_3',
'--model op_dwc_1_960_7_7_1_3',
            """
                for model in [
                    # '--model bert_all',
                    # '--model bert_embeddings',
                    # '--model bert_encoder',
                    # '--model bert_pooler',
                    # '--model bert_layer',
                    # '--model bert_attention',
                    # '--model bert_intermediate',
                    # '--model bert_output',
                    # '--model op_gemm_128_768_768',   # 5
                    # '--model op_gemm_128_3072_768',  # 1
                    # '--model op_gemm_128_768_3072',  # 1
                    # '--model op_gemm_128_768_3072',  # 1
                    '--model op_dwc_64_128_62_62_1_7',
                    # '--model op_dwc_1_384_14_14_1_3',
                    # '--model op_dwc_1_32_112_112_1_3',
                    # '--model op_dwc_1_96_56_56_2_3',
                    # '--model op_dwc_1_144_56_56_1_3',
                    # '--model op_dwc_1_144_28_28_2_3',
                    # '--model op_dwc_1_192_28_28_1_3',
                    # '--model op_dwc_1_192_28_28_1_3',
                    # '--model op_dwc_1_192_14_14_2_3',
                    # '--model op_dwc_1_384_14_14_1_3',
                    # '--model op_dwc_1_384_14_14_1_3',
                    # '--model op_dwc_1_384_14_14_1_3',
                    # '--model op_dwc_1_384_14_14_1_3',
                    # '--model op_dwc_1_576_14_14_1_3',
                    # '--model op_dwc_1_576_14_14_1_3',
                    # '--model op_dwc_1_576_7_7_2_3',
                    # '--model op_dwc_1_960_7_7_1_3',
                    # '--model op_dwc_1_960_7_7_1_3',
                    # '--model op_dwc_1_960_7_7_1_3',
                    # '--model op_gemm_128_768_3072',
                    # '--model op_gemm_1024_1024_1024',
                    # '--model op_gemm_131_769_3079',
                    # '--model bert_self_attention',
                    # '--model bert_self_output',
                    # '--model bert_self_at_query',
                    # '--model bert_self_at_qkv',
                    # '--model bert_self_at_qkv_v2',
                    # '--model bert_self_at_softmax',
                    # '--model bert_self_at_context',
                ]:
                    # extra = '--number 1 --repeat 1 --warmup 0 --nocheck'
                    extra = ''
                    bench.main('{} {} {} {}'.format(executor, bs, model, extra))


if __name__ == '__main__':
    main_old()
    # main()
