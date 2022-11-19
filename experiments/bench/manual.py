import bench
import hidet


def main():
    with hidet.utils.CacheDir('./outs/cache'):
        for executor in [
            '--exec trt',
            '--exec trt --precision f16 --trt_fp16',
            '--exec hidet --precision f32 --reduce_precision f32 --mma simt --hidet_space 2',
            '--exec hidet --precision f16 --reduce_precision f16 --mma wmma --hidet_space 2',
            '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 2',
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
            # '--exec ort --ort_provider cuda',
            # '--exec ort --ort_provider trt'
        ]:
            for bs in [
                '--bs 1',
                # '--bs 2',
                # '--bs 4',
                # '--bs 8',
                # '--bs 16'
            ]:
                for model in [
                    '--model bert_all',
                    '--model bert_embeddings',
                    '--model bert_encoder',
                    '--model bert_pooler',
                    '--model bert_layer',
                    '--model bert_attention',
                    '--model bert_intermediate',
                    '--model bert_output',
                    # '--model op_gemm_128_768_768',   # 5
                    # '--model op_gemm_128_3072_768',  # 1
                    # '--model op_gemm_128_768_3072',  # 1
                    # '--model op_gemm_128_768_3072',  # 1
                    # '--model op_gemm_128_768_3072',
                    # '--model op_gemm_1024_1024_1024',
                    # '--model op_gemm_131_769_3079',
                    '--model bert_self_attention',
                    '--model bert_self_output',
                    '--model bert_self_at_query',
                    '--model bert_self_at_qkv',
                    '--model bert_self_at_qkv_v2',
                    '--model bert_self_at_softmax',
                    '--model bert_self_at_context',
                ]:
                    # extra = '--number 1 --repeat 1 --warmup 0 --nocheck'
                    extra = ''
                    bench.main('{} {} {} {}'.format(executor, bs, model, extra))


if __name__ == '__main__':
    main()


