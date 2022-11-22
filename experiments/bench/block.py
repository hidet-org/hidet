import bench
import hidet

hidet.option.cache_dir('./outs/cache')
hidet.option.parallel_build(False)
# hidet.option.save_lower_ir(True)


def main():
    for executor in [
        # '--exec trt',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --hidet_space 2',
        '--exec trt --precision f16 --trt_fp16',
        '--exec hidet --precision f16 --reduce_precision f16 --mma wmma --hidet_space 2',
        '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 2',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 1 --parallel_k disabled',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 1 --parallel_k default',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 1 --parallel_k 6',
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
                # '--model bert_all',
                # '--model bert_embeddings',
                # '--model bert_encoder',
                # '--model bert_pooler',
                '--model bert_layer',
                # '--model bert_attention',
                # '--model bert_intermediate',
                # '--model bert_output',
                # '--model bert_self_attention',
                # '--model bert_self_output',
                # '--model bert_self_at_query',
                # '--model bert_self_at_qkv',
                # '--model bert_self_at_qkv_v2',
                # '--model bert_self_at_softmax',
                # '--model bert_self_at_context',
            ]:
                extra = '--number 1 --repeat 1 --warmup 0 --nocheck'
                # extra = ''
                with hidet.utils.nvtx_annotate(message=executor.split()[1], color='green'):
                    bench.main('{} {} {} {}'.format(executor, bs, model, extra))


if __name__ == '__main__':
    main()
