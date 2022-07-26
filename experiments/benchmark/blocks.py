import bench_model


def main():
    for executor in [
        # '--exec trt',
        '--exec trt --precision f16 --trt_fp16',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma wmma',
        '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 1 --parallel_k disabled',
        '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 1 --parallel_k default',
        '--exec hidet --precision f16 --reduce_precision f16 --mma mma --hidet_space 1 --parallel_k 6',
        # '--exec ort --ort_provider cuda',
        # '--exec ort --ort_provider trt'
        # '--exec ansor --tvm_trial 800',
        # '--exec autotvm --tvm_trial 1000',
        # '--exec ort --ort_provider cuda',
        # '--exec ort --ort_provider trt'
        # '--exec ansor --tvm_trial 800',
        # '--exec autotvm --tvm_trial 1000',
    ]:
        for bs in [
            # '--bs 1',
            '--bs 16'
        ]:
            for model in [
                # '--model bert_all',
                # '--model bert_embeddings',
                # '--model bert_encoder',
                # '--model bert_pooler',
                # '--model bert_layer',
                # '--model bert_attention',
                # '--model bert_intermediate',
                '--model bert_output',
                # '--model bert_self_attention',
                # '--model bert_self_output',
                # '--model bert_self_at_query',
                # '--model bert_self_at_qkv',
                # '--model bert_self_at_qkv_v2',
                # '--model bert_self_at_softmax',
                # '--model bert_self_at_context',
            ]:
                extra = '--number 3 --repeat 1 --warmup 0'
                # extra = ''
                bench_model.main('{} {} {} {}'.format(executor, bs, model, extra))


if __name__ == '__main__':
    main()
