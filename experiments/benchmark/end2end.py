import bench


def main():
    for executor in [
        # '--exec torch --number 100',
        # '--exec ort --ort_provider cuda',
        # '--exec trt',
        # '--exec trt --trt_fp16',
        '--exec hidet --precision f32 --reduce_precision f32 --mma simt',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --hidet_space 0 --disable-graph-cache',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --hidet_space 0 --parallel_k search --disable-graph-cache',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --hidet_space 2 --parallel_k search --disable-graph-cache',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --hidet_space 2 --parallel_k 4 --disable-graph-cache',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k search --disable-graph-cache',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k disabled',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k 2',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k 4',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k 8',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma wmma',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma mma',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma mma_custom',
        # '--exec ort --ort_provider trt'
        # '--exec ansor --tvm_trial 800',
        # '--exec autotvm --tvm_trial 1000',
    ]:
        for bs in [
            '--bs 1',
            # '--bs 4',
            # '--bs 8',
            # '--bs 16'
        ]:
            for model in [
                '--model resnet50',
                '--model inception_v3',
                '--model mobilenet_v2',
                '--model bert',
                '--model gpt2'
            ]:
                bench.main('{} {} {}'.format(executor, bs, model))


if __name__ == '__main__':
    main()
