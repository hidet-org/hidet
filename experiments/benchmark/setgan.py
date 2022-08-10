import bench


def main():
    for executor in [
        # '--exec torch --number 100',
        '--exec trt',
        # '--exec trt --trt_fp16',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k search',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k disabled',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k 2',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k 4',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k 8',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma wmma',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma mma',
        '--exec ort --ort_provider cuda',
        '--exec hidet --precision f32 --reduce_precision f32 --mma simt',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt --parallel_k search',
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
                '--model op_setgan_conv_0',
                # '--model op_setgan_conv_1',
                # '--model op_setgan_conv_2',
                # '--model op_setgan_conv_3',
                # '--model op_setgan_conv_4',
                # '--model op_setgan_conv_5',
                # '--model op_setgan_conv_6',
                # '--model op_setgan_conv_7',
                # '--model op_setgan_conv_8',
                # '--model op_setgan_conv_9',
                # '--model op_setgan_conv_10',
                # '--model op_setgan_conv_11',
                # '--model op_setgan_conv_12',
                # '--model op_setgan_conv_13',
                # '--model op_setgan_conv_14',
                # '--model op_setgan_conv_15',
            ]:
                extra = '--number 1 --repeat 1 --warmup 0'
                # extra = ''
                bench.main('{} {} {} {}'.format(executor, bs, model, extra))


if __name__ == '__main__':
    main()
