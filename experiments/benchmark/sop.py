import bench


def main():
    for executor in [
        # '--exec trt',
        # '--exec trt --trt_fp16',
        # '--exec hidet --precision f32 --reduce_precision f32 --mma simt',
        # '--exec hidet --precision f16 --reduce_precision f16 --mma wmma',
        '--exec hidet --precision f16 --reduce_precision f16 --mma mma',
        '--exec ort --ort_provider cuda',
        # '--exec ort --ort_provider trt'
        # '--exec ansor --tvm_trial 800',
        # '--exec autotvm --tvm_trial 1000',
    ]:
        for bs in [
            '--bs 1',
            '--bs 16'
        ]:
            for model in [
                '--model resnet50_conv_0',
                '--model resnet50_conv_1',
                '--model resnet50_conv_2',
                '--model resnet50_conv_3',
                '--model resnet50_conv_4',
                '--model resnet50_conv_5',
                '--model resnet50_conv_6',
                '--model resnet50_conv_7',
                '--model resnet50_conv_8',
                '--model resnet50_conv_9',
                '--model resnet50_conv_10',
                '--model resnet50_conv_11',
                '--model resnet50_conv_12',
                '--model resnet50_conv_13',
                '--model resnet50_conv_14',
                '--model resnet50_conv_15',
                '--model resnet50_conv_16',
                '--model resnet50_conv_17',
                '--model resnet50_conv_18',
                '--model resnet50_conv_19',
                '--model resnet50_conv_20',
                '--model resnet50_conv_21',
                '--model resnet50_conv_22',
            ]:
                bench.main('{} {} {}'.format(executor, bs, model))


if __name__ == '__main__':
    main()
