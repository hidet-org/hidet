from typing import Optional
import os
from bench.common import BenchResult, get_onnx_model, benchmark_run
from hidet.utils import hidet_cache_file
import hidet


def bench_tf(args, out_dir) -> BenchResult:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    import tensorflow as tf
    from tensorflow.config.optimizer import set_jit
    tf.get_logger().setLevel('ERROR')

    if args.exec == 'tf_xla':  # turn on xla
        set_jit(True)

    pb_path = hidet_cache_file('tf', '{}.pb'.format(args.model))
    onnx_path, input_names, input_tensors = get_onnx_model(name=args.model, batch_size=args.bs, precision=args.precision)
    if not os.path.exists(pb_path):
        from onnx_tf.backend import prepare, TensorflowRep
        import onnx
        model = onnx.load_model(onnx_path)
        tf_rep: Optional[TensorflowRep] = prepare(model)
        tf_rep.export_graph(pb_path + '.tmp')
        os.rename(pb_path + '.tmp', pb_path)
        assert os.path.exists(pb_path)

    with tf.gfile.FastGFile(pb_path, 'rb') as fin:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fin.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    out_names = {
        'op_matmul_nn_1': ['2']
    }
    if args.model not in out_names:
        raise ValueError('You should provide the names of output tensors in tensorflow graph. '
                         'This is why people do not like tf. :(')

    tf_input_tensors = [graph.get_tensor_by_name(name + ':0') for name in input_names]
    tf_output_tensors = [graph.get_tensor_by_name(name + ':0') for name in out_names[args.model]]

    with tf.Session(graph=graph) as sess:
        result = BenchResult()
        run_options = tf.RunOptions()
        run_metadata = tf.RunMetadata()
        feed_dict = {tf_tensor: v.numpy() for tf_tensor, v in zip(tf_input_tensors, input_tensors)}
        result.outputs = [hidet.array(v) for v in sess.run(
            tf_output_tensors,
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata
        )]

        def run_func():
            return sess.run(tf_output_tensors,
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)

        result.latencies = benchmark_run(run_func, warmup=args.warmup, number=args.number * args.repeat, repeat=1)  # for tensorflow, we need large number
        result.configs = 'fp32'
        return result

