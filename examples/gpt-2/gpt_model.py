from typing import Optional
import os
import numpy.testing
import numpy as np
import hidet
from hidet import ops
from hidet import FlowGraph
from utils import load_params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    w = hidet.asarray(w).cuda()
    b = hidet.asarray(b).cuda()
    return x @ w + b


def layer_norm(x, g, b, eps: float = 1e-5):
    g = hidet.asarray(g).cuda()
    b = hidet.asarray(b).cuda()
    x = ops.layer_norm(x, epsilon=eps)
    return g * x + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = ops.gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return ops.softmax(q @ ops.transpose(k, [-1, -2]) / float(np.sqrt(q.shape[-1])) + mask, axis=-1) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    n_seq = x.shape[0]
    causal_mask = hidet.asarray((1 - np.tri(x.shape[0])) * -1e10, dtype=x.dtype).cuda()  # [n_seq, n_seq]
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, n_embd * 3]

    # [n_seq, n_embed * 3] -> [n_seq, 3, n_head, n_embed // n_head]
    x = ops.reshape(x, [x.shape[0], 3, n_head, x.shape[1] // (3 * n_head)])
    # [n_seq, 3, n_head, n_embed // n_head] -> [3, n_head, n_seq, n_embed // n_head]
    x = ops.transpose(x, [1, 2, 0, 3])
    # [3, n_head, n_seq, n_embed // n_head] -> [3, n_head, n_seq, n_embed // n_head]
    q, k, v = [t for t in ops.split(x, 3, axis=0)]

    # impl 1:
    # o = ops.attention(q, k, v, causal_mask)  # [1, n_head, n_seq, n_embed // n_head]

    # impl 2:
    o = attention(q, k, v, causal_mask)

    o = ops.rearrange(o, [[2], [0, 1, 3]])
    o = linear(o, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return o


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2_forward(ids, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte = hidet.asarray(wte).cuda()
    wpe = hidet.asarray(wpe).cuda()

    # [n_seq] -> [n_seq, n_embd]
    x = hidet.ops.take(wte, ids) + hidet.ops.take(wpe, hidet.ops.arange(ids.shape[0]).cuda())

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)

    x = layer_norm(x, **ln_f)
    x = ops.matmul(x, ops.transpose(wte))
    return x


def gpt2(model_size: str = "124M", seq_length: Optional[int] = 1000, use_fp16=False) -> FlowGraph:
    cache_dir = hidet.utils.hidet_cache_dir('./examples/gpt-2/')
    model_name = 'model_{}_seq{}_{}.hf'.format(model_size, seq_length, 'fp16' if use_fp16 else 'fp32')
    hf_path = os.path.join(cache_dir, model_name)
    if os.path.exists(hf_path):
        return hidet.load_graph(hf_path)
    else:
        print("Building hidet graph for GPT-2 ({}) with sequence length {}".format(model_size, seq_length))
        hparams, params = load_params(model_size, models_dir=cache_dir)
        if seq_length > hparams["n_ctx"]:
            raise ValueError(f"seq_length should be less than or equal to {hparams['n_ctx']}")

        ids = hidet.symbol([seq_length], dtype='int32', device='cuda')
        out = gpt2_forward(ids, **params, n_head=hparams["n_head"])
        graph = hidet.trace_from(out, inputs=[ids])
        with hidet.graph.PassContext() as ctx:
            if use_fp16:
                ctx.set_precision('float16')
                ctx.set_mma('mma')
            graph_opt = hidet.graph.optimize(graph)

        hidet.save_graph(graph_opt, hf_path)
        return graph_opt
