"""
Visualize Flow Graph
====================

Visualization is a key component of a machine learning tool to allow us have a better understanding of the model.

We customized the popular `Netron <https://github.com/lutzroeder/netron>`_ viewer to visualize the flow graph of a
hidet model. The customized Netron viewer can be found at `here </netron>`_, you can also find a link on the
bottom of the documentation side bar.

In this tutorial, we will show you how to visualize the flow graph of a model.

Define model
------------

We first define a model with a self-attention layer.

"""
import math
import hidet
from hidet import Tensor
from hidet.graph import nn, ops


class SelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        batch_size, seq_length, hidden_size = x.shape
        x = x.reshape(
            [batch_size, seq_length, self.num_attention_heads, self.attention_head_size]
        )
        x = x.rearrange([[0, 2], [1], [3]])
        return x  # [batch_size * num_attention_heads, seq_length, attention_head_size]

    def forward(self, hidden_states: Tensor, attention_mask: Tensor):
        batch_size, seq_length, _ = hidden_states.shape
        query = self.transpose_for_scores(self.query_layer(hidden_states))
        key = self.transpose_for_scores(self.key_layer(hidden_states))
        value = self.transpose_for_scores(self.value_layer(hidden_states))
        attention_scores = ops.matmul(query, ops.transpose(key, [-1, -2])) / math.sqrt(
            self.attention_head_size
        )
        attention_scores = attention_scores + attention_mask
        attention_probs = ops.softmax(attention_scores, axis=-1)
        context = ops.matmul(attention_probs, value)
        context = context.reshape(
            [batch_size, self.num_attention_heads, seq_length, self.attention_head_size]
        )
        context = context.rearrange([[0], [2], [1, 3]])
        return context


model = SelfAttention()
print(model)

# %%
# Generate flow graph
# -------------------
# Then we generate the flow graph of the model.

graph = model.flow_graph_for(
    inputs=[hidet.randn([1, 128, 768]), hidet.ones([1, 128], dtype='int32')]
)
print(graph)

# %%
# Dump netron graph
# -----------------
# To visualize the flow graph, we need to dump the graph structure to a json file using
# :py:func:`hidet.utils.netron.dump` function.
from hidet.utils import netron

with open('attention-graph.json', 'w') as f:
    netron.dump(graph, f)

# %%
# Above code will generate a json file named ``attention-graph.json``.
#
# You can download the generated json file
# :download:`attention-graph.json <../../../../gallery/how-to-guides/attention-graph.json>`
# and open it with the `customized Netron viewer </netron>`_.
#

# %%
# Visualize optimization intermediate graphs
# ------------------------------------------
#
# Hidet also provides a way to visualize the intermediate graphs of the optimization passes.
#
# To get the json files for the intermediate graphs, we need to add an instrument that dumps the graph in the
# pass context before optimize it. We can use
# :py:meth:`PassContext.save_graph_instrument() <hidet.graph.transforms.PassContext.save_graph_instrument>`
# method to do that.

with hidet.graph.PassContext() as ctx:
    # print the time cost of each pass
    ctx.profile_pass_instrument(print_stdout=True)

    # save the intermediate graph of each pass to './outs' directory
    ctx.save_graph_instrument(out_dir='./outs')

    # run the optimization passes
    graph_opt = hidet.graph.optimize(graph)

# %%
# Above code will generate a directory named ``outs`` that contains the json files for the intermediate graphs.
# The optimized graph:

print(graph_opt)

# %%
# The dumped netron graphs that can be visualized:
#
# :download:`Download 1_FoldConstantPass.json <../../../../gallery/how-to-guides/outs/1_FoldConstantPass.json>`
#
# :download:`Download 2_PatternTransformPass.json <../../../../gallery/how-to-guides/outs/2_SubgraphRewritePass.json>`
#
# :download:`Download 4_ResolveVariantPass.json <../../../../gallery/how-to-guides/outs/4_ResolveVariantPass.json>`
#
# :download:`Download 5_FuseOperatorPass.json <../../../../gallery/how-to-guides/outs/5_FuseOperatorPass.json>`

# %%
# Summary
# -------
# This tutorial shows how to visualize the flow graph of a model and the intermediate graphs of the optimization passes.
#
