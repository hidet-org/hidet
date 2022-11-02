"""
Visualize Flow Graph
====================

Define model
------------

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
        x = x.reshape([batch_size, seq_length, self.num_attention_heads, self.attention_head_size])
        x = x.rearrange([[0, 2], [1], [3]])
        return x  # [batch_size * num_attention_heads, seq_length, attention_head_size]

    def forward(self, hidden_states: Tensor, attention_mask: Tensor):
        batch_size, seq_length, _ = hidden_states.shape
        query = self.transpose_for_scores(self.query_layer(hidden_states))
        key = self.transpose_for_scores(self.key_layer(hidden_states))
        value = self.transpose_for_scores(self.value_layer(hidden_states))
        attention_scores = ops.matmul(query, key.transpose([-1, -2])) / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = ops.softmax(attention_scores, axis=-1)
        context = ops.matmul(attention_probs, value)
        context = context.reshape([batch_size, self.num_attention_heads, seq_length, self.attention_head_size])
        context = context.rearrange([[0], [2], [1, 3]])
        return context

# %%
# Generate flow graph
# -------------------

model = SelfAttention()
graph = model.flow_graph_for(inputs=[
    hidet.randn([1, 128, 768], device='cpu'),
    hidet.ones([1, 128], dtype='int32', device='cpu')
])
print(graph)

# %%
# Dump netron graph
# -----------------
from hidet.utils import netron

with open('attention-graph.json', 'w') as f:
    netron.dump(graph, f)

# %%
# :download:`Download attention-graph.json <../../../../gallery/how-to-guides/attention-graph.json>`

# %%
# Visualize optimization intermediate graphs
# ------------------------------------------

with hidet.graph.PassContext() as ctx:

    # print the time cost of each pass
    ctx.profile_pass_instrument(print_stdout=True)

    # save the intermediate graph of each pass to './outs' directory
    ctx.save_graph_instrument(out_dir='./outs')

    # run the optimization passes
    graph_opt = hidet.graph.optimize(graph)

# %%
# The optimized graph:

print(graph_opt)

# %%
# The dumped netron graphs that can be visualized:
#
# :download:`Download 1_FoldConstantPass.json <../../../../gallery/how-to-guides/outs/1_FoldConstantPass.json>`
#
# :download:`Download 2_PatternTransformPass.json <../../../../gallery/how-to-guides/outs/2_PatternTransformPass.json>`
#
# :download:`Download 4_ResolveVariantPass.json <../../../../gallery/how-to-guides/outs/4_ResolveVariantPass.json>`
#
# :download:`Download 5_FuseOperatorPass.json <../../../../gallery/how-to-guides/outs/5_FuseOperatorPass.json>`

# %%
# Summary
# -------