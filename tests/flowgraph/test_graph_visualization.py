# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pytest
import hidet
from hidet.graph.impl.graph_impl import draw_graph


def test_draw_graph_simple():
    """Test drawing a simple flow graph with a few operators."""
    # Create a simple graph
    x = hidet.symbol([3, 4], device='cuda')
    y = x + 3.0
    y = hidet.ops.square(x) - y
    y = hidet.ops.square(y) - x
    graph = hidet.trace_from(y, x)

    # Output directory for test
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Draw graph to file - use base name without extension
    filename = os.path.join(out_dir, 'simple_graph')
    draw_graph(graph, filename)

    # Verify the DOT file was created
    dot_filename = os.path.join(out_dir, 'simple_graph.dot')
    assert os.path.exists(dot_filename), f"Failed to create DOT file at {dot_filename}"

    # Test with show_tensors=True
    filename_with_tensors = os.path.join(out_dir, 'simple_graph_with_tensors')
    draw_graph(graph, filename_with_tensors)

    # Verify the DOT file with tensors was created
    dot_filename_with_tensors = os.path.join(out_dir, 'simple_graph_with_tensors.dot')
    assert os.path.exists(dot_filename_with_tensors), f"Failed to create DOT file at {dot_filename_with_tensors}"

    print(f"Graph visualizations saved to {out_dir}")


def test_draw_graph_complex():
    """Test drawing a more complex flow graph with multiple layers."""
    # Create a more complex graph with multiple operations
    x = hidet.symbol([16, 32], device='cuda')

    # Create a small network with multiple layers
    y = hidet.ops.matmul(x, hidet.randn([32, 64], device='cuda'))
    y = hidet.ops.relu(y)
    y = hidet.ops.matmul(y, hidet.randn([64, 32], device='cuda'))
    y = hidet.ops.sigmoid(y)
    y = y + x  # Add a residual connection
    graph = hidet.trace_from(y, x)

    # Output directory for test
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Draw graph to file
    filename = os.path.join(out_dir, 'complex_graph')
    draw_graph(graph, filename)

    # Verify the DOT file was created
    dot_filename = os.path.join(out_dir, 'complex_graph.dot')
    assert os.path.exists(dot_filename), f"Failed to create DOT file at {dot_filename}"

    print(f"Complex graph visualization saved to {out_dir}")


def test_draw_graph_large():
    """Test drawing a large flow graph with approximately 50 nodes."""
    # Create a larger graph with many operations
    x = hidet.symbol([32, 32], device='cuda')

    # Create a deep network with many layers to generate ~50 nodes
    # First create a multi-layer network
    hidden_sizes = [64, 128, 256, 128, 64, 32]
    y = x

    # Add many layers with activations and operations
    for i, size in enumerate(hidden_sizes):
        # Linear transformation
        w = hidet.randn([y.shape[1], size], device='cuda')
        y = hidet.ops.matmul(y, w)

        # Add bias
        b = hidet.randn([size], device='cuda')
        y = y + b

        # Add activation - alternate between different activations
        if i % 3 == 0:
            y = hidet.ops.relu(y)
        elif i % 3 == 1:
            y = hidet.ops.sigmoid(y)
        else:
            y = hidet.ops.tanh(y)

        # Add more operations to increase node count
        if i % 2 == 0:
            # Add a residual-like connection if shapes match
            if i > 0 and y.shape[1] == hidden_sizes[i - 1]:
                prev_layer = hidet.randn([32, size], device='cuda')
                y = y + prev_layer

        # Add normalization-like operations
        y_mean = hidet.ops.mean(y, dims=1, keep_dim=True)
        y_var = hidet.ops.var(y, dims=1, keep_dim=True)
        y = (y - y_mean) / hidet.ops.sqrt(y_var + 1e-5)

        # Scale and shift
        gamma = hidet.randn([size], device='cuda')
        beta = hidet.randn([size], device='cuda')
        y = y * gamma + beta

    # Add final operations
    y = hidet.ops.softmax(y, axis=1)

    # Trace the graph
    graph = hidet.trace_from(y, x)

    # Output directory for test
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Draw graph to file
    filename = os.path.join(out_dir, 'large_graph')
    draw_graph(graph, filename, show_tensors=True)

    # Verify the DOT file was created
    dot_filename = os.path.join(out_dir, 'large_graph.dot')
    assert os.path.exists(dot_filename), f"Failed to create DOT file at {dot_filename}"

    print(f"Large graph visualization saved to {out_dir}")
    print(f"The large graph has {len(graph.nodes)} nodes")


if __name__ == "__main__":
    # Run tests directly
    test_draw_graph_simple()
    test_draw_graph_complex()
    test_draw_graph_large()
    print("All tests passed!")
