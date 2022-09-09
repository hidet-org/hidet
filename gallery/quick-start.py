"""
Quick Start
===========

This guide walks through the key functionality of Hidet for tensor computation.
"""

# %%
# We should first import hidet.
import hidet

# %%
# Define tensors
# --------------
# A *tensor* is a n-dim array. As other machine learning framework,
# Hidet takes tensor as the core object to compute and manipulate.
# The following code defines a tensor with shape [2, 3] with randomly initialized values.

a = hidet.randn([2, 3])
print(a)


# %%
# Run operators
# -------------


# %%
# Symbolic execution and FlowGraph
# --------------------------------
#

# %%
# Optimize FlowGraph
# ------------------
