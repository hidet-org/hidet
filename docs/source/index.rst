Welcome to Hidet's Documentation
================================

Hidet is an open-source DNN inference framework, it features

- **Ease of Use**: Support end to end inference for PyTorch and ONNX models.
- **High Performance**: Graph-level optimizations and operator-level kernel tuning.
- **Extensibility**: Easy to add new operators, and fusion patterns.
- **Python Oriented**: All core components are written in Python.


.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  getting-started/install
  gallery/getting-started/quick-start

.. toctree::
  :maxdepth: 1
  :caption: Tutorials

  gallery/tutorials/optimize-pytorch-model
  gallery/tutorials/optimize-onnx-model

.. toctree::
  :maxdepth: 1
  :caption: How-to Guide

  gallery/how-to-guides/visualize-flow-graph

.. toctree::
  :maxdepth: 1
  :caption: Developer Guide

  how-to-guides/add-new-operator/index
  gallery/developer-guides/add-operator-resolve-rule
  gallery/developer-guides/add-subgraph-rewrite-rule
  developer-guides/contributing.rst

.. toctree::
  :maxdepth: 1
  :caption: Hidet Script

  hidet-script/index

.. toctree::
  :maxdepth: 1
  :caption: Notes

  notes/operator-cache

.. toctree::
  :maxdepth: 1
  :caption: Reference

  python_api/index.rst
  genindex
