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
  gallery/tutorials/run-onnx-model

.. toctree::
  :maxdepth: 1
  :caption: How-to Guide

  how-to-guides/add-new-operator/index
  gallery/how-to-guides/add-operator-resolve-rule
  gallery/how-to-guides/add-subgraph-rewrite-rule
  gallery/how-to-guides/visualize-flow-graph

.. toctree::
  :maxdepth: 1
  :caption: Developer Guide

  developer-guides/contributing.rst
  developer-guides/hidet-script/index

.. toctree::
  :maxdepth: 1
  :caption: Notes

  notes/operator-cache

.. toctree::
  :maxdepth: 1
  :caption: Reference

  python_api/index.rst
  genindex
