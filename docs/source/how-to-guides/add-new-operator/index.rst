Add New Operator
================

Hidet is designed to be extensible. It is easy to add new operators to Hidet. There are two ways to add and schedule
an operator.

1. **Rule-based Scheduling**
   Define the mathematical computation of the operator, and use Hidet's rule-based scheduler to implement the
   computation.
2. **Template-based Scheduling**
   Besides the computation, also give the concrete implementation of the operator.

.. toctree::
  :maxdepth: 1
  :caption: Two Methods

  ../../gallery/how-to-guides/add-new-operator-rule-based
  ../../gallery/how-to-guides/add-new-operator-template-based
