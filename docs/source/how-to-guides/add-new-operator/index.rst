Add New Operator
================

Hidet is designed to be extensible. It is easy to add new operators to Hidet. There are two ways to add and schedule
an operator.

1. **Rule-based Scheduling**
   Define the mathematical computation of the operator, and Hidet will automatically schedule the computation into
   parallel tensor program with Hidet's rule-based scheduler.
2. **Template-based Scheduling**
   Besides the computation, user can also give the concrete implementation of the operator to achieve better performance
   for complex operators.

.. toctree::
  :maxdepth: 1
  :caption: Define Computation

  ../../gallery/how-to-guides/add-new-operator-compute-definition

.. toctree::
  :maxdepth: 1
  :caption: Two Scheduling Methods

  ../../gallery/how-to-guides/add-new-operator-rule-based
  ../../gallery/how-to-guides/add-new-operator-template-based
