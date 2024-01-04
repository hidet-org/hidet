Expressions
===========

Hidet script supports the following expressions:

Literals
--------

The literal expressions are the expressions that represent constant values. An integer literal (e.g., ``1``)
has data type ``i32`` by default. A floating point literal (e.g., ``1.0``) has data type ``float32`` by default.
A boolean literal (i.e., ``True`` and ``False``) has data type ``bool`` by default. To define a literal with a
specific data type, we can use the form ``<data type>(<value>)`` like ``f16(1.0)`` in the hidet script.

Variables
---------

A variable is an expression that represents a memory location. A variable has a name and a data type. A
variable can be defined in 1) the function parameters, 2) the variable declaration statement, 3) the
for loop statement, 4) the for-mapping statement.

Unary expressions
-----------------

A unary expression is an expression that applies a unary operator to a single operand. The unary operators
supported in hidet script are:

- ``+e``: unary plus
- ``-e``: unary minus
- ``~e``: get the address of ``e``
- ``bitwise_not(e)``: bitwise not, where ``bitwise_not`` refers to ``hidet.lang.bitwise_not``
- ``not cond``: logical not

Binary expressions
------------------

A binary expression is an expression that applies a binary operator to two operands. The binary operators
supported in hidet script are:

- ``e1 + e2``: addition
- ``e1 - e2``: subtraction
- ``e1 * e2``: multiplication
- ``e1 / e2``: division (we follow the semantics of c/c++ instead of python)
- ``e1 % e2``: remainder
- ``e1 ** e2``: power
- ``e1 << e2``: left shift
- ``e1 >> e2``: right shift
- ``e1 & e2``: bitwise and
- ``e1 | e2``: bitwise or
- ``e1 ^ e2``: bitwise xor
- ``e1 and e2``: logical and
- ``e1 or e2``: logical or
- ``e1 == e2``: equal
- ``e1 != e2``: not equal
- ``e1 < e2``: less than
- ``e1 <= e2``: less than or equal
- ``e1 > e2``: greater than
- ``e1 >= e2``: greater than or equal


Note on division: in python, the division operator ``/`` will produce a floating point result even if the
operands are integers. However, in hidet script, we follow the semantics of c/c++: if the operands are integers,
the division operator ``/`` will produce an integer result with floor(a / b) value; if the operands are floating
point numbers, the division operator ``/`` will produce a floating point result.

Ternary expressions
-------------------

A ternary expression is an expression that applies a ternary operator to three operands. The ternary operator
supported in hidet script is:

- ``true_expr if cond else false_expr``: conditional expression

This operator has the same semantics as the conditional expression in c/c++: ``cond ? true_expr : false_expr``.

Subscript and slice expressions
-------------------------------

For a tensor ``t`` or tensor pointer

- ``e1[p1, p2, ..., pn]``: subscript expression
- ``e1[p1, p2:q2, p3:, :p4, :, p5]``: slice expression
- ``func(e1, e2, ..., en)``: function call expression
- ``address(e)``: get the address of ``e``, where ``address`` refers to ``hidet.lang.address``
