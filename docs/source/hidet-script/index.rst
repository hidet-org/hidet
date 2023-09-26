Introduction
============

Hidet Script is a domain specific language (DSL) for writing tensor programs directly in python.
The users can write the tensor programs with python's syntax with some constrains and extensions.
A transpiler is used to translate the python abstract syntax tree (AST) to Hidet's tensor program IR.
Then, the translated tensor programs in Hidet IR is optimized and compiled to the target binary for execution.
The tensor program writer works in the python environment in the whole process.

