# %%
import sys
from lark import Lark, Transformer

# with open('hidet.lark') as f:
#     hidet_grammar = f.read()

# json_parser = Lark(hidet_grammar, start='module')

# with open('test.txt') as f:
#     tree = json_parser.parse(f.read())
#     print(tree.pretty())


grammar = """
    expr: or_expr
    
    ?or_expr: xor_expr ("|" xor_expr)*
    ?xor_expr: and_expr ("^" and_expr)*
    ?and_expr: shift_expr ("&" shift_expr)*
    ?shift_expr: arith_expr (_shift_op arith_expr)*
    ?arith_expr: term (_add_op term)*
    ?term: factor (_mul_op factor)*
    ?factor: _unary_op factor | power
    ?power: atom ("**" factor)?

    ?atom: "(" expr ")" 
        | "True"    -> const_true
        | "False"   -> const_false
        | SIGNED_NUMBER -> number
        | IDENT -> ident

    !_unary_op: "+"|"-"
    !_add_op: "+"|"-"
    !_shift_op: "<<"|">>"
    !_mul_op: "*"|"/"|"%"
    !comp_op: "<"|">"|"=="|">="|"<="|"<>"|"!="

    IDENT : /[^\W\d]\w*/

    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""

grammar = """
    start: [static] [expr ("," expr)*]
    static: "static"
    expr : "a"

    %import common.WS
    %ignore WS
"""

parser = Lark(grammar, start="start")

tree = parser.parse("")
print(tree.pretty())
# %%
 