# %%
import sys
from lark import Lark, Transformer


with open('hidet/python/hidet/ir/hidet.lark') as f:
    hidet_grammar = f.read()

json_parser = Lark(hidet_grammar, start='module')

with open('test.txt') as f:
    tree = json_parser.parse(f.read())
    print(tree.pretty())

# %%

grammar = """
    expr: expr "+" expr | SIGNED_NUMBER
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""

parser = Lark(grammar, start="expr")

tree = parser.parse("42 + 3 + 4")
print(tree.pretty())