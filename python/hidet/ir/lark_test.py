# %%
import sys
from lark import Lark, Transformer


with open('hidet/python/hidet/ir/hidet.lark') as f:
    hidet_grammar = f.read()

json_parser = Lark(hidet_grammar, start='module')

with open('test.txt') as f:
    tree = json_parser.parse(f.read())
    print(tree.pretty())

