import pytest
from hidet.ir.expr import *
from hidet.ir.dialects.compute import compute
from hidet.ir.dialects.pattern import match, AnyExpr, UnionPattern, any_const_int


def check_pairs(pairs):
    for pattern, target, expect in pairs:
        actual, msg = match(pattern, target)
        if expect is None:
            assert actual is None
        else:
            for p in expect:
                assert p in actual
                assert expect[p] is actual[p]


def test_normal_expr():
    a, b, c, d = var('a'), var('b'), var('c'), var('d')

    pairs = [
        (Add(a, b), Add(c, d), {a: c, b: d}),
        (Add(a, a), Add(c, c), {a: c}),
        (Add(a, a), Add(c, d), None),
        (Add(a, b), Add(c, c), {a: c, b: c}),
    ]
    check_pairs(pairs)


def test_any_pattern():
    a, b, c, d = var('a'), var('b'), var('c'), var('d')
    s = Add(a, b)
    m = Multiply(a, b)
    any_expr = AnyExpr()
    any_var = AnyExpr(Var)
    any_add = AnyExpr(Add)

    pairs = [
        (any_expr, s, {any_expr: s}),
        (Add(any_expr, b), Add(c, d), {any_expr: c, b: d}),
        (any_var, Add(c, d), None),
        (any_add, s, {any_add: s}),
        (any_add, m, None)
    ]
    check_pairs(pairs)


def test_union_pattern():
    a, b, c, d = var('a'), var('b'), var('c'), var('d')
    add_cd = Add(c, d)
    mul_cd = Multiply(c, d)
    union = UnionPattern([Add(a, b), Multiply(a, b), a])

    pairs = [
        (union, c, {union: c, a: c}),
        (union, add_cd, {union: add_cd, a: c, b: d}),
        (union, mul_cd, {union: mul_cd, a: c, b: d}),
        (union, Mod(c, d), None)
    ]
    check_pairs(pairs)


if __name__ == '__main__':
    pytest.main(__file__)
