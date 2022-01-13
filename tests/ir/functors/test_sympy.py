import pytest
from hidet.ir.expr import convert, var
from hidet.ir.functors import to_sympy, from_sympy, equal, coefficients


def test_equality():
    a, b, c = var('a'), var('b'), var('c')
    pairs = [
        ((a + b) * (a + b), a * a + b * b + a * b * 2, True),
        (a + b * 3 + c, a + c + b / 3, False)
    ]
    for p, q, r in pairs:
        if r:
            assert equal(p, q)
        else:
            assert not equal(a, b)


def test_to_sympy():
    a, b, c = var('a'), var('b'), var('c')
    samples = [
        ((a * b), 'a*b'),
        ((a / 3), 'a/3')
    ]
    for e, s in samples:
        assert str(to_sympy(e)[0]) == s


def test_to_sympy_from_sympy():
    a, b, c = var('a'), var('b'), var('c')
    exprs = [
        a * b + c
    ]
    for e in exprs:
        sexpr, smap = to_sympy(e)
        new_e = from_sympy(sexpr, smap)
        assert equal(new_e, e)


def test_coefficient():
    x = var('x')
    i, j = var('i'), var('j')
    p, q, r = var('p'), var('q'), var('r')

    samples = [
        # bases, polynomial about bases, terms
        (
            [x],
            x * (p + q) * r,
            {
                (1,): p * r + q * r,
            }
        ),
        (
            [x],
            r * (p * x + q * x),
            {
                (1,): r * (p + q),
            }
        ),
        (
            [x],
            x * convert(32) + convert(16),
            {
                (1,): convert(32),
                (0,): convert(16)
            }
        ),
        (
            [i, j],
            i * 32 + j * 16 + i * j * 8 + 4,
            {
                (1, 1): convert(8),
                (1, 0): convert(32),
                (0, 1): convert(16),
                (0, 0): convert(4)
            }
        ),
        (
            [i, j],
            (i + j + x) * (i + j + x),
            {
                (1, 1): convert(2),
                (1, 0): x * 2,
                (0, 1): x * 2,
                (2, 0): convert(1),
                (0, 2): convert(1),
                (0, 0): x * x
            }
        )
    ]

    for bases, poly_expr, terms in samples:
        actual_terms = coefficients(poly_expr, bases)
        assert len(actual_terms) == len(terms)
        for term in actual_terms:
            actual_value = actual_terms[term]
            expect_value = terms[term]
            assert equal(actual_value, expect_value), str(actual_value) + ' ' + str(expect_value)


if __name__ == '__main__':
    pytest.main(__file__)
