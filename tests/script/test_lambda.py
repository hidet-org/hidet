import hidet
from hidet.lang import script, attrs, printf


def test_hidet_script_lambda():
    def print_grid(m: int, n: int, f_grid):
        from hidet.ir.builders import StmtBuilder

        sb = StmtBuilder()
        with sb.for_range(m) as i:
            with sb.for_range(n) as j:
                sb += printf("%2d ", f_grid(i, j))
            sb += printf("\n")
        return sb.finish()

    with hidet.script_module() as script_module:

        @script
        def launch():
            attrs.func_kind = 'public'
            print_grid(9, 9, lambda i, j: (i + 1) * (j + 1))

    built = script_module.build()
    built()
