import hidet
from hidet import int32


def test_inline_let_pass():
    from hidet.lang import attrs

    with hidet.script_module() as script_module:

        @hidet.script
        def launch() -> int:
            attrs.func_kind = 'public'
            pid: int32 = 128
            bidy: int32 = pid
            pid = pid + 1
            foo: int32 = bidy
            return foo

    func = script_module.build()
    ret = func()
    assert ret == 128
