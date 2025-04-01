import hidet
from hidet.ir.stmt import BlackBoxStmt
from hidet.lang import attrs
from hidet.runtime import BackendException


def test_catch_runtime_exception():

    with hidet.script_module() as script_module:

        @hidet.script
        def launch():
            attrs.func_kind = 'public'

            BlackBoxStmt('throw HidetException("This is a runtime exception.");')

    func = script_module.build()
    print(func.source())
    """
    DLL void hidet_launch() {
      try {
        throw HidetException("This is a runtime exception.");
      } catch (HidetException &e) { 
        hidet_set_last_error(e.what());
        return ;
      }
    }
    """
    try:
        func()
    except BackendException as e:
        print('Caught a runtime exception: ', e)
    else:
        raise AssertionError('Should have raised a runtime exception')
