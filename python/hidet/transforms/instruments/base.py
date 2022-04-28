from hidet.ir.func import IRModule


class PassInstrument:
    def before_all_passes(self, ir_module: IRModule):
        pass

    def before_pass(self, pass_name: str, ir_module: IRModule):
        pass

    def after_pass(self, pass_name: str, ir_module: IRModule):
        pass

    def after_all_passes(self, ir_module: IRModule):
        pass


