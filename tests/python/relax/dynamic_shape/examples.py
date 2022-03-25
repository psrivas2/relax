from __future__ import annotations  # must import to defer parsing of annotations
import os
import numpy as np
import tvm
from tvm.relay import Call
from tvm import relax, tir, topi, te
from tvm.runtime import container
from tvm.relax.testing import nn

import tvm.script
from tvm.script import tir as T, relax as R


class ShapeExamples:
    """A class to express shape examples"""

    # symbolic dimensions to be used in various shape expressions
    def start():
        m, n, k = tir.Var("m", "int64"), tir.Var("n", "int64"), tir.Var("k", "int64")

        def te_func_concat(A, B):
            """A te function to concatenate A & B"""
            C = te.compute((n + m), lambda i: tvm.tir.if_then_else(i < n, A[i], B[i - n]))
            return C

        def build_concat(a, b):
            bb = relax.BlockBuilder()
            with bb.function("concat", [a, b]):
                gv0 = bb.emit_te(te_func_concat, a, b)
                bb.emit_func_output(gv0)
            mod = bb.get()
            return mod

        def run():
            # create variables
            data = relax.Var(
                "data",
                [n],
                relax.DynTensorType(1, "float32"),
            )
            weight = relax.Var(
                "weight",
                [m],
                relax.DynTensorType(1, "float32"),
            )

            # construct a concat expression
            mod = build_concat(data, weight)
            print(R.parser.astext(mod))
            # build and create vm executor
            target = tvm.target.Target("llvm", host="llvm")
            ex = relax.vm.build(mod, target)
            vm = relax.VirtualMachine(ex, tvm.cpu())
            # run the mlp model on relax vm
            data = tvm.nd.array(
                np.random.rand(
                    10,
                ).astype(np.float32)
            )
            weight = tvm.nd.array(
                np.random.rand(
                    16,
                ).astype(np.float32)
            )
            res = vm["concat"](data, weight)
            print(res)

        run()


if __name__ == "__main__":
    ShapeExamples.start()
