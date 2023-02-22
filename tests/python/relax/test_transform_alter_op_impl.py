# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest
import tvm.testing

from tvm import relax, tir, topi
from tvm.script import tir as T, ir as I, relax as R
from tvm.tir import IndexMap

kOperatorKind = "operator_kind"
kFrozenLayout = "frozen_layout"


def test_replace():
    func_name = "conv2d"

    def gen_mod(data_layout="NCHW"):
        bb = relax.BlockBuilder()

        # a symbolic variable to represent minibatch size
        N, C, H, W = 32, 3, 224, 224
        O, kH, kW = 64, 5, 5
        if data_layout == "NCHW":
            kernel_layout = "OIHW"
            data_shape = (N, C, H, W)
            kernel_shape = (O, C, kH, kW)
        else:
            kernel_layout = "HWIO"
            data_shape = (N, H, W, C)
            kernel_shape = (kH, kW, C, O)

        x = relax.Var("x", relax.TensorStructInfo(shape=data_shape))
        filter = relax.Var("filter", relax.TensorStructInfo(shape=kernel_shape))
        # build a three linear-layer neural network for a classification task
        with bb.function("main", params=[x, filter]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, x)
                lv1 = bb.emit_te(
                    topi.nn.conv2d,
                    lv0,
                    filter,
                    strides=1,
                    padding=1,
                    dilation=1,
                    data_layout=data_layout,
                    kernel_layout=kernel_layout,
                    primfunc_name_hint=func_name,
                )
                lv2 = bb.emit_te(topi.nn.relu, lv1)
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    @T.prim_func
    def conv2d_NHWC(
        rxplaceholder: T.Buffer((T.int64(32), T.int64(224), T.int64(224), T.int64(3)), "float32"),
        rxplaceholder_1: T.Buffer((T.int64(5), T.int64(5), T.int64(3), T.int64(64)), "float32"),
        conv2d_nhwc: T.Buffer((T.int64(32), T.int64(222), T.int64(222), T.int64(64)), "float32"),
    ):
        T.func_attr({"operator_kind": "conv2d", "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(32), T.int64(226), T.int64(226), T.int64(3)))
        for i0, i1, i2, i3 in T.grid(T.int64(32), T.int64(226), T.int64(226), T.int64(3)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(rxplaceholder[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                    T.int64(1) <= v_i1
                    and v_i1 < T.int64(225)
                    and T.int64(1) <= v_i2
                    and v_i2 < T.int64(225),
                    rxplaceholder[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3],
                    T.float32(0),
                )
        for nn, yy, xx, ff, ry, rx, rc in T.grid(
            T.int64(32),
            T.int64(222),
            T.int64(222),
            T.int64(64),
            T.int64(5),
            T.int64(5),
            T.int64(3),
        ):
            with T.block("conv2d_nhwc"):
                v_nn, v_yy, v_xx, v_ff, v_ry, v_rx, v_rc = T.axis.remap(
                    "SSSSRRR", [nn, yy, xx, ff, ry, rx, rc]
                )
                T.reads(
                    pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc],
                    rxplaceholder_1[v_ry, v_rx, v_rc, v_ff],
                )
                T.writes(conv2d_nhwc[v_nn, v_yy, v_xx, v_ff])
                with T.init():
                    conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] = T.float32(0)
                conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] = (
                    conv2d_nhwc[v_nn, v_yy, v_xx, v_ff]
                    + pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc]
                    * rxplaceholder_1[v_ry, v_rx, v_rc, v_ff]
                )

    # get and print the IRmodule being built
    before = gen_mod(data_layout="NCHW")
    before[func_name] = before[func_name].with_attr(kOperatorKind, func_name)
    before["main"].show()
    after = relax.transform.AlterOpImpl(
        {func_name: conv2d_NHWC},
        {
            func_name: [
                IndexMap.from_func(lambda n, c, h, w: (n, h, w, c)),
                IndexMap.from_func(lambda o, i, h, w: (h, w, i, o)),
                IndexMap.from_func(lambda n, c, h, w: (n, h, w, c)),
            ]
        },
    )(before)
    after.show()


def test_single_output():
    @I.ir_module
    class InputModule:
        @R.function
        def foo(
            x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")
        ) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv = R.call_tir(add, (x, y), out_sinfo=R.Tensor((16,), dtype="float32"))
                gv: R.Tensor((16,), dtype="float32") = lv
                R.output(gv)
            return gv

        @T.prim_func
        def add(
            arg0: T.Buffer((16,), "float32"),
            arg1: T.Buffer((16,), "float32"),
            output: T.Buffer((16,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            for ax0 in T.grid(16):
                with T.block("T_add"):
                    v_ax0 = T.axis.remap("S", [ax0])
                    T.reads(arg0[v_ax0], arg1[v_ax0])
                    T.writes(output[v_ax0])
                    output[v_ax0] = arg0[v_ax0] + arg1[v_ax0]

    before = InputModule
    func_name = "add"
    before[func_name] = before[func_name].with_attr(kOperatorKind, func_name)

    @T.prim_func
    def add_2d(
        arg0: T.Buffer((4, 4), "float32"),
        arg1: T.Buffer((4, 4), "float32"),
        output: T.Buffer((4, 4), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        for ax0, ax1 in T.grid(4, 4):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                T.writes(output[v_ax0, v_ax1])
                output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

    index_map = lambda i: (i // 4, i % 4)
    after = relax.transform.AlterOpImpl(
        {func_name: add_2d}, {func_name: [index_map, index_map, index_map]}
    )(before)
    after.show()


def test_multiple_outputs():
    @I.ir_module
    class InputModule:
        @R.function
        def foo(
            x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")
        ) -> R.Tuple(R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")):
            with R.dataflow():
                gv = R.call_tir(
                    some_op,
                    (x, y),
                    out_sinfo=[R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")],
                )
                R.output(gv)
            return gv

        @T.prim_func
        def some_op(
            arg0: T.Buffer((16,), "float32"),
            arg1: T.Buffer((16,), "float32"),
            output0: T.Buffer((16,), "float32"),
            output1: T.Buffer((16,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            for ax0 in T.grid(16):
                with T.block("T_add"):
                    v_ax0 = T.axis.remap("S", [ax0])
                    T.reads(arg0[v_ax0], arg1[v_ax0])
                    T.writes(output0[v_ax0], output1[v_ax0])
                    output0[v_ax0] = arg0[v_ax0] + arg1[v_ax0]
                    output1[v_ax0] = arg0[v_ax0] - arg1[v_ax0]

    before = InputModule
    func_name = "some_op"
    before[func_name] = before[func_name].with_attr(kOperatorKind, func_name)

    @T.prim_func
    def some_op_2d(
        arg0: T.Buffer((4, 4), "float32"),
        arg1: T.Buffer((4, 4), "float32"),
        output0: T.Buffer((4, 4), "float32"),
        output1: T.Buffer((4, 4), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        for ax0, ax1 in T.grid(4, 4):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                T.writes(output0[v_ax0, v_ax1], output1[v_ax0, v_ax1])
                output0[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]
                output1[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] - arg1[v_ax0, v_ax1]

    index_map = lambda i: (i // 4, i % 4)
    after = relax.transform.AlterOpImpl(
        {func_name: some_op_2d}, {func_name: [index_map, index_map, index_map, index_map]}
    )(before)
    after.show()


def test_implicit_padding_output():
    @I.ir_module
    class InputModule:
        @R.function
        def foo(x: R.Tensor((14,), dtype="float32")) -> R.Tensor((14,), dtype="float32"):
            with R.dataflow():
                lv = R.call_tir(relu, (x,), out_sinfo=R.Tensor((14,), dtype="float32"))
                gv: R.Tensor((14,), dtype="float32") = lv
                R.output(gv)
            return gv

        @T.prim_func
        def relu(arg0: T.Buffer((14,), "float32"), output: T.Buffer((14,), "float32")):
            T.func_attr({"tir.noalias": True})
            for ax0 in T.grid(14):
                with T.block("T_add"):
                    v_ax0 = T.axis.remap("S", [ax0])
                    T.reads(arg0[v_ax0])
                    T.writes(output[v_ax0])
                    output[v_ax0] = T.max(arg0[v_ax0], T.float32(0))

    before = InputModule
    func_name = "relu"
    before[func_name] = before[func_name].with_attr(kOperatorKind, func_name)

    @T.prim_func
    def relu_pad(arg0: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")):
        T.func_attr({"tir.noalias": True})
        for ax0 in T.grid(16):
            with T.block("T_add"):
                v_ax0 = T.axis.remap("S", [ax0])
                T.reads(arg0[v_ax0])
                T.writes(output[v_ax0])
                output[v_ax0] = T.max(arg0[v_ax0], T.float32(0))

    index_map = lambda i: (i % 16)
    with pytest.raises(
        tvm.TVMError, match="Non bijective transforms on input and output buffers are not supported"
    ):
        _ = relax.transform.AlterOpImpl({func_name: relu_pad}, {func_name: [index_map, index_map]})(
            before
        )


if __name__ == "__main__":
    tvm.testing.main()
