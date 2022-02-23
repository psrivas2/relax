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
"""Tuning Resnet with AutoTIR."""

import tvm
import tvm.testing
from tvm.relay import testing
from tvm import relax, relay
from tvm.relax.testing import relay_translator, nn
from tvm.runtime import vm as vm_rt
import numpy as np
import os
import tempfile
import time
from tvm.meta_schedule import ReplayTraceConfig, tune_tir
from tvm.meta_schedule.integration import extract_task_from_relax
from tvm.meta_schedule.database import JSONDatabase
from tvm import transform


from tvm.meta_schedule.builder import LocalBuilder
from tvm.meta_schedule.runner import LocalRunner


def autotir_tune(batch_size, target, database, is_tune, layout="NHWC", dtype="float32"):
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)

    relay_mod, params = testing.resnet.get_workload(
        num_layers=50,
        batch_size=batch_size,
        dtype=dtype,
        layout=layout,
        image_shape=image_shape,
    )

    relax_mod = relay_translator.from_relay(relay_mod["main"])

    if is_tune:
        print("tuning")
        runner = LocalRunner()
        builder = LocalBuilder()
        tasks = extract_task_from_relax(relax_mod, target=target)
        print("total tasks:", len(tasks))
        for task in tasks:
            print(f"Extracted task: {task.task_name}, {task.target}")
            with tempfile.TemporaryDirectory() as work_dir:
                sch = tune_tir(
                    mod=task.mod,
                    target=target,
                    config=ReplayTraceConfig(
                        num_trials_per_iter=64,
                        num_trials_total=2000,
                    ),
                    builder=builder,
                    runner=runner,
                    work_dir=work_dir,
                    database=database,
                    num_threads=16,
                )
    return relay_mod, params, relax_mod


if __name__ == "__main__":
    target = tvm.target.Target("llvm -mcpu=core-avx2 --num-cores=16")
    network = "resnet-50"
    batch_size = 1
    layout = "NCHW"
    # layout = "NHWC"
    dtype = "float32"

    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)
    input_shape = (batch_size,) + image_shape

    workload_file = "autotir_logs/" + "%s-%s-B%d-%s-workload.json" % (
        network,
        layout,
        batch_size,
        target.kind.name,
    )

    tuning_record_file = "autotir_logs/" + "%s-%s-B%d-%s-tuning_record.json" % (
        network,
        layout,
        batch_size,
        target.kind.name,
    )

    os.makedirs(os.path.dirname(workload_file), exist_ok=True)
    is_tune = not os.path.exists(workload_file)

    database = JSONDatabase(
        path_workload=workload_file,
        path_tuning_record=tuning_record_file,
    )

    # tune the model
    relay_mod, relay_params, relax_mod = autotir_tune(
        batch_size, target, database, is_tune, layout, dtype
    )

    # resnet benchmarking
    with transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyHistoryBest(database, target)(relax_mod)
        relax_ex, relax_lib = relax.vm.build(relax_mod, target)

    input_shape = (1, 3, 224, 224)
    data = tvm.nd.array(np.random.rand(*input_shape).astype(np.float32))
    params = nn.init_params(relax_mod)

    # measure relay performance
    with transform.PassContext(opt_level=3):
        exe = relay.vm.compile(relay_mod, target, params=relay_params)
    relay_vm = vm_rt.VirtualMachine(exe, tvm.cpu())
    # inputs = [data] + params
    result = relay_vm.run(data)
    tic = time.time()
    for i in range(100):
        relay_vm.run(data)
    toc = time.time()
    e0 = toc - tic

    # measure relax performance
    relax_vm = relax.VirtualMachine(relax_ex, tvm.cpu(), mod=relax_lib)
    res = relax_vm["main"](data, *params)
    tic = time.time()
    for i in range(100):
        relax_vm["main"](data, *params)
    toc = time.time()
    e1 = toc - tic

    print(f"relay: {e0}")
    print(f"relax: {e1}")