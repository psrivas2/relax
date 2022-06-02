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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""Pytorch call-packed library generator"""
import yaml


def read_pytorch_ops_from_yaml():
    """open native_functions.yaml and return all contents"""
    with open("native_functions.yaml", "r") as f:
        dictionary = yaml.full_load(f)
    return dictionary


def gen_call_packed():
    """Generate call packed library"""
    native_functions = read_pytorch_ops_from_yaml()
    for item in native_functions[:]:
        print(item["func"])
        print(item["variants"])


if __name__ == "__main__":
    gen_call_packed()
