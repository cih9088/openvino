# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def get_debuggers_by_target(algo_name, debuggers=[]):
    selected_debuggers = []
    for debugger in debuggers:
        if algo_name in debugger.targets:
            selected_debuggers.append(debugger)
    return selected_debuggers


def get_debugger_by_name(name, debuggers=[]):
    for debugger in debuggers:
        if name == debugger.name:
            return debugger
    return None
