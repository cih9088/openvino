# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from addict import Dict
from copy import deepcopy


class POTDebugger(ABC):
    def __init__(self, config, engine):
        self.config = Dict(deepcopy(config))
        self._engine = engine
