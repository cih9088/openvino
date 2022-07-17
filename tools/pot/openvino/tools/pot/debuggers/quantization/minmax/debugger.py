# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
from copy import deepcopy
from functools import partial

import numpy as np
from openvino.tools.pot.algorithms.quantization import fake_quantize as fqut
from openvino.tools.pot.algorithms.quantization.utils import get_tensor_statistics
from openvino.tools.pot.graph import model_utils as mu
from openvino.tools.pot.graph import node_utils as nu
from openvino.tools.pot.statistics.statistics import TensorStatistic

from ...debugger import POTDebugger
from .plots import _plot_fake_quantize_node, _plot_node, _plot_weight_node


PREFIX = "debugger_"
STAT_KEY_SEPARATOR = "@@"
NAME_NORMALIZER_DICT = {"/": "##"}

FAKE_QUANTIZE_CONFIG_KEY = "fake_quantize_config.json"
FAKE_QUANTIZE_MAPPING_KEY = "fake_quantize_mapping.pkl"
FAKE_QUANTIZE_PARAMS_KEY = "fake_quantize_params.pkl"
FAKE_QUANTIZE_ATTRS_KEY = "fake_quantize_attrs.pkl"
DEBUGGER_CONFIG = "config.json"


def normalize_name(name):
    if isinstance(name, tuple):
        name = f"{name[0]}.{name[1]}"

    for old, new in NAME_NORMALIZER_DICT.items():
        name = name.replace(old, new)
    return name


def denormalize_name(name, return_tuple=False):
    for old, new in NAME_NORMALIZER_DICT.items():
        name = name.replace(new, old)

    if return_tuple:
        try:
            port = int(name.split(".")[-1])
            name = (".".join(name.split(".")[:-1]), port)
        except:
            pass

    return name


def _get_stats_collector(seed, p, custom_stats_fn):
    prng = np.random.RandomState(seed)

    # default stats collector
    stats_collector = {
        PREFIX + "identity_p": lambda x: x if prng.uniform(0, 1) < p else np.array([])
    }

    # user defined stats collector
    if custom_stats_fn:
        for k, v in custom_stats_fn.items():
            k_prefixed = PREFIX + k
            assert (
                k_prefixed not in stats_collector
            ), f"{k} is reservied for default collector"
            stats_collector[k_prefixed] = v

    # convert function to tensorstatistic
    for k, v in stats_collector.items():
        stats_collector[k] = TensorStatistic(v)

    return stats_collector


def _load_all(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


class MinMaxQuantizationDebugger(POTDebugger):
    name = "MinMaxQuantizationDebugger"
    targets = ["MinMaxQuantization", "DefaultQuantization"]

    def __init__(self, config, engine=None):
        super().__init__(config, engine)

        self._open_files = dict()
        self._dump_path = self.config.get("dump_path", None)
        if self._dump_path is not None:
            os.makedirs(self._dump_path, exist_ok=True)

        self._fake_quantize_config = None
        self._fake_quantize_mapping = None
        self._fake_quantize_params = None
        self._fake_quantize_attrs = None

        self._quantized_stats = None
        self._original_stats = None

        self._dumpable = False

    @property
    def stats_collector(self):
        seed = self.config.get("seed", 0)
        identity_p = self.config.get("identity_p", 0.2)
        custom_stats_fn = self.config.get("stats_collector", None)
        return _get_stats_collector(seed, identity_p, custom_stats_fn)

    @property
    def fake_quantize_config(self):
        if self._fake_quantize_config is None:
            if self._dump_path is not None and os.path.exists(self._dump_path):
                config_path = os.path.join(self._dump_path, FAKE_QUANTIZE_CONFIG_KEY)
                with open(config_path, "r") as f:
                    self._fake_quantize_config = json.load(f)

        assert (
            self._fake_quantize_config is not None
        ), "Not able to load fake_quantize_config"
        return self._fake_quantize_config

    @fake_quantize_config.setter
    def fake_quantize_config(self, fake_quantize_config):
        self._fake_quantize_config = deepcopy(fake_quantize_config)
        if self._dump_path is not None and os.path.exists(self._dump_path):
            config_path = os.path.join(self._dump_path, FAKE_QUANTIZE_CONFIG_KEY)
            with open(config_path, "w") as f:
                json.dump(self._fake_quantize_config, f)

    @property
    def fake_quantize_mapping(self):
        if self._fake_quantize_mapping is None:
            if self._dump_path is not None and os.path.exists(self._dump_path):
                mapping_path = os.path.join(self._dump_path, FAKE_QUANTIZE_MAPPING_KEY)
                with open(mapping_path, "rb") as f:
                    self._fake_quantize_mapping = pickle.load(f)

        assert (
            self._fake_quantize_mapping is not None
        ), "Not able to load fake_quantize_mapping"
        return self._fake_quantize_mapping

    @fake_quantize_mapping.setter
    def fake_quantize_mapping(self, fake_quantize_mapping):
        self._fake_quantize_mapping = deepcopy(fake_quantize_mapping)
        if self._dump_path is not None and os.path.exists(self._dump_path):
            mapping_path = os.path.join(self._dump_path, FAKE_QUANTIZE_MAPPING_KEY)
            with open(mapping_path, "wb") as f:
                pickle.dump(self._fake_quantize_mapping, f)

    @property
    def fake_quantize_params(self):
        if self._fake_quantize_params is None:
            if self._dump_path is not None and os.path.exists(self._dump_path):
                params_path = os.path.join(self._dump_path, FAKE_QUANTIZE_PARAMS_KEY)
                with open(params_path, "rb") as f:
                    self._fake_quantize_params = pickle.load(f)

        assert (
            self._fake_quantize_params is not None
        ), "Not able to load fake_quantize_params"
        return self._fake_quantize_params

    @fake_quantize_params.setter
    def fake_quantize_params(self, fake_quantize_params):
        self._fake_quantize_params = deepcopy(fake_quantize_params)
        if self._dump_path is not None and os.path.exists(self._dump_path):
            params_path = os.path.join(self._dump_path, FAKE_QUANTIZE_PARAMS_KEY)
            with open(params_path, "wb") as f:
                pickle.dump(self._fake_quantize_params, f)

    @property
    def fake_quantize_attrs(self):
        if self._fake_quantize_attrs is None:
            if self._dump_path is not None and os.path.exists(self._dump_path):
                attrs_path = os.path.join(self._dump_path, FAKE_QUANTIZE_ATTRS_KEY)
                with open(attrs_path, "rb") as f:
                    self._fake_quantize_attrs = pickle.load(f)

        assert (
            self._fake_quantize_attrs is not None
        ), "Not able to load fake_quantize_attrs"
        return self._fake_quantize_attrs

    @fake_quantize_attrs.setter
    def fake_quantize_attrs(self, fake_quantize_attrs):
        self._fake_quantize_attrs = deepcopy(fake_quantize_attrs)
        if self._dump_path is not None and os.path.exists(self._dump_path):
            attrs_path = os.path.join(self._dump_path, FAKE_QUANTIZE_ATTRS_KEY)
            with open(attrs_path, "wb") as f:
                pickle.dump(self._fake_quantize_attrs, f)

    @property
    def original_stats(self):
        return self._original_stats

    @original_stats.setter
    def original_stats(self, original_stats):
        if self._open_files:
            for f in self._open_files.values():
                if not f.closed:
                    f.close()
        self._open_files = dict()

        identity_key = PREFIX + "identity_p"
        for stats in original_stats.values():
            if identity_key in stats:
                stats[identity_key] = [
                    stat
                    for stat in stats[identity_key]
                    if stat is not None and stat.size != 0
                ]
        self._original_stats = original_stats

    @property
    def quantized_stats(self):
        return self._quantized_stats

    @quantized_stats.setter
    def quantized_stats(self, quantized_stats):
        if self._open_files:
            for f in self._open_files.values():
                if not f.closed:
                    f.close()
        self._open_files = dict()

        identity_key = PREFIX + "identity_p"
        for stats in quantized_stats.values():
            if identity_key in stats:
                stats[identity_key] = [
                    stat
                    for stat in stats[identity_key]
                    if stat is not None and stat.size != 0
                ]
        self._quantized_stats = quantized_stats

    def dump(self, dump_path=None):
        """Dump collect statistics
        :param dump_path: a path whre node activations would be dumpped
        """
        if dump_path is None and self._dump_path is not None:
            dump_path = self._dump_path
        assert dump_path is not None

        assert (
            self._dumpable
        ), "Debugger is not able to dump itself. It did not collect statistics yet."

        os.makedirs(dump_path, exist_ok=True)

        before_path = os.path.join(dump_path, "before")
        after_path = os.path.join(dump_path, "after")
        os.makedirs(before_path, exist_ok=True)
        os.makedirs(after_path, exist_ok=True)

        for node_name, stats in self.original_stats.items():
            for key, stat in stats.items():
                if len(stat):
                    dump_key = self._get_dump_key(node_name, key)
                    path = os.path.join(before_path, f"{dump_key}.pkl")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        for stat_ in stat:
                            pickle.dump(stat_, f)

        for node_name, stats in self.quantized_stats.items():
            for key, stat in stats.items():
                if len(stat):
                    dump_key = self._get_dump_key(node_name, key)
                    path = os.path.join(after_path, f"{dump_key}.pkl")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        for stat_ in stat:
                            pickle.dump(stat_, f)

        if self._fake_quantize_params is not None:
            fake_quantize_parameters_path = os.path.join(
                dump_path, FAKE_QUANTIZE_PARAMS_KEY
            )
            with open(fake_quantize_parameters_path, "wb") as f:
                pickle.dump(self.fake_quantize_params, f)

        if self._fake_quantize_mapping is not None:
            mapping_path = os.path.join(dump_path, FAKE_QUANTIZE_MAPPING_KEY)
            with open(mapping_path, "wb") as f:
                pickle.dump(self.fake_quantize_mapping, f)

        if self._fake_quantize_config is not None:
            config_path = os.path.join(dump_path, FAKE_QUANTIZE_CONFIG_KEY)
            with open(config_path, "w") as f:
                json.dump(self.fake_quantize_config, f)

        if self._fake_quantize_attrs is not None:
            attrs_path = os.path.join(dump_path, FAKE_QUANTIZE_ATTRS_KEY)
            with open(attrs_path, "wb") as f:
                pickle.dump(self.fake_quantize_attrs, f)

    def run(self, quantized_ir_model, sampler):
        assert (
            self._engine is not None
        ), "engine is not given when initializing debugger"

        self.fake_quantize_mapping = self._get_fake_quantize_mapping(quantized_ir_model)
        self.fake_quantize_params = self._get_fake_quantize_params(quantized_ir_model)
        self.fake_quantize_attrs = self._get_fake_quantize_attrs(quantized_ir_model)

        ignored = self.config.get("ignored", dict())
        ignored_scope = ignored.get("scope", [])
        ignored_ops = ignored.get("operations", [])
        stats_layout = self.create_stats_layout(
            quantized_ir_model, ignored_scope, ignored_ops, "after"
        )

        self._engine.calculate_metrics = True
        self._engine.set_model(quantized_ir_model)
        _, quantized_stats = self._engine.predict(stats_layout, sampler)

        self.quantized_stats = quantized_stats

        self._collect_weights(quantized_ir_model)
        self._dumpable = True

    def get_dumpped_node_names(self, suffix, stat_key="identity_p", dump_path=None):
        """List dumpped node names
        :param suffix: 'before' or 'after'
        :param stat_key: a name of statistic to load
        :param dump_path: a directory where node activations were dumpped
        :return a list of node names
        """
        assert suffix in ["before", "after"]

        if dump_path is None:
            dump_path = self._dump_path
        assert dump_path is not None and os.path.exists(dump_path)

        nodes = []
        dump_path = os.path.join(dump_path, suffix)
        for root, dirs, files in os.walk(dump_path):
            for file in files:
                node_name, stat_key_ = file.replace(".pkl", "").split(
                    STAT_KEY_SEPARATOR
                )
                stat_key_ = stat_key_.replace(PREFIX, "")
                if stat_key_ == stat_key:
                    node_name = denormalize_name(node_name)
                    nodes.append(node_name)
        return nodes

    def load_statistic(self, node_name, suffix, stat_key="identity_p", dump_path=None):
        """Load statistics
        :param node_name: node to load statistic
        :param suffix: 'before' or 'after'
        :param stat_key: a name of statistic to load
        :param dump_path: a directory where node activations were dumpped
        :return a list of statistics
        """
        if dump_path is None:
            dump_path = self._dump_path
        assert dump_path is not None and os.path.exists(dump_path)

        stat_key = PREFIX + stat_key

        node_name = normalize_name(node_name)
        dump_path = os.path.join(dump_path, suffix)
        file_name = os.path.join(
            dump_path, f"{node_name}{STAT_KEY_SEPARATOR}{stat_key}.pkl"
        )
        stats = []
        for data in _load_all(file_name):
            stats.append(data)
        return stats

    def visualize(self, path_to_save=None, dump_path=None):
        """Visualize all node in dumpped path
        :param path_to_save: a path to save.
            If it is not given, a list of figures is returned
        :param dump_path: a directory where node activations were dumpped
        :return a list of figures if `path_to_save` is not given, else `None`
        """
        node_names = self.get_dumpped_node_names("after", dump_path=dump_path)
        node_names.extend(
            self.get_dumpped_node_names(
                "after", stat_key="weights", dump_path=dump_path
            )
        )
        out = None
        if path_to_save is None:
            out = dict()
        else:
            os.makedirs(path_to_save, exist_ok=True)
        for node_name in node_names:
            fig = self.visualize_node(node_name, dump_path=dump_path)
            node_name = normalize_name(node_name)
            if path_to_save is None:
                out[node_name] = fig
            else:
                fig.savefig(os.path.join(path_to_save, node_name + ".png"))
        return out

    def visualize_node(self, node_name, *args, **kwargs):
        """Visualize given node
        :param node_name: node to visualize
        :param dump_path: a directory where node activation was dumpped
        :return a figure
        """
        weight_nodes = self.get_dumpped_node_names(
            "after", stat_key="weights", dump_path=kwargs.get("dump_path", None)
        )

        if node_name in self.fake_quantize_mapping:
            return self._visualize_fake_quantize_node(node_name, *args, **kwargs)
        elif node_name in weight_nodes:
            return self._visualize_weight_node(node_name, *args, **kwargs)
        else:
            return self._visualize_node(node_name, *args, **kwargs)

    def _visualize_node(self, node_name, batch_dim=0, channel_axis=-1, dump_path=None):

        if dump_path is None:
            dump_path = self._dump_path
        assert dump_path is not None and os.path.exists(dump_path)

        before_stats = self.load_statistic(node_name, "before", dump_path=dump_path)
        after_stats = self.load_statistic(node_name, "after", dump_path=dump_path)

        before_stats = np.concatenate(before_stats, axis=batch_dim)
        after_stats = np.concatenate(after_stats, axis=batch_dim)

        return _plot_node(
            before_stats=before_stats,
            after_stats=after_stats,
            node_name=node_name,
            channel_axis=channel_axis,
        )

    def _visualize_fake_quantize_node(
        self, fq_name, batch_dim=0, channel_axis=-1, dump_path=None
    ):
        if dump_path is None:
            dump_path = self._dump_path
        assert dump_path is not None and os.path.exists(dump_path)

        assert fq_name in self.fake_quantize_config

        fq_input_name = self.fake_quantize_mapping[fq_name]

        before_stats = self.load_statistic(fq_input_name, "before", dump_path=dump_path)
        after_stats = self.load_statistic(fq_input_name, "after", dump_path=dump_path)
        fq_stats = self.load_statistic(fq_name, "after", dump_path=dump_path)

        before_stats = np.concatenate(before_stats, axis=batch_dim)
        after_stats = np.concatenate(after_stats, axis=batch_dim)
        fq_stats = np.concatenate(fq_stats, axis=batch_dim)

        return _plot_fake_quantize_node(
            before_stats=before_stats,
            after_stats=after_stats,
            fq_stats=fq_stats,
            channel_axis=channel_axis,
            fq_attrs=self.fake_quantize_attrs[fq_name],
            fq_config=self.fake_quantize_config[fq_name],
            fq_params=self.fake_quantize_params[fq_name],
            fq_name=fq_name,
            fq_input_name=fq_input_name,
        )

    def _visualize_weight_node(
        self, node_name, batch_dim=0, channel_axis=0, top_k=5, dump_path=None
    ):
        if dump_path is None:
            dump_path = self._dump_path
        assert dump_path is not None and os.path.exists(dump_path)

        reverse_mapping = {
            v: k
            for k, v in self.fake_quantize_mapping.items()
            if "fq_weight" in k or "fq_recurrent_weight" in k
        }

        fq_name = reverse_mapping[node_name]

        weights = self.load_statistic(
            node_name, "after", stat_key="weights", dump_path=dump_path
        )
        weights = np.concatenate(weights, axis=batch_dim)

        return _plot_weight_node(
            weights=weights,
            channel_axis=channel_axis,
            fq_attrs=self.fake_quantize_attrs[fq_name],
            fq_config=self.fake_quantize_config[fq_name],
            fq_params=self.fake_quantize_params[fq_name],
            fq_name=fq_name,
            weight_name=node_name,
            top_k=top_k,
        )

    @staticmethod
    def _get_dump_key(node_name, stat_key):
        node_name = normalize_name(node_name)
        dump_key = f"{node_name}{STAT_KEY_SEPARATOR}{stat_key}"
        return dump_key

    def _collect_weights(self, model):
        for fq in mu.get_nodes_by_type(model, ["FakeQuantize"], recursively=True):
            fq_input = fqut.get_fake_quantize_input(fq)
            fq_input_value = fqut.get_fake_quantize_input_value(fq)
            is_weights = fq_input.type == "Const" or fq_input_value is not None
            if is_weights:
                if self._dump_path:
                    dump_key = self._get_dump_key(fq_input.fullname, PREFIX + "weights")
                    with open(
                        os.path.join(self._dump_path, "after", f"{dump_key}.pkl"), "wb"
                    ) as f:
                        pickle.dump(fq_input_value, f)
                else:
                    self.quantized_stats[fq_input.fullname] = {
                        PREFIX + "weights": [fq_input_value]
                    }

    def _get_fake_quantize_mapping(self, model):
        mapping = dict()
        fq_nodes = mu.get_nodes_by_type(model, ["FakeQuantize"], recursively=True)
        for fq in fq_nodes:
            fq_input_key = nu.get_quantized_input_key(fq)
            if isinstance(fq_input_key, tuple):
                fq_input_key = f"{fq_input_key[0]}.{fq_input_key[1]}"
            mapping[fq.fullname] = fq_input_key
        return mapping

    def _get_fake_quantize_params(self, model):
        fq_nodes = mu.get_nodes_by_type(model, ["FakeQuantize"], recursively=True)
        output = dict()
        for fq in fq_nodes:
            output[fq.fullname] = fqut.get_fake_quantize_parameters(fq)
        return output

    def _get_fake_quantize_attrs(self, model):
        fq_nodes = mu.get_nodes_by_type(model, ["FakeQuantize"], recursively=True)
        attrs = ["levels", "auto_broadcast"]
        output = dict()
        for fq in fq_nodes:
            output[fq.fullname] = dict()
            for attr in attrs:
                output[fq.fullname][attr] = fq.attrs()[attr]
        return output

    def create_stats_layout(self, model, ignore_names=[], ignore_types=[], suffix=""):
        """Create statistic layout for this debugger
        :param model: model
        :param ignore_names: node names to ignore
        :param ignore_types: node types to ignore
        :param suffix: suffix
        :return a dictionary to collect statistic
        """

        def _get_dump_key(node_name, key):
            dump_key = self._get_dump_key(node_name, key)
            if node_name not in self._open_files:
                path = os.path.join(dump_path, suffix, f"{dump_key}.pkl")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                f = open(path, "wb")
                self._open_files[dump_key] = f
            return dump_key

        def _dump(data, fn, key):
            data = fn(data)
            if data.size != 0:
                f = self._open_files[key]
                pickle.dump(data, f)

        dump_path = self._dump_path

        stats_layout = dict()
        for node in mu.get_all_operation_nodes(model, recursively=True):

            if (
                node.fullname in ignore_names
                or node.fullname in stats_layout
                or node.type in ignore_types
                or node.type in ["Const", "result", "Reshape", "Assign"]
            ):
                continue

            # only collect parameters from outisde of graph
            if node.type == "Parameter" and nu.get_node_inputs(node):
                continue

            for port_number in range(len(node.out_ports())):
                if node.type in ["TensorIterator", "Loop"] and port_number > 0:
                    break
                node_name = node.fullname
                if len(node.out_ports()) > 1:
                    node_name = (node.fullname, port_number)

                temp_stats = dict()
                for key, fn in self.stats_collector.items():
                    if dump_path:
                        dump_key = _get_dump_key(node_name, key)
                        temp_stats[key] = partial(_dump, fn=fn, key=dump_key)
                    else:
                        temp_stats[key] = fn
                stats_layout[node_name] = temp_stats

                if node.type == "FakeQuantize":
                    fq_input = fqut.get_fake_quantize_input(node)
                    fq_input_value = fqut.get_fake_quantize_input_value(node)
                    is_weights = fq_input.type == "Const" or fq_input_value is not None
                    if not is_weights:
                        fq_input_key = nu.get_quantized_input_key(node)
                        if fq_input_key not in stats_layout:
                            temp_stats = dict()
                            for key, fn in self.stats_collector.items():
                                if dump_path:
                                    dump_key = _get_dump_key(fq_input_key, key)
                                    temp_stats[key] = partial(
                                        _dump, fn=fn, key=dump_key
                                    )
                                else:
                                    temp_stats[key] = fn
                            stats_layout[fq_input_key] = temp_stats
                    else:
                        layouts = stats_layout.pop(node_name)
                        for layout in layouts.values():
                            if isinstance(layout, partial):
                                key = layout.keywords["key"]
                                f = self._open_files[key]
                                f_name = f.name
                                f.close()
                                os.remove(f_name)

        if dump_path:
            for node_name, layouts in stats_layout.items():
                node_name = normalize_name(node_name)
                for layout in layouts.values():
                    if isinstance(layout, partial):
                        key = layout.keywords["key"]
                        assert key.startswith(node_name)
        return stats_layout

    def _augment_stats_layout(self, stats_layout, model):
        ignored = self.config.get("ignored", dict())
        ignored_scope = ignored.get("scope", [])
        ignored_ops = ignored.get("operations", [])
        if "FakeQuantize" not in ignored_ops:
            ignored_ops.append("FakeQuantize")
        debugger_stats_layout = self.create_stats_layout(
            model, ignored_scope, ignored_ops, "before"
        )

        stats_layout = deepcopy(stats_layout)
        for node_name, layouts in debugger_stats_layout.items():
            if node_name not in stats_layout:
                stats_layout[node_name] = layouts
            else:
                for key, layout in layouts.items():
                    stats_layout[node_name][key] = layout
        return stats_layout

    def __repr__(self):
        names = list(
            map(
                lambda x: x.replace(PREFIX, ""),
                list(self.stats_collector.keys()),
            )
        )
        return f"{self.__class__.__name__}(" f"stats_collector={names}" ")"
