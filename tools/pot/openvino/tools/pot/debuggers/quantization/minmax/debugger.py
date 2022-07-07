# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from openvino.tools.pot.algorithms.quantization import fake_quantize as fqut
from openvino.tools.pot.algorithms.quantization.utils import get_tensor_statistics
from openvino.tools.pot.graph import model_utils as mu
from openvino.tools.pot.graph import node_utils as nu
from openvino.tools.pot.statistics.statistics import TensorStatistic

from ...debugger import POTDebugger


sns.set()


PREFIX = "debugger_"


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


class MinMaxQuantizationDebugger(POTDebugger):
    name = "MinMaxQuantizationDebugger"
    targets = ["MinMaxQuantization", "DefaultQuantization"]

    def __init__(self, config, engine):
        super().__init__(config, engine)

        self._fake_quantize_config = None
        self._fake_quantize_mapping = None

        self._quantized_stats = None

        self._original_stats = None

    @property
    def stats_collector(self):
        seed = self.config.get("seed", 0)
        identity_p = self.config.get("identity_p", 0.2)
        custom_stats_fn = self.config.get("stats_collector")
        return _get_stats_collector(seed, identity_p, custom_stats_fn)

    @property
    def fake_quantize_config(self):
        return self._fake_quantize_config

    @fake_quantize_config.setter
    def fake_quantize_config(self, fake_quantize_config):
        self._fake_quantize_config = deepcopy(fake_quantize_config)

    @property
    def fake_quantize_mapping(self):
        return self._fake_quantize_mapping

    @fake_quantize_mapping.setter
    def fake_quantize_mapping(self, fake_quantize_mapping):
        self._fake_quantize_mapping = deepcopy(fake_quantize_mapping)

    @property
    def fake_quantize_params(self):
        return self._fake_quantize_params

    @fake_quantize_params.setter
    def fake_quantize_params(self, fake_quantize_params):
        self._fake_quantize_params = deepcopy(fake_quantize_params)

    @property
    def original_stats(self):
        return self._original_stats

    @original_stats.setter
    def original_stats(self, original_stats):
        for stats in original_stats.values():
            stats[PREFIX + "identity_p"] = [
                stat for stat in stats[PREFIX + "identity_p"] if stat.size != 0
            ]
        self._original_stats = original_stats

    @property
    def quantized_stats(self):
        return self._quantized_stats

    @quantized_stats.setter
    def quantized_stats(self, quantized_stats):
        for stats in quantized_stats.values():
            stats[PREFIX + "identity_p"] = [
                stat for stat in stats[PREFIX + "identity_p"] if stat.size != 0
            ]
        self._quantized_stats = quantized_stats

    def run(self, quantized_ir_model, sampler):
        self.fake_quantize_mapping = self._get_fake_quantize_mapping(quantized_ir_model)
        self.fake_quantize_params = self._get_fake_quantize_params(quantized_ir_model)

        stats_layout = self.create_stats_layout(
            self.fake_quantize_config, quantized_ir_model, for_weights=False
        )

        self._engine.calculate_metrics = True
        self._engine.set_model(quantized_ir_model)
        _, quantized_stats = self._engine.predict(stats_layout, sampler)

        self.quantized_stats = quantized_stats

    def visualize(self, path):
        os.makedirs(path, exist_ok=True)

        for fq_name, fq_input_name in self.fake_quantize_mapping.items():
            data_in = np.array(
                self._original_stats[fq_input_name][PREFIX + "identity_p"]
            )

            data_in_quantized = np.array(
                self._quantized_stats[fq_input_name][PREFIX + "identity_p"]
            )
            data_out = np.array(self._quantized_stats[fq_name][PREFIX + "identity_p"])

            input_low, input_high, output_low, output_high = self.fake_quantize_params[
                fq_name
            ]

            fq_config = self.fake_quantize_config[fq_name]

            levels = 2 ** fq_config["bits"]

            min_index = data_in <= min(input_low, input_high)
            max_index = data_in > max(input_low, input_high)
            mid_index = np.logical_and(
                data_in > min(input_low, input_high),
                data_in <= max(input_low, input_high),
            )

            data_q = data_in.copy()
            data_q[min_index] = min(input_low, input_high)
            data_q[max_index] = max(input_low, input_high)
            data_q = np.round(
                (data_q - input_low) / (input_high - input_low) * (levels - 1)
            )

            data_dq = data_q / (levels - 1) * (output_high - output_low) + output_low

            scale = (output_high - output_low) / (levels - 1)
            zero_point = -output_low / (output_high - output_low) * (levels - 1)

            rang = levels * scale
            fq_mse = np.square(data_in - data_dq).mean()
            fq_rmse = np.sqrt(fq_mse)
            fq_err = fq_rmse / rang

            in_mse = np.square(data_in - data_in_quantized).mean()
            in_rmse = np.sqrt(in_mse)
            in_err = in_rmse / rang

            data = {
                "FQ Config": fq_config,
                "Scale": scale.item(),
                "Zero Point": zero_point.item(),
                "input original - output dequantized": {
                    "MSE": fq_mse.item(),
                    "RMSE": fq_rmse.item(),
                    "RMSE/range": fq_err.item(),
                },
                "input original - input quantized": {
                    "MSE": in_mse.item(),
                    "RMSE": in_rmse.item(),
                    "RMSE/range": in_err.item(),
                },
            }

            fig = plt.figure(figsize=(30, 25))
            gs = GridSpec(nrows=5, ncols=2)
            fig.suptitle(fq_name, fontsize=20)

            data_str = json.dumps(data.pop("FQ Config"), indent=8)
            ax = fig.add_subplot(gs[0, 0])
            ax.set_facecolor((1, 1, 1))
            ax.axis("off")
            ax.text(0, 0, data_str, transform=ax.transAxes)
            ax.set_title("FQ info")

            data_str = json.dumps(data, indent=8)
            ax = fig.add_subplot(gs[0, 1])
            ax.set_facecolor((1, 1, 1))
            ax.axis("off")
            ax.text(0, 0, data_str, transform=ax.transAxes)
            ax.set_title("Error")

            ax = fig.add_subplot(gs[1, :])
            sns.histplot(np.reshape(data_in, (-1)), bins="scott", ax=ax)
            ax.axvline(x=input_low, color="r")
            ax.axvline(x=input_high, color="r")
            ax.set_yscale("log")
            ax.set_ylabel("log(Count)")
            ax.set_title("Input data distribution")

            ax = fig.add_subplot(gs[2, :])
            sns.histplot(np.reshape(data_q, (-1)), bins="scott", ax=ax)
            ax.axvline(x=0, color="r")
            ax.axvline(x=levels, color="r")
            ax.set_yscale("log")
            ax.set_ylabel("log(Count)")
            ax.set_title("Quantized data distribution")

            ax = fig.add_subplot(gs[3, :])
            sns.histplot(np.reshape(data_dq, (-1)), bins="scott", ax=ax)
            ax.axvline(x=output_low, color="r")
            ax.axvline(x=output_high, color="r")
            ax.set_yscale("log")
            ax.set_ylabel("log(Count)")
            ax.set_title("Dequantized data distribution")

            ax = fig.add_subplot(gs[4, :])
            sns.histplot(np.reshape(data_out, (-1)), bins="scott", ax=ax)
            ax.axvline(x=output_low, color="r")
            ax.axvline(x=output_high, color="r")
            ax.set_yscale("log")
            ax.set_ylabel("log(Count)")
            ax.set_title("FQ output data distribution")

            plt.tight_layout()
            fig_name = os.path.join(path, fq_name.replace("/", ":"))
            plt.savefig(fig_name)

    def _get_fake_quantize_mapping(self, model):
        mapping = dict()
        fq_nodes = mu.get_nodes_by_type(model, ["FakeQuantize"], recursively=True)
        for fq in fq_nodes:
            fq_input = fqut.get_fake_quantize_input(fq)
            fq_input_value = fqut.get_fake_quantize_input_value(fq)
            is_weights = fq_input.type == "Const" or fq_input_value is not None
            if is_weights is False:
                fq_input_key = nu.get_quantized_input_key(fq)
                mapping[fq.fullname] = fq_input_key
        return mapping

    def _get_fake_quantize_params(self, model):
        fq_nodes = mu.get_nodes_by_type(model, ["FakeQuantize"], recursively=True)
        output = dict()
        for fq in fq_nodes:
            output[fq.fullname] = fqut.get_fake_quantize_parameters(fq)
        return output

    def create_stats_layout(
        self, fake_quantize_config, quantized_ir_model, for_weights
    ):

        stats_layout = dict()
        for fq in mu.get_nodes_by_type(
            quantized_ir_model, ["FakeQuantize"], recursively=True
        ):
            fq_input = fqut.get_fake_quantize_input(fq)
            fq_input_value = fqut.get_fake_quantize_input_value(fq)
            is_weights = fq_input.type == "Const" or fq_input_value is not None
            layer_config = fake_quantize_config[fq.fullname]
            if is_weights and for_weights:
                stats_layout[fq.fullname] = get_tensor_statistics(
                    layer_config["range_estimator"], for_weights=True
                )
            elif not is_weights and not for_weights:
                fq_input_key = nu.get_quantized_input_key(fq)
                stats_layout[fq_input_key] = get_tensor_statistics(
                    layer_config["range_estimator"], for_weights=False
                )
                stats_layout[fq.fullname] = get_tensor_statistics(
                    layer_config["range_estimator"], for_weights=False
                )
                for key, fn in self.stats_collector.items():
                    stats_layout[fq.fullname][key] = fn

                for key, fn in self.stats_collector.items():
                    stats_layout[fq_input_key][key] = fn

        return stats_layout

    def _augment_stats_layout(self, stats_layout):
        stats_layout = deepcopy(stats_layout)
        for _, stats in stats_layout.items():
            for key, fn in self.stats_collector.items():
                assert key not in stats
                stats[key] = fn
        return stats_layout

    def dump(self, path):
        os.makedirs(path, exist_ok=True)

        original_stats_path = os.path.join(path, "original_stats.pkl")
        with open(original_stats_path, "wb") as f:
            pickle.dump(self._original_stats, f)

        quantized_stats_path = os.path.join(path, "quantized_stats.pkl")
        with open(quantized_stats_path, "wb") as f:
            pickle.dump(self._quantized_stats, f)

        fake_quantize_parameters_path = os.path.join(
            path, "fake_quantize_parameters.pkl"
        )
        with open(fake_quantize_parameters_path, "wb") as f:
            pickle.dump(self.fake_quantize_params, f)

        mapping_path = os.path.join(path, "fake_quantize_mapping.pk")
        with open(mapping_path, "wb") as f:
            pickle.dump(self.fake_quantize_mapping, f)

        config_path = os.path.join(path, "fake_quantize_config.json")
        with open(config_path, "w") as f:
            json.dump(self.fake_quantize_config, f)

    def __repr__(self):
        names = list(
            map(
                lambda x: x.replace(PREFIX, ""),
                list(self.stats_collector.keys()),
            )
        )
        return f"{self.__class__.__name__}(" f"stats_collector={names}" ")"
