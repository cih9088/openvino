# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict

import random
from copy import deepcopy
import numpy as np
import torch

from .utils import get_optimization_params
from ..quantization.accuracy_aware_common.utils import evaluate_model, create_metric_config
from ...algorithms.algorithm import Algorithm
from ...engines.simplified_engine import SimplifiedEngine
from ...graph import editor as ge
from ...graph import model_utils as mu, node_utils as nu
from ...graph.special_operations import OPERATIONS_WITH_WEIGHTS
from ...samplers.batch_sampler import BatchSampler
from ...statistics.collector import collect_statistics
from ...statistics.statistics import TensorStatistic
from ...utils.logger import get_logger

logger = get_logger(__name__)


# pylint: disable=E1102,C0415,R0902,R0912
class LayerwiseModelFinetuning(Algorithm):
    name = 'LayerwiseModelFinetuning'

    @property
    def change_original_model(self):
        return True

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._tconf = {
            'optimizer': 'Adam',
            'loss': 'l2',
            'seed': 0,
            'weight_decay': 0,
            'loss_logging_freq': 10,
            'calibration_indices_pool': 300,
            'use_only_fp_inputs': True,
            'calculate_grads_on_loss_increase_only': True,
            'update_every_batch': False,
            'use_ranking_subset': False,
            'tuning_ignored_scope': self._config.ignored.get('scope', []),
            'batch_size': 1
        }
        for key, value in self._tconf.items():
            self._tconf[key] = self._config.get(key, value)

        self._device = 'cpu'
        self._current_best_loss = 0.0
        self._iteration = 0
        self._safety_eps = 1e-8
        self._dataset_size = len(self._engine.data_loader)
        self._samples_indices_pool = range(self._dataset_size)
        self._weighted_operations = [op['type'] for op in OPERATIONS_WITH_WEIGHTS]
        self._is_variable_resolution_model = False
        self._optimization_dataset_size = self._dataset_size
        self._metric_subset_ratio = (10 * self._tconf['calibration_indices_pool'] / self._optimization_dataset_size)
        self._ranking_subset_size = self._tconf['calibration_indices_pool']

        self._original_model = None
        self._initial_losses = {}
        self._nodes_to_tune = {}
        self._nodes_to_tune_input = {}
        self._nodes_to_tune_output = {}
        self._layer_ops_wrapped = {}
        self._is_simplified_evaluation = isinstance(self._engine, SimplifiedEngine)
        self._base_algo_config = deepcopy(self._config)
        self._base_algo = None
        self._base_algo_args = None
        self._metrics_config = None

        self.set_seed(self._tconf['seed'], self._device)
        self.set_default_parameters()

    def set_default_parameters(self):
        if self._tconf['use_ranking_subset']:
            if self._is_simplified_evaluation:
                logger.info('Cannot use ranking subset in simplified mode')
                self._tconf['use_ranking_subset'] = False
            else:
                self._metrics_config = create_metric_config(
                    self._engine,
                    self._config,
                    force_logit_comparison=True,
                    logit_distance_type='mse',
                )

        if (self._tconf['calibration_indices_pool'] is not None
                and self._tconf['calibration_indices_pool'] < self._optimization_dataset_size):
            self._samples_indices_pool = random.sample(
                range(self._optimization_dataset_size), self._tconf['calibration_indices_pool'])

    def run(self, model):
        raise NotImplementedError

    def _collect_nodes_to_tune(self, modified_model):
        raise NotImplementedError

    def _wrap_nodes(self, modified_model, nodes_to_tune):
        raise NotImplementedError

    def _calculate_gradients(self, losses):
        pass

    def _get_optimizer_and_criterion(self, wrapped_ops_parameters):
        criterion, optimizer_algorithm = get_optimization_params(self._tconf['loss'], self._tconf['optimizer'])
        optimizers = {
            name: optimizer_algorithm(params=param, weight_decay=self._tconf['weight_decay'])
            for name, param in wrapped_ops_parameters.items()
        }
        return optimizers, criterion

    def _wrap_node(self, op_node, wrapper, op_info):
        params = []
        wrapped_op = None
        if wrapper.is_able_to_wrap(op_node):
            wrapped_op = wrapper(op_node, device=self._device, **op_info)
            for name, param in wrapped_op.named_parameters():
                lr_name = name + '_lr'
                if lr_name in self._tconf.keys():
                    params.append({'lr': self._tconf[lr_name], 'params': [param]})
                else:
                    logger.warning('Undefined parameter found: {}'.format(name))
                    continue
        else:
            logger.warning('Was not able to wrap layer {} with PyTorch'.format(op_node.fullname))
        return wrapped_op, params

    def _fine_tuning_loop(
            self,
            modified_model,
            optimizers,
            criterion,
            n_batches,
            fp_model_callbacks,
            modified_model_callbacks=None
    ):
        for layer in self._layer_ops_wrapped.values():
            layer.to(self._device)

        for optimizer in optimizers.values():
            optimizer.zero_grad()

        try:
            # Calculate feature maps for the original model beforehand on the used batch
            batch_indices_sample = self._random_samples()
            fp_activations = self._update_batch_from_model(self._original_model,
                                                           batch_indices_sample,
                                                           fp_model_callbacks)

            ops_list = [op for op in modified_model.pseudo_topological_sort() 
                        if op.kind == 'op' and op.name in self._nodes_to_tune]

            print("Total number of layers to tune: ", len(ops_list))
            tuning_groups = []
            tuned_nodes = set()

            for o in ops_list:
                if not o.fullname in tuned_nodes:
                    group = {o.fullname:o}
                    tuned_nodes.add(o.fullname)
                    if o.type == 'FakeQuantize':
                        for n in ops_list:
                            if n.fullname in o.source_names and not n.fullname in tuned_nodes:
                                group[n.fullname] = n
                                tuned_nodes.add(n.fullname)
                    tuning_groups.append(group)
            
            print("Total number of groups to tune: ", len(tuning_groups))
            #print([g.keys() for g in tuning_groups])

            for group in tuning_groups:
                new_modified_model_callbacks = {}
                new_fp_model_callbacks = {}
                model_copy = deepcopy(modified_model)

                for _, modified_node in group.items():
                    node_copy = mu.get_node_by_name(model_copy, modified_node.name)
                    model_copy, params = self._prepare_model_and_params(model_copy, node_copy)

                    input_node = self._get_input_node(modified_node)
                    output_node = input_node
                    output_node = input_node
                    if modified_node.type in self._weighted_operations:
                        bias_node = nu.get_bias_for_node(modified_node)
                        output_node = modified_node
                        if bias_node is not None:
                            output_node = nu.get_node_output(bias_node, 0)[0]
                    
                    input_node_name = self._get_input_node_name(modified_node)

                    new_fp_model_callbacks[output_node.fullname] = fp_model_callbacks[output_node.fullname]

                    new_modified_model_callbacks[input_node_name] = modified_model_callbacks[input_node_name]
                
                    print("Started to tune operation: ", modified_node.fullname)
                    print("Input node name: ", input_node_name)
                    print("Output node name: ", output_node.fullname)
                
                if self._tconf['update_every_batch']:
                    logger.debug('Batch update')
                    batch_indices_sample = self._random_samples()
                    fp_activations = self._update_batch_from_model(self._original_model,
                                                                   batch_indices_sample,
                                                                   new_fp_model_callbacks)

                modified_activations = fp_activations
                if modified_model_callbacks:
                    modified_activations = self._update_batch_from_model(modified_model,
                                                                         batch_indices_sample,
                                                                         new_modified_model_callbacks)

                self._fine_tuning_step(
                    optimizers,
                    criterion,
                    group,
                    fp_activations,
                    modified_activations,
                    n_batches,
                    new_fp_model_callbacks,
                    new_modified_model_callbacks,
                    modified_model
                )
            return 0

        except MemoryError:
            return -1

    def _random_samples(self):
        batch_indices_sample = random.sample(self._samples_indices_pool, self._tconf['batch_size'])
        if self._is_simplified_evaluation:
            batch_indices_sample = BatchSampler(batch_indices_sample)
        return batch_indices_sample

    def _update_batch_from_model(self, model, batch_indices_sample, model_callbacks):
        self._engine.set_model(model)

        _, output_activations = self._engine.predict(model_callbacks, batch_indices_sample)
        return self._activation_maps_to_torch(output_activations)

    def _fine_tuning_step(
            self,
            optimizers,
            criterion,
            group,
            fp_activations,
            modified_activations,
            n_batches,
            fp_model_callbacks,
            modified_model_callbacks,
            modified_model
    ):
        accumulated_losses = {o: 0.0 for o in group.keys()}
        losses = {}
        for batch_idx in range(n_batches):
            for op_name in group.keys():
                if batch_idx != 0 and self._tconf['update_every_batch']:
                    logger.debug('Batch update')
                    batch_indices_sample = self._random_samples()
                    fp_activations = self._update_batch_from_model(self._original_model,
                                                                    batch_indices_sample,
                                                                    fp_model_callbacks)
                    modified_activations = fp_activations
                    if modified_model_callbacks:
                        modified_activations = self._update_batch_from_model(modified_model,
                                                                            batch_indices_sample,
                                                                            modified_model_callbacks)
                
                torch_wrapped_op = self._layer_ops_wrapped[op_name]
                input_name = self._nodes_to_tune_input[op_name]
                output_name = self._nodes_to_tune_output[op_name]

                in_blobs = modified_activations[input_name]['output']
                if self._tconf['use_only_fp_inputs']:
                    in_blobs = fp_activations[input_name]['output']
                fp_out_blobs = fp_activations[output_name]['output']

                if not self._is_variable_resolution_model:
                    modified_out_blobs = torch_wrapped_op(in_blobs)
                    losses[op_name] = criterion(modified_out_blobs, fp_out_blobs)
                else:
                    for blob_idx, modified_in_blob in enumerate(in_blobs):
                        modified_out_blob = torch_wrapped_op(torch.unsqueeze(modified_in_blob, 0))
                        losses[op_name] += criterion(
                            modified_out_blob, torch.unsqueeze(fp_out_blobs[blob_idx], 0)
                        )

            for name, loss in losses.items():
                accumulated_losses[name] = loss.data

            if batch_idx == 0 and self._iteration == 0:
                self._initial_losses = deepcopy(accumulated_losses)
                self._initial_losses = {
                    name: val + self._safety_eps
                    for name, val in self._initial_losses.items()
                }

            weighted_loss = 0
            init_loss = self._initial_losses[op_name]
            accumulated_loss = accumulated_losses[op_name]
            weighted_loss += accumulated_loss / init_loss / len(self._initial_losses)

            if batch_idx % self._tconf['loss_logging_freq'] == 0:
                printable_loss = weighted_loss.to('cpu').numpy()
                logger.info(
                    'Batch #%s/%s, weighted_loss: %s',
                    batch_idx + 1,
                    n_batches,
                    printable_loss,
                )

            if self._tconf['calculate_grads_on_loss_increase_only']:
                if weighted_loss >= self._current_best_loss:
                    self._current_best_loss = weighted_loss
                    self._calculate_gradients(losses)
                for _, optimizer in optimizers.items():
                    optimizer.step()
                    if self._current_best_loss == weighted_loss:
                        optimizer.zero_grad()
                self._current_best_loss = weighted_loss
            else:
                self._calculate_gradients(losses)
                for _, optimizer in optimizers.items():
                    optimizer.step()
                    optimizer.zero_grad()
            if self._tconf['update_every_batch']:
                for layer in self._layer_ops_wrapped.values():
                    layer.update_node_params()

    def _activation_maps_to_torch(self, activations):
        for layer_name in activations:
            activations[layer_name]['output'] = [
                torch.tensor(activations[layer_name]['output'][index][0]).to(self._device) for index in
                range(len(activations[layer_name]['output']))]
            if len({feature_map.shape for feature_map in activations[layer_name]['output']}) > 1:
                self._is_variable_resolution_model = True
        if not self._is_variable_resolution_model:
            for layer_name in activations:
                activations[layer_name]['output'] = torch.stack(activations[layer_name]['output'])
        return activations

    def _get_ranking_subset(self):
        """
        Find a subset of samples with the highest distance between
        outputs of original and compressed model (a ranking subset)
        :return: ranking data subset indices
        """
        base_algo = self._base_algo(**self._base_algo_args)
        base_algo.register_statistics(self._original_model, self.algo_collector)
        collect_statistics(self._engine, self._original_model, [base_algo])
        base_model = base_algo.run(deepcopy(self._original_model))
        output_node_name = nu.get_node_input(self._original_model.get_final_output_nodes()[0], 0).fullname

        stats_layout = {output_node_name: {'output_logits': TensorStatistic(lambda logits: logits)}}
        metric_subset_size = int(self._dataset_size * self._metric_subset_ratio)
        diff_subset_indices = (
            sorted(random.sample(range(self._dataset_size), metric_subset_size))
            if metric_subset_size < self._dataset_size
            else list(range(self._dataset_size))
        )

        _, original_per_sample_metrics = evaluate_model(
            self._original_model,
            self._engine,
            self._dataset_size,
            subset_indices=diff_subset_indices,
            metrics_config=self._metrics_config,
            output_node_name=output_node_name,
            stats_layout=stats_layout,
        )
        _, base_model_per_sample_metrics = evaluate_model(
            base_model,
            self._engine,
            self._dataset_size,
            subset_indices=diff_subset_indices,
            metrics_config=self._metrics_config,
            output_node_name=output_node_name,
            stats_layout=stats_layout,
        )

        persample_metric = list(self._metrics_config.values())[0].persample
        sorted_sample_importance = persample_metric.sort_fn(
            original_per_sample_metrics[persample_metric.name],
            base_model_per_sample_metrics[persample_metric.name],
            reverse=True,
        )
        ranking_indices = sorted_sample_importance[: self._ranking_subset_size]
        ranking_subset = list(np.array(diff_subset_indices)[ranking_indices])
        return ranking_subset

    def _create_layer_callbacks(self, modified_model):
        fp_model_callbacks = {}
        modified_model_callbacks = {}

        for op_name in self._nodes_to_tune:
            modified_node = mu.get_node_by_name(modified_model, op_name)

            input_node = self._get_input_node(modified_node)
            output_node = input_node
            if modified_node.type in self._weighted_operations:
                bias_node = nu.get_bias_for_node(modified_node)
                output_node = modified_node
                if bias_node is not None:
                    output_node = nu.get_node_output(bias_node, 0)[0]
            input_node_name = self._get_input_node_name(modified_node)

            if self._tconf['use_only_fp_inputs']:
                fp_model_callbacks[input_node_name] = {'output': lambda tensor: tensor}
            else:
                modified_model_callbacks[input_node_name] = {'output': lambda tensor: tensor}
            fp_model_callbacks[output_node.fullname] = {'output': lambda tensor: tensor}
            self._nodes_to_tune_input[op_name] = input_node_name
            self._nodes_to_tune_output[op_name] = output_node.fullname

        return fp_model_callbacks, modified_model_callbacks

    def register_statistics(self, model, stats_collector):
        self.algo_collector = stats_collector

    def _check_batch_size(self):
        if self._tconf['batch_size'] > self._dataset_size:
            logger.debug('Batch size changed from - {} to dataset size - {}.'.format(
                self._tconf['batch_size'], self._dataset_size))
            self._tconf['batch_size'] = self._dataset_size

    @staticmethod
    def set_seed(seed, device):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if device != 'cpu':
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False

    @staticmethod
    def _get_input_node(node):
        return nu.get_node_input(node, 0)

    @staticmethod
    def _get_input_node_name(node):
        return nu.get_quantized_input_key(node)
    
    def _prepare_model_and_params(self, model, node):
        params = {}

        if model.is_cascade:
            model.clean_up()
        else:
            input_nodes = [mu.get_node_by_name(model, name) for name in
                           self._subgraphs_data[node.fullname]['input_nodes']]
            stats_nodes = [mu.get_node_by_name(model, name) for name in
                           self._subgraphs_data[node.fullname]['stats_nodes']]
            output_nodes = [mu.get_node_by_name(model, name) for name in
                            self._subgraphs_data[node.fullname]['output_nodes']]

            self._remove_default_results(node.graph)

            params['parameters_data_dict'] = self._create_parameters_for_input_nodes(input_nodes)
            params['results_data_dict'] = self._create_results_after_nodes(stats_nodes)
            remaining_outputs = list(set(output_nodes) - set(stats_nodes))
            self._create_results_after_nodes(remaining_outputs)

            model.clean_up()

            self._update_split_subgraphs(model)
            params['feed_dicts'] = self._create_feed_dicts(params['parameters_data_dict'])
        return model, params

    def _fill_subgraphs_data(self, model):
        def skip_node(node):
            if not nu.node_with_quantized_weights(node) and not self._apply_for_all_nodes:
                logger.debug('%s skipped because it does not have FQ weights.', node.fullname)
                return True

            if not nu.check_const_input(node):
                logger.debug('%s skipped because channel axis is not defined', node.fullname)
                return True

            bias_node = nu.get_bias_for_node(node)
            if bias_node is None:
                logger.debug('%s skipped because its bias is empty.', node.fullname)
                return True

            return False

        model_copy = deepcopy(model)
        subgraphs_data = OrderedDict()
        self._remove_fq_from_inputs(model_copy)
        for node_name in self._nodes_with_bias_names:
            node = mu.get_node_by_name(model_copy, node_name)
            if skip_node(node):
                continue
            input_nodes, stats_nodes, output_nodes = self._get_subgraph_data_for_node(node)
            subgraphs_data[node_name] = {
                'input_nodes': [n.fullname for n in input_nodes],
                'stats_nodes': [n.fullname for n in stats_nodes],
                'output_nodes': [n.fullname for n in output_nodes]
            }
        del model_copy
        return subgraphs_data

    def _get_subgraph_data_for_node(self, main_node):
        stats_nodes = []
        checked_stat_names = []
        input_nodes = []
        checked_input_names = []
        output_nodes = []

        def fill_stats_nodes():
            main_node_children = self.get_node_children(main_node)
            for main_node_child in main_node_children:
                walk_to_children(main_node_child)

        def fill_input_nodes():
            stat_nodes_list = stats_nodes if stats_nodes else [main_node]
            for stat_node in stat_nodes_list:
                stat_node_parents = self.get_node_parents(stat_node)
                for stat_node_parent in stat_node_parents:
                    walk_to_parents(stat_node_parent)

        def fill_output_nodes():
            assigns = ge.get_nodes_by_type(main_node.graph, ['Assign'], recursively=False)
            for node_name in checked_input_names:
                node = ge.get_node_by_name(main_node.graph, node_name, recursively=False)

                if node.type == 'ReadValue':
                    output_nodes.extend(nu.get_lstm_ends(node, assigns, checked_input_names))

        def walk_to_children(node, is_this_branch_node=False):
            node_parents = self.get_node_parents(node)
            node_input_0 = nu.get_node_input(node, 0)
            if is_this_branch_node:
                # Jump over Split nodes
                if node_input_0.type in self._split_types:
                    node_input_0 = nu.get_node_input(node_input_0, 0)

            node_input_0_name = nu.create_node_name(node_input_0)
            if node.type in self._types_with_bias \
                    and (nu.node_with_quantized_weights(node) and not self._apply_for_all_nodes):
                if node_input_0.fullname not in checked_stat_names:
                    checked_stat_names.append(node_input_0.fullname)
                    checked_input_names.append(node_input_0.fullname)
                    stats_nodes.append(node_input_0)
                    self._collected_stat_inputs.append(node_input_0_name)
            elif is_this_branch_node and len(node_parents) > 1:
                return
            else:
                node_children = self.get_node_children(node)
                is_branching = len(node_children) > 1
                for node_child in node_children:
                    walk_to_children(node_child, is_branching)

        def walk_to_parents(node):
            node_parents = self.get_node_parents(node)
            if node.fullname in checked_input_names:
                return
            checked_input_names.append(node.fullname)
            if node.fullname in self._collected_stat_inputs:
                if node not in input_nodes:
                    input_nodes.append(node)
            else:
                for node_parent in node_parents:
                    walk_to_parents(node_parent)

        fill_stats_nodes()
        fill_input_nodes()
        fill_output_nodes()

        output_nodes = output_nodes if output_nodes else stats_nodes

        return input_nodes, stats_nodes, output_nodes
