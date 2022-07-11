# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import deque, defaultdict
from copy import deepcopy

from .range_estimator import get_range_estimator_config
from .utils import get_hardware_config_operation_type, load_hardware_config
from ...graph import editor as ge
from ...graph.special_operations import QUANTIZE_AGNOSTIC_OPERATIONS, CONCAT_UNIFY_OUTPUTS, \
    CONCAT_UNIFY_INPUTS, RECURRENT_OPERATIONS
from ...graph.utils import find_operation_matches, get_operation_list, is_data_type_quantizable
from ...graph.model_utils import get_nodes_by_type, get_node_by_name
from ...graph.node_utils import get_input_shape, get_all_node_outputs,\
    get_node_input, get_node_inputs, get_node_data_type, check_const_input, \
    get_mapped_node_in_subgraph

from ...utils.logger import get_logger

logger = get_logger(__name__)


QUANTIZATION_PARAMETERS = [
    'level_high',
    'level_low',
    'mode',
    'granularity',
    'bits'
]


def get_fake_quantize_configuration(config):
    """ Create fake quantization configuration from the tool configuration
    :param config: dictionary with compression section from toolkit config file
    :return dictionary with fake quantization configuration
     """
    q_config = {'weights': {}, 'activations': {}}
    for op_type, q_params in q_config.items():
        op_type_config = config.get(op_type, {})
        for param_name, param_value in op_type_config.items():
            if param_name in QUANTIZATION_PARAMETERS:
                q_params[param_name] = param_value
    return q_config


def intersect_configs(left, right):
    """ intersect two sets of configurations """
    def _get_main_param_for_config(config):
        """ check main parameters intersection """
        main_params = {}
        for field_name in ['mode', 'granularity']:
            main_params[field_name] = config[field_name]
        return main_params

    def _intersect_configs(left_, right_):
        """ intersect two sets of configurations """
        result = []
        offset = 0
        for l_ in left_:
            l_main = _get_main_param_for_config(l_)
            for idx, r_ in enumerate(right_[offset:]):
                r_main = _get_main_param_for_config(r_)
                if l_main == r_main:
                    if l_['bits'] <= r_['bits']:
                        result.append(l_)
                    else:
                        result.append(r_)
                    offset += idx
                    break
        return result

    def _extend_configs(config):
        """ extend the set of configurations by adding configurations expressed
        through configurations from the input configuration set """
        def _insert_to_front(config_, item_):
            if item_ not in config_:
                config_.insert(0, item_)

        config_ext = []
        for item in reversed(config):
            _insert_to_front(config_ext, item)
            mode_asymmetric = 'mode' in item and item['mode'] == 'asymmetric'
            granularity_perchannel = 'granularity' in item and item['granularity'] == 'perchannel'
            if granularity_perchannel:
                item_ext = deepcopy(item)
                item_ext['granularity'] = 'pertensor'
                _insert_to_front(config_ext, item_ext)
            if mode_asymmetric:
                item_ext = deepcopy(item)
                item_ext['mode'] = 'symmetric'
                _insert_to_front(config_ext, item_ext)
            if mode_asymmetric and granularity_perchannel:
                item_ext = deepcopy(item)
                item_ext['mode'] = 'symmetric'
                item_ext['granularity'] = 'pertensor'
                _insert_to_front(config_ext, item_ext)
        return config_ext

    res = _intersect_configs(left, right)
    if not res:
        left_ext = _extend_configs(left)
        right_ext = _extend_configs(right)
        res = _intersect_configs(left_ext, right_ext)
    return res


def read_all_fake_quantize_configurations(config, hardware_config, model):
    """ Read all fake quantize configurations from hardware config which are suitable to
    every fake quantize node based on toolkit config file and sub graph of every fake quantize node
    :param config: dictionary with compression section from toolkit config file
    :param hardware_config: dictionary with hardware config
    :param model: CompressedModel instance to quantize
    :return dictionary with fake quantize names as keys and
     list of corresponding configurations as values
     """
    def _is_subset(left: dict, right: dict):
        """ Checks that x is a subset of y
        :param left: supposed to be subset of set 'right'
        :param right: set to check that left belongs to"""
        for key in left.keys():
            if key not in right.keys() or\
                    left[key] != right[key]:
                return False
        return True

    def _find_configurations(fq_name_, fq_type_):
        res_conf = []
        for op in ops:
            if fq_type_ in op['quantization']:
                confs = [conf for conf in op['quantization'][fq_type_]
                         if _is_subset(q_config[fq_type_], conf)]
                if confs:
                    res_conf = intersect_configs(res_conf, confs) if res_conf else confs
                else:
                    logger.warning('Fake quantize node %s does not support configuration '
                                   'from tool config file (mismatch with hardware config)',
                                   fq_name_)
                    res_conf = intersect_configs(res_conf, q_config[fq_type_]) \
                        if res_conf else [q_config[fq_type_]]
                if not res_conf:
                    raise Exception('Fake quantize configuration cannot be empty')
        return res_conf

    q_config = get_fake_quantize_configuration(config)

    res_fq_to_hw_conf = {}
    for fq_name, (types, is_weights) in _fake_quantize_to_types(model, hardware_config).items():
        fq_type = 'weights' if is_weights else 'activations'
        res_fq_to_hw_conf[fq_name] = {fq_type: []}
        for type_ in types:
            child_name, op_type = type_
            ops = [op for op in hardware_config if op_type == op['type']]
            conf = _find_configurations(fq_name, fq_type)
            if conf:
                res_fq_to_hw_conf[fq_name][fq_type].append((child_name, conf))
    return res_fq_to_hw_conf


def add_range_estimator_configs(fq_to_hw_confs, config):
    """ Expand fake quantize configuration with range_estimator config
    :param fq_to_hw_confs: dictionary with fake quantize names as keys and its configurations as values
    :param config: tool config used to create range_estimator config
    :return dictionary with fake quantize nodes names as keys and its configurations as values
     extended with range_estimator config"""
    for confs in fq_to_hw_confs.values():
        for i_type, conf in confs.items():
            conf['range_estimator'] = get_range_estimator_config(config, i_type, conf['granularity'], conf['mode'])
    return fq_to_hw_confs


def get_configurations_by_preset(config, model, fq_to_hw_confs):
    """ Choose fake quantize configuration by preset
    :param config: dictionary with params algo section from toolkit config
    :param model: CompressedModel instance
    :param fq_to_hw_confs: dictionary with fake quantize names as keys and
     list of its configurations as values (read_all_fake_quantize_configurations(..) return value)
    :return dictionary with fake quantize nodes names as keys and
     suitable configuration chose by preset as values"""

    def _apply_preset_rule(preset_, fq_name, param_type, confs, to_skip=None):
        if param_type == 'weights':
            if preset_ == 'accuracy':
                return confs[-1]
            return confs[0]
        if not to_skip or fq_name not in [fq for _, fqs in to_skip for fq in fqs]:
            if preset_ == 'performance':
                return confs[0]
            return confs[-1]
        return confs

    def _intersect_and_apply_preset(preset_, fq_to_hw_confs_, fqs_to_unify_):

        def _unify_and_apply_preset(preset_, cur_conf, fqs_to_unify_):
            def _test_shapes(shapes):
                return any([s[0] != shapes[0][0] or len(s) == 1 or s[1] != shapes[0][1] for s in shapes])

            for bridges, fqs in fqs_to_unify_:
                i_type = set()
                for fq in fqs:
                    i_type.update([k for k in cur_conf[fq].keys()])
                assert len(i_type) == 1, (
                    f"Unifying FakeQuantize must be the same type, but {i_type} given. {fqs}."
                )
                i_type = next(iter(i_type))
                assert i_type in ["activations", "weights"]

                res_conf = []
                with_concat = 'Concat' in [get_node_by_name(model, bridge, recursively=True).type for bridge in bridges]
                fq_input_shapes = [get_input_shape(get_node_by_name(model, fq, recursively=True), 0) for fq in fqs]
                unclear_layout = _test_shapes(fq_input_shapes)
                bridge_layers = [get_node_by_name(model, bridge, recursively=True) for bridge in bridges]
                bridge_input_shapes = [get_input_shape(layer, i) for layer in bridge_layers for i in layer.in_ports()]
                broadcasting = _test_shapes(bridge_input_shapes)
                for fq in fqs:
                    if with_concat or unclear_layout or broadcasting:
                        if 'activations' in cur_conf[fq]:
                            configuration = [c for c in cur_conf[fq]['activations'] if c['granularity'] == 'pertensor']
                        elif "weights" in cur_conf[fq]:
                            if isinstance(cur_conf[fq]["weights"], list):
                                configuration = [c for c in cur_conf[fq]['weights']]
                            else:
                                configuration = [cur_conf[fq][i_type]]
                    else:
                        configuration = cur_conf[fq][i_type]
                    res_conf = intersect_configs(res_conf, configuration) if res_conf else configuration
                if not res_conf:
                    raise Exception('Fake quantize nodes {} cannot be unified'.format(fqs))
                for fq in fqs:
                    cur_conf[fq][i_type] = _apply_preset_rule(preset_, fq, i_type, res_conf)
            return cur_conf

        res = {}
        for key, value in fq_to_hw_confs_.items():
            conf = dict()
            for i_type in ['activations', 'weights']:
                if i_type in value:
                    res_conf = []
                    for _, configuration in value[i_type]:
                        res_conf = intersect_configs(res_conf, configuration) if res_conf else configuration
                    if not res_conf:
                        raise Exception('Fake quantize node {} does not have a suitable configuration'
                                        ' for layers {}'.format(key, [layer for layer, _ in value[i_type]]))
                    conf[i_type] = _apply_preset_rule(preset_, key, i_type, res_conf, fqs_to_unify_)
            res[key] = conf
        res = _unify_and_apply_preset(preset_, res, fqs_to_unify_)
        return res

    available_presets = ['accuracy', 'mixed', 'performance']
    preset = config.preset
    if preset not in available_presets:
        raise Exception('Unsupported preset value: {}.'
                        ' Supported values are {}'.format(preset, available_presets))

    fqs_to_unify = find_fqs_to_unify(model, config)
    result = _intersect_and_apply_preset(preset, fq_to_hw_confs, fqs_to_unify)

    return result


def get_configurations_by_qscheme(fq_to_hw_confs, qscheme):
    """ Choose fake quantize configuration by qscheme
    :param fq_to_hw_confs: dictionary with fake quantize names as keys and
     list of its configurations as values (read_all_fake_quantize_configurations(..) return value)
    :param qscheme: The quantization scheme generated from the space
    :return dictionary with fake quantize nodes names as keys and
     suitable configuration chose by preset as values"""

    def _set_config(conf_by_layer, fq_type_):
        out = {}
        for node_name, _ in conf_by_layer:
            qscheme[node_name]['quantize'] = 1
        (node_name, _) = conf_by_layer[0]
        if qscheme[node_name]:
            out = qscheme[node_name][fq_type_]
        return out

    res = {}
    for key, value in fq_to_hw_confs.items():
        # fake quantize node can only have one type, so value dictionary will always have 1 element
        fq_type, confs = list(value.items())[0]
        res[key] = {fq_type: _set_config(confs, fq_type)}
    return res


def find_fqs_to_unify(model, config):
    def _get_ops_with_attribute(hw_ops_, att_name):
        ops_with_attr_ = dict()
        for hw_op in hw_ops_:
            if 'attributes' in hw_op and att_name in hw_op['attributes']:
                attribute = hw_op["attributes"].pop(att_name)
                if not hw_op["attributes"]:
                    del hw_op["attributes"]

                type = hw_op["type"]
                ops_with_attr_[type] = attribute
        return ops_with_attr_

    def _is_special_unify_conditions(node):
        check_map = {
            'Concat': _is_concat_unify_condition
        }
        if node.type in check_map:
            logger.debug('Checking {} node with {} type'.format(node.fullname, node.type))
            return check_map[node.type](node)
        return True

    def _is_concat_unify_condition(node):
        def _is_followed_by_conv(input_node):
            if _is_quantize_agnostic_op(input_node):
                concat_stack.extend(get_all_node_outputs(input_node))
            elif input_node.type in [n['type'] for n in CONCAT_UNIFY_OUTPUTS]:
                concat_stack.clear()
                logger.debug('Found %s %s as Concat %s output',
                             input_node.type, input_node.fullname, node.fullname)
                return True
            return False

        res = False
        concat_inputs = get_node_inputs(node)
        for concat_input in concat_inputs:
            if concat_input.type not in [n['type'] for n in CONCAT_UNIFY_INPUTS]:
                logger.debug('Concat %s without FQ or Concat as input will not unified',
                             node.fullname)
                return res
        concat_stack = [node]
        while concat_stack:
            node_to_check = concat_stack.pop()
            res = _is_followed_by_conv(node_to_check)
        return res

    def _is_agnostic_branching_op(node_):
        return node_.type == 'Concat'

    def _is_quantize_agnostic_op(node_):
        return bool(find_operation_matches(quantize_agnostic_ops, node_))

    def _is_unified_scales_op(node_):
        if bool(find_operation_matches(unified_scales_ops, node_)):
            return _is_special_unify_conditions(node_)
        return False

    def _has_const_input(layer):
        return 'Const' in [parent.type for parent in get_node_inputs(layer) if parent]

    def _is_recurrent_ops(layer):
        return layer.type in [op['type'] for op in RECURRENT_OPERATIONS]

    def _is_valid_unify(to_unify):
        if (
            to_unify[0]
            and any(
                [
                    _is_unified_scales_op(
                        get_node_by_name(model, bridge, recursively=True)
                    )
                    for bridge in to_unify[0]
                ]
            )
            and len(to_unify[1]) > 1
        ):
            return True

    def _find_target_node(fq):
        if fq.type == "FakeQuantize":
            candidate = get_all_node_outputs(fq)[0]
            for port_idx, port in candidate.in_ports().items():
                if port.get_source().node == fq:
                    break
            while _is_quantize_agnostic_op(candidate):
                new_candidate = get_all_node_outputs(candidate)[0]
                for port_idx, port in new_candidate.in_ports().items():
                    if port.get_source().node == candidate:
                        break
                candidate = new_candidate
            return candidate, port_idx
        else:
            return fq, None

    def _find_consecutive_node(node, target_type):
        candidates = get_all_node_outputs(node)
        for candidate in candidates:
            if _is_quantize_agnostic_op(candidate) or (
                candidate.kind == "op" and candidate.type == "FakeQuantize"
            ):
                candidates.extend(get_all_node_outputs(candidate))
            if candidate.type == target_type:
                return candidate
        return None

    def _find_fq(node, port):
        node = get_node_input(node, port)
        while node.type != "FakeQuantize":
            node = get_node_input(node, 0)
            if not _is_quantize_agnostic_op(node):
                break
        return node if node.type == "FakeQuantize" else None

    def _get_subsequent_node_by_types(query_node, types):
        next_candidates = [(i, query_node) for i in get_all_node_outputs(query_node)]
        for cur_node, prev_node in next_candidates:
            if _is_quantize_agnostic_op(cur_node):
                next_candidates.extend([(i, cur_node) for i in get_all_node_outputs(cur_node)])
                continue
            if cur_node.kind == "op" and cur_node.type in types:
                return cur_node, prev_node
        return None, None

    def _process_node(node_, stack_, visited_, to_unify_, scale_config_):
        visited_[node_.fullname][0] = True
        if _is_unified_scales_op(node_) or _is_agnostic_branching_op(node_):
            if (not _has_const_input(node_) or _is_recurrent_ops(node)) and node_.fullname not in to_unify[0]:
                to_unify_[0].append(node_.fullname)
        elif node_.type == 'FakeQuantize':
            target_node, in_port_idx = _find_target_node(node_)
            if target_node.type in scale_config_:
                to_unify_[1].append((node_.fullname, in_port_idx))
            elif get_node_input(node_, 0).type != 'Const':
                to_unify_[1].append((node_.fullname, in_port_idx))
        # traverse down
        if node_.type == 'FakeQuantize' or _is_quantize_agnostic_op(node_):
            for child in get_all_node_outputs(node_):
                node_data_type = get_node_data_type(child)
                if not all(visited_[child.fullname]) and is_data_type_quantizable(node_data_type) and \
                        (_is_quantize_agnostic_op(child) or _is_unified_scales_op(child)):
                    stack_.append(child)
        # traverse up
        if node_.type != 'FakeQuantize':
            for port_idx, parent in enumerate(get_node_inputs(node_)):
                node_data_type = get_node_data_type(parent)
                if parent and not all(visited_[parent.fullname]) and is_data_type_quantizable(node_data_type) and \
                        (parent.type == 'FakeQuantize' or _is_quantize_agnostic_op(parent)):
                    if node_.type in scale_config_:
                        if len(visited[node_.fullname]) == 1:
                            visited[node_.fullname] = \
                                visited[node_.fullname] + [False] * len(scale_config_[node_.type])
                        unify_port_idx = to_unify_[1][-1][1]
                        for idx, scale in enumerate(scale_config_[node_.type]):
                            if unify_port_idx in scale and port_idx in scale:
                                if len(to_unify_[1]) == len(scale) - 1:
                                    visited[node_.fullname][idx + 1] = True
                                stack_.append(parent)
                    else:
                        stack_.append(parent)

    def _process_consecutive_node(node_, to_unify_, scale_config_):
        if isinstance(node_, tuple):
            target_node, in_port_idx = node_
            is_in_same_graph = False
        elif node_.type == "FakeQuantize":
            target_node, in_port_idx = _find_target_node(node_)
            if target_node.type not in scale_config_:
                return
            is_in_same_graph = True
        else:
            return

        for consecutive_type in scale_config_.keys():
            consecutive_node = _find_consecutive_node(target_node, consecutive_type)
            if consecutive_node is None:
                continue
            target_ports, consecutive_ports = scale_config_[consecutive_type]
            if in_port_idx not in target_ports:
                continue

            if consecutive_node not in to_unify_[0]:
                to_unify_[0].append(consecutive_node.fullname)
            for consecutive_port in consecutive_ports:
                fq = _find_fq(consecutive_node, consecutive_port)
                to_unify_[1].append((fq.fullname, consecutive_port))

            if is_in_same_graph:
                if target_node not in to_unify_[0]:
                    to_unify_[0].append(target_node.fullname)
                for target_port in target_ports:
                    fq = _find_fq(target_node, target_port)
                    to_unify_[1].append((fq.fullname, target_port))

    if model is None:
        return []

    hardware_config = load_hardware_config(config)
    hw_ops = get_operation_list(hardware_config)
    quantize_agnostic_ops = [op[1] for op in find_operation_matches(QUANTIZE_AGNOSTIC_OPERATIONS, hw_ops)]

    unified_scales_ops_dict = _get_ops_with_attribute(hw_ops, "scales_unified")
    unified_scales_ops = [{"type": k} for k in unified_scales_ops_dict.keys()]
    for k in list(unified_scales_ops_dict.keys()):
        if unified_scales_ops_dict[k] == "unified":
            unified_scales_ops_dict.pop(k)

    fqs_to_unify = []
    if unified_scales_ops:
        visited = defaultdict(lambda: [False])
        for fq in get_nodes_by_type(model, ['FakeQuantize'], recursively=True):
            target_node, _ = _find_target_node(fq)
            if not all(visited[fq.fullname]) and \
                    (get_node_input(fq, 0).type != 'Const' or target_node.type in unified_scales_ops_dict):
                stack = [fq]
                to_unify = [[], []]
                while stack:
                    node = stack.pop()
                    _process_node(node, stack, visited, to_unify, unified_scales_ops_dict)

                if _is_valid_unify(to_unify):
                    fqs_to_unify.append([to_unify[0], [name for (name, _) in to_unify[1]]])

    consecutive_unified_scales_ops_dict = _get_ops_with_attribute(
        hw_ops, "consecutive_scales_unified"
    )
    if consecutive_unified_scales_ops_dict:
        types_with_subgraph = ["TensorIterator"]
        fqs_to_consecutive_unify = []
        for sub_model in model.models:
            graph = sub_model["model"]
            for fq in ge.get_nodes_by_type(graph, ["FakeQuantize"], recursively=True):
                target_node, target_in_port_idx = _find_target_node(fq)
                if target_node.type in consecutive_unified_scales_ops_dict:
                    scale_config = consecutive_unified_scales_ops_dict[target_node.type]
                    to_unify = [[], []]
                    _process_consecutive_node(fq, to_unify, scale_config)

                    # not found and fq is in subgraph
                    if not to_unify[0] and not to_unify[1] and graph != target_node.graph:
                        # get node with subgraph where target_node is in
                        node_with_subgraph = ge.get_node_with_subgraph_by_node(
                            graph, target_node
                        )
                        _process_consecutive_node(
                            (node_with_subgraph, target_in_port_idx), to_unify, scale_config
                        )
                        # found in main graph
                        if to_unify[0] and to_unify[1]:
                            to_unify[0].append(target_node.fullname)
                            to_unify[1].append((fq.fullname, target_in_port_idx))
                        else:
                            # get subsequent node
                            next_node_with_subgraph, input_node = _get_subsequent_node_by_types(
                                node_with_subgraph,
                                types_with_subgraph
                            )
                            if input_node is None and next_node_with_subgraph is None:
                                continue

                            if next_node_with_subgraph.type == "TensorIterator":
                                node_in_ti = get_mapped_node_in_subgraph(
                                    next_node_with_subgraph, input_node
                                )
                            else:
                                raise NotImplementedError
                            _process_consecutive_node(
                                (node_in_ti, target_in_port_idx), to_unify, scale_config
                            )
                            # found in the next tensoriterator
                            if to_unify[0] and to_unify[1]:
                                to_unify[0].append(target_node.fullname)
                                to_unify[1].append((fq.fullname, target_in_port_idx))

                    if _is_valid_unify(to_unify):
                        to_unify = [to_unify[0], [name for (name, _) in to_unify[1]]]

                        for item in fqs_to_consecutive_unify:
                            if item[0] == to_unify[0]:
                                item[1] = list(set(item[1]).union(set(to_unify.pop(1))))
                                break
                            elif set(item[1]).intersection(set(to_unify[1])):
                                item[1] = list(set(item[1]).union(set(to_unify.pop(1))))
                                item[0] = list(set(item[0]).union(set(to_unify.pop(0))))
                                break
                        if len(to_unify) > 1:
                            fqs_to_consecutive_unify.append(to_unify)
        # concat fqs to unify
        temp = []
        for consecutive_bridges, consecutive_fqs in fqs_to_consecutive_unify:
            indices = set()
            for consecutive_fq in consecutive_fqs:
                for idx, (_, fqs) in enumerate(fqs_to_unify):
                    if consecutive_fq in fqs:
                        indices.add(idx)
            if indices:
                indices = sorted(list(indices), reverse=True)
                bridges = set()
                fqs = set()
                for idx in indices:
                    bridge, fq = fqs_to_unify.pop(idx)
                    bridges.update(bridge)
                    fqs.update(fq)
                bridges.update(consecutive_bridges)
                fqs.update(consecutive_fqs)
                temp.append([list(bridges), list(fqs)])
            else:
                temp.append([consecutive_bridges, consecutive_fqs])
        fqs_to_unify.extend(temp)

    fqs_to_unify = sorted([[sorted(c[0]), sorted(c[1])] for c in fqs_to_unify])
    logger.debug('Operations and corresponding fake quantize nodes to unify scales:')
    for ops, fqs in fqs_to_unify:
        logger.debug('Operations: {}'.format(ops))
        logger.debug('Fake quantize nodes: {}'.format(fqs))
    logger.debug('')

    return fqs_to_unify


def _fake_quantize_to_types(model, hardware_config):
    """ Helper function to bypass graph and get fake quantize node
    children nodes with predefined types
    :return dictionary with fake quantize node name as a key and tuple with list of
    its quantizable descendant types and boolean specifying if fake quantize node is weights
    """

    def _is_quantizable(node):
        return not find_operation_matches(quantize_agnostic_ops, node)

    def _get_node_valuable_descendant(node):
        descendants = []
        queue = deque([node])
        while queue:
            current = queue.popleft()
            children = get_all_node_outputs(current)
            for child in children:
                if not _is_quantizable(child):
                    queue.append(child)
                elif child.type not in descendants:
                    descendants.append((child.fullname,
                                        get_hardware_config_operation_type(child, available_types)))
                if current.type == 'Split' \
                        and child.type == 'Concat' \
                        and len({child_.fullname for child_ in children}) == 1:
                    break
        return descendants

    hw_ops = get_operation_list(hardware_config)
    quantize_agnostic_ops = [op[1] for op in
                             find_operation_matches(QUANTIZE_AGNOSTIC_OPERATIONS, hw_ops)]

    out = {}
    available_types = [layer['type'] for layer in hardware_config]
    for fq in get_nodes_by_type(model, ['FakeQuantize'], recursively=True):
        node_input = get_node_input(fq, 0)
        out[fq.fullname] = (_get_node_valuable_descendant(fq), node_input.type == 'Const')

    return out


def change_configurations_by_model_type(model, config, fq_configuration, hardware_config):
    if config['model_type'] == 'transformer' and config['target_device'] in ['ANY', 'CPU', 'GPU']:
        change_configurations_by_model_type_transformer(model, fq_configuration, hardware_config)


def change_configurations_by_model_type_transformer(model, fq_configuration, hardware_config):
    fq_types = _fake_quantize_to_types(model, hardware_config)
    for fq in get_nodes_by_type(model, ['FakeQuantize']):
        node_creator_fq, is_weights = fq_types[fq.name]
        node_name = None
        for name, type_node in node_creator_fq:
            if type_node == 'MatMul':
                node_name = name

        if node_name is None or is_weights:
            continue

        node = get_node_by_name(model, node_name)

        if not check_const_input(node):
            fq_configuration[fq.name]['activations'] = deepcopy(fq_configuration[fq.name]['activations'])
            fq_configuration[fq.name]['activations']['mode'] = 'symmetric'
