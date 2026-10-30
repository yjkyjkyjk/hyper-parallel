# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Scheduler nodes for local shared expert work."""

from dataclasses import dataclass

from hyper_parallel.core.multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN, EVENT_INVALID_ID, RuntimeConfigC, TaskAiCoreType,
    TaskDescC, TaskSplitValue, TaskType,
)
from hyper_parallel.core.multicore.scheduler.graph import (
    ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec,
)
from hyper_parallel.core.multicore.tasks.utils import make_tensor_desc


FORWARD_POINTER_SLOT = 31
FORWARD_RUNTIME_SLOT = 32
BACKWARD_POINTER_SLOT = 36
BACKWARD_RUNTIME_SLOT = 37


def fuse_shared_output(graph: ComputeGraph, tsv: TaskSplitValue, *, backward: bool) -> None:
    """Add a BF16-rounded shared residual to the existing route reduction."""
    name = "unpermute_grad" if backward else "weighted_unpermute"
    endpoint = graph.get_op(name)
    slot = 11 if backward else 5
    pointer_slot = BACKWARD_POINTER_SLOT if backward else FORWARD_POINTER_SLOT
    endpoint.outputs[0].tensor_type = 2
    endpoint.outputs[0].param_position = pointer_slot
    endpoint.param_positions[-1] = pointer_slot
    endpoint.fill_config.shared_output_slot = slot
    endpoint.fill_config.add_shared_residual = not backward
    tsv.shared_output_fused = True


@dataclass(frozen=True)
class SharedFillConfig:
    """Fill a shared task using addresses from the pointer table."""

    pointer_slot: int
    runtime_slot: int
    tiling_offset: int
    dependency: int
    trigger: int

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        cube = op.op_type == OpType.GMM
        queue = cfg.cube_task_indices if cube else cfg.vector_task_indices
        queue_start = tsv.pre_cube_task_num if cube else tsv.pre_vector_task_num
        task_type = {
            OpType.GMM: TaskType.TASK_GROUPED_MATMUL,
            OpType.SWIGLU: TaskType.TASK_SWI_GLU,
            OpType.SWIGLU_GRAD: TaskType.TASK_SWI_GLU_GRAD,
            OpType.ADD_RESIDUAL: TaskType.TASK_ADD_CUSTOM,
        }[op.op_type]
        for index in range(op.task_num):
            task = TaskDescC()
            task.task_type = task_type
            task.task_aicore_type = (
                TaskAiCoreType.TASK_AICORE_CUBE if cube else TaskAiCoreType.TASK_AICORE_VECTOR
            )
            task.num_inputs = len(op.inputs)
            task.num_outputs = len(op.outputs)
            for target, specs in ((task.inputs, op.inputs), (task.outputs, op.outputs)):
                for position, spec in enumerate(specs):
                    is_group = spec.name == "shared_group_list"
                    target[position] = make_tensor_desc(
                        tensor_type=0 if is_group else 2,
                        input_position=self.runtime_slot if is_group else self.pointer_slot,
                        dynamic_dim=0 if is_group else spec.param_position,
                        base_ptr_offset=(0 if cube else index * op.split_value * spec.shape[1]),
                        data_type=spec.dtype_size,
                        transpose_flag=int(spec.transpose),
                        shape=spec.shape,
                    )
            task.dependent_event = self.dependency
            task.trigger_event = self.trigger
            task.task_index = index
            task.task_split_num = op.task_num
            task.task_split_value = op.split_value
            task.tiling_data_position = self.runtime_slot
            task.tiling_data_offset = self.tiling_offset
            task_id = tsv.pre_task_num + index
            cfg.all_tasks[task_id] = task
            queue[queue_start + index] = task_id
        cfg.all_event_num_triggers[self.trigger] += op.task_num
        cfg.task_index_num[0 if cube else 1] += op.task_num
        tsv.pre_task_num += op.task_num
        if cube:
            tsv.pre_cube_task_num += op.task_num
        else:
            tsv.pre_vector_task_num += op.task_num


def add_shared_graph(
    graph: ComputeGraph, tsv: TaskSplitValue, hidden: int, intermediate: int,
    cores: int, *, backward: bool,
) -> None:
    """Append a static local shared SwiGLU branch to a routed graph."""
    if intermediate <= 0 or tsv.per_expert_seq_to_other % 128:
        raise ValueError("shared expert requires a positive width and 128-aligned tokens")
    tsv.shared_expert = True
    tokens = tsv.per_expert_seq_to_other
    pointer_slot, runtime_slot = (
        (BACKWARD_POINTER_SLOT, BACKWARD_RUNTIME_SLOT)
        if backward else (FORWARD_POINTER_SLOT, FORWARD_RUNTIME_SLOT)
    )
    event_base = tsv.all_event_num + 3 + 2 * ATOMIC_ADD_VALUE_LEN
    endpoint_name = "unpermute_grad" if backward else "weighted_unpermute"
    endpoint = graph.get_op(endpoint_name) if graph.has_op(endpoint_name) else None
    inline_residual = endpoint is not None and not backward
    if inline_residual:
        for predecessor in endpoint.predecessors:
            predecessor.successors.remove(endpoint)
        endpoint.predecessors.clear()
    if endpoint is not None:
        fuse_shared_output(graph, tsv, backward=backward)
    route_event = tsv.all_event_num - int(inline_residual)
    route_terminals = [op for op in graph.topological_sort() if not op.successors and op is not endpoint]
    stride = 2016 * cores
    group = TensorSpec("shared_group_list", [1], runtime_slot, dtype_size=8)

    def tensor(name: str, shape: list[int], index: int, transpose: bool = False) -> TensorSpec:
        return TensorSpec(name, shape, index, transpose=transpose)

    def add(name: str, kind: OpType, inputs: list[TensorSpec], output: TensorSpec,
            offset: int, dependency: int, trigger: int) -> OperatorNode:
        cube = kind == OpType.GMM
        op = OperatorNode(
            name=name, diagnostic_name=name, op_type=kind,
            inputs=inputs + [group] if cube else inputs, outputs=[output],
            param_positions=[], split_value=tokens if cube else 128,
            split_spec=SplitSpec(None, lambda _tsv, count=(cores if cube else tokens // 128): count),
            tiling_position=runtime_slot,
            fill_config=SharedFillConfig(pointer_slot, runtime_slot, offset, dependency, trigger),
        )
        graph.add_op(op)
        return op

    if not backward:
        x = tensor("shared_x", [tokens, hidden], 0)
        w1 = tensor("shared_w1", [1, 2 * intermediate, hidden], 1, True)
        w2 = tensor("shared_w2", [1, hidden, intermediate], 2, True)
        up = tensor("shared_up", [tokens, 2 * intermediate], 3)
        act = tensor("shared_activation", [tokens, intermediate], 4)
        out = tensor("shared_output", [tokens, hidden], 5)
        first = add("shared_up_proj", OpType.GMM, [x, w1], up, 8, route_event, event_base)
        middle = add("shared_swiglu", OpType.SWIGLU, [up], act, 8 + 2 * stride, event_base, event_base + 1)
        last = add("shared_down_proj", OpType.GMM, [act, w2], out, 8 + stride, event_base + 1, event_base + 2)
    else:
        dy = tensor("shared_dy", [tokens, hidden], 0)
        act = tensor("shared_activation", [tokens, intermediate], 1)
        dw2 = tensor("shared_dw2", [1, hidden, intermediate], 2)
        w2 = tensor("shared_w2", [1, hidden, intermediate], 3)
        dact = tensor("shared_dact", [tokens, intermediate], 4)
        up = tensor("shared_up", [tokens, 2 * intermediate], 5)
        dup = tensor("shared_dup", [tokens, 2 * intermediate], 6)
        w1 = tensor("shared_w1", [1, 2 * intermediate, hidden], 7)
        dx = tensor("shared_dx", [tokens, hidden], 8)
        x = tensor("shared_x", [tokens, hidden], 9)
        dw1 = tensor("shared_dw1", [1, 2 * intermediate, hidden], 10)
        dy_transposed = tensor("shared_dy", [tokens, hidden], 0, True)
        dup_transposed = tensor("shared_dup", [tokens, 2 * intermediate], 6, True)
        first = add("shared_act_grad", OpType.GMM, [dy, w2], dact, 8, route_event, event_base)
        w2op = add("shared_w2_grad", OpType.GMM, [dy_transposed, act], dw2, 8 + 3 * stride, route_event, event_base + 4)
        middle = add("shared_swiglu_grad", OpType.SWIGLU_GRAD, [dact, up], dup, 8 + 4 * stride, event_base, event_base + 2)
        gate_event = event_base + (3 if endpoint is not None else 4)
        gate = add("shared_gate_grad", OpType.GMM, [dup, w1], dx, 8 + stride, event_base + 2, gate_event)
        weight = add("shared_w1_grad", OpType.GMM, [dup_transposed, x], dw1, 8 + 2 * stride, event_base + 2, event_base + 4)
        graph.add_edge(first, middle).add_edge(middle, gate).add_edge(middle, weight)
        for terminal in route_terminals:
            graph.add_edge(terminal, first).add_edge(terminal, w2op)
        tsv.shared_completion_event = event_base + 4
        tsv.shared_completion_count = 3 * cores
        if endpoint is not None:
            total = tensor("shared_total_dx", [tokens, hidden], 11)
            residual = add("shared_dx_add", OpType.ADD_RESIDUAL, [dx, total], total,
                           0, gate_event, event_base + 4)
            graph.add_edge(gate, residual)
            graph.add_edge(endpoint, first).add_edge(endpoint, w2op)
            tsv.shared_completion_count = 2 * cores + tokens // 128
        return
    graph.add_edge(first, middle).add_edge(middle, last)
    for terminal in route_terminals:
        graph.add_edge(terminal, first)
    tsv.shared_completion_event = event_base + 2
    tsv.shared_completion_count = cores
    if endpoint is not None:
        graph.add_edge(last, endpoint)
        endpoint.fill_config.dependent_event = tsv.shared_completion_event
