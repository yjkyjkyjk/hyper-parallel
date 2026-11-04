# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Host regression for independent routing and shared-expert switches."""

import itertools
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.diagnostics import bind_shared_diagnostic_ports
from hyper_parallel.core.multicore.diagnostics.common.graph import validate_mega_kernel_diagnostic_contract
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.modules.mega_moe.shared import shared_backward_buffers
from hyper_parallel.core.multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_parallel.core.multicore.scheduler.config import TaskType


class TestSharedExpertSchedule(unittest.TestCase):
    """Check shared dependencies without launching an accelerator kernel."""

    def test_shared_scratch_ownership_and_capacity(self):
        """Serial shared scratch stays disjoint and returned dX owns compact storage."""
        x, w1, w2 = torch.empty(4, 8), torch.empty(6, 8), torch.empty(8, 3)
        saved = (x, w1, w2, torch.empty(4, 6), torch.empty(4, 3))
        for capacity in (1, 128):
            with self.subTest(capacity=capacity), patch(
                    "hyper_parallel.core.multicore.modules.mega_moe.shared.pointer_table",
                    return_value=torch.empty(0)):
                act, up = torch.empty(capacity), torch.empty(capacity)
                values, _, reuse = shared_backward_buffers(x, saved, scratch=(act, up), integrated_output=True)
                self.assertEqual(reuse, capacity == 128)
                self.assertEqual(values[11].untyped_storage().nbytes(), x.numel() * x.element_size())
                self.assertNotEqual(values[11].data_ptr(), values[8].data_ptr())
                if reuse:
                    self.assertEqual(values[4].data_ptr(), act.data_ptr())
                    self.assertEqual(values[6].data_ptr(), up.data_ptr())
                    self.assertEqual(values[8].storage_offset(), saved[4].numel())
                    self.assertEqual(values[2].storage_offset(), saved[3].numel())
                    self.assertEqual(values[10].storage_offset(), saved[3].numel() + w2.numel())
                else:
                    self.assertNotEqual(values[4].data_ptr(), act.data_ptr())
                    self.assertNotEqual(values[6].data_ptr(), up.data_ptr())

    def test_independent_direction_and_route_modes(self):
        """All mode combinations retain route semantics and complete every shared output."""
        modes = ("legacy", "permute", "unpermute", "both")
        for forward, backward, shared_forward, shared_backward in itertools.product(
                modes, modes, (False, True), (False, True)):
            with self.subTest(forward=forward, backward=backward,
                              shared_forward=shared_forward, shared_backward=shared_backward):
                spec = MegaMoeSpec(
                    128, 5120, 128, 4, 2, None, 512, 2, None, 0, 20,
                    forward_fusion_mode=forward, backward_fusion_mode=backward,
                    shared_intermediate_size=256, forward_shared_expert=shared_forward,
                    backward_shared_expert=shared_backward,
                )
                fg, fc, bg, bc = _build_runtime_artifacts(spec)
                self.assertTrue(can_reuse_backward_dispatch(bc, 2, 20))
                for graph, cfg, enabled, terminal_stage in (
                        (fg, fc, shared_forward, "shared_down_proj"),
                        (bg, bc, shared_backward, "shared_w1_grad")):
                    self.assertEqual(graph.has_op(terminal_stage), enabled)
                    if not enabled:
                        continue
                    ordered = graph.topological_sort()
                    names = [op.name for op in ordered]
                    first_shared = next(i for i, name in enumerate(names) if name.startswith("shared_"))
                    endpoint_name = "weighted_unpermute" if graph is fg else "unpermute_grad"
                    self.assertTrue(all(name.startswith("shared_") or name == endpoint_name
                                        for name in names[first_shared:]))
                    if graph.has_op(endpoint_name):
                        endpoint = graph.get_op(endpoint_name)
                        self.assertEqual(names[-1], endpoint_name if graph is fg else "shared_dx_add")
                        self.assertEqual(endpoint.fill_config.shared_output_slot, 5 if graph is fg else 11)
                        task_type = (TaskType.TASK_WEIGHTED_UNPERMUTE if graph is fg
                                     else TaskType.TASK_UNPERMUTE_REDUCE_GRAD)
                        endpoint_tasks = [task for task in cfg.all_tasks[:cfg.task_num]
                                          if task.task_type == task_type]
                        self.assertTrue(endpoint_tasks)
                        for task in endpoint_tasks:
                            self.assertEqual(task.outputs[0].tensor_type, 2)
                            self.assertEqual(task.outputs[0].dynamic_dim, 5 if graph is fg else 11)
                            self.assertEqual(task.extra_value_1 & 2, 2 if graph is fg else 0)
                        if graph is fg:
                            self.assertTrue(all(op.name.startswith("shared_") for op in endpoint.predecessors))
                        else:
                            self.assertIn(endpoint, graph.get_op("shared_act_grad").predecessors)
                    task_count = sum(op.task_num for op in ordered)
                    terminal = cfg.all_tasks[task_count]
                    self.assertEqual(terminal.task_type, TaskType.TASK_TERMINATE)
                    producers = sum(cfg.all_tasks[i].trigger_event == terminal.dependent_event
                                    for i in range(task_count))
                    self.assertEqual(cfg.all_event_num_triggers[terminal.dependent_event], producers)
                    self.assertGreater(producers, 0)
                    diagnostic_graph = bind_shared_diagnostic_ports(graph, cfg, forward=graph is fg)
                    shared_ops = [op for op in diagnostic_graph.topological_sort()
                                  if op.name.startswith("shared_")]
                    self.assertTrue(all(op.diagnostic_host_ports for op in shared_ops))
                    self.assertTrue(all("shared_group_list" not in {tensor.name for tensor in op.inputs}
                                        for op in shared_ops))
                    sidecar_base = 29 if graph is fg else 35
                    self.assertTrue(all(position >= sidecar_base for op in shared_ops
                                        for position in op.param_positions))
                    self.assertFalse(any(op.diagnostic_host_ports for op in graph.topological_sort()
                                         if op.name.startswith("shared_")))
                    validate_mega_kernel_diagnostic_contract(
                        diagnostic_graph, cfg, 35 if graph is fg else (47 if graph.has_op(endpoint_name) else 46),
                    )

    def test_reuse_rejects_changed_routed_swiglu_ports(self):
        """Shared tasks must not hide an unknown direct reader or broken route chain."""
        spec = MegaMoeSpec(
            128, 5120, 128, 4, 2, None, 512, 2, None, 0, 20,
            forward_fusion_mode="both", backward_fusion_mode="both",
            shared_intermediate_size=256, forward_shared_expert=True, backward_shared_expert=True,
        )
        _, _, _, cfg = _build_runtime_artifacts(spec)
        self.assertTrue(can_reuse_backward_dispatch(cfg, 2, 20))
        task = next(task for task in cfg.all_tasks[:cfg.task_num]
                    if task.task_type == TaskType.TASK_SWI_GLU_GRAD and task.inputs[0].tensor_type != 2)
        task.inputs[0].input_position = 0
        self.assertFalse(can_reuse_backward_dispatch(cfg, 2, 20))
        task.inputs[0].input_position = 8
        shared = next(task for task in cfg.all_tasks[:cfg.task_num]
                      if task.task_type == TaskType.TASK_SWI_GLU_GRAD and task.inputs[0].tensor_type == 2)
        shared.inputs[0].input_position = 35
        self.assertFalse(can_reuse_backward_dispatch(cfg, 2, 20))
        shared.inputs[0].input_position = 36
        cfg.all_event_num_triggers[task.trigger_event] += 1
        self.assertFalse(can_reuse_backward_dispatch(cfg, 2, 20))

    def test_representative_full_fusion_storage_and_fences(self):
        """The performance shape joins all routed work before residual reduction."""
        spec = MegaMoeSpec(
            4096, 5120, 1792, 32, 8, None, 49152, 4, None, 0, 20,
            forward_fusion_mode="both", backward_fusion_mode="both",
            shared_intermediate_size=1792, forward_shared_expert=True, backward_shared_expert=True,
        )
        fg, fc, bg, bc = _build_runtime_artifacts(spec)
        self.assertTrue(can_reuse_backward_dispatch(bc, 8, 20))
        for graph, cfg, endpoint, slot in ((fg, fc, "weighted_unpermute", 5),
                                           (bg, bc, "unpermute_grad", 11)):
            self.assertEqual(graph.topological_sort()[-1].name, endpoint if graph is fg else "shared_dx_add")
            self.assertEqual(graph.get_op(endpoint).fill_config.shared_output_slot, slot)
            shared_first = graph.get_op("shared_up_proj" if graph is fg else "shared_act_grad")
            event = shared_first.fill_config.dependency
            if graph is fg:
                self.assertEqual(event, 0xFFFFFFFF)
                route_event = graph.get_op(endpoint).fill_config.dependent_event
                producers = [task for task in cfg.all_tasks[:cfg.task_num]
                             if task.trigger_event == route_event]
                self.assertEqual(cfg.all_event_num_triggers[route_event], len(producers))
                self.assertTrue(any(task.task_type == TaskType.TASK_GROUPED_MATMUL
                                    and task.dependent_event != 0xFFFFFFFF
                                    for task in producers))
            else:
                producers = [task for task in cfg.all_tasks[:cfg.task_num] if task.trigger_event == event]
                self.assertEqual(cfg.all_event_num_triggers[event], len(producers))
                self.assertTrue(producers)
        residual = bg.get_op("shared_dx_add")
        gate = bg.get_op("shared_gate_grad")
        self.assertEqual(residual.predecessors, [gate])
        self.assertEqual(residual.fill_config.dependency, gate.fill_config.trigger)
        self.assertNotEqual(gate.fill_config.trigger, bg.get_op("shared_w1_grad").fill_config.trigger)
