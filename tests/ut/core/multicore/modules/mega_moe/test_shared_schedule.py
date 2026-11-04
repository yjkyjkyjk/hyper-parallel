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
"""Check progress of the static shared-forward worker queues and events."""

import itertools
import unittest

from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_forward_graph, build_route_forward_graph,
)
from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_INVALID_ID, TaskSplitValue, TaskType,
)


def _assert_worker_progress(configs):
    """Interpret event signaling and fixed per-worker queues across all ranks."""
    events = [[0] * cfg.event_capacity for cfg in configs]
    workers = []
    for rank, cfg in enumerate(configs):
        if cfg.ready_event:
            events[rank][cfg.ready_event] = 1
        lanes = cfg.num_workers // 2
        for index, task_id in enumerate(cfg.cube_task_indices[:cfg.task_index_num[0]]):
            assert cfg.all_tasks[task_id].task_index % lanes == index % lanes, (
                "changed GMM core assignment"
            )
        queued = []
        for queue, count in ((cfg.cube_task_indices, cfg.task_index_num[0]),
                             (cfg.vector_task_indices, cfg.task_index_num[1])):
            task_ids = list(queue[:count])
            queued.extend(task_ids)
            workers.extend([rank, task_ids[lane::lanes], 0] for lane in range(lanes))
        assert sorted(queued) == list(range(cfg.task_num + 1)), "lost or duplicated task"
    while True:
        pending = progress = 0
        for worker in workers:
            rank, task_ids, index = worker
            if index == len(task_ids):
                continue
            pending += 1
            cfg = configs[rank]
            task = cfg.all_tasks[task_ids[index]]
            dependencies = [task.dependent_event]
            if task.task_type == TaskType.TASK_WEIGHTED_UNPERMUTE and task.inputs[3].dim[3]:
                dependencies.append(task.inputs[3].dim[3])
            if any(event != EVENT_INVALID_ID
                   and events[rank][event] < cfg.all_event_num_triggers[event]
                   for event in dependencies):
                continue
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                peer = task.task_index // (task.task_split_num // len(configs))
                events[peer][task.trigger_event] += 1
                if task.extra_value_2:
                    events[rank][task.extra_value_2] += 1
            else:
                events[rank][task.trigger_event] += 1
            worker[2] += 1
            progress += 1
        if not pending:
            return
        assert progress, "worker queues contain an event dependency cycle"


class TestSharedForwardSchedule(unittest.TestCase):
    """Exercise event joins with partial worker rounds and multiple EP ranks."""

    def test_worker_progress_and_completion_join(self):
        """Every scheduled task runs and reduction waits on both branches."""
        shapes = ((128, 4, 2, 2), (4096, 32, 4, 8), (128, 32, 4, 8),
                  (512, 4, 2, 2), (128, 4, 1, 2), (384, 6, 2, 2))
        for shape, cores, builder in itertools.product(
                shapes, (1, 20, 24), (build_forward_graph, build_route_forward_graph)):
            tokens, experts, ep, top_k = shape
            with self.subTest(shape=shape, cores=cores, builder=builder.__name__):
                values = TaskSplitValue(tp=1, ep=ep, seq_size=tokens,
                                       all_expert_num=experts, top_k=top_k)
                graph = builder(values, hidden_size=5120, intermediate_size=1792,
                                num_cube_cores=cores, integrated_unpermute=True,
                                shared_intermediate_size=1792, fuse_shared_expert=True)
                graph.propagate_splits(values)
                configs = [build_config_for_rank(graph, values, rank, cores) for rank in range(ep)]
                join = graph.get_op("weighted_unpermute").fill_config.dependent_event
                self.assertEqual(graph.get_op("shared_up_proj").fill_config.dependency,
                                 EVENT_INVALID_ID)
                self.assertEqual(graph.get_op("shared_down_proj").fill_config.trigger, join)
                for cfg in configs:
                    self.assertEqual(cfg.all_event_num_triggers[join],
                                     graph.get_op("combine").task_num + cores)
                _assert_worker_progress(configs)
