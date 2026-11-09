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
# ============================================================================

"""Unit tests for native Torch adapter registration diagnostics."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


from hyper_parallel.core.multicore._loader import NativeComponentUnavailableError
from hyper_parallel.core.multicore.torch import ops
from tests.common.mark_utils import arg_mark


# Registration-cache and alias-ABI checks require Torch's private operator schema and loader.
class TestTorchOps(unittest.TestCase):
    """Native initialization is cached, but Torch itself is imported normally."""

    # pylint: disable=protected-access

    def setUp(self) -> None:
        """Isolate backend registration from the native adapter schema checks."""
        patcher = patch.object(ops, "load_torch_npu")
        self.backend = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        """Never retain a mocked native-registration cache between tests."""
        ops._load_native.cache_clear()

    def test_shared_indirect_payloads_record_launch_stream(self):
        """Keep every indirect allocation alive on the stream that consumes its address."""
        stream = object()
        resources = [Mock(device=SimpleNamespace(type="npu")) for _ in range(5)]
        for index, resource in enumerate(resources):
            resource.data_ptr.return_value = 4096 * (index + 1)
        addresses = ops.torch.zeros(2, dtype=ops.torch.int64)
        with patch.object(ops.torch, "tensor", return_value=addresses), patch.object(
                ops.torch.npu, "current_stream", return_value=stream):
            runtime = ops.torch.zeros(64, dtype=ops.torch.uint8)
            ops._bind_shared_resources(runtime, resources[0], resources[1], tuple(resources[2:]))
        for resource in resources:
            resource.record_stream.assert_called_once_with(stream)

    @staticmethod
    def _backward_op(alias: str = "a") -> SimpleNamespace:
        """Build a real schema for the mutable receive and expert input-gradient slots."""
        operands = ["Tensor(a!) dispatch"]
        for index in range(1, 32):
            mutation = alias if index == 12 else ("b" if index in (25, 29) else None)
            operands.append(f"Tensor({mutation}!) input_{index}" if mutation else f"Tensor input_{index}")
        schema = ops.torch._C.parse_schema(
            f"mega_moe_grad({', '.join(operands)}) -> ()"
        )
        return SimpleNamespace(default=SimpleNamespace(_schema=schema))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_adapter_failure_reports_original_cause(self):
        """Feature: Native adapter load diagnostics.
        Description: Register the backend and fail the actual adapter library load.
        Expectation: Preserve the failed library and original ABI error.
        """
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("bad.so"))),
            patch.object(ops, "preload_vendor_library"),
            patch.object(ops.torch.ops, "load_library", side_effect=OSError("bad ABI")),
            self.assertRaisesRegex(NativeComponentUnavailableError, "ADAPTER-LOAD-FAILED.*bad ABI"),
        ):
            ops._load_native()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_successful_registration_is_cached(self):
        """Feature: Cached native registration.
        Description: Load the backend and adapter twice, then reject mismatched native schemas.
        Expectation: Register once per cache lifetime and retain transport and alias validation.
        """
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("good.so"))),
            patch.object(ops, "preload_vendor_library") as preload,
            patch.object(ops.torch.ops, "load_library") as load,
            patch.object(ops.torch.ops.hyper_parallel, "mega_moe_transport_version",
                         return_value=6, create=True) as version,
            patch.object(ops.torch.ops.hyper_parallel, "mega_moe_grad", self._backward_op(), create=True) as backward,
        ):
            ops._load_native()
            ops._load_native()
            self.backend.assert_called_once_with()
            preload.assert_called_once_with(Path("vendor"))
            load.assert_called_once_with("good.so")
            for protocol, alias, message in ((0, "a", "transport ABI"), (6, "e", "storage reuse")):
                ops._load_native.cache_clear()
                version.return_value = protocol
                backward.default._schema = self._backward_op(alias).default._schema
                with self.assertRaisesRegex(NativeComponentUnavailableError, message):
                    ops._load_native()
