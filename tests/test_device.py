import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

DEVICE_MODULE = Path(__file__).parents[1] / "utils" / "device.py"


def load_device_module(device_type="cpu", available_backends=()):
    torch: Any = types.ModuleType("torch")
    torch.device = lambda value: value
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: device_type == "cuda",
        device_count=lambda: 2,
        set_device=lambda rank: None,
        synchronize=lambda: None,
        nvtx=object(),
        profiler=object(),
    )
    torch.xpu = types.SimpleNamespace(
        is_available=lambda: device_type == "xpu",
        device_count=lambda: 2,
        set_device=lambda rank: None,
        synchronize=lambda: None,
    )
    torch.distributed = types.SimpleNamespace(
        is_backend_available=lambda backend: backend in available_backends,
    )

    module_name = f"test_device_{device_type}"
    spec = importlib.util.spec_from_file_location(module_name, DEVICE_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch}):
        spec.loader.exec_module(module)
    return module


class TestDistributedBackend(unittest.TestCase):
    def test_xpu_uses_xccl(self):
        device = load_device_module("xpu", available_backends={"xccl"})
        self.assertEqual(device.dist_backend(), "xccl")

    def test_backend_override_is_honored_when_available(self):
        device = load_device_module("xpu", available_backends={"gloo"})
        with patch.dict(os.environ, {"TORCH_DISTRIBUTED_BACKEND": "gloo"}):
            self.assertEqual(device.dist_backend(), "gloo")

    def test_unavailable_backend_fails_clearly(self):
        device = load_device_module("xpu")
        with self.assertRaisesRegex(RuntimeError, "xccl.*not available"):
            device.dist_backend()


class TestDistributedEnvironment(unittest.TestCase):
    def test_single_process_defaults_rank_values_to_zero(self):
        device = load_device_module()
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
            self.assertEqual(device.distributed_env(), (1, 0, 0))

    def test_multi_process_requires_all_env_settings(self):
        device = load_device_module()
        env = {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "localhost",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaisesRegex(RuntimeError, "MASTER_PORT"),
        ):
            device.distributed_env()

    def test_multi_process_returns_validated_rank_values(self):
        device = load_device_module()
        env = {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(device.distributed_env(), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
