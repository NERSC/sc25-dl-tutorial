"""Device-portability helpers.

Picks the right torch accelerator (cuda / xpu / cpu) based on what's available,
and exposes no-op shims for the cuda-only nvtx/profiler/synchronize calls so the
training loop runs unchanged on Intel XPU.
"""
import contextlib
import os
import torch


def _detect_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


DEVICE_TYPE: str = _detect_device_type()


def device_count() -> int:
    if DEVICE_TYPE == "cuda":
        return torch.cuda.device_count()
    if DEVICE_TYPE == "xpu":
        return torch.xpu.device_count()
    return 1


def set_device(local_rank: int) -> torch.device:
    if DEVICE_TYPE == "cuda":
        torch.cuda.set_device(local_rank)
    elif DEVICE_TYPE == "xpu":
        torch.xpu.set_device(local_rank)
    return torch.device(f"{DEVICE_TYPE}:{local_rank}" if DEVICE_TYPE != "cpu" else "cpu")


def synchronize() -> None:
    if DEVICE_TYPE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE_TYPE == "xpu":
        torch.xpu.synchronize()


def dist_backend() -> str:
    if DEVICE_TYPE == "cuda":
        return "nccl"
    if DEVICE_TYPE == "xpu":
        return "ccl"
    return "gloo"


# nvtx / profiler shims --------------------------------------------------------
# On XPU/CPU we replace push/pop with no-ops. We keep cuda calls real so any
# nsys profiling on cuda hosts still works.

class _NvtxShim:
    def range_push(self, _msg: str) -> None: pass
    def range_pop(self) -> None: pass


class _ProfilerShim:
    def start(self) -> None: pass
    def stop(self) -> None: pass


if DEVICE_TYPE == "cuda":
    nvtx = torch.cuda.nvtx
    profiler = torch.cuda.profiler
else:
    nvtx = _NvtxShim()
    profiler = _ProfilerShim()
