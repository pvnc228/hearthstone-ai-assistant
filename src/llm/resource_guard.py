"""Conservative local-ML resource limits for an interactive Windows workstation."""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable


class ResourceLimitExceeded(RuntimeError):
    """Raised before local inference can exhaust workstation resources."""


@dataclass(frozen=True)
class ResourceSnapshot:
    gpu_total_mib: int
    gpu_used_mib: int
    gpu_free_mib: int
    gpu_temperature_c: int
    gpu_utilization_percent: int
    ram_total_mib: int
    ram_available_mib: int


@dataclass(frozen=True)
class ResourceBudget:
    cpu_threads: int = 2
    interop_threads: int = 1
    max_batch_size: int = 2
    max_process_gpu_fraction: float = 0.40
    max_gpu_used_mib: int = 4096
    min_gpu_free_mib: int = 4096
    min_ram_available_mib: int = 8192
    max_gpu_temperature_c: int = 75
    cooldown_seconds: float = 0.05
    telemetry_every_batches: int = 8
    pause_poll_seconds: float = 2.0
    max_pause_seconds: float = 60.0

    def validate(self) -> None:
        if not 1 <= self.cpu_threads <= 4:
            raise ValueError("cpu_threads must be between 1 and 4")
        if not 1 <= self.interop_threads <= 2:
            raise ValueError("interop_threads must be between 1 and 2")
        if not 1 <= self.max_batch_size <= 4:
            raise ValueError("max_batch_size must be between 1 and 4")
        if not 0.1 <= self.max_process_gpu_fraction <= 0.5:
            raise ValueError("max_process_gpu_fraction must be between 0.1 and 0.5")
        if self.telemetry_every_batches < 1:
            raise ValueError("telemetry_every_batches must be positive")

    def validate_batch_size(self, batch_size: int) -> None:
        self.validate()
        if not 1 <= batch_size <= self.max_batch_size:
            raise ResourceLimitExceeded(
                f"batch_size={batch_size} exceeds safe maximum {self.max_batch_size}"
            )


def conservative_environment(budget: ResourceBudget) -> dict[str, str]:
    budget.validate()
    return {
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(budget.cpu_threads),
        "MKL_NUM_THREADS": str(budget.cpu_threads),
        "OPENBLAS_NUM_THREADS": str(budget.cpu_threads),
        "NUMEXPR_NUM_THREADS": str(budget.cpu_threads),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
    }


def configure_conservative_process(budget: ResourceBudget) -> None:
    """Apply limits before model loading; safe to call once at process startup."""
    for key, value in conservative_environment(budget).items():
        os.environ[key] = value

    if os.name == "nt":
        below_normal_priority_class = 0x00004000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal_priority_class):
            raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")

    import torch

    torch.set_num_threads(budget.cpu_threads)
    try:
        torch.set_num_interop_threads(budget.interop_threads)
    except RuntimeError as exc:
        if "cannot set number of interop threads" not in str(exc).lower():
            raise
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(budget.max_process_gpu_fraction, device=0)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def windows_ram_mib() -> tuple[int, int]:
    if os.name != "nt":
        raise RuntimeError("Windows RAM telemetry is only available on Windows")
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return status.ullTotalPhys // 1024 // 1024, status.ullAvailPhys // 1024 // 1024


def parse_nvidia_smi_row(row: str) -> tuple[int, int, int, int, int]:
    values = [int(part.strip()) for part in row.strip().split(",")]
    if len(values) != 5:
        raise ValueError(f"Unexpected nvidia-smi output: {row!r}")
    return tuple(values)  # type: ignore[return-value]


def query_resource_snapshot() -> ResourceSnapshot:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    first_gpu = result.stdout.splitlines()[0]
    gpu_total, gpu_used, gpu_free, temperature, utilization = parse_nvidia_smi_row(first_gpu)
    ram_total, ram_available = windows_ram_mib()
    return ResourceSnapshot(
        gpu_total_mib=gpu_total,
        gpu_used_mib=gpu_used,
        gpu_free_mib=gpu_free,
        gpu_temperature_c=temperature,
        gpu_utilization_percent=utilization,
        ram_total_mib=ram_total,
        ram_available_mib=ram_available,
    )


def unsafe_reasons(snapshot: ResourceSnapshot, budget: ResourceBudget) -> list[str]:
    reasons: list[str] = []
    if snapshot.gpu_used_mib > budget.max_gpu_used_mib:
        reasons.append(f"GPU used {snapshot.gpu_used_mib} MiB > {budget.max_gpu_used_mib} MiB")
    if snapshot.gpu_free_mib < budget.min_gpu_free_mib:
        reasons.append(f"GPU free {snapshot.gpu_free_mib} MiB < {budget.min_gpu_free_mib} MiB")
    if snapshot.ram_available_mib < budget.min_ram_available_mib:
        reasons.append(f"RAM free {snapshot.ram_available_mib} MiB < {budget.min_ram_available_mib} MiB")
    if snapshot.gpu_temperature_c > budget.max_gpu_temperature_c:
        reasons.append(
            f"GPU temperature {snapshot.gpu_temperature_c} C > {budget.max_gpu_temperature_c} C"
        )
    return reasons


class ResourceGuard:
    def __init__(
        self,
        budget: ResourceBudget,
        *,
        snapshot_provider: Callable[[], ResourceSnapshot] = query_resource_snapshot,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        budget.validate()
        self.budget = budget
        self.snapshot_provider = snapshot_provider
        self.sleeper = sleeper
        self.clock = clock
        self.batch_count = 0
        self.sample_count = 0
        self.max_gpu_used_mib = 0
        self.min_gpu_free_mib: int | None = None
        self.min_ram_available_mib: int | None = None
        self.max_gpu_temperature_c = 0

    def preflight(self) -> ResourceSnapshot:
        return self.wait_until_safe()

    def observe(self, snapshot: ResourceSnapshot) -> None:
        self.sample_count += 1
        self.max_gpu_used_mib = max(self.max_gpu_used_mib, snapshot.gpu_used_mib)
        self.min_gpu_free_mib = (
            snapshot.gpu_free_mib
            if self.min_gpu_free_mib is None
            else min(self.min_gpu_free_mib, snapshot.gpu_free_mib)
        )
        self.min_ram_available_mib = (
            snapshot.ram_available_mib
            if self.min_ram_available_mib is None
            else min(self.min_ram_available_mib, snapshot.ram_available_mib)
        )
        self.max_gpu_temperature_c = max(
            self.max_gpu_temperature_c, snapshot.gpu_temperature_c
        )

    def telemetry_summary(self) -> dict[str, int | None]:
        return {
            "samples": self.sample_count,
            "max_gpu_used_mib": self.max_gpu_used_mib,
            "min_gpu_free_mib": self.min_gpu_free_mib,
            "min_ram_available_mib": self.min_ram_available_mib,
            "max_gpu_temperature_c": self.max_gpu_temperature_c,
        }

    def wait_until_safe(self) -> ResourceSnapshot:
        started = self.clock()
        while True:
            snapshot = self.snapshot_provider()
            self.observe(snapshot)
            reasons = unsafe_reasons(snapshot, self.budget)
            if not reasons:
                return snapshot
            if self.clock() - started >= self.budget.max_pause_seconds:
                raise ResourceLimitExceeded("Resource watchdog stopped the run: " + "; ".join(reasons))
            self.sleeper(self.budget.pause_poll_seconds)

    def after_batch(self) -> ResourceSnapshot | None:
        self.batch_count += 1
        self.sleeper(self.budget.cooldown_seconds)
        if self.batch_count % self.budget.telemetry_every_batches == 0:
            return self.wait_until_safe()
        return None
