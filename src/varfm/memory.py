"""Memory guards: a run must never push the machine to its memory limits.

When the GPU has its own memory, running out of it raises a CUDA out-of-memory error, which the runner catches (it
halves the batch and tries again). When the GPU shares the system memory (an NVIDIA GB10 / DGX Spark, a Jetson: torch
reports the device as `is_integrated`), the same pressure can freeze the whole machine instead. There the guard

- keeps a reserve of system memory free: a model is not loaded if loading it would eat into the reserve, and
  PyTorch's allocator is capped, so that a batch that does not fit raises a catchable out-of-memory error;
- reads /proc/meminfo every second from a thread and, below the critical floor, kills the process at once, before the
  machine starts to freeze (the run resumes from its checkpoint);
- makes the process the first one the kernel's OOM killer would pick.

The thresholds are those used on the GB10 (MemTotal 130.6 GB): 24 GB always free, the batch halves below 36 GB, the
process stops below 8 GB, PyTorch may take at most 60% of the memory. Other shared-memory machines get the same
proportions; on other machines the thresholds only protect the system memory (10%, 15%, 4%).
"""

import gc
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

GREEN, YELLOW, RED, CRITICAL = "green", "yellow", "red", "critical"
GB10_TOTAL_GB = 130.6


@dataclass(frozen=True)
class Limits:
    reserve_gb: float                 # always left free: red below it (pause, then stop)
    yellow_gb: float                  # yellow below it: the batch halves
    critical_gb: float                # below it the run stops at once
    swap_growth_gb: float = 0.256     # red if the swap in use grows more than this
    gpu_fraction: float | None = None  # cap of PyTorch's allocator, as a share of the memory (shared memory only)


def limits_for(total_gb: float, shared: bool) -> Limits:
    if shared:
        s = total_gb / GB10_TOTAL_GB
        return Limits(24 * s, 36 * s, 8 * s, 0.256, 0.6)
    return Limits(0.10 * total_gb, 0.15 * total_gb, 0.04 * total_gb)


def read_meminfo(path: str = "/proc/meminfo") -> tuple[float, float, float]:
    """MemTotal, MemAvailable and the swap in use, in GB."""
    values = {}
    with open(path) as fh:
        for line in fh:
            name, _, rest = line.partition(":")
            if rest.split():
                values[name] = int(rest.split()[0]) * 1024 / 1e9
    return values["MemTotal"], values["MemAvailable"], values.get("SwapTotal", 0.0) - values.get("SwapFree", 0.0)


def is_oom(exc: BaseException) -> bool:
    """An out-of-memory error of PyTorch (CUDA or host), cuBLAS or Python."""
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cublas_status_alloc_failed" in text


def free_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class MemoryGuard:
    """Watches the system memory while a model runs; use it as a context manager around loading and batches."""

    def __init__(self, device: str = "cpu", meminfo: str = "/proc/meminfo"):
        self.device, self.meminfo = device, meminfo
        self.enabled = Path(meminfo).exists()
        self.shared = False
        if device.startswith("cuda"):
            import torch
            self.shared = bool(torch.cuda.get_device_properties(torch.device(device)).is_integrated)
        self.total_gb, self.available_gb, self._swap0 = read_meminfo(meminfo) if self.enabled else (0.0, 0.0, 0.0)
        self.lowest_gb = self.available_gb
        self.limits = limits_for(self.total_gb, self.shared)
        self.cap_gb = None
        self.critical = False
        self._halt = threading.Event()
        self._thread = None

    def state(self) -> str:
        if not self.enabled:
            return GREEN
        _, self.available_gb, swap = read_meminfo(self.meminfo)
        self.lowest_gb = min(self.lowest_gb, self.available_gb)
        lim = self.limits
        if self.available_gb < lim.critical_gb:
            return CRITICAL
        if self.available_gb < lim.reserve_gb or swap - self._swap0 > lim.swap_growth_gb:
            return RED
        return YELLOW if self.available_gb < lim.yellow_gb else GREEN

    def require(self, need_gb: float, what: str) -> None:
        """Refuse to start `what` (e.g. loading a model) if it would leave less than the reserve free."""
        self.state()
        if self.enabled and self.available_gb - need_gb < self.limits.reserve_gb:
            raise MemoryError(f"{what} takes up to {need_gb:.0f} GB and {self.limits.reserve_gb:.0f} GB must stay "
                              f"free, but {self.available_gb:.1f} GB are available: free some memory first")

    def cap_gpu(self) -> None:
        """Shared memory only: cap PyTorch's allocator below what keeps the reserve free."""
        if not (self.shared and self.enabled):
            return
        import torch
        self.state()
        fraction = min(self.limits.gpu_fraction, (self.available_gb - self.limits.reserve_gb) / self.total_gb)
        if fraction <= 0.05:
            raise MemoryError(f"only {self.available_gb:.1f} GB available: free some memory first")
        index = torch.device(self.device).index
        index = torch.cuda.current_device() if index is None else index
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        self.cap_gb = fraction * torch.cuda.get_device_properties(index).total_memory / 1e9
        print(f"memory guard: shared GPU memory, PyTorch capped at {self.cap_gb:.0f} GB, "
              f"{self.limits.reserve_gb:.0f} GB kept free", flush=True)

    def gpu_room_gb(self) -> float:
        """GPU memory that batches may still take, with the model loaded."""
        import torch
        if self.cap_gb is not None:
            return self.cap_gb - torch.cuda.memory_allocated() / 1e9
        free, _ = torch.cuda.mem_get_info()
        return (free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1e9

    def wait(self, timeout_s: float = 60.0) -> bool:
        """Red: wait for memory to come back (True) or give up after timeout_s (False)."""
        free_memory()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.state() in (GREEN, YELLOW):
                return True
            time.sleep(1.0)
        return False

    def __enter__(self):
        if self.shared:
            try:
                Path("/proc/self/oom_score_adj").write_text("1000")
            except OSError:
                pass
        if self.enabled:
            self._halt.clear()
            self._thread = threading.Thread(target=self._watch, name="varfm-memory", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return False

    def _watch(self) -> None:
        while not self._halt.wait(1.0):
            if self.state() == CRITICAL:
                self.critical = True
                if self.shared:  # freezing the machine is worse than losing the batch in flight
                    print(f"\nmemory guard: only {self.available_gb:.1f} GB available, stopping now "
                          "(run again to resume)", file=sys.stderr, flush=True)
                    os.kill(os.getpid(), signal.SIGKILL)


@contextmanager
def gpu_lock():
    """One model on the GPU at a time on this machine: an exclusive lock on a file, refused at once if it is taken."""
    if os.name != "posix":
        yield
        return
    import fcntl
    path = os.environ.get("VARFM_GPU_LOCK", "/tmp/varfm_gpu.lock")
    fh = open(path, "a")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise RuntimeError(f"another run is using the GPU (lock {path})") from None
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
