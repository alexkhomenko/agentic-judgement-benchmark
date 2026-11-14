"""Controller-side host memory sampler and hardware facts (macOS).

Runs in the controller process, not the worker, so an out-of-memory exit of the worker leaves its
evidence behind. Reads kernel counters directly (sysctl, host_statistics64, proc_pid_rusage); no
subprocesses and no privileges. On other platforms every measurement is reported as unavailable with a
reason rather than as zero.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
import threading
import time
from typing import Any

PRESSURE_LEVELS = {1: "normal", 2: "warning", 4: "critical"}
HOST_VM_INFO64 = 4


class _SwapUsage(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("avail", ctypes.c_uint64), ("used", ctypes.c_uint64),
                ("pagesize", ctypes.c_uint32), ("encrypted", ctypes.c_bool)]


class _VMStatistics64(ctypes.Structure):
    _fields_ = [("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
                ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
                ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
                ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64), ("faults", ctypes.c_uint64),
                ("cow_faults", ctypes.c_uint64), ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64),
                ("purges", ctypes.c_uint64), ("purgeable_count", ctypes.c_uint32),
                ("speculative_count", ctypes.c_uint32), ("decompressions", ctypes.c_uint64),
                ("compressions", ctypes.c_uint64), ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
                ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
                ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
                ("total_uncompressed_pages_in_compressor", ctypes.c_uint64)]


class _RusageInfoV0(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in ("user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
                                             "pageins", "wired_size", "resident_size", "phys_footprint",
                                             "proc_start_abstime", "proc_exit_abstime")]


class HostProbe:
    """Thin wrapper over the kernel counters; every reader returns None when unavailable."""

    def __init__(self) -> None:
        self.available = platform.system() == "Darwin"
        self._libc: Any = None
        if self.available:
            self._libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            self._libc.mach_host_self.restype = ctypes.c_uint32

    def _sysctl(self, name: str, value: Any) -> bool:
        size = ctypes.c_size_t(ctypes.sizeof(value))
        return bool(self._libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None,
                                            ctypes.c_size_t(0)) == 0)

    def _sysctl_int(self, name: str, ctype: Any = ctypes.c_int) -> int | None:
        if not self.available:
            return None
        value = ctype()
        return int(value.value) if self._sysctl(name, value) else None

    def _sysctl_str(self, name: str) -> str | None:
        if not self.available:
            return None
        buffer = ctypes.create_string_buffer(256)
        size = ctypes.c_size_t(256)
        ok = self._libc.sysctlbyname(name.encode(), buffer, ctypes.byref(size), None, ctypes.c_size_t(0)) == 0
        return buffer.value.decode(errors="replace") if ok else None

    def pressure_level(self) -> int | None:
        return self._sysctl_int("kern.memorystatus_vm_pressure_level")

    def free_percent(self) -> int | None:
        return self._sysctl_int("kern.memorystatus_level")

    def swap_used(self) -> int | None:
        if not self.available:
            return None
        usage = _SwapUsage()
        return int(usage.used) if self._sysctl("vm.swapusage", usage) else None

    def vm_counters(self) -> dict[str, int] | None:
        if not self.available:
            return None
        stats = _VMStatistics64()
        count = ctypes.c_uint32(ctypes.sizeof(_VMStatistics64) // 4)
        if self._libc.host_statistics64(self._libc.mach_host_self(), HOST_VM_INFO64, ctypes.byref(stats),
                                        ctypes.byref(count)) != 0:
            return None
        return {"swapins": int(stats.swapins), "swapouts": int(stats.swapouts), "pageouts": int(stats.pageouts)}

    def page_size(self) -> int | None:
        return self._sysctl_int("hw.pagesize", ctypes.c_int64)

    def footprint(self, pid: int) -> int | None:
        if not self.available:
            return None
        info = _RusageInfoV0()
        return int(info.phys_footprint) if self._libc.proc_pid_rusage(pid, 0, ctypes.byref(info)) == 0 else None

    def host_info(self) -> dict[str, Any]:
        if not self.available:
            return {"system": platform.system(), "machine": platform.machine(),
                    "unavailable": "hardware facts are collected on macOS only"}
        return {"system": "Darwin", "machine": platform.machine(), "model": self._sysctl_str("hw.model"),
                "chip": self._sysctl_str("machdep.cpu.brand_string"),
                "memory_bytes": self._sysctl_int("hw.memsize", ctypes.c_int64),
                "cpu_cores": self._sysctl_int("hw.ncpu"), "performance_cores": self._sysctl_int("hw.perflevel0.physicalcpu"),
                "efficiency_cores": self._sysctl_int("hw.perflevel1.physicalcpu"),
                "os_version": self._sysctl_str("kern.osproductversion"), "os_build": self._sysctl_str("kern.osversion")}


class MemorySampler:
    """Samples host memory pressure, swap, and one process's footprint on a background thread.

    Usage: `with MemorySampler(pid) as sampler: ...; summary = sampler.summary()`. The summary survives the
    sampled process dying; samples taken after it exits simply lack a footprint.
    """

    def __init__(self, pid: int | None, *, interval_s: float = 0.25, probe: HostProbe | None = None) -> None:
        self.pid, self.interval_s = pid, interval_s
        self.probe = probe or HostProbe()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = 0
        self._pressure_max: int | None = None
        self._free_min: int | None = None
        self._footprint_max: int | None = None
        self._swap_max: int | None = None
        self._start_counters: dict[str, int] | None = None
        self._end_counters: dict[str, int] | None = None
        self._swap_start: int | None = None
        self._swap_end: int | None = None
        self._started = self._ended = 0.0

    def __enter__(self) -> MemorySampler:
        self._started = time.monotonic()
        self._start_counters = self.probe.vm_counters()
        self._swap_start = self.probe.swap_used()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._sample()
        self._end_counters = self.probe.vm_counters()
        self._swap_end = self.probe.swap_used()
        self._ended = time.monotonic()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        self._samples += 1
        level, free = self.probe.pressure_level(), self.probe.free_percent()
        swap = self.probe.swap_used()
        footprint = self.probe.footprint(self.pid) if self.pid is not None else None
        if level is not None:
            self._pressure_max = level if self._pressure_max is None else max(self._pressure_max, level)
        if free is not None:
            self._free_min = free if self._free_min is None else min(self._free_min, free)
        if swap is not None:
            self._swap_max = swap if self._swap_max is None else max(self._swap_max, swap)
        if footprint is not None:
            self._footprint_max = footprint if self._footprint_max is None else max(self._footprint_max, footprint)

    def summary(self) -> dict[str, Any]:
        if not self.probe.available:
            return {"available": False, "reason": "host memory sampling is implemented for macOS only"}
        page = self.probe.page_size()
        deltas: dict[str, int | None] = {}
        for name in ("swapins", "swapouts", "pageouts"):
            start, end = (self._start_counters or {}).get(name), (self._end_counters or {}).get(name)
            deltas[f"{name}_bytes"] = (end - start) * page if None not in (start, end, page) else None  # type: ignore[operator]
        return {
            "available": True,
            "source": "controller sampler: sysctl kern.memorystatus_*, vm.swapusage, host_statistics64, proc_pid_rusage",
            "interval_s": self.interval_s,
            "samples": self._samples,
            "duration_s": round(self._ended - self._started, 3),
            "pressure_level_max": self._pressure_max,
            "pressure_max": PRESSURE_LEVELS.get(self._pressure_max or 0),
            "free_percent_min": self._free_min,
            "swap_used_start_bytes": self._swap_start,
            "swap_used_max_bytes": self._swap_max,
            "swap_used_end_bytes": self._swap_end,
            **deltas,
            "worker_footprint_max_bytes": self._footprint_max,
        }
