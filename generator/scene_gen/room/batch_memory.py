"""Check available memory and measure memory used during generation."""
import ctypes
import os
from pathlib import Path
import re
import subprocess
import sys


ROOM_BYTES = 2_000_000_000
AVAILABLE_BYTES = 2_000_000_000
PROJECT_BYTES = 4_000_000_000
POLL_SECONDS = .10
METRIC = 'macOS physical footprint' if sys.platform == 'darwin' else 'resident set size'


if sys.platform == 'darwin':
    # Darwin SDK sys/resource.h: RUSAGE_INFO_V4 and struct rusage_info_v4.
    class _RusageV4(ctypes.Structure):
        _fields_ = [
            ('before_footprint', ctypes.c_uint64 * 9),
            ('footprint', ctypes.c_uint64),
            ('before_peak', ctypes.c_uint64 * 20),
            ('peak', ctypes.c_uint64),
            ('after_peak', ctypes.c_uint64 * 6),
        ]

    _pid_rusage = ctypes.CDLL('/usr/lib/libSystem.B.dylib').proc_pid_rusage
    _pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RusageV4))
    _pid_rusage.restype = ctypes.c_int


def process_memory(pid, rss):
    if sys.platform == 'darwin':
        usage = _RusageV4()
        if _pid_rusage(pid, 4, ctypes.byref(usage)) == 0:
            return usage.footprint, max(usage.footprint, usage.peak)
    return rss, rss


def system_memory():
    if sys.platform == 'darwin':
        level = int(subprocess.check_output(
            ['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True))
        total = int(subprocess.check_output(['/usr/sbin/sysctl', '-n', 'hw.memsize'], text=True))
        ram = subprocess.check_output(['/usr/bin/memory_pressure', '-Q'], text=True)
        percent = int(re.search(r'free percentage: (\d+)%', ram)[1])
        # The query reports whole percentages. Reserve the possible rounding remainder.
        usable = (max(0, percent-1)*total)//100
        return {'available_bytes': usable, 'pressure_normal': level == 1, 'macos_level': level}
    meminfo = Path('/proc/meminfo').read_text()
    available = int(re.search(r'^MemAvailable:\s+(\d+)', meminfo, re.M)[1])*1024
    return {'available_bytes': available, 'pressure_normal': True}


def scan_processes():
    """Find worker groups and scene commands already running on this host."""
    ps = subprocess.check_output(['/bin/ps', '-ax', '-o', 'pid=,pgid=,rss=,command='], text=True)
    rows = []
    for line in ps.splitlines():
        pid, pgid, rss, command = line.strip().split(maxsplit=3)
        rows.append((int(pid), int(pgid), int(rss)*1024, command))
    return rows


def other_scene_processes(rows, workers):
    busy = []
    for pid, pgid, _rss, command in rows:
        if pid == os.getpid() or pgid in workers:
            continue
        if re.search(r'^\S*python[^\s]* .*?-m scene_gen\.room(?:\s|\.(?:full_batch|serve_material_\w+|viewer)\b)', command):
            busy.append(pid)
    return busy
