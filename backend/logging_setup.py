"""One logging entry point for all three AuK processes.

Why one file per role instead of one shared file: the model worker is a *spawned*
process, and two processes holding the same RotatingFileHandler on Windows interleave
records and then fail to rename the file at rollover (the other handle is still open).
So: logs/launcher.log (run.py), logs/server.log (FastAPI), logs/worker.log (model).

Nothing in here may raise. Losing a log line is acceptable; taking down a generation
because the log directory is read-only is not — hence the guarded handlers and the
silent degradation to console-only.
"""
from __future__ import annotations

import contextvars
import ctypes
import logging
import logging.handlers
import os
import platform
import sys
import time
import uuid
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_GB = 1024.0 ** 3
_MAX_BYTES = 10 * 1024 * 1024
_BACKUPS = 5
_FORMAT = "%(asctime)s %(levelname)-7s [%(rid)s] %(name)-14s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 8 hex chars: short enough to read aloud over the phone, and grep-able across both
# files, which is the whole point — server.log and worker.log share one request id.
rid_var: contextvars.ContextVar[str] = contextvars.ContextVar("auk_rid", default="-")

_ROLES = ("launcher", "server", "worker")


def new_rid() -> str:
    return uuid.uuid4().hex[:8]


def set_rid(value: str | None = None) -> str:
    """Start a new correlation id (or adopt one from another process) and return it."""
    rid = (value or "").strip()[:16] or new_rid()
    rid_var.set(rid)
    return rid


def get_rid() -> str:
    return rid_var.get()


def log_dir() -> Path:
    # Overridable so a second instance (a test server on another port) never shares the
    # same rotating files with the user's running one.
    return Path(os.getenv("AUK_LOG_DIR", str(ROOT / "logs"))).expanduser()


def level_from_env() -> int:
    name = (os.getenv("AUK_LOG_LEVEL") or "").strip().upper()
    if name:
        return getattr(logging, name, logging.INFO)
    return logging.DEBUG if (os.getenv("AUK_DEBUG") or "").strip().lower() in {
        "1", "true", "yes", "on"} else logging.INFO


class _RidFilter(logging.Filter):
    def filter(self, record):
        record.rid = rid_var.get()
        return True


def _console_stream():
    """stdout with errors='replace' — a GBK console would otherwise raise inside the
    handler for every CJK line, and logging would swallow it as '--- Logging error ---'."""
    stream = sys.stdout
    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace", line_buffering=True)
    except Exception:
        pass
    return stream


def _file_handler(role: str, fmt: logging.Formatter, rid_filter: logging.Filter):
    try:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            directory / ("%s.log" % role), maxBytes=_MAX_BYTES, backupCount=_BACKUPS,
            encoding="utf-8", delay=True)
    except Exception as exc:
        print("[日志] 无法写 %s：%s（只输出到控制台）" % (role, exc))
        return None
    handler.setFormatter(fmt)
    handler.addFilter(rid_filter)
    handler.setLevel(logging.DEBUG)
    return handler


def attach(role: str) -> logging.Logger:
    """Wire THIS process' root logger to logs/<role>.log plus the console. Idempotent."""
    root = logging.getLogger()
    if getattr(root, "_auk_role", None) == role:
        return logging.getLogger("auk")
    rid_filter = _RidFilter()
    fmt = logging.Formatter(_FORMAT, _DATE_FORMAT)
    handlers = []
    file_handler = _file_handler(role, fmt, rid_filter)
    if file_handler is not None:
        handlers.append(file_handler)
    try:
        console = logging.StreamHandler(stream=_console_stream())
        console.setFormatter(fmt)
        console.addFilter(rid_filter)
        # The console is the user's launcher window: keep it readable, let the file hold
        # the DEBUG detail.
        console.setLevel(max(level_from_env(), logging.INFO))
        handlers.append(console)
    except Exception:
        pass
    root.handlers[:] = handlers
    root.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    # uvicorn installs its own handlers with propagate=False during its dictConfig, which
    # runs *before* the app's startup event — re-point them so server.log actually sees
    # startup, shutdown and "Exception in ASGI application". uvicorn.access is replaced by
    # our own request middleware (which carries the rid), so it goes to WARNING.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lib = logging.getLogger(name)
        lib.handlers = []
        lib.propagate = True
    lib_levels = {"uvicorn.access": logging.WARNING, "watchfiles": logging.WARNING,
                  "transformers": logging.WARNING, "accelerate": logging.WARNING,
                  "httpx": logging.WARNING, "filelock": logging.WARNING}
    if role != "worker":
        # torch/numba are chatty in the worker only; the server process never imports torch.
        lib_levels["torch"] = logging.WARNING
    for name, lvl in lib_levels.items():
        logging.getLogger(name).setLevel(lvl)
    root._auk_role = role
    return logging.getLogger("auk")


def tail(path, lines: int = 20) -> str:
    """Last N lines of a log file — so a parent process can report *why* its child died
    without the user opening a second file."""
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
            rows = handle.readlines()
    except Exception as exc:
        return "(读不到 %s：%s)" % (path, exc)
    return "".join(rows[-max(1, int(lines)):])


# ------------------------------------------------------------------ resource readings
class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_line() -> str:
    """ram used/total and the system COMMIT charge — the number behind WinError 1455.

    Commit (pagefile-backed reservation) is the budget that actually kills a load here,
    not physical RAM: Windows lets you commit past the pagefile and only fails at the
    moment of allocation.
    """
    if os.name != "nt":
        return "mem=non-windows"
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return "mem=?"
        return ("ram %.1f/%.1fGB  commit %.1f/%.1fGB"
                % ((status.ullTotalPhys - status.ullAvailPhys) / _GB, status.ullTotalPhys / _GB,
                   (status.ullTotalPageFile - status.ullAvailPageFile) / _GB,
                   status.ullTotalPageFile / _GB))
    except Exception:
        return "mem=?"


def proc_line() -> str:
    """This process' working set and private (pagefile-backed) bytes."""
    try:
        import psutil

        info = psutil.Process().memory_full_info()
        return "pid %d  rss %.2fGB  private %.2fGB" % (
            os.getpid(), info.rss / _GB, info.pagefile / _GB)
    except Exception:
        return "pid %d  rss=?  private=?" % os.getpid()


def gpu_line(timeout: float = 6.0) -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout).stdout.strip()
        if not out:
            return "gpu=?"
        parts = [p.strip() for p in out.split(",")]
        return "gpu %s  %.2f/%.2fGB" % (parts[0], float(parts[1]) / 1024.0, float(parts[2]) / 1024.0)
    except Exception:
        return "gpu=?"


def resource_line(with_gpu: bool = False) -> str:
    line = "%s | %s" % (memory_line(), proc_line())
    return line + (" | " + gpu_line() if with_gpu else "")


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "-"


def env_fingerprint(extra: dict | None = None) -> str:
    """Multi-line snapshot logged once per process. This is what makes a remote report
    diagnosable: 'their machine has 16GB and a 15GB commit ceiling' has to be readable
    out of the log, not asked about afterwards."""
    rows = [
        "python %s  %s %s  pid %d" % (platform.python_version(), platform.system(),
                                      platform.release(), os.getpid()),
        "root=%s  log_dir=%s  cwd=%s" % (ROOT, log_dir(), os.getcwd()),
        "fastapi %s  uvicorn %s  pydantic %s" % (_package_version("fastapi"),
                                                _package_version("uvicorn"),
                                                _package_version("pydantic")),
        # importlib.metadata reads the dist-info, so this costs no import and still proves
        # whether the model stack (and psutil, which is only a transitive dep) exists here.
        "torch %s  torchaudio %s  transformers %s  accelerate %s  soundfile %s  psutil %s"
        % (_package_version("torch"), _package_version("torchaudio"),
           _package_version("transformers"), _package_version("accelerate"),
           _package_version("soundfile"), _package_version("psutil")),
        resource_line(with_gpu=True),
        "log_level=%s" % logging.getLevelName(level_from_env()),
        "env " + ("  ".join("%s=%s" % (k, v) for k, v in sorted(os.environ.items())
                             if k.startswith("AUK_")) or "(无 AUK_* 变量)"),
    ]
    for key, value in sorted((extra or {}).items()):
        rows.append("%s=%s" % (key, value))
    return "\n  | " + "\n  | ".join(rows)


def debug_on() -> bool:
    return logging.getLogger("auk").isEnabledFor(logging.DEBUG)


def elapsed(seconds: float) -> str:
    return "%.1fs" % seconds if seconds < 120 else "%.1fmin" % (seconds / 60.0)


def stamp() -> float:
    return time.perf_counter()
