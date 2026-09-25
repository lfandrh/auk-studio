"""AuK Studio launcher.

Two failure modes this exists to prevent, both observed on this box on 2026-09-24:
  * two servers listening at once (`0.0.0.0:8080` + `127.0.0.1:8080` can coexist on
    Windows, and localhost traffic only reaches the more specific one, so the second is
    invisible while its model worker eats ~18GB of committed RAM);
  * a model worker outliving the server, because nothing in Python runs when the process is
    killed outright -- console X and `taskkill /F` skip atexit and daemon-process cleanup.
"""
import argparse
import ctypes
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
MIN_FREE_COMMIT_GB = 18.0   # one paged load commits ~16.9GB of weights (measured 2026-09-24)
_KILL_ON_JOB_CLOSE = 0x2000
_job_handle = None          # must stay referenced: closing the last handle is the kill
logger = None               # set by _start_logging(); None until then


def _start_logging():
    """Wire the launcher to logs/launcher.log and mirror everything to the console.

    The launcher's own window is its diagnostic surface, and that window is the first thing
    to disappear (X button, or a crash), so every decision it prints has to be on disk too.
    One call to ``say`` writes both. When the server starts in this same process, the app's
    startup hook re-attaches to logs/server.log, so the two phases stay in separate files.
    """
    global logger
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from backend import logging_setup as logkit
    except Exception as exc:
        print("[警告] 日志模块不可用（%s: %s），本次只输出到控制台。" % (type(exc).__name__, exc))
        return None
    logger = logkit.attach("launcher")
    logger = logging.getLogger("auk.launcher")   # same handlers, clearer name in the file
    logger.info("启动器开始 %s", logkit.env_fingerprint())
    print("  日志目录: %s（launcher.log=启动器 / server.log=服务 / worker.log=模型进程）"
          % logkit.log_dir())
    return logger


def say(msg, *args):
    """Print to the console AND keep the line: same call, both sinks."""
    if logger is None:
        print(msg % args if args else msg)
        return
    logger.info(msg, *args)


def dbg(msg, *args):
    if logger is not None:
        logger.debug(msg, *args)


def _console_safe():
    """CJK is written through WriteConsoleW on a real console, but a redirected pipe falls
    back to the ANSI codepage and raises on anything it cannot encode. Line buffering matters
    too: block-buffered stdout hides the replace/warning lines from a redirected log until the
    process exits, and the server never exits."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def _die_with_children():
    """Put this process in a Job Object the kernel empties when it dies, however it dies.
    Members created afterwards (uvicorn's loop, the spawn model worker) inherit membership."""
    global _job_handle
    if os.name != "nt":
        return
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_errno=True)
    # Explicit typing matters: a HANDLE truncated to c_int is not a valid job handle, and
    # AssignProcessToJobObject then fails for a reason GetLastError will not explain.
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_uint32]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.GetCurrentProcess.argtypes = []
    k32.GetCurrentProcess.restype = wintypes.HANDLE

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32)]

    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    info = Extended()
    info.Basic.LimitFlags = _KILL_ON_JOB_CLOSE
    handle = k32.CreateJobObjectW(None, None)
    if not handle:
        _job_warning(ctypes.get_last_error()); return
    if not k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        err = ctypes.get_last_error(); k32.CloseHandle(handle); _job_warning(err); return
    if not k32.AssignProcessToJobObject(handle, k32.GetCurrentProcess()):
        err = ctypes.get_last_error(); k32.CloseHandle(handle); _job_warning(err); return
    _job_handle = handle


def _job_warning(err):
    # Already being inside a job that forbids nesting is the usual cause; Ctrl+C still
    # shuts the worker down through the app's shutdown hook, so warn instead of refusing.
    say("[提示] 无法创建进程作业对象 (WinError %s)，异常退出时模型子进程可能残留，"
        "可用 python run.py --stop 清理" % (err or "?"))


def _run(args):
    proc = subprocess.run(args, capture_output=True, text=True, timeout=30, errors="replace")
    return proc.stdout or ""


def _listeners(port):
    """[(address, pid)] currently LISTENING on `port`. A bind()/connect() probe cannot answer
    this: binding succeeds even when a wildcard listener already shadows the port.

    No `-p tcp` filter: that protocol family selector hides the IPv6 table, so a dual-stack
    `[::]:port` listener -- which does shadow 127.0.0.1 on Windows -- would read as "port free".
    Those rows are still labelled TCP (with `[::]` addresses); startswith() keeps both that
    spelling and the "TCPv6" one some builds use, while never matching UDP.
    """
    try:
        out = _run(["netstat", "-ano"])
    except (OSError, subprocess.SubprocessError) as exc:
        dbg("netstat 失败，按“端口空闲”处理: %s", exc)
        return []
    found = []
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 5 and f[0].upper().startswith("TCP") and f[3].upper() == "LISTENING" \
                and f[1].rsplit(":", 1)[-1] == str(port):
            try:
                found.append((f[1], int(f[4])))
            except ValueError:
                continue
    return found


def _image(pid):
    for line in _run(["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"]).splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2 and parts[1] == str(pid):
            return parts[0]
    return "?"


def _free_commit_gb():
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]

    st = MemoryStatusEx()
    st.dwLength = ctypes.sizeof(st)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st.ullAvailPageFile / 2 ** 30


def _health(port):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/api/health" % port, timeout=3) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as exc:
        # Normal when the port is free; only interesting together with a live listener.
        dbg("健康探测 %d 无应答: %s: %s", port, type(exc).__name__, exc)
        return None


def _code_is_newer(port):
    """True when backend/app.py changed after the running instance imported it, i.e. the
    live server would answer with behaviour that no longer exists on disk. An instance that
    predates the code_mtime field is itself the oldest code imaginable, so it counts."""
    live = _health(port)
    if not live:
        return None
    if "code_mtime" not in live:
        return True
    try:
        return os.path.getmtime(os.path.join(ROOT, "backend", "app.py")) > live["code_mtime"] + 1
    except OSError as exc:
        dbg("读不到 backend/app.py 的 mtime: %s", exc)
        return None


def _open_when_ready(url, probe_host, port):
    def wait():
        deadline = time.time() + 120
        while time.time() < deadline:
            with socket.socket() as sock:
                sock.settimeout(0.4)
                if sock.connect_ex((probe_host, port)) == 0:
                    break
            time.sleep(0.25)
        else:
            return
        try:
            webbrowser.open(url)
        except Exception:
            print("请手动打开 %s" % url)
    threading.Thread(target=wait, daemon=True).start()


def _stop(port, proven=False):
    """Kill every listener on the port together with its whole process tree (the model
    worker is a child, so /T reaches it), then any spawn worker left parentless.

    The start path refuses to kill a process it cannot prove is AuK, and --stop is where that
    promise is easiest to break: the port is the only lookup key, so without the proof a wrong
    PID here is someone else's database or editor. `proven` lets the caller pass on a verdict
    it already reached (a live /api/health response beats a command-line match).
    """
    killed = []
    say("--stop：清理端口 %d 上的实例及其模型子进程" % port)
    listeners = _listeners(port)
    if not proven and listeners and not _owned_by_auk({p for _a, p in listeners}):
        desc = ", ".join("%s PID %d %s" % (a, p, _image(p)) for a, p in listeners)
        say("[错误] 端口 %d 被无法证明是 AuK 的进程占着（%s），不去结束它，也不清理模型子进程。\n"
            "       先看一眼再决定：tasklist /FI \"PID eq <pid>\" /V，确认后再手动 "
            "taskkill /F /T /PID <pid>。" % (port, desc))
        return 1
    for addr, pid in listeners:
        if pid in killed:
            continue
        say("结束 %s  (PID %d, %s) 及其子进程" % (addr, pid, _image(pid)))
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True, errors="replace")
        killed.append(pid)
    orphans = _parentless_workers()
    for pid, cmd in orphans:
        say("发现无父进程的模型子进程 PID %d，一并结束\n       命令行：%s" % (pid, cmd[:200]))
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        killed.append(pid)
    if not killed:
        say("端口 %d 上没有正在监听的 AuK 服务，也没有残留的模型子进程。" % port)
        return 0
    for _ in range(20):
        if not _listeners(port):
            say("完成，端口 %d 已释放。" % port)
            return 0
        time.sleep(0.3)
    say("[注意] 仍有进程在监听 %s：%s —— 请手动结束这些 PID。", port, _listeners(port))
    return 1


def _python_procs():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" "
          "| Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    try:
        raw = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps])
        rows = json.loads(raw.strip() or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return rows if isinstance(rows, list) else [rows]


def _parentless_workers():
    """multiprocessing spawn children whose parent is gone -- the shape an orphaned model
    worker has when the server died without cleanup. Requires a process snapshot, so this
    only runs from --stop."""
    rows = _python_procs()
    live = {int(r["ProcessId"]) for r in rows if r.get("ProcessId")}
    out = []
    for r in rows:
        cmd = r.get("CommandLine") or ""
        ppid = int(r.get("ParentProcessId") or 0)
        if "multiprocessing" in cmd and "spawn_main" in cmd and ppid not in live and ppid != 4:
            out.append((int(r["ProcessId"]), cmd))
    return out


def _owned_by_auk(pids):
    """Whether these listener PIDs belong to an AuK Studio server. Kills are not reversible,
    so an unrecognised process on the port is reported instead of terminated."""
    marks = ("run.py", "backend.app", "uvicorn")
    for r in _python_procs():
        cmd = r.get("CommandLine") or ""
        if int(r.get("ProcessId") or 0) in pids and any(m in cmd for m in marks):
            return True
    return False


def main(argv=None):
    _console_safe()
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description="AuK Studio 启动器")
    parser.add_argument("--stop", action="store_true",
                        help="结束占用该端口的服务及其模型子进程，用于清理卡住/残留的实例")
    parser.add_argument("--no-browser", action="store_true", help="就绪后不自动打开页面")
    opts, _unknown = parser.parse_known_args(argv)

    host = os.getenv("AUK_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("AUK_PORT", "8080"))
    except ValueError as exc:
        print("[错误] AUK_PORT=%r 不是端口号，按 8080 处理。" % os.getenv("AUK_PORT"))
        say("AUK_PORT 无法解析（%s: %s），回退 8080" % (type(exc).__name__, exc))
        port = 8080
    _start_logging()      # before any decision worth keeping a record of
    say("参数：host=%s port=%d stop=%s no_browser=%s", host, port, opts.stop, opts.no_browser)
    if opts.stop:
        return _stop(port)

    _die_with_children()
    url = "http://127.0.0.1:%d/" % port
    taken = _listeners(port)
    if taken:
        addr, pid = taken[0]
        live = _health(port)
        is_auk = bool(live and live.get("service") == "online" and "models" in live) \
            or _owned_by_auk({p for _a, p in taken})
        if not is_auk:
            say("[错误] 端口 %d 被其它进程占着（%s, PID %d, %s），不会去结束它。\n"
                "       想并行运行请换端口：set AUK_PORT=8081 后重新启动，或先手动结束该进程。"
                % (port, addr, pid, _image(pid)))
            return 1
        # Starting again means "give me a working server on the current code", so the old
        # instance is replaced rather than reused -- including mid-generation, on purpose.
        say("端口 %d 上已有 AuK 服务（PID %d），结束它并启动当前代码的新实例%s" % (
            port, pid, "（原实例跑的是被改之前的 backend/app.py）"
            if _code_is_newer(port) else ""))
        if (live or {}).get("loaded_model"):
            say("  原实例已加载 %s，新实例首次使用还要重新加载模型（约 20s）"
                  % live["loaded_model"])
        if _stop(port, proven=is_auk):
            return 1

    free = _free_commit_gb()
    if free is not None and free < MIN_FREE_COMMIT_GB:
        say("[警告] 可用提交内存(commit)仅 %.1f GB，加载一次模型约需 17GB，"
            "可能仍有未退出的模型进程。若加载报 WinError 1455，先运行 python run.py --stop。"
            % free)
    say("AuK Studio  http://%s:%d/  (Ctrl+C 停止；异常退出时模型子进程会一并结束)" % (host, port))
    if not opts.no_browser and os.getenv("AUK_OPEN_BROWSER", "1") != "0":
        _open_when_ready(url, "127.0.0.1" if host in ("0.0.0.0", "::", "") else host, port)

    import uvicorn
    say("进入 uvicorn：host=%s port=%s reload=False（改后端代码需重启才生效）", host, port)
    try:
        uvicorn.run("backend.app:app", host=host, port=port, reload=False)
    except (KeyboardInterrupt, SystemExit):
        say("服务已停止（用户中断 / 正常退出）")
        return 0
    except BaseException:
        # The window closes on the next line, so this is the only copy of why.
        if logger is not None:
            logger.exception("uvicorn 退出：服务进程异常结束")
        raise
    say("uvicorn 正常返回，服务结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
