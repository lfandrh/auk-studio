from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
FRONTEND_DIR = ROOT / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Uploaded clips stay on disk so the page can rebuild itself after a refresh
# (A/B audition + re-run with the same input without re-uploading). Names are
# always uuid4().hex plus a whitelisted suffix, so a strict match here is also
# the path-traversal guard for /api/input/{name}.
_UPLOAD_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}
_UPLOAD_MEDIA_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".webm": "audio/webm",
}
_UPLOAD_NAME_RE = re.compile(r"^[0-9a-f]{32}\.(?:wav|mp3|flac|m4a|ogg|webm)$")
_UPLOAD_TTL_SEC = 7 * 24 * 3600


def _resolve_upload(name: str) -> Path:
    name = (name or "").strip()
    if not _UPLOAD_NAME_RE.match(name):
        raise HTTPException(400, "invalid input_name")
    path = UPLOAD_DIR / name
    if not path.is_file():
        raise HTTPException(404, "Uploaded audio not found — uploads are kept for 7 days")
    return path


def prune_uploads(now: Optional[float] = None) -> int:
    cutoff = (now if now is not None else time.time()) - _UPLOAD_TTL_SEC
    removed = 0
    for path in UPLOAD_DIR.glob("*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


_last_prune = 0.0


def _prune_uploads_if_stale():
    global _last_prune
    if time.time() - _last_prune < 600:
        return
    _last_prune = time.time()
    prune_uploads()


prune_uploads()

# Emotion is a separate AuK paralinguistic-edit pass applied to generated
# speech. AuK supports exactly these 8 emotions (pe.config.yaml emotion_edit);
# anything else is rejected by the official PromptEnhancer as unsupported.
_EMOTION_INSTRUCTIONS = {e: f"Change the emotion to {e}." for e in (
    "happy", "angry", "sad", "fearful", "surprised", "disgusted", "calm", "excited",
)}
# pe.config.yaml emotion_edit.duration: equal_length x per-emotion coefficient.
# Plain equal length truncates sad/fearful delivery, which the reference scales longer.
_EMOTION_DURATION_COEF = {
    "sad": 1.22, "fearful": 1.16,
    "happy": 1.06, "angry": 1.06, "surprised": 1.06,
    "disgusted": 1.06, "calm": 1.06, "excited": 1.06,
}

# infer_gradio LOUDNESS_LIMIT_TASK_TYPES: extracted vocals otherwise come back much
# hotter than the source mix. Attenuate toward -14 LUFS, then clamp the peak to 0.95.
# pyloudnorm is an optional official dependency (gradio itself wraps it in try/except),
# so without it we degrade to peak-only instead of failing.
_LOUDNESS_LIMIT_TASKS = frozenset({"extract_vocals", "vocal_edit"})
_KNOWN_TASKS = _LOUDNESS_LIMIT_TASKS | {
    "content_edit", "emotion_edit", "enhance_speech", "separate_speech",
    "improve_quality", "instruct_tts", "zero_shot_tts", "speed_edit",
    "volume_edit", "pitch_edit", "voice_edit", "nonverbal_edit",
    "whisper_edit", "accent_edit",
}
_VOCAL_TARGET_LUFS = -14.0
_VOCAL_PEAK_CEILING = 0.95


def _limit_task_output(waveform, task, sample_rate: int):
    if str(task or "") not in _LOUDNESS_LIMIT_TASKS:
        return waveform
    return _limit_vocal_output_loudness(waveform, sample_rate)


def _limit_vocal_output_loudness(waveform, sample_rate: int):
    import math

    out = waveform.detach().double().cpu()
    if out.ndim == 2:
        out = out.mean(dim=0) if out.shape[0] <= out.shape[1] else out.mean(dim=1)
    try:
        import pyloudnorm as pyln

        measured = float(pyln.Meter(int(sample_rate)).integrated_loudness(out.numpy()))
        if math.isfinite(measured) and measured > _VOCAL_TARGET_LUFS:
            out = out * min(1.0, 10.0 ** ((_VOCAL_TARGET_LUFS - measured) / 20.0))
    except Exception:
        pass
    if out.numel():
        peak = float(out.abs().max())
        if math.isfinite(peak) and peak > _VOCAL_PEAK_CEILING:
            out = out * (_VOCAL_PEAK_CEILING / peak)
    return out

app = FastAPI(title="AuK Studio API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def model_settings(model: str):
    if model == "flash":
        return (
            env_path("AUK_FLASH_CONFIG", str(ROOT / "ckpts/AuK-Flash/config.yaml")),
            env_path("AUK_FLASH_CKPT", str(ROOT / "ckpts/AuK-Flash/auk_flash.safetensors")),
        )
    if model == "base":
        return (
            env_path("AUK_BASE_CONFIG", str(ROOT / "ckpts/AuK/config.yaml")),
            env_path("AUK_BASE_CKPT", str(ROOT / "ckpts/AuK/auk_base.safetensors")),
        )
    raise ValueError("model must be base or flash")


def _model_worker(conn, root: str):
    """Isolated AuK process. A native CUDA/safetensors crash here must not kill FastAPI."""
    engine = None
    loaded_model = None
    loaded_resident = False
    try:
        import torch  # imported only inside the worker
        from auk.infer.infer_auk import AukInfer, save_audio
    except Exception as exc:
        conn.send({"ok": False, "error": f"AuK import failed: {type(exc).__name__}: {exc}"})
        conn.close()
        return

    def cleanup():
        nonlocal engine, loaded_model, loaded_resident
        engine = None
        loaded_model = None
        loaded_resident = False
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    try:
        while True:
            cmd = conn.recv()
            op = cmd.get("op")
            if op == "ping":
                conn.send({"ok": True, "loaded_model": loaded_model})
                continue
            if op == "load":
                model = cmd["model"]
                resident = bool(cmd.get("resident", False))
                config, ckpt = _worker_model_settings(root, model)
                if not config.exists():
                    conn.send({"ok": False, "error": f"AuK config not found: {config}"})
                    continue
                if not ckpt.exists():
                    conn.send({"ok": False, "error": f"AuK checkpoint not found: {ckpt}"})
                    continue
                if loaded_model == model and engine is not None and loaded_resident == resident:
                    conn.send({"ok": True, "loaded_model": loaded_model, "resident": loaded_resident, "already_loaded": True})
                    continue
                cleanup()


                # 强制使用 GPU
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA 不可用，AuK 不会使用 CPU。请检查 PyTorch CUDA 环境。"
                    )

                print(f"[AuK Worker] CUDA: {torch.version.cuda}")
                print(f"[AuK Worker] GPU: {torch.cuda.get_device_name(0)}")

                engine = AukInfer(
                    str(config),
                    str(ckpt),
                    device="cuda:0",
                    dtype="bf16",
                    cpu_offload=not resident,
                )
                print("=== Model dtype check ===")

                for name, module in engine.__dict__.items():
                    if hasattr(module, "parameters"):
                        try:
                            p = next(module.parameters())
                            print(name, p.device, p.dtype)
                        except:
                            pass

                loaded_model = model
                loaded_resident = resident

                print(f"[AuK Worker] Model loaded on GPU: {model}")
                print(
                    f"[AuK Worker] VRAM allocated: "
                    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                )
                conn.send({"ok": True, "loaded_model": loaded_model, "resident": loaded_resident, "already_loaded": False,
                           "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2)})
                continue
            if op == "unload":
                cleanup()
                conn.send({"ok": True, "loaded_model": None})
                continue
            if op == "generate":
                if engine is None or loaded_model != cmd["model"]:
                    conn.send({"ok": False, "error": "Requested model is not loaded"})
                    continue
                messages = cmd["messages"]
                emotion = str(cmd.get("emotion") or "").strip().lower()
                output_path = Path(cmd["output_path"])
                # gen_seconds <= 0 means "match the input audio length" (AuK's
                # equal_length strategy for every editing task, mirroring gradio's
                # get_gen_duration which returns the ref seconds when no target is set).
                _gs = float(cmd.get("gen_seconds") or 0)
                _seed_raw = str(cmd.get("seed") or "").strip()
                _seed = int(_seed_raw) if _seed_raw.lstrip("-").isdigit() else None
                audio_out, sample_rate = engine.generate(
                    messages, gen_seconds=(_gs if _gs > 0 else None), seed=_seed)
                # AuK has no single-pass "clone + emotion" instruction: emotion is a
                # separate paralinguistic-edit task run on already-generated speech
                # (see infer_gradio emotion examples). So clone first, then re-render.
                if emotion in _EMOTION_INSTRUCTIONS:
                    import tempfile

                    stage1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    stage1.close()
                    try:
                        save_audio(audio_out, sample_rate, stage1.name)
                        emo_messages = [{"role": "user", "content": [
                            {"type": "text", "text": _EMOTION_INSTRUCTIONS[emotion]},
                            {"type": "audio", "audio": stage1.name},
                        ]}]
                        _stage1_sec = audio_out.shape[-1] / float(sample_rate)
                        audio_out, sample_rate = engine.generate(
                            emo_messages, audio=stage1.name, seed=_seed,
                            gen_seconds=_stage1_sec * _EMOTION_DURATION_COEF.get(emotion, 1.06))
                    finally:
                        try:
                            os.unlink(stage1.name)
                        except OSError:
                            pass
                save_audio(_limit_task_output(audio_out, cmd.get("task"), sample_rate), sample_rate, str(output_path))
                conn.send({"ok": True, "sample_rate": int(sample_rate),
                           "out_frames": int(audio_out.shape[-1]),
                           "output_path": str(output_path)})
                continue
            if op == "shutdown":
                cleanup()
                conn.send({"ok": True})
                break
            conn.send({"ok": False, "error": f"Unknown worker operation: {op}"})
    except (EOFError, BrokenPipeError, OSError):
        pass
    except Exception as exc:
        # Python-level inference error: return it. Native crashes cannot reach here;
        # the parent detects the dead worker and starts a fresh one on the next request.
        try:
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _worker_model_settings(root: str, model: str):
    r = Path(root)
    if model == "flash":
        return Path(os.getenv("AUK_FLASH_CONFIG", str(r / "ckpts/AuK-Flash/config.yaml"))).expanduser().resolve(), Path(os.getenv("AUK_FLASH_CKPT", str(r / "ckpts/AuK-Flash/auk_flash.safetensors"))).expanduser().resolve()
    return Path(os.getenv("AUK_BASE_CONFIG", str(r / "ckpts/AuK/config.yaml"))).expanduser().resolve(), Path(os.getenv("AUK_BASE_CKPT", str(r / "ckpts/AuK/auk_base.safetensors"))).expanduser().resolve()


class ModelSupervisor:
    """Keeps FastAPI independent from the native AuK/CUDA process."""
    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self.conn = None
        self.loaded_model = None
        self.last_error = None
        self.generation_busy = False
        self.ctx = mp.get_context("spawn")
        self.timeout = float(os.getenv("AUK_MODEL_TIMEOUT", "600"))
        # Desired VRAM strategy, chosen in the UI (模型与设置 → 显存策略).
        self.resident = os.getenv("AUK_RESIDENT", "").strip().lower() in {"1", "true", "yes", "on"}
        self.loaded_resident = False
        self.loaded_vram_gb = None

    def _start(self):
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=_model_worker, args=(child, str(ROOT)), daemon=True)
        proc.start()
        child.close()
        self.proc, self.conn = proc, parent
        self.loaded_model = None
        self.loaded_resident = False
        self.loaded_vram_gb = None
        self.last_error = None

    def _stop(self, force=False):
        conn, proc = self.conn, self.proc
        self.conn = None
        self.proc = None
        self.loaded_model = None
        self.loaded_resident = False
        self.loaded_vram_gb = None
        if conn:
            if not force:
                try:
                    conn.send({"op": "shutdown"})
                    if conn.poll(2):
                        conn.recv()
                except Exception:
                    pass
            try:
                conn.close()
            except Exception:
                pass
        if proc:
            proc.join(2)
            if proc.is_alive():
                proc.terminate()
                proc.join(3)

    def _ensure_process(self):
        if self.proc is None or not self.proc.is_alive() or self.conn is None:
            self._stop(force=True)
            self._start()

    def _request(self, payload, timeout=None):
        timeout = self.timeout if timeout is None else timeout
        self._ensure_process()
        try:
            self.conn.send(payload)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.conn.poll(0.25):
                    return self.conn.recv()
                if not self.proc.is_alive():
                    raise RuntimeError("AuK worker exited unexpectedly (possible CUDA/driver/model crash)")
            raise TimeoutError(f"AuK worker timed out after {timeout:.0f}s")
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise RuntimeError("AuK worker exited unexpectedly (possible CUDA/driver/model crash)") from exc

    def load(self, model, resident=None):
        if model not in {"base", "flash"}:
            raise ValueError("model must be base or flash")
        if resident is None:
            resident = self.resident
        resident = bool(resident)
        with self.lock:
            config, ckpt = model_settings(model)
            if not config.exists():
                raise FileNotFoundError(f"AuK config not found: {config}")
            if not ckpt.exists():
                raise FileNotFoundError(f"AuK checkpoint not found: {ckpt}")
            if self.loaded_model == model and self.loaded_resident == resident and self.proc and self.proc.is_alive():
                return {"ok": True, "loaded_model": model, "resident": resident, "already_loaded": True}
            # Same model but a different VRAM strategy still needs a real reload:
            # paging on/off is baked into the module placement at load time.
            if self.loaded_model and (self.loaded_model != model or self.loaded_resident != resident):
                self._stop(force=False)
            self.resident = resident
            result = self._request({"op": "load", "model": model, "resident": resident}, timeout=self.timeout)
            if not result.get("ok"):
                self.last_error = result.get("error")
                self._stop(force=True)
                raise RuntimeError(self.last_error or "AuK model load failed")
            self.loaded_model = model
            self.loaded_resident = resident
            self.loaded_vram_gb = result.get("vram_allocated_gb")
            return result

    def unload(self):
        with self.lock:
            if self.proc is None:
                self.loaded_model = None
                return {"ok": True, "loaded_model": None}
            try:
                result = self._request({"op": "unload"}, timeout=30)
            except Exception:
                self._stop(force=True)
                return {"ok": True, "loaded_model": None, "forced": True}
            self._stop(force=False)
            return result

    def generate(self, model, messages, seconds, output_path, emotion="", seed="", task=""):
        with self.lock:
            if self.generation_busy:
                raise RuntimeError("Another generation is already running")
            self.generation_busy = True
            try:
                self.load(model)
                # two-pass (clone + emotion) roughly doubles inference time;
                # seconds == 0 (auto/match-input) has no known length, so assume 60s.
                budget = (seconds if seconds > 0 else 60) * (40 if emotion else 20) + 120
                result = self._request({"op": "generate", "model": model, "messages": messages, "gen_seconds": seconds, "output_path": str(output_path), "emotion": emotion, "seed": seed, "task": task}, timeout=max(self.timeout, budget))
                if not result.get("ok"):
                    self.last_error = result.get("error")
                    raise RuntimeError(self.last_error or "AuK inference failed")
                return result
            except (RuntimeError, TimeoutError) as exc:
                self.last_error = str(exc)
                # If the worker crashed, kill any broken process. The next request will spawn a fresh one.
                if self.proc is None or not self.proc.is_alive() or isinstance(exc, TimeoutError):
                    self._stop(force=True)
                raise
            finally:
                self.generation_busy = False

    def status(self):
        with self.lock:
            alive = bool(self.proc and self.proc.is_alive())
            return {"worker_alive": alive, "loaded_model": self.loaded_model if alive else None,
                    "resident": self.loaded_resident if alive else False,
                    "vram_allocated_gb": self.loaded_vram_gb if alive else None,
                    "busy": self.generation_busy, "last_error": self.last_error}


SUPERVISOR = ModelSupervisor()

try:  # prime psutil's delta window so the first header reading is not a bogus 0%
    import psutil as _psutil_prime

    _psutil_prime.cpu_percent(interval=None)
except Exception:
    pass

_GPU_STATIC = None
_GPU_SNAP = (0.0, None)  # (monotonic stamp, snapshot) — nvidia-smi costs ~0.2s on Windows
_GPU_SNAP_TTL = 1.5


def _nvidia_smi(query):
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8,
    )
    lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
    return lines[0] if lines else None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _gpu_snapshot():
    """Driver-reported, one subprocess call shared by /api/health and /api/system.

    The driver is the only trustworthy source here: under Windows WDDM an 8GB card
    will "successfully" let torch allocate 11GB by spilling into system RAM, so
    torch.cuda.memory_* over-reports exactly when residency is impossible.
    """
    global _GPU_SNAP
    now = time.monotonic()
    if _GPU_SNAP[1] is not None and now - _GPU_SNAP[0] < _GPU_SNAP_TTL:
        return _GPU_SNAP[1]
    snap = {"name": None, "vram_gb": None, "used_gb": None, "util": None}
    try:
        row = _nvidia_smi("name,utilization.gpu,memory.used,memory.total")
        if row:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 4:
                util, used, total = _f(parts[1]), _f(parts[2]), _f(parts[3])
                snap.update({
                    "name": parts[0],
                    "vram_gb": round(total / 1024, 1) if total is not None else None,
                    "used_gb": round(used / 1024, 2) if used is not None else None,
                    "util": util,
                })
    except Exception:
        pass
    _GPU_SNAP = (now, snap)
    return snap


def gpu_info():
    """Total VRAM only — cached for the process lifetime; the 显存策略 badges need
    it before any model worker exists."""
    global _GPU_STATIC
    if _GPU_STATIC is None:
        snap = _gpu_snapshot()
        _GPU_STATIC = {"name": snap["name"], "vram_gb": snap["vram_gb"]}
    return dict(_GPU_STATIC)


def gpu_used_gb():
    return _gpu_snapshot()["used_gb"]


# bf16 weights ≈ 12.8GB (Qwen 8.1 + DiT 3.1 + fp32 VAE 0.6) + CUDA context and
# activation headroom. Below ~14GB the weights alone don't fit; 30GB is the
# threshold where residency needs no thought at all.
_RESIDENT_MIN_GB = 14.0
_RESIDENT_SAFE_GB = 30.0


def resident_verdict(vram_gb):
    if vram_gb is None:
        return "unknown"
    if vram_gb < _RESIDENT_MIN_GB:
        return "impossible"
    if vram_gb < _RESIDENT_SAFE_GB:
        return "tight"
    return "safe"


async def save_upload(file: UploadFile) -> Path:
    _prune_uploads_if_stale()
    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(400, "Unsupported audio format")
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio file")
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(413, "Audio file is too large (max 100 MB)")
    path.write_bytes(data)
    return path


def api_error(exc: Exception):
    msg = str(exc)
    lower = msg.lower()
    if "checkpoint not found" in lower or "config not found" in lower:
        return HTTPException(404, msg)
    if "out of memory" in lower or "cudnn_status_failure" in lower:
        return HTTPException(503, "显存不足：当前显卡放不下全常驻的权重（bf16 约 12.8GB）。请在「模型与设置 → 显存策略」切回分页加载。")
    if "worker exited" in lower or "access violation" in lower:
        return HTTPException(503, "AuK 模型进程异常退出，服务本身仍在运行。请再次点击加载模型或重新生成，系统会自动重新拉起模型进程。")
    if isinstance(exc, TimeoutError):
        return HTTPException(504, "AuK 模型响应超时，模型进程已重置。请稍后重试。")
    return HTTPException(500, msg)


@app.get("/api/health")
def health():
    s = SUPERVISOR.status()
    gpu = gpu_info()
    return {
        "ok": True,
        "service": "online",
        "worker_alive": s["worker_alive"],
        "loaded_model": s["loaded_model"],
        "resident": s["resident"],
        "vram_used_gb": gpu_used_gb() if s["worker_alive"] else None,
        "gpu": {**gpu, "used_gb": gpu_used_gb()},
        "resident_verdict": resident_verdict(gpu["vram_gb"]),
        "busy": s["busy"],
        "last_error": s["last_error"],
        "models": {
            "base": {"config": str(model_settings("base")[0]), "checkpoint": str(model_settings("base")[1]), "available": model_settings("base")[0].exists() and model_settings("base")[1].exists()},
            "flash": {"config": str(model_settings("flash")[0]), "checkpoint": str(model_settings("flash")[1]), "available": model_settings("flash")[0].exists() and model_settings("flash")[1].exists()},
        },
    }


def system_stats():
    """psutil is a transitive dep, not a declared one — the header readout degrades
    to CPU/内存 dashes instead of breaking the page when it is absent."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        cpu = round(psutil.cpu_percent(interval=None), 1)
        ram_used = round((vm.total - vm.available) / 1024**3, 1)
        ram_total = round(vm.total / 1024**3, 1)
    except Exception:
        cpu = ram_used = ram_total = None
    return {"cpu": cpu, "ram_used_gb": ram_used, "ram_total_gb": ram_total}


@app.get("/api/system")
def system():
    g = _gpu_snapshot()
    return {"ok": True, **system_stats(),
            "gpu": {"name": g["name"], "util": g["util"], "used_gb": g["used_gb"], "total_gb": g["vram_gb"]}}


@app.post("/api/model/load")
def load_model(model: str = Form(...), resident: str = Form("0")):
    try:
        return SUPERVISOR.load(model, resident=resident.strip().lower() in {"1", "true", "yes", "on"})
    except Exception as exc:
        raise api_error(exc)


@app.post("/api/model/unload")
def unload_model():
    try:
        return SUPERVISOR.unload()
    except Exception as exc:
        raise api_error(exc)


@app.get("/api/model/status")
def model_status():
    return SUPERVISOR.status()


@app.post("/api/generate")
async def generate(instruction: str = Form(...), model: str = Form("flash"), gen_seconds: float = Form(0), emotion: str = Form(""), seed: str = Form(""), task: str = Form(""), input_name: str = Form(""), audio: Optional[UploadFile] = File(None)):
    if not instruction.strip():
        raise HTTPException(400, "instruction is required")
    if model not in {"base", "flash"}:
        raise HTTPException(400, "model must be base or flash")
    if gen_seconds < 0 or gen_seconds > 120:
        raise HTTPException(400, "gen_seconds must be between 0 and 120")
    seed_raw = seed.strip()
    if seed_raw and not seed_raw.isdigit():
        raise HTTPException(400, "seed must be a non-negative integer")

    # input_name lets the page re-run against a clip it already uploaded (survives a
    # refresh, when the browser no longer holds the File). A fresh upload wins.
    audio_path = _resolve_upload(input_name) if input_name.strip() else None
    if audio_path is None:
        audio_path = await save_upload(audio) if audio else None
    # gen_seconds == 0 means "match the input audio length". That only works when
    # there IS an input clip; text-only instruct TTS has nothing to measure, so it
    # must carry an explicit positive duration.
    if gen_seconds == 0 and audio_path is None:
        raise HTTPException(400, "text-only generation needs gen_seconds > 0")
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction.strip()}]}]
    if audio_path:
        messages[0]["content"].append({"type": "audio", "audio": str(audio_path)})
    job_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{job_id}.wav"
    emo = emotion.strip().lower()
    emo = emo if emo in _EMOTION_INSTRUCTIONS else ""
    # Official gradio gates post-processing on task_type; the Web path has no PromptEnhancer
    # to classify it, so the page states it explicitly. Unknown names simply mean "no gating".
    task_name = task.strip().lower()
    task_name = task_name if task_name in _KNOWN_TASKS else ""
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(SUPERVISOR.generate, model, messages, float(gen_seconds), output_path, emo, seed_raw, task_name)
        elapsed = round(time.perf_counter() - started, 1)
        rate = int(result.get("sample_rate") or 0)
        frames = int(result.get("out_frames") or 0)
        return {"job_id": job_id, "model": model, "sample_rate": rate or None,
                "duration_sec": round(frames / rate, 2) if rate and frames else None,
                "target_sec": round(float(gen_seconds), 2) if gen_seconds > 0 else None,
                "elapsed_sec": elapsed, "seed": int(seed_raw) if seed_raw else None,
                "emotion": emo or None, "instruction": instruction.strip(), "task": task_name or None,
                "input_url": f"/api/input/{audio_path.name}" if audio_path else None,
                "audio_url": f"/api/audio/{job_id}"}
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        err = api_error(exc)
        raise err
    finally:
        if audio_path:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(400, "invalid job id")
    path = OUTPUT_DIR / f"{job_id}.wav"
    if not path.exists():
        raise HTTPException(404, "Audio result not found")
    return FileResponse(path, media_type="audio/wav", filename=f"auk-{job_id}.wav")


@app.get("/api/input/{name}")
def get_input(name: str):
    path = _resolve_upload(name)
    return FileResponse(path, media_type=_UPLOAD_MEDIA_TYPES[path.suffix])


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
