from __future__ import annotations

import asyncio
import json
import logging
import math
import multiprocessing as mp
import os
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import logging_setup as logkit

# Named after the file, not the role: this module is imported by both the server and the
# spawn worker, and the two write to different files by handler-per-process, not by name.
log = logging.getLogger("auk.app")

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
FRONTEND_DIR = ROOT / "frontend"
# The page's 51 demo cases play their reference clips from here. The library owns this
# directory (infer_gradio.py reads the same files); it is served under /demo-audio instead
# of being copied into frontend/, because two copies drift: the official one can be updated
# while the page keeps playing the stale one, and nothing notices.
DEMO_AUDIO_DIR = ROOT / "assets" / "demo-input-audio"
# When this file was last edited. The launcher compares it against the value the running
# server reports, so "an instance is already on the port" can be answered either as
# "fine, here is your page" or as "that instance is stale, stop it first".
CODE_MTIME = os.path.getmtime(__file__)
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
# The optional `_r<startMs>_<endMs>` tail is a trimmed slice of an upload (see
# _trimmed_name): kept as its own file so the page can audition exactly what the model
# heard, and reused verbatim when the same range is requested again.
_UPLOAD_NAME_RE = re.compile(r"^[0-9a-f]{32}(?:_r\d+_\d+)?\.(?:wav|mp3|flac|m4a|ogg|webm)$")
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
        except OSError as exc:
            # A clip held open by the browser's player is the normal case; the next prune
            # (every 10min, or on the next start) gets another shot at it.
            log.debug("清理上传文件跳过 %s: %s", path.name, exc)
            continue
    if removed:
        log.info("清理过期上传文件 %d 个（保留 %d 天）", removed, _UPLOAD_TTL_SEC // 86400)
    return removed


_last_prune = 0.0


def _prune_uploads_if_stale():
    global _last_prune
    if time.time() - _last_prune < 600:
        return
    _last_prune = time.time()
    prune_uploads()


# Pruning is deferred to a startup hook (see below) so a cold-start directory scan
# never delays the first request.

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

# ---------------------------------------------------------------- sequence budget
# The reference clip and the generated target share ONE latent sequence
# (infer_auk.py:226 total_latent_len = ref_latent_len + gen_latent_len) and training only
# ever saw 0.3-30s of it (train.py:164, and comfyui/nodes.py MAX_SEQUENCE_SECONDS=30).
# Nothing in the inference path enforces that ceiling, so an over-budget job comes back
# "successfully" and sounds wrong. Everything below exists to keep jobs inside the window.
_SEQ_BUDGET_SEC = 30.0
_SEQ_MARGIN_SEC = 2.5          # VAE/frame rounding plus boundary slop
_REF_MAX_SEC = 10.0            # longer clone references get trimmed, never fed in whole
# The model returns exactly the number of seconds it is asked for, so asking for less than
# the text needs can only be answered by speaking faster (measured 2.33x on a 666-char text
# squeezed into 60s). Headroom buys the opposite, gentler, failure.
_SEQ_HEADROOM = 1.08
_SEGMENT_GAP_SEC = 0.2         # silence between stitched segments (IndexTTS default)
_MIN_TAIL_SEC = 2.0            # a shorter last segment gets folded back when it still fits
_MODEL_FRAME_SEC = 0.02        # 24000 Hz / 480 downsample = 50 latent frames per second

_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
# Lookbehind keeps the delimiter attached to the clause it closes, so re-joining the pieces
# reproduces the original text exactly.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[。！？；?!;\n])")
_COMMA_SPLIT_RE = re.compile(r"(?<=[，、：,:])")


def spoken_seconds(text: str) -> float:
    """Official PromptEnhancer rule (pe.py:1770): 0.21s per CJK char, 0.30s per latin word."""
    return len(_CJK_CHAR_RE.findall(text)) * 0.21 + len(_LATIN_WORD_RE.findall(text)) * 0.30


def sequence_budget(ref_seconds: float, emotion: str = "") -> float:
    """How many seconds of *target* a single pass may ask for, given the reference."""
    usable = _SEQ_BUDGET_SEC - _SEQ_MARGIN_SEC - max(0.0, float(ref_seconds))
    if emotion in _EMOTION_INSTRUCTIONS:
        # The emotion pass re-uses stage-1 as its own reference and asks for
        # stage1 x coefficient (app.py _EMOTION_DURATION_COEF), so that pass carries
        # (1 + coef) x segment_length inside the sequence.
        coef = _EMOTION_DURATION_COEF.get(emotion, 1.06)
        usable = min(usable, (_SEQ_BUDGET_SEC - _SEQ_MARGIN_SEC) / (1.0 + coef))
    return max(0.3, usable)


def plan_segments(text: str, ref_seconds: float, emotion: str = "") -> dict:
    """Split `text` so every pass keeps ref+target inside the window. One segment when the
    text already fits, which is the common case and keeps today's behaviour untouched."""
    text = str(text or "").strip()
    cap = sequence_budget(ref_seconds, emotion)
    empty = {"segments": [], "count": 0, "cap_seconds": round(cap, 2),
             "ref_seconds": round(float(ref_seconds), 2), "total_seconds": 0.0,
             "gap_seconds": 0.0, "split": False}
    if not text:
        return empty

    pieces = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        if not clause:
            continue
        if not clause.strip():
            # Blank space between two delimiters carries no speech, but it must stay
            # attached to what came before or the pieces would no longer re-join to the
            # original text -- and losing characters silently is the last thing this
            # page should do.
            if pieces:
                pieces[-1] += clause
            continue
        if spoken_seconds(clause) <= cap:
            pieces.append(clause)
            continue
        for sub in (s for s in _COMMA_SPLIT_RE.split(clause) if s.strip()):
            if spoken_seconds(sub) <= cap:
                pieces.append(sub)
                continue
            # Nothing to cut on inside this clause: slice by estimated time instead of
            # letting one unsplittable sentence blow the window.
            per_char = spoken_seconds(sub) / max(1, len(sub))
            take = max(1, int(cap / max(per_char, 1e-6) / _SEQ_HEADROOM))
            for i in range(0, len(sub), take):
                pieces.append(sub[i:i + take])

    segments = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if current and spoken_seconds(candidate) * _SEQ_HEADROOM > cap:
            segments.append(current)
            current = piece
        else:
            current = candidate
    if current.strip():
        segments.append(current)

    if len(segments) > 1 and spoken_seconds(segments[-1]) < _MIN_TAIL_SEC:
        merged = segments[-2] + segments[-1]
        if spoken_seconds(merged) * _SEQ_HEADROOM <= cap:
            segments = segments[:-2] + [merged]

    plan = [{"text": s, "estimate": round(spoken_seconds(s), 2),
             "seconds": round(max(0.3, min(cap, math.ceil(spoken_seconds(s) * _SEQ_HEADROOM
                                / _MODEL_FRAME_SEC) * _MODEL_FRAME_SEC)), 2)}
            for s in segments]
    gaps = _SEGMENT_GAP_SEC * max(0, len(plan) - 1)
    return {"segments": plan, "count": len(plan), "cap_seconds": round(cap, 2),
            "ref_seconds": round(float(ref_seconds), 2),
            "total_seconds": round(sum(p["seconds"] for p in plan) + gaps, 2),
            "gap_seconds": round(gaps, 2), "split": len(plan) > 1}



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
    except Exception as exc:
        # Optional dependency (it is not in pyproject). The peak ceiling below is the
        # part that actually protects against clipping, so this only costs loudness.
        log.debug("LUFS 归一不可用（pyloudnorm 未安装？），仅按峰值限幅: %s", exc)
    if out.numel():
        peak = float(out.abs().max())
        if math.isfinite(peak) and peak > _VOCAL_PEAK_CEILING:
            out = out * (_VOCAL_PEAK_CEILING / peak)
    return out

app = FastAPI(title="AuK Studio API", version="2.0.0")


class RequestLog:
    """Pure-ASGI request tagger: one rid per request, one line per real API call.

    Deliberately not a BaseHTTPMiddleware: that one runs the downstream app in a separate
    task, which is exactly where contextvars get lost, and the rid has to survive into the
    model worker's own log lines for ``grep <rid> logs/*.log`` to rebuild a failure.

    The page polls /api/health and /api/system twice a second, so those (and every static
    asset) drop to DEBUG; without that the interesting lines drown within a minute.
    """

    _QUIET = ("/api/health", "/api/system", "/api/model/status", "/api/log/client")
    _DETAIL_CAP = 2048

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = logkit.set_rid()
        method = scope.get("method", "?")
        path = scope.get("path", "")
        api = path.startswith("/api/")
        quiet = not api or path in self._QUIET
        started = time.monotonic()
        state = {"status": None, "detail": b"", "watch": False}

        async def send_wrapper(message):
            kind = message["type"]
            if kind == "http.response.start":
                state["status"] = message["status"]
                headers = list(message.get("headers") or [])
                headers.append((b"x-auk-rid", rid.encode()))
                message = {**message, "headers": headers}
                # A 4xx/5xx JSON body here is the exact sentence the user saw, and it is
                # the only copy that survives once the tab is closed.
                state["watch"] = message["status"] >= 400 and any(
                    k.lower() == b"content-type" and b"json" in v.lower()
                    for k, v in headers if isinstance(v, bytes))
            elif kind == "http.response.body" and state["watch"]:
                room = self._DETAIL_CAP - len(state["detail"])
                if room > 0:
                    state["detail"] += (message.get("body") or b"")[:room]
            await send(message)

        if api and not quiet:
            log.info("%s %s 开始", method, path)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            line = "%s %s → %s %.0fms" % (method, path, state["status"] or "-",
                                          (time.monotonic() - started) * 1000)
            if quiet:
                log.debug(line)
            elif (state["status"] or 500) >= 400:
                # UnicodeDecodeError on a truncated multi-byte CJK char is the normal case
                # when the cap cuts a string in half, and a partial line still identifies it.
                text = state["detail"].decode("utf-8", "replace")[:self._DETAIL_CAP]
                log.warning("%s | %s%s", line, text,
                            " " + logkit.resource_line() if (state["status"] or 0) >= 500 else "")
            else:
                log.info(line)


app.add_middleware(RequestLog)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled_exception(request, exc):
    """Last line of defence: FastAPI would otherwise log a traceback to a console that
    closes, and hand the browser a bare 'Internal Server Error' with nothing to quote."""
    rid = logkit.get_rid()
    log.error("未捕获异常 %s %s | %s", request.method, request.url.path,
              logkit.resource_line(), exc_info=exc)
    return JSONResponse(500, {"detail": f"服务器内部错误：{type(exc).__name__}: {exc}（日志 {rid}）"})


@app.on_event("startup")
async def _start_logging_and_prune():
    # attach() has to run per process, not at import: a spawn worker re-imports this module
    # and would otherwise hand its own file handle to the same logs/server.log.
    logkit.attach("server")
    log.info("进程环境 %s", logkit.env_fingerprint())
    threading.Thread(target=prune_uploads, daemon=True).start()


@app.on_event("shutdown")
async def _release_model_worker():
    # Daemon cleanup in multiprocessing only runs on a *normal* interpreter exit; stopping
    # explicitly here keeps the ~18GB of committed weights from outliving the request that
    # shut the server down. run.py's job object covers the exits that never get here.
    SUPERVISOR._stop(force=True)


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


def _check_reference_cap(path, max_sec: float) -> Optional[str]:
    """Authoritative second look at the reference length, from the process that can decode
    anything the model can. Returns an error string when the clip is over the cap.

    m4a/webm cannot be probed by libsndfile, so the parent plans against the cap blindly;
    without this check a 60s phone recording would slip through as "10s" and hand the whole
    thing to the model, which is the exact silent degradation this whole path exists to stop.
    """
    if not path:
        return None
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        seconds = float(info.num_frames) / float(info.sample_rate)
    except Exception as exc:
        # WARNING, not debug: this guard is the second opinion that is supposed to catch a
        # 60s phone recording the parent could not probe, so when it cannot read the file
        # either, the job proceeds with no length check at all.
        log.warning("参考音频二次测量失败（%s），本次不做长度拦截: %s", path, exc)
        return None  # unreadable here too; the model itself will report the real failure
    if seconds > float(max_sec) + 0.05:
        return (f"参考音频实测 {seconds:.1f}s，超过 {float(max_sec):.0f}s 上限："
                "参考与目标共用同一条序列，官方训练预算只有 30s。请在页面上截取其中一段（5–8s 最佳）。")
    return None


def _materialize_reference(prep: dict) -> Path:
    """Write out the slice of a reference clip the job should actually use.

    The trim has to be a real file, not a tensor slice: the reference is consumed twice —
    once as VAE latents and again as Qwen audio tokens, which `build_cond_inputs` reads from
    the path inside `messages` (cfm_edit.py:99-111). Slicing only the tensor would leave the
    language-model half looking at the full-length clip.
    """
    import torchaudio  # worker-side only; the FastAPI process stays torch-free

    target = Path(prep["target"])
    if target.is_file() and target.stat().st_size > 44:
        log.debug("复用已存在的参考切片 %s", target.name)
        return target
    audio, sr = torchaudio.load(str(prep["source"]))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    total = audio.shape[-1]
    lo = max(0, min(int(round(float(prep["start"]) * sr)), total - 1))
    hi = max(lo + 1, min(int(round(float(prep["end"]) * sr)), total))
    target.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(target), audio[:, lo:hi], sr)
    log.info("截取参考切片 %s: %.2f-%.2fs / 全长 %.2fs @%dHz", target.name, lo / sr, hi / sr,
             total / sr, sr)
    return target


def _emotion_pass(engine, audio_out, sample_rate: int, emotion: str, seed):
    """Re-render generated speech with an emotion instruction (AuK's second pass).

    The stage-1 audio becomes this pass's reference, so it carries
    (1 + coefficient) x its own length inside the sequence — which is why the segment
    planner shrinks the budget beforehand rather than discovering it here.
    """
    import tempfile

    from auk.infer.infer_auk import save_audio  # torch-bound, worker-side only

    stage1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    stage1.close()
    log.info("情感第二遍: %s（在已生成的 %.2fs 语音上重渲染）", emotion,
             audio_out.shape[-1] / float(sample_rate))
    try:
        save_audio(audio_out, sample_rate, stage1.name)
        emo_messages = [{"role": "user", "content": [
            {"type": "text", "text": _EMOTION_INSTRUCTIONS[emotion]},
            {"type": "audio", "audio": stage1.name},
        ]}]
        stage1_sec = audio_out.shape[-1] / float(sample_rate)
        return engine.generate(
            emo_messages, audio=stage1.name, seed=seed,
            gen_seconds=stage1_sec * _EMOTION_DURATION_COEF.get(emotion, 1.06))
    finally:
        try:
            os.unlink(stage1.name)
        except OSError as exc:
            log.debug("临时情感输入未删除 %s: %s", stage1.name, exc)


def audio_duration(path: Path) -> Optional[float]:
    """Seconds in an uploaded clip, or None when libsndfile cannot read the container.

    Runs in the FastAPI process, so it must not import torch; soundfile is a plain
    libsndfile binding and already a dependency of the model. m4a/webm are the cases that
    return None — the worker re-checks those authoritatively with torchaudio.info.
    """
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception as exc:
        # Expected for m4a/webm: the page measured the duration in the browser and the
        # worker re-checks authoritatively, so a None here is a fallback not a fault.
        log.debug("soundfile 无法测量 %s: %s", path.name, exc)
    return None


def _trimmed_name(source: Path, start: float, end: float) -> Path:
    """Deterministic so re-running the same job reuses the slice instead of re-writing it."""
    return source.with_name(
        f"{source.stem}_r{int(round(float(start) * 1000))}_{int(round(float(end) * 1000))}.wav")


def resolve_reference(audio_path: Optional[Path], ref_seconds: Optional[float],
                      ref_range: Optional[tuple], allow_over: bool = False,
                      cap: Optional[float] = _REF_MAX_SEC):
    """Work out what the reference slot will really contain: length after any trim, and the
    file to feed the model (the slice, which is a real file because Qwen reads the path).

    `cap` is None for editing jobs: there the clip is the material being processed, not a
    voice reference to choose, so the right ceiling is the sequence window itself (checked
    in /api/generate) rather than "keep the reference short".

    `allow_over` is for /api/plan only: a preview must not fail, so an over-cap reference is
    reported via the `capped` flag and planned as if it were trimmed to exactly the cap.
    """
    if audio_path is None:
        if ref_range:
            raise HTTPException(400, "ref_range needs an uploaded reference")
        return {"seconds": 0.0, "path": None, "prep": None, "source": None, "guard": None,
                "untrimmed_seconds": 0.0, "capped": False}
    known = ref_seconds if ref_seconds and ref_seconds > 0 else audio_duration(audio_path)
    # An unprobeable container is planned against the cap itself: sizing segments for a
    # 10s reference when the real one is 5s only makes the job a little more granular.
    full = known if known and known > 0 else (cap or 0.0)
    if ref_range:
        start, end = float(ref_range[0]), float(ref_range[1])
        if start < 0 or end <= start:
            raise HTTPException(400, "ref_range must be start,end with end > start")
        # full == 0.0 means "we could not measure this clip at all" (an editing job has no
        # cap to fall back to, and m4a/webm often survives ffmpeg). Rejecting then would
        # call every range past the end of a 0.0s clip; the trim itself is harmless.
        if full > 0 and end > full + 0.05:
            raise HTTPException(400, f"ref_range end {end:.1f}s is past the end of a {full:.1f}s clip")
        used = min(end - start, cap or (end - start))
        trimmed = _trimmed_name(audio_path, start, end)
        return {"seconds": round(used, 2), "path": trimmed,
                "prep": {"source": str(audio_path), "start": start, "end": end,
                         "target": str(trimmed)},
                "source": audio_path, "untrimmed_seconds": round(full, 2), "capped": False,
                "guard": {"path": str(trimmed), "max_sec": cap} if cap else None}
    if cap is not None and full > cap:
        if allow_over:
            return {"seconds": cap, "path": audio_path, "prep": None,
                    "source": audio_path, "untrimmed_seconds": round(full, 2), "capped": True,
                    "guard": {"path": str(audio_path), "max_sec": cap}}
        # No silent truncation: IndexTTS head-cuts the reference to 15s without telling
        # anyone, which is exactly the kind of surprise this page has been fixing.
        raise HTTPException(400,
                            f"参考音频 {full:.1f}s，超过 {cap:.0f}s 上限：序列里参考和目标共用同一段长度，"
                            f"官方训练窗口只有 0.3–{_SEQ_BUDGET_SEC:.0f}s。请在页面上截取其中一段（5–8s 最佳），"
                            f"或换一段更短的。")
    return {"seconds": round(full, 2), "path": audio_path, "prep": None, "source": audio_path,
            "untrimmed_seconds": round(full, 2), "capped": False,
            "guard": {"path": str(audio_path), "max_sec": cap} if cap else None}


wlog = logging.getLogger("auk.worker")


def _send_error(conn, error: str, exc=None):
    """Reply {ok: False} and keep the traceback.

    The page shows `error` (a short sentence it can render); the traceback goes both to
    logs/worker.log and to the parent's log, because a native-level CUDA failure inside a
    worker that never returns is diagnosed by *which* line was the last one written.
    """
    payload = {"ok": False, "error": error}
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        payload["traceback"] = tb[-4000:]
        wlog.error("%s\n%s", error, tb)
    else:
        wlog.error(error)
    try:
        conn.send(payload)
    except Exception as send_exc:
        wlog.error("错误无法回传给服务进程（%s: %s）", type(send_exc).__name__, send_exc)


def _mem_report(torch) -> str:
    """torch's own counters plus the real budgets. The torch numbers are only meaningful
    relative to themselves here: under WDDM an 8GB card will happily let torch *believe* it
    allocated 11GB by spilling into system RAM, which is why the commit line comes first."""
    try:
        return ("torch allocated %.2fGB / reserved %.2fGB / peak %.2fGB | %s"
                % (torch.cuda.memory_allocated() / 1024**3,
                   torch.cuda.memory_reserved() / 1024**3,
                   torch.cuda.max_memory_allocated() / 1024**3,
                   logkit.resource_line()))
    except Exception:
        return logkit.resource_line()


def _model_worker(conn, root: str):
    """Isolated AuK process. A native CUDA/safetensors crash here must not kill FastAPI."""
    logkit.attach("worker")
    engine = None
    loaded_model = None
    loaded_resident = False
    try:
        import torch  # imported only inside the worker
        from auk.infer.infer_auk import AukInfer, save_audio
    except BaseException as exc:
        # BaseException: an import failure here (a broken CUDA build, a missing DLL) is the
        # one error that otherwise leaves the parent waiting on a pipe nobody will answer.
        _send_error(conn, f"AuK import failed: {type(exc).__name__}: {exc}", exc)
        conn.close()
        return
    try:
        gpu_desc = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                    else "无可用 CUDA")
    except Exception as exc:
        gpu_desc = "读取失败（%s）" % exc
    wlog.info("模型进程就绪 pid=%d%s", os.getpid(), logkit.env_fingerprint({
        "torch_cuda": torch.version.cuda or "-",
        "gpu": gpu_desc,
        "worker_root": root}))

    def cleanup():
        nonlocal engine, loaded_model, loaded_resident
        engine = None
        loaded_model = None
        loaded_resident = False
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            wlog.debug("清理显存缓存失败（不影响功能）: %s", exc)

    def _recover_after_error(op):
        """Hand the allocator back whatever the failed operation was holding, then report
        whether the CUDA context is still worth serving from. A bad input or a full disk is
        worth surviving; an illegal memory access is not, and continuing on one would only
        make every later request fail the same way."""
        try:
            if op == "load":
                cleanup()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException as exc:
            wlog.debug("出错后清理显存失败（%s: %s）", type(exc).__name__, exc)
        try:
            if not torch.cuda.is_available():
                return True
            torch.cuda.synchronize()
            return float((torch.ones(8, device="cuda:0") * 2).sum()) == 16.0
        except BaseException as exc:
            wlog.error("出错后 CUDA 自检失败（%s: %s），模型进程退出，下次操作自动重新拉起",
                       type(exc).__name__, exc)
            return False


    try:
        while True:
            op = None
            try:
                cmd = conn.recv()
                op = cmd.get("op")
                # One id for the whole request, adopted from the parent, so server.log and
                # worker.log line up under the same `grep`.
                logkit.set_rid(cmd.get("rid"))
                if op != "ping":
                    wlog.debug("收到 %s", op)
                if op == "ping":
                    conn.send({"ok": True, "loaded_model": loaded_model})
                    continue
                if op == "load":
                    model = cmd["model"]
                    resident = bool(cmd.get("resident", False))
                    config, ckpt = _worker_model_settings(root, model)
                    if not config.exists():
                        _send_error(conn, f"AuK config not found: {config}")
                        continue
                    if not ckpt.exists():
                        _send_error(conn, f"AuK checkpoint not found: {ckpt}")
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

                    wlog.info("开始加载 %s：分页=%s config=%s ckpt=%s (%.2fGB) | %s",
                              model, not resident, config.name, ckpt.name,
                              ckpt.stat().st_size / 1024**3, _mem_report(torch))
                    # The line above is the marker for "the load started": a worker that dies
                    # without writing the next one died inside AukInfer (safetensors read,
                    # commit exhaustion, driver crash), which is a different bug from an import
                    # or config problem.
                    started = time.perf_counter()
                    engine = AukInfer(
                        str(config),
                        str(ckpt),
                        device="cuda:0",
                        dtype="bf16",
                        cpu_offload=not resident,
                    )
                    load_sec = time.perf_counter() - started
                    if os.getenv("AUK_DEBUG"):
                        for name, module in engine.__dict__.items():
                            if hasattr(module, "parameters"):
                                try:
                                    p = next(module.parameters())
                                    wlog.debug("模块 %s: %s %s", name, p.device, p.dtype)
                                except Exception as exc:
                                    wlog.debug("模块 %s 无参数可探测: %s", name, exc)

                    loaded_model = model
                    loaded_resident = resident

                    wlog.info("加载完成 %s 用时 %s | %s", model, logkit.elapsed(load_sec),
                              _mem_report(torch))
                    conn.send({"ok": True, "loaded_model": loaded_model, "resident": loaded_resident, "already_loaded": False,
                               "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2)})
                    continue
                if op == "unload":
                    wlog.info("卸载模型 %s", loaded_model)
                    cleanup()
                    conn.send({"ok": True, "loaded_model": None})
                    continue
                if op == "generate":
                    if engine is None or loaded_model != cmd["model"]:
                        _send_error(conn, "Requested model is not loaded")
                        continue
                    emotion = str(cmd.get("emotion") or "").strip().lower()
                    output_path = Path(cmd["output_path"])
                    _seed_raw = str(cmd.get("seed") or "").strip()
                    _seed_base = int(_seed_raw) if _seed_raw.lstrip("-").isdigit() else None
                    # gen_seconds <= 0 means "match the input audio length" (AuK's
                    # equal_length strategy for every editing task, mirroring gradio's
                    # get_gen_duration which returns the ref seconds when no target is set).
                    _gs = float(cmd.get("gen_seconds") or 0)
                    # One entry per inference pass. A text that fits the sequence window
                    # produces exactly the single pass the page always asked for.
                    work = [(s["messages"], float(s["seconds"]),
                             None if _seed_base is None else _seed_base + i)
                            for i, s in enumerate(cmd.get("segments") or [])] or \
                           [(cmd["messages"], _gs if _gs > 0 else None, _seed_base)]
                    if cmd.get("ref_prep"):
                        _materialize_reference(cmd["ref_prep"])
                    _guard = cmd.get("ref_guard")
                    if _guard:
                        _err = _check_reference_cap(_guard.get("path"), _guard.get("max_sec", _REF_MAX_SEC))
                        if _err:
                            _send_error(conn, _err)
                            continue
                    wlog.info("推理开始：模型=%s %s 段数=%d 各段=%s 情感=%s 任务=%s 输出=%s",
                              loaded_model, "全常驻" if loaded_resident else "分页加载", len(work),
                              " ".join("%.1f" % s for _m, s, _x in work) or "-", emotion or "-",
                              cmd.get("task") or "-", output_path.name)
                    parts = []
                    # engine.batch() holds the paged weights on the GPU across every segment:
                    # without it each pass would pay a full 8GB-card offload/restore cycle,
                    # which dwarfs the inference itself.
                    with engine.batch():
                        for idx, (_msgs, _secs, _seed) in enumerate(work):
                            if len(work) > 1:
                                conn.send({"ok": True, "progress": {"segment": idx + 1, "total": len(work)}})
                            torch.cuda.reset_peak_memory_stats()
                            passed = time.perf_counter()
                            audio_out, sample_rate = engine.generate(_msgs, gen_seconds=_secs, seed=_seed)
                            wlog.info("第 %d/%d 段完成：目标 %s 实际 %.2fs 用时 %s | %s",
                                      idx + 1, len(work), "%.1fs" % _secs if _secs else "等长",
                                      audio_out.shape[-1] / float(sample_rate),
                                      logkit.elapsed(time.perf_counter() - passed), _mem_report(torch))
                            if emotion in _EMOTION_INSTRUCTIONS:
                                # AuK has no single-pass "clone + emotion" instruction: emotion is a
                                # separate paralinguistic-edit task run on already-generated speech
                                # (see infer_gradio emotion examples). Applied per segment so the
                                # second pass never carries more than one segment inside its sequence.
                                audio_out, sample_rate = _emotion_pass(
                                    engine, audio_out, sample_rate, emotion, _seed)
                            if parts:
                                gap = int(round(sample_rate * _SEGMENT_GAP_SEC))
                                if gap > 0:
                                    parts.append(torch.zeros(1, gap, dtype=audio_out.dtype))
                            parts.append(audio_out)
                    audio_out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
                    save_audio(_limit_task_output(audio_out, cmd.get("task"), sample_rate), sample_rate, str(output_path))
                    wlog.info("推理完成：%d 段 / %.2fs 音频 @%dHz，文件 %s",
                              len(work), audio_out.shape[-1] / float(sample_rate), sample_rate,
                              "%.1f MB" % (output_path.stat().st_size / 1048576)
                              if output_path.exists() else "未写出!")
                    conn.send({"ok": True, "sample_rate": int(sample_rate),
                               "out_frames": int(audio_out.shape[-1]),
                               "segments": len(work),
                               "output_path": str(output_path)})
                    continue
                if op == "shutdown":
                    wlog.info("收到 shutdown，模型进程退出（当前模型 %s）", loaded_model)
                    cleanup()
                    conn.send({"ok": True})
                    break
                _send_error(conn, f"Unknown worker operation: {op}")
            except (EOFError, ConnectionError, KeyboardInterrupt):
                # The pipe dying means the parent is gone, and a closed console reaches a
                # daemon child as KeyboardInterrupt: both still end the worker (outer handler).
                raise
            except BaseException as exc:
                # A Python-level failure belongs to the operation that raised it: report it
                # and keep serving. Dying here made the next request pay a process respawn
                # plus a full model load (~35s) for what is usually a bad input or a full
                # disk -- unless the CUDA context itself is what failed.
                _send_error(conn, "%s: %s" % (type(exc).__name__, exc), exc)
                parts = audio_out = work = None
                if not _recover_after_error(op):
                    break
    except (EOFError, ConnectionError) as exc:
        # Only the pipe dying means "the parent is gone" — and there is nobody left to tell,
        # so the log line is the only record that this process ended cleanly-ish.
        wlog.info("与服务进程断开（%s: %s），模型进程退出。当前模型=%s", type(exc).__name__, exc,
                  loaded_model)
    except BaseException as exc:
        # Python-level error from load/generate: return it. Native crashes cannot reach
        # here; the parent detects the dead worker and starts a fresh one on the next
        # request. Only the pipe itself (EOFError / ConnectionError above) means "the
        # parent is gone" -- a plain OSError, such as Windows surfacing a commit-limit
        # failure as ``[WinError 1455] 页面文件太小``, is a real load error the user needs
        # to see instead of a generic "worker exited unexpectedly".
        # BaseException also covers KeyboardInterrupt, which is how a closed console window
        # reaches a daemon child: log it, then try to say so over the pipe anyway.
        _send_error(conn, f"{type(exc).__name__}: {exc}", exc)
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


class _Cancelled(RuntimeError):
    """Raised inside _request when the user cancels; the worker is force-stopped so a
    late reply cannot desync the pipe for the next operation."""


_CANCEL_MSG = "已取消：模型进程已终止，下次操作会自动重新拉起"
_CRASH_MSG = "AuK worker exited unexpectedly (possible CUDA/driver/model crash)"


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ModelSupervisor:
    """Keeps FastAPI independent from the native AuK/CUDA process."""
    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self.conn = None
        self.loaded_model = None
        self.last_error = None
        self.generation_busy = False
        # {"segment": 2, "total": 5} while a multi-segment job runs; surfaced through
        # /api/health so a long job can report more than "it is still working".
        self.progress = None
        self.ctx = mp.get_context("spawn")
        # Wall-clock cap, 0 (default) = unlimited. Slow machines must not be punished
        # for legitimately needing more than N seconds: a dead worker is already caught
        # instantly by the is_alive() check in _request, so the cap only ever kills a
        # load/generation that was going to succeed. Set AUK_MODEL_TIMEOUT=<seconds>
        # to opt back in; the UI's 取消等待 covers the alive-but-hung case.
        self.timeout = float(os.getenv("AUK_MODEL_TIMEOUT", "0"))
        self._cancel = threading.Event()
        # Desired VRAM strategy, chosen in the UI (模型与设置 → 显存策略).
        self.resident = _truthy(os.getenv("AUK_RESIDENT"))
        self.loaded_resident = False
        self.loaded_vram_gb = None
        # What the worker was last asked to do, and for how long. When it dies the parent
        # is the only one left that knows, and "died 4s into a load" vs "died 200s into a
        # generation" are different bugs.
        self.last_op = None
        self.op_started = 0.0

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
        log.info("模型进程已拉起 pid=%s（分页加载=%s）| %s", proc.pid, not self.resident,
                 logkit.resource_line(with_gpu=True))

    def _stop(self, force=False):
        conn, proc = self.conn, self.proc
        self.conn = None
        self.proc = None
        self.loaded_model = None
        self.loaded_resident = False
        self.loaded_vram_gb = None
        reason = "强制终止" if force else "正常关闭"
        if conn:
            if not force:
                try:
                    conn.send({"op": "shutdown"})
                    if conn.poll(2):
                        conn.recv()
                    else:
                        log.warning("模型进程未在 2s 内应答 shutdown，将强制终止")
                except Exception as exc:
                    log.debug("shutdown 握手失败（进程可能已经退出）: %s", exc)
            try:
                conn.close()
            except Exception:
                pass
        if proc:
            proc.join(2)
            if proc.is_alive():
                log.warning("%s：模型进程 pid=%s 3s 内未退出，terminate()", reason, proc.pid)
                proc.terminate()
                proc.join(3)
            else:
                log.info("%s：模型进程 pid=%s 已退出（exitcode=%s）", reason, proc.pid,
                         proc.exitcode)

    def _ensure_process(self):
        if self.proc is None or not self.proc.is_alive() or self.conn is None:
            if self.proc is not None:
                log.warning("模型进程已不在（上次操作 %s），重新拉起", self.last_op or "-")
            self._stop(force=True)
            self._start()

    def _worker_death(self, cause=None):
        """Build the log line for a worker that is gone: what it was doing, how long it got,
        the resource snapshot (a 1455 commit failure and a CUDA access violation look the
        same from here), and the tail of its own log so one file is enough to triage."""
        log.error("模型进程失联：上次操作 %s，已等待 %s | %s\n%s",
                  self.last_op or "-", logkit.elapsed(time.monotonic() - self.op_started),
                  logkit.resource_line(with_gpu=True),
                  logkit.tail(logkit.log_dir() / "worker.log", 20), exc_info=cause)

    def _request(self, payload, timeout=None):
        timeout = self.timeout if timeout is None else timeout
        # Clear before the process comes up, not after: _ensure_process can spend seconds
        # stopping a half-dead worker and spawning a replacement, and a 取消等待 pressed in
        # that window used to be wiped here and then silently ignored.
        self._cancel.clear()
        self._ensure_process()
        self.progress = None
        started = time.monotonic()
        self.last_op = payload.get("op")
        self.op_started = started
        # The worker sets the same rid from this field, so its lines and ours share an id.
        payload = {**payload, "rid": logkit.get_rid()}
        result = failure = None
        death_logged = False
        # Local refs: the shutdown hook reaches _stop() without this lock and nulls both
        # attributes mid-poll, which raised AttributeError past the OSError handler below.
        conn, proc = self.conn, self.proc
        try:
            conn.send(payload)
            while True:
                if conn.poll(0.25):
                    result = conn.recv()
                    # A worker running several segments reports each one before the final
                    # reply; anything carrying "progress" keeps the loop waiting.
                    if isinstance(result, dict) and "progress" in result:
                        self.progress = result.get("progress")
                        continue
                    break
                if self._cancel.is_set():
                    failure = _Cancelled(_CANCEL_MSG)
                    break
                if not proc.is_alive():
                    failure = RuntimeError(_CRASH_MSG)
                    self._worker_death()
                    death_logged = True
                    break
                # timeout <= 0 means "no cap": keep waiting while the worker is alive.
                if 0 < timeout < time.monotonic() - started:
                    failure = TimeoutError(f"AuK worker timed out after {timeout:.0f}s")
                    break
        except (EOFError, BrokenPipeError, OSError) as exc:
            # Only transport failures reach here. TimeoutError is itself an OSError
            # subclass in 3.10+, which is why the outcomes above are assigned to
            # `failure` and raised outside this try instead of being raised inside it.
            if self._cancel.is_set():
                failure = _Cancelled(_CANCEL_MSG)
            else:
                failure = RuntimeError(_CRASH_MSG)
                failure.__cause__ = exc
                self._worker_death(exc)
                death_logged = True
        took = time.monotonic() - started
        if failure is not None:
            # _worker_death already wrote the forensic line for the cases where the process
            # is gone; one more line here covers the rest (cancel, timeout) so every
            # failure has a server.log entry even if api_error is never reached.
            if not death_logged:
                if isinstance(failure, _Cancelled):
                    log.warning("模型进程 %s 被用户取消（%s）", payload.get("op"),
                                logkit.elapsed(took))
                else:
                    log.error("模型进程 %s 未正常返回（%s）: %s", payload.get("op"),
                              logkit.elapsed(took), failure)
            raise failure
        if not result.get("ok"):
            log.error("模型进程 %s 报错（%s）: %s%s", payload.get("op"), logkit.elapsed(took),
                      str(result.get("error"))[:400],
                      "\n" + result["traceback"] if result.get("traceback") else "")
        else:
            (log.info if payload.get("op") != "ping" else log.debug)(
                "模型进程 %s 完成（%s）", payload.get("op"), logkit.elapsed(took))
        return result

    def abort(self):
        """Cancel the in-flight load/generation without needing self.lock (held by that
        very operation). _request notices within one poll and force-stops the worker."""
        alive = bool(self.proc and self.proc.is_alive())
        self._cancel.set()
        return {"ok": True, "had_worker": alive}

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
                log.debug("模型 %s（%s）已就绪，跳过加载", model, "全常驻" if resident else "分页")
                return {"ok": True, "loaded_model": model, "resident": resident, "already_loaded": True}
            # Same model but a different VRAM strategy still needs a real reload:
            # paging on/off is baked into the module placement at load time.
            if self.loaded_model and (self.loaded_model != model or self.loaded_resident != resident):
                log.info("切换 %s → %s / %s，先停掉当前模型进程", self.loaded_model, model,
                         "全常驻" if resident else "分页加载")
                self._stop(force=False)
            self.resident = resident
            log.info("加载模型 %s（%s）| %s", model, "全常驻" if resident else "分页加载",
                     logkit.resource_line(with_gpu=True))
            started = time.monotonic()
            try:
                result = self._request({"op": "load", "model": model, "resident": resident})
            except _Cancelled:
                # The worker may still be mid-load; killing it is the only way to make
                # sure its late reply isn't consumed by the next operation.
                log.warning("加载 %s 被取消（%s 后），强制终止模型进程", model,
                            logkit.elapsed(time.monotonic() - started))
                self._stop(force=True)
                raise
            if not result.get("ok"):
                self.last_error = result.get("error")
                self._stop(force=True)
                raise RuntimeError(self.last_error or "AuK model load failed")
            self.loaded_model = model
            self.loaded_resident = resident
            self.loaded_vram_gb = result.get("vram_allocated_gb")
            log.info("模型 %s 就绪，用时 %s，torch 报告显存 %s GB | %s", model,
                     logkit.elapsed(time.monotonic() - started), self.loaded_vram_gb,
                     logkit.resource_line(with_gpu=True))
            return result

    def unload(self):
        with self.lock:
            if self.proc is None:
                self.loaded_model = None
                return {"ok": True, "loaded_model": None}
            log.info("卸载模型 %s", self.loaded_model)
            try:
                result = self._request({"op": "unload"}, timeout=30)
            except Exception as exc:
                log.warning("卸载未正常完成（%s: %s），强制终止模型进程", type(exc).__name__, exc)
                self._stop(force=True)
                return {"ok": True, "loaded_model": None, "forced": True}
            self._stop(force=False)
            return result

    def generate(self, model, messages, seconds, output_path, emotion="", seed="", task="",
                 segments=None, ref_prep=None, ref_guard=None):
        with self.lock:
            if self.generation_busy:
                raise RuntimeError("Another generation is already running")
            self.generation_busy = True
            try:
                self.load(model)
                # two-pass (clone + emotion) roughly doubles inference time;
                # seconds == 0 (auto/match-input) has no known length, so assume 60s.
                budget = (seconds if seconds > 0 else 60) * (40 if emotion else 20) + 120
                if segments:
                    total = sum(float(s["seconds"]) for s in segments)
                    budget = total * (40 if emotion else 20) + 120 * len(segments)
                # With no configured cap, a paged 8GB generation runs as long as it needs
                # to; the 取消等待 button covers the alive-but-hung case.
                cap = 0 if self.timeout <= 0 else max(self.timeout, budget)
                log.info("提交模型进程：任务=%s 段数=%s 目标=%.2fs 情感=%s 种子=%s 输出=%s 时限=%s",
                         task or "编辑/未声明", len(segments) if segments else 1, seconds,
                         emotion or "-", seed or "-", Path(output_path).name,
                         f"{cap:.0f}s" if cap > 0 else "不限")
                result = self._request({"op": "generate", "model": model, "messages": messages, "gen_seconds": seconds, "output_path": str(output_path), "emotion": emotion, "seed": seed, "task": task,
                                        "segments": segments, "ref_prep": ref_prep, "ref_guard": ref_guard}, timeout=cap)
                if not result.get("ok"):
                    self.last_error = result.get("error")
                    raise RuntimeError(self.last_error or "AuK inference failed")
                # A generation that worked is proof the earlier failure is behind the user;
                # leaving it set kept the ⓘ badge showing an error from a previous attempt.
                self.last_error = None
                return result
            except (RuntimeError, TimeoutError) as exc:
                self.last_error = str(exc)
                # If the worker crashed, was cancelled, or blew the configured cap, kill
                # it: the next request spawns a fresh one and no stale reply can desync.
                if (self.proc is None or not self.proc.is_alive()
                        or isinstance(exc, (TimeoutError, _Cancelled))):
                    self._stop(force=True)
                raise
            finally:
                self.generation_busy = False
                self.progress = None

    def status(self):
        # Deliberately lock-free. self.lock is held for the whole of a load or generation,
        # and the page polls this every second -- taking the lock here made health
        # unanswerable until the job ended (measured: 15/15 polls timed out during a 57s
        # split job), so the progress readout, the busy flag and the resource bar all
        # froze. Each attribute read is atomic under the GIL; proc goes into a local
        # because it can be swapped out from under us between the check and the call.
        proc = self.proc
        alive = bool(proc and proc.is_alive())
        return {"worker_alive": alive, "loaded_model": self.loaded_model if alive else None,
                "resident": self.loaded_resident if alive else False,
                "vram_allocated_gb": self.loaded_vram_gb if alive else None,
                "timeout_s": self.timeout,
                "busy": self.generation_busy, "progress": self.progress,
                "last_error": self.last_error}


SUPERVISOR = ModelSupervisor()

try:  # prime psutil's delta window so the first header reading is not a bogus 0%
    import psutil as _psutil_prime

    _psutil_prime.cpu_percent(interval=None)
except Exception:
    # Silent on purpose: this runs at import, before attach() has a handler to write to.
    # The startup fingerprint line reports whether psutil is present instead.
    pass

_GPU_STATIC = None
_GPU_SNAP = (0.0, None)  # (monotonic stamp, snapshot) — nvidia-smi costs ~0.2s on Windows
_GPU_SNAP_TTL = 1.5
# Only one thread pays for a probe at a time; see _gpu_snapshot.
_GPU_PROBE = threading.Lock()
_warned = set()


def warn_once(key: str, msg: str, *args):
    """For a fault that repeats on every poll (health is hit twice a second): say it once
    at WARNING so it is findable, then stay quiet instead of burying the log.
    """
    if key in _warned:
        log.debug(msg, *args)
        return
    _warned.add(key)
    log.warning(msg, *args)


def _gpu_probe_failed(reason: str):
    """Both the header resource bar and the 显存策略 badges hide themselves when the driver
    cannot be read, which is otherwise invisible: the pill just never appears."""
    warn_once("gpu-probe", "读不到 nvidia-smi（%s）：显存/显卡信息将不可用，资源条只显示 CPU 与内存。",
              reason)


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
    cached = _GPU_SNAP[1]
    if cached is not None and now - _GPU_SNAP[0] < _GPU_SNAP_TTL:
        return cached
    # nvidia-smi is a subprocess (~0.2s here, seconds when the driver is busy) and the page
    # polls health twice a second per tab. Without single-flight every poll that lands on an
    # expired cache starts its own probe in its own anyio threadpool slot, which is how a
    # busy card turns the resource bar into a stall. Losers serve the previous reading; only
    # the very first caller (nothing cached yet) waits for a real probe.
    if not _GPU_PROBE.acquire(blocking=cached is None):
        return cached
    try:
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
            else:
                _gpu_probe_failed("nvidia-smi 无输出")
        except Exception as exc:
            _gpu_probe_failed("%s: %s" % (type(exc).__name__, exc))
        _GPU_SNAP = (now, snap)
        return snap
    finally:
        _GPU_PROBE.release()


def gpu_info():
    """Total VRAM only — cached for the process lifetime; the 显存策略 badges need
    it before any model worker exists."""
    global _GPU_STATIC
    if _GPU_STATIC is None:
        snap = _gpu_snapshot()
        # Do not freeze a failed probe in: nvidia-smi missing once at startup would hide the
        # GPU name and the residency verdict for the rest of the server's life.
        if snap["vram_gb"] is not None:
            _GPU_STATIC = {"name": snap["name"], "vram_gb": snap["vram_gb"]}
        else:
            return {"name": snap["name"], "vram_gb": None}
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


_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


async def save_upload(file: UploadFile) -> Path:
    _prune_uploads_if_stale()
    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(400, "Unsupported audio format")
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    # Stream to disk in chunks so an oversized upload can't be fully buffered in RAM
    # before we reject it — the read-then-check pattern made the 100 MB cap a suggestion.
    written = 0
    try:
        with path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Audio file is too large (max 100 MB)")
                out.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if written == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "Empty audio file")
    log.info("上传已保存 %s（%s，%.1f MB）", path.name, suffix, written / 1048576)
    return path


def api_error(exc: Exception):
    """Map a model/service failure onto the HTTP reply the page shows.

    Graded by outcome: a 4xx is the user's input and only needs one line, while a 5xx is a
    real fault and logs the traceback plus a resource snapshot — on this box an OOM and a
    WinError 1455 commit failure look identical in the message but not in the numbers. 5xx
    replies also carry the request id, so a screenshot alone is enough to find the entry.
    """
    msg = str(exc)
    lower = msg.lower()
    if isinstance(exc, _Cancelled) or "已取消" in msg:
        status, detail = 499, "已取消：模型进程已终止，下次操作会自动重新拉起。"
    elif "checkpoint not found" in lower or "config not found" in lower:
        status, detail = 404, msg
    elif "out of memory" in lower or "cudnn_status_failure" in lower:
        status, detail = 503, "显存不足：当前显卡放不下全常驻的权重（bf16 约 12.8GB）。请在「模型与设置 → 显存策略」切回分页加载。"
    elif "worker exited" in lower or "access violation" in lower:
        status, detail = 503, "AuK 模型进程异常退出，服务本身仍在运行。请再次点击加载模型或重新生成，系统会自动重新拉起模型进程。"
    elif isinstance(exc, TimeoutError):
        status, detail = 504, "AuK 模型响应超时，模型进程已重置。请稍后重试。"
    elif "model must be base or flash" in msg:
        # A hand-made request, not a server fault: without this it logged a traceback and
        # filled the error log with something no user ever caused.
        status, detail = 400, msg
    else:
        status, detail = 500, msg
    if status >= 500:
        rid = logkit.get_rid()
        log.error("请求失败 [%s] %s | %s", rid, msg[:300], logkit.resource_line(with_gpu=True),
                  exc_info=exc)
        detail = f"{detail}（日志编号 {rid}）"
    elif status == 499:
        log.info("操作已取消（模型进程已终止）")
    else:
        log.warning("拒绝请求 %s: %s", status, msg[:300])
    return HTTPException(status, detail)


@app.get("/api/health")
def health():
    s = SUPERVISOR.status()
    # One driver read feeds every VRAM field below. Reading it three times per poll used to
    # mean three chances to pay for a subprocess on a card that is already busy generating.
    snap = _gpu_snapshot()
    gpu = {"name": snap["name"], "vram_gb": snap["vram_gb"]}
    # model_settings() resolves absolute paths through the filesystem on every call: once
    # per model, not once per JSON key.
    models = {}
    for key in ("base", "flash"):
        cfg, ckpt = model_settings(key)
        models[key] = {"config": str(cfg), "checkpoint": str(ckpt),
                       "available": cfg.exists() and ckpt.exists()}
    return {
        "ok": True,
        "service": "online",
        "worker_alive": s["worker_alive"],
        "loaded_model": s["loaded_model"],
        "resident": s["resident"],
        "vram_used_gb": snap["used_gb"] if s["worker_alive"] else None,
        "gpu": {**gpu, "used_gb": snap["used_gb"]},
        "resident_verdict": resident_verdict(gpu["vram_gb"]),
        "busy": s["busy"],
        "progress": s.get("progress"),
        "model_timeout_s": s["timeout_s"],
        "code_mtime": CODE_MTIME,
        "last_error": s["last_error"],
        "models": models,
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
    except Exception as exc:
        warn_once("psutil", "psutil 不可用（%s: %s）：顶部资源条没有 CPU/内存读数。模型加载本身不依赖它。",
                  type(exc).__name__, exc)
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
        return SUPERVISOR.load(model, resident=_truthy(resident))
    except Exception as exc:
        raise api_error(exc)


@app.post("/api/model/unload")
def unload_model():
    try:
        return SUPERVISOR.unload()
    except Exception as exc:
        raise api_error(exc)


@app.post("/api/model/abort")
def abort_model():
    """Cancel the current load/generation. Safe to call at any time: it only sets a flag
    that the in-flight _request poll picks up, which then force-stops the worker."""
    return SUPERVISOR.abort()


@app.get("/api/model/status")
def model_status():
    return SUPERVISOR.status()


# Jobs whose *text* is the thing being spoken, so the text can be split across passes.
# Deliberately excludes "": an undeclared task means an editing job, where the clip itself
# is both reference and target and there is nothing to split.
_TASKS_WITH_TEXT = {"instruct_tts", "zero_shot_tts"}


def _build_segments(messages: list, instruction: str, text: str, plan: dict) -> list:
    """Per-segment copies of the request, with only the spoken text swapped.

    Returns [] when the page's instruction does not contain the text exactly once (a
    hand-written advanced-mode prompt, or a text that also appears inside the voice
    description): silently splitting something we cannot locate would corrupt the
    instruction, so those jobs stay single-pass and get checked against the window instead.
    """
    if not plan.get("split") or not text or instruction.count(text) != 1:
        # 0 = a hand-written instruction that does not carry the text; >1 = replace(...,1)
        # would swap the first occurrence, which may not be the spoken text at all.
        return []
    out = []
    for seg in plan["segments"]:
        copied = json.loads(json.dumps(messages))
        for block in copied[0]["content"]:
            if block.get("type") == "text":
                block["text"] = instruction.replace(text, seg["text"], 1)
        out.append({"messages": copied, "seconds": seg["seconds"]})
    return out


@app.post("/api/plan")
def plan_job(text: str = Form(...), ref_name: str = Form(""), ref_seconds: float = Form(0),
             ref_range: str = Form(""), emotion: str = Form("")):
    """Segment the same way /api/generate will, without loading the model, so the page can
    say "分 8 段" before the user commits to a multi-minute job."""
    text = text.strip()
    if not text:
        raise HTTPException(400, "text is required")
    rng = _parse_range(ref_range)
    path = _resolve_upload(ref_name) if ref_name.strip() else None
    ref_seconds = _require_finite(float(ref_seconds), "ref_seconds")
    known = ref_seconds if ref_seconds > 0 else (audio_duration(path) if path else 0.0)
    if path is None and known > 0:
        # The page measured the duration in the browser and has no stored upload to point
        # at yet (the clip itself travels with /api/generate). Planning as if there were no
        # reference at all would over-promise ~5s of target capacity per segment.
        # A range trims what the model actually sees, so plan against the trimmed length --
        # otherwise the preview promises fewer segments than the run (generate slices too).
        secs = (rng[1] - rng[0]) if rng else known
        ref = {"seconds": min(max(secs, 0.0), _REF_MAX_SEC), "capped": secs > _REF_MAX_SEC}
    else:
        ref = resolve_reference(path, known, rng, allow_over=True)
    emo = emotion.strip().lower()
    emo = emo if emo in _EMOTION_INSTRUCTIONS else ""
    plan = plan_segments(text, ref["seconds"], emo)
    plan["reference_capped"] = ref["capped"]
    plan["ref_max_sec"] = _REF_MAX_SEC
    plan["budget_sec"] = _SEQ_BUDGET_SEC
    passes = 2 if emo else 1
    # Wall-clock is hardware-dependent, so this is a bracket rather than a promise. Measured
    # on this box (3060 Ti 8GB, paged, AuK-Flash nfe=4): 666 chars -> 8 segments -> 137.14s of
    # audio took 59.9s end to end, on top of a 23s model load, i.e. ~0.22s of compute per
    # second of audio plus ~2s of per-segment overhead. The low end is that measurement; the
    # high end covers a slower or more heavily paged card. The emotion pass renders roughly
    # the same amount of audio a second time.
    compute = sum(s["seconds"] for s in plan["segments"]) * passes
    plan["est_work_sec"] = round(compute * 0.22 + 22 + 2 * max(0, plan["count"] - 1), 0)
    plan["est_work_max_sec"] = round(compute * 1.0 + 50 + 10 * max(0, plan["count"] - 1), 0)
    return plan


def _require_finite(value: float, name: str) -> float:
    """A plain `float` Form field lets nan/inf through, and every guard downstream is a
    comparison: `nan < 0` and `nan > cap` are both False, so nan reaches the model."""
    if not math.isfinite(value):
        raise HTTPException(400, f"{name} must be a finite number")
    return value


def _parse_range(value: str):
    parts = [p.strip() for p in str(value or "").split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) != 2:
        raise HTTPException(400, "ref_range must be start,end in seconds")
    try:
        start, end = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise HTTPException(400, "ref_range must be two numbers") from exc
    # float("nan")/float("inf") parse fine and then defeat every comparison below them
    # (nan < 0 is False), so a hand-made request would slice the reference to nan seconds.
    if not (math.isfinite(start) and math.isfinite(end)):
        raise HTTPException(400, "ref_range must be finite numbers")
    return (start, end)


@app.post("/api/generate")
async def generate(instruction: str = Form(...), model: str = Form("flash"), gen_seconds: float = Form(0), emotion: str = Form(""), seed: str = Form(""), task: str = Form(""), input_name: str = Form(""), text: str = Form(""), ref_range: str = Form(""), input_duration: float = Form(0), audio: Optional[UploadFile] = File(None)):
    if not instruction.strip():
        raise HTTPException(400, "instruction is required")
    if model not in {"base", "flash"}:
        raise HTTPException(400, "model must be base or flash")
    gen_seconds = _require_finite(float(gen_seconds), "gen_seconds")
    input_duration = _require_finite(float(input_duration), "input_duration")
    if gen_seconds < 0:
        # No upper bound: the honest text-length estimate is allowed to ask for as much
        # audio as the text needs. A ceiling here silently lies (the page used to clamp to
        # 60s and command the model to speak 140s of text in 60s — measured 2.33x rate).
        raise HTTPException(400, "gen_seconds must be 0 or greater")
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
    if gen_seconds == 0 and audio_path is None and not text.strip():
        raise HTTPException(400, "text-only generation needs gen_seconds > 0")
    emo = emotion.strip().lower()
    emo = emo if emo in _EMOTION_INSTRUCTIONS else ""
    # Official gradio gates post-processing on task_type; the Web path has no PromptEnhancer
    # to classify it, so the page states it explicitly. Unknown names simply mean "no gating".
    task_name = task.strip().lower()
    task_name = task_name if task_name in _KNOWN_TASKS else ""

    ref_cap = _REF_MAX_SEC if task_name == "zero_shot_tts" else None
    ref = resolve_reference(audio_path, input_duration or None, _parse_range(ref_range),
                            cap=ref_cap)
    instruction = instruction.strip()
    spoken = text.strip()
    base_messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    if ref["path"]:
        base_messages[0]["content"].append({"type": "audio", "audio": str(ref["path"])})

    # Text-driven jobs (clone / instruct TTS) are sized from the text and split so every
    # pass stays inside the window. Editing jobs keep the page's duration and are checked
    # against it instead, because their source clip is both reference and target.
    plan = None
    segments = []
    if spoken and task_name in _TASKS_WITH_TEXT:
        plan = plan_segments(spoken, ref["seconds"], emo)
        segments = _build_segments(base_messages, instruction, spoken, plan)
    if plan and plan["split"] and not segments:
        raise HTTPException(400,
                            f"这段文本约需 {plan['total_seconds']:.0f}s，超出模型单条 {_SEQ_BUDGET_SEC:.0f}s 序列窗口，"
                            "但无法在指令文本里唯一定位它（手写指令，或同一段文字出现了多次），因此不能自动分段。请在普通模式下把要念的文字单独填写。")

    if segments:
        messages = segments[0]["messages"]
        seconds = float(segments[0]["seconds"])
    else:
        messages = base_messages
        if plan and plan["segments"]:
            seconds = float(plan["segments"][0]["seconds"])
        else:
            seconds = float(gen_seconds) or (ref["seconds"] if audio_path is not None else 0.0)
        cap = sequence_budget(ref["seconds"], emo)
        if seconds > cap + 0.05:
            # Only reachable for jobs the planner did not size (no separate text field, or a
            # hand-written instruction): a text-driven job gets clamped into segments instead.
            role = "参考" if task_name in _TASKS_WITH_TEXT else "输入"
            detail = f"{role} {ref['seconds']:.1f}s + 目标 {seconds:.1f}s"
            if emo:
                detail += f"，情感第二遍还要 ×{1 + _EMOTION_DURATION_COEF.get(emo, 1.06):.2f}"
            hint = ("请缩短参考音频或降低目标时长。" if task_name in _TASKS_WITH_TEXT else
                    f"编辑类任务是整段处理、不自动分段，等长任务请把音频控制在 "
                    f"{(_SEQ_BUDGET_SEC - _SEQ_MARGIN_SEC) / 2:.0f}s 以内。")
            raise HTTPException(400,
                                f"超出模型序列窗口（{detail}）：{role}音频和目标输出共用同一条序列，官方训练"
                                f"预算只有 {_SEQ_BUDGET_SEC:.0f}s、留余量后单条最多 {cap:.1f}s 目标。{hint}")

    job_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{job_id}.wav"
    log.info("生成 %s 受理：模型=%s 任务=%s 情感=%s 输入=%s 参考=%.2fs(%s) 目标=%.2fs 段数=%d "
             "文本=%d字 指令=%d字 种子=%s", job_id[:8], model, task_name or "-", emo or "-",
             audio_path.name if audio_path else "-", ref["seconds"],
             "截取" if ref.get("prep") else ("原样" if ref["path"] else "无"),
             seconds, len(segments) or 1, len(spoken), len(instruction), seed_raw or "-")
    log.debug("生成 %s 分段秒数: %s | 计划=%s", job_id[:8],
              " ".join("%.2f" % s["seconds"] for s in segments) or "-",
              (plan or {}).get("segments") and [s["seconds"] for s in plan["segments"]])
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            SUPERVISOR.generate, model, messages, seconds, output_path,
            emo, seed_raw, task_name, segments or None, ref["prep"], ref["guard"])
        elapsed = round(time.perf_counter() - started, 1)
        rate = int(result.get("sample_rate") or 0)
        frames = int(result.get("out_frames") or 0)
        log.info("生成 %s 完成：%.2fs 音频（%d 帧 @%sHz）用时 %s", job_id[:8],
                 frames / rate if rate and frames else -1, frames, rate or "-",
                 logkit.elapsed(elapsed))
        return {"job_id": job_id, "model": model, "sample_rate": rate or None,
                "duration_sec": round(frames / rate, 2) if rate and frames else None,
                "target_sec": round(float(gen_seconds), 2) if gen_seconds > 0 else None,
                "elapsed_sec": elapsed, "seed": int(seed_raw) if seed_raw else None,
                "emotion": emo or None, "instruction": instruction.strip(), "task": task_name or None,
                "segments": (plan or {}).get("count") or 1,
                "segment_plan": [s["seconds"] for s in (plan or {}).get("segments") or []],
                "ref_seconds": ref["seconds"] if ref["seconds"] else None,
                "input_url": f"/api/input/{ref['path'].name}" if ref["path"] else None,
                "audio_url": f"/api/audio/{job_id}"}
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        err = api_error(exc)
        raise err
    # NOTE: the input clip is intentionally kept on disk (not deleted here). The page
    # stores input_url and re-references it via input_name for A/B audition and
    # same-input re-runs across a refresh; uploads are reclaimed by the 7-day prune.


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


# ------------------------------------------------------------------ 浏览器侧错误回传
clog = logging.getLogger("auk.client")
_CLIENT_BUDGET_PER_MIN = 60
_CLIENT_DEDUPE_SEC = 10.0
_CLIENT_BODY_MAX = 64 * 1024
_client_seen = {}
_client_window = [0.0, 0]


def _one_line(value, limit: int) -> str:
    """A record has to stay one line: a browser-supplied stack is the one input here that
    could otherwise inject forged entries into the log file."""
    return str(value if value is not None else "")[:limit].replace("\r", " ").replace(
        "\n", " \\n ").replace("\x00", "")


@app.post("/api/log/client")
async def client_log(request: Request):
    """Log what the page could not show.

    The server only sees requests it answered: a JS exception, a playback URL that failed,
    or a button that stopped responding leaves no trace anywhere. This is the page's black
    box. The budget and the dedupe exist because a page fault loop could otherwise refill a
    10MB rotating log in minutes.
    """
    try:
        # Read with a cap instead of request.json(): a page stuck in a fault loop posts its
        # whole environment every time, and a single oversized body would be buffered in
        # full before anything looked at it.
        raw = b""
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > _CLIENT_BODY_MAX:
                raise HTTPException(413, "浏览器日志回传超过 %dKB" % (_CLIENT_BODY_MAX // 1024))
        payload = json.loads(raw or b"{}")
    except HTTPException:
        raise
    except Exception as exc:
        log.debug("浏览器日志回传无法解析: %s: %s", type(exc).__name__, exc)
        raise HTTPException(400, "body must be a JSON object")
    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be a JSON object")
    now = time.monotonic()
    if now - _client_window[0] > 60:
        _client_window[0], _client_window[1] = now, 0
    _client_window[1] += 1
    if _client_window[1] > _CLIENT_BUDGET_PER_MIN:
        raise HTTPException(429, "浏览器日志回传超过每分钟 %d 条" % _CLIENT_BUDGET_PER_MIN)
    kind = _one_line(payload.get("kind") or "error", 24)
    message = _one_line(payload.get("message"), 600)
    key = kind + "|" + message
    if now - _client_seen.get(key, -1e9) < _CLIENT_DEDUPE_SEC:
        return {"ok": True, "deduped": True}
    _client_seen[key] = now
    if len(_client_seen) > 500:
        _client_seen.clear()
    # Adopt the id of the request that broke, so this line sits beside it in server.log.
    logkit.set_rid(_one_line(payload.get("rid"), 16))
    where = _one_line(payload.get("source"), 200)
    lineno = _one_line(payload.get("line"), 12)
    clog.warning("浏览器 %s: %s%s", kind, message,
                 " @%s%s" % (where, ":" + lineno if lineno else ""))
    stack = _one_line(payload.get("stack"), 1500)
    if stack:
        clog.warning("浏览器堆栈 %s", stack)
    return {"ok": True}


# Registered before the catch-all below: Starlette matches routes in registration order, so
# a mount added after "/" would never see a request.
app.mount("/demo-audio", StaticFiles(directory=DEMO_AUDIO_DIR), name="demo-audio")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
