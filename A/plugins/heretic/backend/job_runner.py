# -*- coding: utf-8 -*-
"""plugins/heretic/backend/job_runner.py

Heretic "Void" Job Runner (content-agnostic)
-------------------------------------------

This module provides a generic asynchronous job runner with:
- file-based logs
- progress/status tracking
- cancel support
- optional GPU/VRAM stats (best-effort)

IMPORTANT:
- This runner intentionally does NOT implement any specific "uncensor/decensor" logic.
- It is designed to run *analysis* or other neutral research routines you plug in.
"""

from __future__ import annotations

import time
import uuid
import threading
import traceback
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class JobStatus:
    job_id: str
    state: str  # queued|running|done|error|canceled
    created_at: float
    updated_at: float
    step: str = ""
    progress: float = 0.0  # 0..1
    message: str = ""
    outputs_dir: str = ""
    log_file: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    gpu: Optional[dict] = None


class CancelToken:
    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    def is_canceled(self) -> bool:
        return self._ev.is_set()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def best_effort_gpu_stats() -> Optional[dict]:
    """Returns GPU stats if available.

    Tries NVML first (pynvml), then nvidia-smi. Returns None if unavailable.
    """
    # NVML
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        name = pynvml.nvmlDeviceGetName(h).decode("utf-8", "ignore")
        pynvml.nvmlShutdown()
        return {
            "name": name,
            "util_gpu": int(util.gpu),
            "util_mem": int(util.memory),
            "vram_used_mb": int(mem.used / 1024 / 1024),
            "vram_total_mb": int(mem.total / 1024 / 1024),
        }
    except Exception:
        pass

    # nvidia-smi
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=2).strip()
        if not out:
            return None
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            return {
                "name": parts[0],
                "util_gpu": int(float(parts[1])),
                "util_mem": int(float(parts[2])),
                "vram_used_mb": int(float(parts[3])),
                "vram_total_mb": int(float(parts[4])),
            }
    except Exception:
        return None
    return None


class HereticJobRunner:
    """Simple in-process job runner.

    You register task functions of the signature:
      fn(ctx: JobContext) -> dict

    The runner executes them in a thread so FastAPI event loop is not blocked.
    """

    def __init__(self, base_outputs_dir: Path) -> None:
        self.base_outputs_dir = base_outputs_dir
        _safe_mkdir(self.base_outputs_dir)
        self._jobs: Dict[str, JobStatus] = {}
        self._tokens: Dict[str, CancelToken] = {}
        self._locks = threading.Lock()

    def get(self, job_id: str) -> Optional[JobStatus]:
        with self._locks:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._locks:
            tok = self._tokens.get(job_id)
            st = self._jobs.get(job_id)
            if not tok or not st:
                return False
            tok.cancel()
            st.state = "canceled"
            st.message = "キャンセル要求を受け付けました"
            st.updated_at = time.time()
            return True

    def _append_log(self, job_id: str, text: str) -> None:
        st = self.get(job_id)
        if not st:
            return
        try:
            Path(st.log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(st.log_file, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    def _update(self, job_id: str, **kwargs: Any) -> None:
        with self._locks:
            st = self._jobs.get(job_id)
            if not st:
                return
            for k, v in kwargs.items():
                setattr(st, k, v)
            st.updated_at = time.time()
            st.gpu = best_effort_gpu_stats()

    def start(
        self,
        task_name: str,
        task_fn: Callable[["JobContext"], dict],
        payload: dict,
        model_ref: Optional[dict] = None,
    ) -> JobStatus:
        job_id = uuid.uuid4().hex[:12]
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_model = (payload.get("model_name") or "unknown").replace(":", "_").replace("/", "_").replace("\\", "_")
        out_dir = self.base_outputs_dir / f"{ts}_{safe_model}_{task_name}"
        log_dir = out_dir / "logs"
        _safe_mkdir(log_dir)
        log_file = log_dir / "run.log"

        st = JobStatus(
            job_id=job_id,
            state="queued",
            created_at=time.time(),
            updated_at=time.time(),
            step="queued",
            progress=0.0,
            message="待機中",
            outputs_dir=str(out_dir),
            log_file=str(log_file),
            gpu=None,
        )
        tok = CancelToken()

        with self._locks:
            self._jobs[job_id] = st
            self._tokens[job_id] = tok

        def _run() -> None:
            self._update(job_id, state="running", step="running", message="実行中", progress=0.01)
            ctx = JobContext(
                job_id=job_id,
                payload=payload,
                model_ref=model_ref,
                outputs_dir=out_dir,
                log_file=log_file,
                token=tok,
                runner=self,
            )
            try:
                res = task_fn(ctx)
                if tok.is_canceled():
                    self._update(job_id, state="canceled", step="canceled", message="キャンセルされました", progress=1.0, result=res)
                else:
                    self._update(job_id, state="done", step="done", message="完了", progress=1.0, result=res)
            except Exception as e:
                err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                self._append_log(job_id, "\n[ERROR]\n" + err + "\n")
                self._update(job_id, state="error", step="error", message="失敗", progress=1.0, error=err)

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        return st

    def public_dict(self, job_id: str) -> Optional[dict]:
        st = self.get(job_id)
        if not st:
            return None
        return asdict(st)


class JobContext:
    def __init__(
        self,
        job_id: str,
        payload: dict,
        model_ref: Optional[dict],
        outputs_dir: Path,
        log_file: Path,
        token: CancelToken,
        runner: HereticJobRunner,
    ) -> None:
        self.job_id = job_id
        self.payload = payload
        self.model_ref = model_ref
        self.outputs_dir = outputs_dir
        self.log_file = log_file
        self.token = token
        self.runner = runner

    def log(self, msg: str) -> None:
        self.runner._append_log(self.job_id, msg + "\n")

    def set_step(self, step: str, progress: float | None = None, message: str | None = None) -> None:
        kw: Dict[str, Any] = {"step": step}
        if progress is not None:
            kw["progress"] = float(progress)
        if message is not None:
            kw["message"] = str(message)
        self.runner._update(self.job_id, **kw)

    def canceled(self) -> bool:
        return self.token.is_canceled()
