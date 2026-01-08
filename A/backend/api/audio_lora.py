from fastapi import APIRouter
from pathlib import Path
import uuid
import threading
import time

from backend.lora.audio.bridge import run_gpt_sovits_vc_lora, run_xtts_tts_lora

router = APIRouter()

_jobs = {}

def _start_job(kind: str, popen, out_log: Path):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"kind": kind, "status": "running", "log_path": str(out_log)}
    def _worker():
        out_log.parent.mkdir(parents=True, exist_ok=True)
        with out_log.open("w", encoding="utf-8") as w:
            for line in iter(popen.stdout.readline, ""):
                if not line:
                    break
                w.write(line)
                w.flush()
        code = popen.wait()
        _jobs[job_id]["status"] = "done" if code == 0 else "failed"
        _jobs[job_id]["exit_code"] = code
    threading.Thread(target=_worker, daemon=True).start()
    return job_id

@router.post("/api/audio/train/tts-lora")
def train_tts_lora(payload: dict):
    dataset_dir = Path(payload.get("dataset_dir", "datasets/audio_tts")).resolve()
    out_dir = Path(payload.get("out_dir", "outputs/lora/tts_xtts")).resolve()
    rank = int(payload.get("lora_rank", 16))
    alpha = int(payload.get("lora_alpha", 32))
    extra = payload.get("extra_args", "")
    proj = Path("third_party/XTTS").resolve()
    out_log = Path("logs") / f"train_xtts_{int(time.time())}.log"
    popen = run_xtts_tts_lora(proj, dataset_dir, out_dir, rank, alpha, extra)
    job_id = _start_job("tts_lora_xtts", popen, out_log)
    return {"job_id": job_id, "log_path": str(out_log)}

@router.post("/api/audio/train/vc-lora")
def train_vc_lora(payload: dict):
    dataset_dir = Path(payload.get("dataset_dir", "datasets/audio_vc")).resolve()
    out_dir = Path(payload.get("out_dir", "outputs/lora/vc_gpt_sovits")).resolve()
    rank = int(payload.get("lora_rank", 16))
    alpha = int(payload.get("lora_alpha", 32))
    extra = payload.get("extra_args", "")
    proj = Path("third_party/GPT-SoVITS").resolve()
    out_log = Path("logs") / f"train_gpt_sovits_{int(time.time())}.log"
    popen = run_gpt_sovits_vc_lora(proj, dataset_dir, out_dir, rank, alpha, extra)
    job_id = _start_job("vc_lora_gpt_sovits", popen, out_log)
    return {"job_id": job_id, "log_path": str(out_log)}

@router.get("/api/audio/train/status/{job_id}")
def train_status(job_id: str):
    return _jobs.get(job_id, {"status": "unknown"})
