import os
import subprocess
from pathlib import Path

def _env_for_subprocess():
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_cmd_template(project_root: Path, filename: str) -> str:
    tpl = project_root / filename
    if not tpl.exists():
        raise RuntimeError(f"学習コマンドが見つかりません: {tpl}")
    text = tpl.read_text(encoding="utf-8").strip()
    if not text or "RUN_" in text and "[ERROR]" in text:
        # allow template but ensure user edited: we still run and it will exit 1
        return text
    return text

def _render(template: str, dataset_dir: Path, out_dir: Path, rank: int, alpha: int) -> str:
    return (template
            .replace("{DATASET_DIR}", str(dataset_dir))
            .replace("{OUT_DIR}", str(out_dir))
            .replace("{RANK}", str(rank))
            .replace("{ALPHA}", str(alpha))).strip()

def run_gpt_sovits_vc_lora(project_root: Path, dataset_dir: Path, out_dir: Path, rank: int, alpha: int, extra: str = "") -> subprocess.Popen:
    _ensure_dir(out_dir)
    template = _load_cmd_template(project_root, "RUN_VC_LORA.cmd")
    cmd = _render(template, dataset_dir, out_dir, rank, alpha)
    if extra:
        cmd = cmd + " " + extra
    return subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=_env_for_subprocess(),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
    )

def run_xtts_tts_lora(project_root: Path, dataset_dir: Path, out_dir: Path, rank: int, alpha: int, extra: str = "") -> subprocess.Popen:
    _ensure_dir(out_dir)
    template = _load_cmd_template(project_root, "RUN_TTS_LORA.cmd")
    cmd = _render(template, dataset_dir, out_dir, rank, alpha)
    if extra:
        cmd = cmd + " " + extra
    return subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=_env_for_subprocess(),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
    )
