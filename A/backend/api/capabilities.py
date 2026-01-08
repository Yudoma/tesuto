from fastapi import APIRouter
from pathlib import Path

router = APIRouter()

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

@router.get("/api/capabilities")
def capabilities():
    gsv = Path("third_party/GPT-SoVITS")
    xtts = Path("third_party/XTTS")
    return {
  "capabilities": {"text": True, "image": True, "audio": True},
  "audio_lora": {
    "vc_gpt_sovits": _exists(Path("third_party/GPT-SoVITS")) and _exists(Path("third_party/GPT-SoVITS") / "GPT_SoVITS") or _exists(Path("third_party/GPT-SoVITS") / "configs"),
    "tts_xtts": _exists(Path("third_party/XTTS")) and _exists(Path("third_party/XTTS") / "TTS"),
  }
}
