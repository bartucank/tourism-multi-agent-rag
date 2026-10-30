from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import importlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from main import clean

_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")


@lru_cache(maxsize=1)
def load_model():
    if importlib.util.find_spec("ema_lightning") is None:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                 "--no-input", "ema-lightning==1.1.2"],
                capture_output=True, text=True, timeout=900,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Seslendirme paketi kurulumu 15 dakika içinde tamamlanamadı.") from exc
        if result.returncode:
            raise RuntimeError(
                "Seslendirme paketi kurulamadı:\n" + (result.stderr or result.stdout)[-6000:]
            )
        importlib.invalidate_caches()
    from ema_lightning import EMA
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return EMA()


def request_speech(answer: str):
    # One worker prevents simultaneous installs and serializes model initialization.
    return _worker.submit(_isolated_speech, answer)


def _isolated_speech(answer: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="tourism-tts-") as directory:
        output = Path(directory) / "speech.wav"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(output)],
            input=answer, capture_output=True, text=True, timeout=1800,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        if result.returncode:
            raise RuntimeError(
                f"Seslendirme süreci durdu (kod {result.returncode}):\n"
                + (result.stderr or result.stdout)[-6000:]
            )
        return output.read_bytes()


def spoken_text(answer: str) -> str:
    text = clean(answer, keep_citations=False)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    # Omit parenthetical asides, including nested ones, from speech only.
    depth = 0
    outside = []
    for character in text:
        if character == "(":
            depth += 1
            outside.append(" ")
        elif character == ")" and depth:
            depth -= 1
        elif not depth:
            outside.append(character)
    text = "".join(outside)
    text = re.sub(r"\bK\s*\d+\b", "", text, flags=re.I)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|>\s*)", "", text)
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"[()[\]{}|]", " ", text)
    # Keep numeric times/ratios intact for the model's Turkish normalizer.
    text = re.sub(r"(?<!\d):|:(?!\d)", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"([,;])(?:\s*[,;])+", r"\1", text)
    text = re.sub(r"[,;]\s*([.!?])", r"\1", text)
    return text.strip(" ,;")


def synthesize(answer: str) -> bytes:
    text = spoken_text(answer)
    if not text:
        raise ValueError("Seslendirilecek metin bulunamadı.")
    speech = load_model().say(text)
    audio = np.asarray(speech.audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("TTS modeli geçerli bir ses üretmedi.")
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(speech.sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


if __name__ == "__main__":
    Path(sys.argv[1]).write_bytes(synthesize(sys.stdin.read()))
