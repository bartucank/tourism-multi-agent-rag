import atexit
import base64
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import importlib
import importlib.util
import io
import json
import os
import re
import select
import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np

from main import clean

_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
_process_lock = threading.Lock()
_speech_process = None


class SpeechWorkerError(RuntimeError):
    pass


def _read_response(process, timeout: int) -> dict:
    if process.stdout is None or not select.select([process.stdout], [], [], timeout)[0]:
        raise TimeoutError("Seslendirme worker'ı zaman aşımına uğradı.")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(
            f"Seslendirme worker'ı beklenmedik şekilde kapandı (kod {process.poll()})."
        )
    try:
        response = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SpeechWorkerError("Seslendirme worker'ından geçersiz yanıt alındı.") from exc
    if "error" in response:
        raise SpeechWorkerError(response["error"])
    return response


def _start_speech_process():
    global _speech_process
    _speech_process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    try:
        _read_response(_speech_process, timeout=1800)
    except Exception:
        _stop_speech_process()
        raise


def _stop_speech_process():
    global _speech_process
    process, _speech_process = _speech_process, None
    if process is None:
        return
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def _persistent_speech(answer: str) -> bytes:
    global _speech_process
    with _process_lock:
        if _speech_process is None or _speech_process.poll() is not None:
            _start_speech_process()
        process = _speech_process
        try:
            if process.stdin is None:
                raise RuntimeError("Seslendirme worker'ının giriş kanalı açılamadı.")
            process.stdin.write(json.dumps({"text": answer}, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = _read_response(process, timeout=1800)
            return base64.b64decode(response["audio"], validate=True)
        except (BrokenPipeError, OSError, RuntimeError, KeyError, ValueError) as exc:
            if not isinstance(exc, SpeechWorkerError) or process.poll() is not None:
                _stop_speech_process()
            raise


atexit.register(_stop_speech_process)


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
    return _worker.submit(_persistent_speech, answer)


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


def synthesize(answer: str, model=None) -> bytes:
    text = spoken_text(answer)
    if not text:
        raise ValueError("Seslendirilecek metin bulunamadı.")
    speech = (model or load_model()).say(text)
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


def _run_worker() -> int:
    try:
        with redirect_stdout(sys.stderr):
            model = load_model()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                audio = synthesize(request["text"], model=model)
            response = {"audio": base64.b64encode(audio).decode("ascii")}
        except Exception as exc:
            response = {"error": str(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_worker())
