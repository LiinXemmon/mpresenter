from __future__ import annotations

import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .config import TTSConfig
from .logger import info, warning
from .utils import ensure_dir


def synthesize_audio(
    scripts: Iterable[dict],
    output_dir: Path,
    target_language: str,
    tts_config: TTSConfig,
) -> List[Path]:
    ensure_dir(output_dir)
    audio_paths: List[Path] = []

    engine_name = (tts_config.engine or "").strip().lower()
    if engine_name in {"none", "silent"}:
        return _synthesize_silent(scripts, output_dir)

    tts_engine = _init_tts_engine(engine_name, target_language, tts_config)

    for index, script in enumerate(scripts, start=1):
        text = script.get("final") or script.get("original") or ""
        segments = [text] if _has_language_features(text) else []
        output_path = output_dir / f"slide_{index:02d}.wav"

        if not segments:
            duration = max(2.0, len(text) / 15.0)
            _write_silence(output_path, duration)
            warning("Synthesizer", f"No language features found; generated silent audio for slide {index}")
            audio_paths.append(output_path)
            continue

        if not tts_engine.synthesize_segments(segments, output_path, target_language):
            raise RuntimeError(f"TTS synthesis failed for slide {index}")
        info("Synthesizer", f"Generated audio for slide {index}")
        audio_paths.append(output_path)

    return audio_paths


def _synthesize_silent(scripts: Iterable[dict], output_dir: Path) -> List[Path]:
    audio_paths: List[Path] = []
    for index, script in enumerate(scripts, start=1):
        text = script.get("final") or script.get("original") or ""
        output_path = output_dir / f"slide_{index:02d}.wav"
        duration = max(2.0, len(text) / 15.0)
        _write_silence(output_path, duration)
        info("Synthesizer", f"Generated silent audio for slide {index}")
        audio_paths.append(output_path)
    return audio_paths


def _has_language_features(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


class CosyVoiceEngine:
    def __init__(
        self,
        model,
        prompt_wav: Path,
        prompt_text: str | None,
        inference: str,
        speed: float,
        sample_rate: int,
        text_frontend: bool,
    ) -> None:
        self._model = model
        self._prompt_wav = prompt_wav
        self._prompt_text = prompt_text
        self._inference = inference
        self._speed = speed
        self._text_frontend = text_frontend
        model_rate = getattr(model, "sample_rate", None)
        if model_rate:
            self._sample_rate = int(model_rate)
        else:
            self._sample_rate = int(sample_rate) if sample_rate else 22050

    def synthesize_segments(self, segments: Iterable[str], output_path: Path, target_language: str) -> bool:
        segment_files: List[Path] = []
        for segment in segments:
            try:
                outputs = self._run_inference(segment)
            except Exception as exc:
                warning("Synthesizer", f"CosyVoice inference failed on a segment: {exc}")
                continue

            samples = _collect_cosyvoice_samples(outputs)
            if samples is None:
                warning("Synthesizer", "CosyVoice returned no audio samples for a segment")
                continue

            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            tmp_file = Path(tmp_path)
            _write_wav_samples(tmp_file, samples, self._sample_rate)
            segment_files.append(tmp_file)

        if not segment_files:
            duration = max(2.0, sum(len(seg) for seg in segments) / 15.0)
            _write_silence(output_path, duration, sample_rate=self._sample_rate)
            warning("Synthesizer", "No valid CosyVoice segments; generated silent audio")
            return True

        ok = _concat_wav(segment_files, output_path)
        for tmp_file in segment_files:
            tmp_file.unlink(missing_ok=True)
        return ok

    def _run_inference(self, text: str):
        prompt_wav = str(self._prompt_wav)
        if self._inference == "cross_lingual":
            return self._model.inference_cross_lingual(
                text,
                prompt_wav,
                stream=False,
                speed=self._speed,
                text_frontend=self._text_frontend,
            )
        if self._inference == "instruct2":
            if not self._prompt_text:
                raise RuntimeError("cosyvoice_prompt_text is required for instruct2 inference")
            return self._model.inference_instruct2(
                text,
                self._prompt_text,
                prompt_wav,
                stream=False,
                speed=self._speed,
                text_frontend=self._text_frontend,
            )
        if self._inference == "zero_shot":
            if not self._prompt_text:
                raise RuntimeError("cosyvoice_prompt_text is required for zero_shot inference")
            return self._model.inference_zero_shot(
                text,
                self._prompt_text,
                prompt_wav,
                stream=False,
                speed=self._speed,
                text_frontend=self._text_frontend,
            )
        raise RuntimeError(f"Unsupported CosyVoice inference mode: {self._inference}")


class Pyttsx3Engine:
    def __init__(self) -> None:
        import pyttsx3  # type: ignore

        self._engine = pyttsx3.init()

    def synthesize_segments(self, segments: Iterable[str], output_path: Path, target_language: str) -> bool:
        segment_files: List[Path] = []
        for segment in segments:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            tmp_file = Path(tmp_path)
            self._engine.save_to_file(segment, str(tmp_file))
            segment_files.append(tmp_file)
        self._engine.runAndWait()

        ok = _concat_wav(segment_files, output_path)
        for tmp_file in segment_files:
            tmp_file.unlink(missing_ok=True)
        return ok


def _init_tts_engine(engine: str, target_language: str, tts_config: TTSConfig):
    if engine in {"cosyvoice", "cosyvoice3", "cosyvoice-3"}:
        return _init_cosyvoice_engine(tts_config)

    if engine == "pyttsx3":
        try:
            return Pyttsx3Engine()
        except ImportError as exc:
            raise RuntimeError("pyttsx3 not installed; configure another TTS engine") from exc

    raise RuntimeError(f"Unsupported TTS engine: {engine}")


def _init_cosyvoice_engine(tts_config: TTSConfig) -> CosyVoiceEngine:
    model_id = (tts_config.cosyvoice_model or "").strip() or "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    root = _resolve_cosyvoice_root(tts_config)
    device = (tts_config.cosyvoice_device or "auto").strip().lower()
    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if root:
        cosyvoice_root = str(root)
        if cosyvoice_root not in sys.path:
            sys.path.insert(0, cosyvoice_root)
        matcha_root = root / "third_party" / "Matcha-TTS"
        if matcha_root.exists():
            matcha_path = str(matcha_root)
            if matcha_path not in sys.path:
                sys.path.insert(0, matcha_path)

    try:
        from cosyvoice.cli.cosyvoice import AutoModel
    except Exception as exc:
        raise RuntimeError(
            "Failed to import CosyVoice. Clone https://github.com/FunAudioLLM/CosyVoice "
            "and set tts.cosyvoice_root, or install CosyVoice as a Python package."
        ) from exc

    model = AutoModel(
        model_dir=model_id,
        fp16=tts_config.cosyvoice_fp16,
        load_trt=tts_config.cosyvoice_load_trt,
        load_vllm=tts_config.cosyvoice_load_vllm,
    )

    prompt_wav = _resolve_cosyvoice_prompt_wav(tts_config, root)
    if not prompt_wav.exists():
        raise RuntimeError(f"CosyVoice prompt wav not found: {prompt_wav}")

    inference = (tts_config.cosyvoice_inference or "zero_shot").strip().lower()
    prompt_text = tts_config.cosyvoice_prompt_text or _default_cosyvoice_prompt_text()
    return CosyVoiceEngine(
        model,
        prompt_wav,
        prompt_text,
        inference,
        tts_config.cosyvoice_speed,
        tts_config.cosyvoice_sample_rate,
        tts_config.cosyvoice_text_frontend,
    )


def _resolve_cosyvoice_root(tts_config: TTSConfig) -> Path | None:
    if tts_config.cosyvoice_root and tts_config.cosyvoice_root.exists():
        return tts_config.cosyvoice_root
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / "third_party" / "CosyVoice"
    if candidate.exists():
        return candidate
    return None


def _resolve_cosyvoice_prompt_wav(tts_config: TTSConfig, root: Path | None) -> Path:
    if tts_config.cosyvoice_prompt_wav:
        return tts_config.cosyvoice_prompt_wav
    if root:
        return root / "asset" / "zero_shot_prompt.wav"
    return Path("third_party") / "CosyVoice" / "asset" / "zero_shot_prompt.wav"


def _default_cosyvoice_prompt_text() -> str:
    return "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


def _collect_cosyvoice_samples(outputs) -> np.ndarray | None:
    if isinstance(outputs, dict):
        outputs = [outputs]
    chunks: list[np.ndarray] = []
    for output in outputs:
        audio = output.get("tts_speech") if isinstance(output, dict) else output
        samples = _coerce_audio_array(audio)
        if samples is not None:
            chunks.append(samples)
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def _extract_audio_payload(result):
    sample_rate = None
    audio = None
    if isinstance(result, dict):
        audio_keys = ["output_wav", "wav", "audio", "output"]
        rate_keys = ["sample_rate", "sampling_rate", "sr"]
        try:
            from modelscope.outputs import OutputKeys

            output_wav = getattr(OutputKeys, "OUTPUT_WAV", None)
            sample_rate_key = getattr(OutputKeys, "SAMPLE_RATE", None)
            if output_wav:
                audio_keys.insert(0, output_wav)
            if sample_rate_key:
                rate_keys.insert(0, sample_rate_key)
        except Exception:
            pass

        audio = _first_present(result, audio_keys)
        sample_rate = _first_present(result, rate_keys)
        if isinstance(audio, dict):
            sample_rate = sample_rate or audio.get("sampling_rate") or audio.get("sample_rate") or audio.get("sr")
            if "array" in audio:
                audio = audio["array"]
            elif "audio" in audio:
                audio = audio["audio"]
    elif isinstance(result, (tuple, list)) and len(result) == 2 and _looks_like_sample_rate(result[1]):
        audio, sample_rate = result
    else:
        audio = result

    if sample_rate is not None:
        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            sample_rate = None
    return audio, sample_rate


def _first_present(mapping: dict, keys: list) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _looks_like_sample_rate(value: object) -> bool:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return False
    return 8000 <= rate <= 192000


def _write_audio_payload(output_path: Path, audio: object, sample_rate: int | None, default_sample_rate: int) -> bool:
    if audio is None:
        return False
    if hasattr(audio, "read") and not isinstance(audio, (str, bytes, bytearray, Path)):
        try:
            data = audio.read()
        except Exception:
            data = None
        if data is not None:
            return _write_audio_payload(output_path, data, sample_rate, default_sample_rate)

    if isinstance(audio, (bytes, bytearray)):
        data = bytes(audio)
        if data[:4] == b"RIFF":
            output_path.write_bytes(data)
            return True
        return _write_pcm_bytes(output_path, data, sample_rate or default_sample_rate or 22050)

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if path.exists():
            output_path.write_bytes(path.read_bytes())
            return True
        if isinstance(audio, str):
            return False

    samples = _coerce_audio_array(audio)
    if samples is None:
        return False
    _write_wav_samples(output_path, samples, sample_rate or default_sample_rate or 22050)
    return True


def _coerce_audio_array(audio: object) -> np.ndarray | None:
    if audio is None:
        return None
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
    except Exception:
        pass
    try:
        arr = np.asarray(audio)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        if arr.shape[0] <= 2 and arr.shape[1] > 2:
            arr = arr[0]
        elif arr.shape[1] <= 2 and arr.shape[0] > 2:
            arr = arr[:, 0]
        else:
            arr = arr.reshape(-1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    return arr


def _write_pcm_bytes(output_path: Path, pcm_bytes: bytes, sample_rate: int) -> bool:
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm_bytes)
    return True


def _write_wav_samples(output_path: Path, samples: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(samples.tobytes())


def _concat_wav(files: Iterable[Path], output_path: Path) -> bool:
    files = list(files)
    if not files:
        return False

    with wave.open(str(files[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
        params_key = _wave_params_key(params)

    for wav_path in files[1:]:
        with wave.open(str(wav_path), "rb") as handle:
            if _wave_params_key(handle.getparams()) != params_key:
                return False
            frames.append(handle.readframes(handle.getnframes()))

    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)

    return True


def _wave_params_key(params: wave._wave_params) -> tuple[int, int, int, str, str]:
    return (params.nchannels, params.sampwidth, params.framerate, params.comptype, params.compname)


def _write_silence(output_path: Path, duration: float, sample_rate: int = 22050) -> None:
    num_frames = int(duration * sample_rate)
    silence = (b"\x00\x00" * num_frames)
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(silence)
