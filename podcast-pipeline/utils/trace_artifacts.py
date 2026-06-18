from __future__ import annotations

import json
import math
import re
import shutil
import struct
import wave
from pathlib import Path
from typing import Any, Iterable


SKIPPED_SEGMENT_KEYS = {
    "enhanced_audio",
    "audio",
    "waveform",
    "audio_segment",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def clean_segment_for_json(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): json_safe(value)
        for key, value in segment.items()
        if key not in SKIPPED_SEGMENT_KEYS and not str(key).startswith("_")
    }


def clean_segments_for_json(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [clean_segment_for_json(segment) for segment in segments]


def _safe_name(value: Any) -> str:
    text = str(value or "segment")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "segment"


def _flatten_waveform(waveform: Any) -> list[float]:
    if hasattr(waveform, "tolist"):
        waveform = waveform.tolist()
    if waveform is None:
        return []
    if isinstance(waveform, (int, float)):
        return [float(waveform)]
    flattened: list[float] = []
    for item in waveform:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten_waveform(item))
        else:
            try:
                flattened.append(float(item))
            except (TypeError, ValueError):
                continue
    return flattened


def write_waveform_wav(path: str | Path, waveform: Any, sample_rate: int) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = _flatten_waveform(waveform)
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        frames = bytearray()
        for sample in samples:
            clipped = max(-1.0, min(1.0, sample))
            frames.extend(struct.pack("<h", int(round(clipped * 32767))))
        handle.writeframes(bytes(frames))
    return out_path


def write_audio_wav(path: str | Path, audio: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_segment = audio.get("audio_segment") if isinstance(audio, dict) else None
    if audio_segment is not None and hasattr(audio_segment, "export"):
        audio_segment.export(out_path, format="wav")
        return out_path
    return write_waveform_wav(out_path, audio.get("waveform", []), int(audio.get("sample_rate", 16000)))


class TraceRunWriter:
    def __init__(
        self,
        run_dir: str | Path | None,
        *,
        source_audio_path: str | Path | None = None,
        logger: Any | None = None,
    ) -> None:
        self.run_dir = Path(run_dir) if run_dir else None
        self.source_audio_path = str(source_audio_path) if source_audio_path else ""
        self.logger = logger

    @property
    def enabled(self) -> bool:
        return self.run_dir is not None

    def _path(self, relative: str) -> Path:
        if self.run_dir is None:
            raise RuntimeError("TraceRunWriter is disabled")
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _metadata(self, stage: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "stage": stage,
            "trace": True,
            "proxy": False,
        }
        if self.source_audio_path:
            payload["source_audio_path"] = self.source_audio_path
        if metadata:
            payload.update(json_safe(metadata))
        return payload

    def _write_json(self, relative: str, payload: dict[str, Any]) -> Path:
        path = self._path(relative)
        path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        if self.logger is not None:
            try:
                self.logger.info(f"Trace artifact saved: {path}")
            except Exception:
                pass
        return path

    def write_diarization(
        self,
        segments: list[dict[str, Any]],
        *,
        micro_overlap_candidates: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        payload = {
            "segments": clean_segments_for_json(segments),
            "metadata": self._metadata("diarization", metadata),
        }
        if micro_overlap_candidates is not None:
            payload["micro_overlap_candidates"] = clean_segments_for_json(micro_overlap_candidates)
        return self._write_json("01_diarization/diarization.json", payload)

    def write_music_clean(
        self,
        segments: list[dict[str, Any]],
        demucs_flags: list[bool],
        audio: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        cleaned_audio_path = self._path("02_music_clean/cleaned_audio.wav")
        write_audio_wav(cleaned_audio_path, audio)
        return self._write_json(
            "02_music_clean/segment_flags.json",
            {
                "segments": clean_segments_for_json(segments),
                "segment_demucs_flags": [bool(flag) for flag in demucs_flags],
                "audio_path": "02_music_clean/cleaned_audio.wav",
                "metadata": self._metadata("music_clean", metadata),
            },
        )

    def write_overlap(
        self,
        segments: list[dict[str, Any]],
        audio: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        sample_rate = int(audio.get("sample_rate", 16000))
        clean_segments: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            clean = clean_segment_for_json(segment)
            enhanced_audio = segment.get("enhanced_audio")
            if enhanced_audio is not None:
                idx = _safe_name(segment.get("index", f"{index:05d}"))
                speaker = _safe_name(segment.get("speaker", "speaker"))
                relative = f"03_overlap/separated_segments/{idx}_{speaker}.wav"
                write_waveform_wav(self._path(relative), enhanced_audio, sample_rate)
                clean["enhanced_audio_path"] = relative
                clean["is_separated"] = True
            clean_segments.append(clean)
        return self._write_json(
            "03_overlap/segments.json",
            {
                "segments": clean_segments,
                "audio_path": "02_music_clean/cleaned_audio.wav",
                "metadata": self._metadata("overlap_separation", metadata),
            },
        )

    def write_asr(
        self,
        segments: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        return self._write_json(
            "04_asr/transcript.json",
            {
                "segments": clean_segments_for_json(segments),
                "audio_path": "02_music_clean/cleaned_audio.wav",
                "metadata": self._metadata("asr", metadata),
            },
        )

    def write_export(
        self,
        final_data: dict[str, Any],
        *,
        source_segments_dir: str | Path | None = None,
        source_json_path: str | Path | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        output = json_safe(final_data)
        output["segments"] = clean_segments_for_json(output.get("segments", []))
        final_dir = self._path("05_export/final")
        data_audio_dir = final_dir / "data_audio"
        data_audio_dir.mkdir(parents=True, exist_ok=True)

        mp3_files: list[Path] = []
        if source_segments_dir:
            source_dir = Path(source_segments_dir)
            if source_dir.exists():
                for path in sorted(source_dir.glob("*.mp3")):
                    destination = data_audio_dir / path.name
                    shutil.copy2(path, destination)
                    mp3_files.append(destination)

        for index, segment in enumerate(output.get("segments", [])):
            if index < len(mp3_files):
                segment["audio_file"] = f"data_audio/{mp3_files[index].name}"

        metadata = output.setdefault("metadata", {})
        metadata["trace"] = True
        metadata["proxy"] = False
        if source_json_path:
            metadata["source_json_path"] = str(source_json_path)
        metadata["copied_mp3_count"] = len(mp3_files)

        return self._write_json("05_export/final/data_audio.json", output)
