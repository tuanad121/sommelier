from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from utils.trace_artifacts import write_audio_wav


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".opus", ".ogg"}


def resolve_config_path(
    config_path: str,
    cwd: Path | None = None,
    script_dir: Path | None = None,
) -> Path:
    """
    Resolve config paths from either the repository root or podcast-pipeline directory.
    """
    cwd = Path.cwd() if cwd is None else Path(cwd)
    script_dir = Path(__file__).resolve().parents[1] if script_dir is None else Path(script_dir)
    requested = Path(config_path)

    if requested.is_absolute():
        if requested.exists():
            return requested
        raise FileNotFoundError(f"config_path not found: {requested}")

    candidates = [
        cwd / requested,
        script_dir / requested,
        script_dir / requested.name,
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in seen)
    raise FileNotFoundError(f"config_path not found: {config_path}. Searched: {searched}")


def build_run_dir(output_root: Path, audio_path: str) -> Path:
    audio_name = Path(audio_path).stem
    run_dir = Path(output_root) / f"run_full_{audio_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_input_artifacts(
    run_dir: Path,
    audio: dict[str, Any],
    chunk_entries: list[dict[str, Any]],
    write_audio: Callable[[Path, dict[str, Any]], None] = write_audio_wav,
) -> None:
    input_dir = Path(run_dir) / "00_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    write_audio(input_dir / "full.wav", audio)

    payload = {
        "audio_path": "00_input/full.wav",
        "chunks": chunk_entries,
        "vad_chunks": chunk_entries,
        "metadata": {"stage": "speaker_diarization", "trace": True},
    }
    for relative in [
        "01_diarization/vad_chunks.json",
        "01_diarization/trace_vad_chunks.json",
    ]:
        out = Path(run_dir) / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_audio_paths(args, cfg: dict[str, Any]) -> list[str]:
    if args.input_audio_path:
        return [args.input_audio_path]

    input_folder = args.input_folder_path or cfg.get("entrypoint", {}).get("input_folder_path", "")
    if not input_folder:
        raise ValueError("Missing input path. Set --input_audio_path or --input_folder_path.")

    folder = Path(input_folder)
    if not folder.exists():
        raise FileNotFoundError(f"input_folder_path not found: {folder}")

    return [
        str(file)
        for file in sorted(folder.iterdir())
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
    ]
