from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.trace_artifacts import write_audio_wav


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".opus", ".ogg"}
SORTFORMER_POSTPROCESSING_FIELDS = (
    ("onset", "sortformer_pp_onset"),
    ("offset", "sortformer_pp_offset"),
    ("pad_onset", "sortformer_pp_pad_onset"),
    ("pad_offset", "sortformer_pp_pad_offset"),
    ("min_duration_on", "sortformer_pp_min_duration_on"),
    ("min_duration_off", "sortformer_pp_min_duration_off"),
)
SORTFORMER_STREAMING_FIELDS = (
    ("chunk_len", "sortformer_chunk_len"),
    ("chunk_left_context", "sortformer_chunk_left_context"),
    ("chunk_right_context", "sortformer_chunk_right_context"),
    ("fifo_len", "sortformer_fifo_len"),
    ("spkcache_update_period", "sortformer_spkcache_update_period"),
    ("spkcache_len", "sortformer_spkcache_len"),
)


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


def build_sortformer_postprocessing_parameters(args) -> dict[str, float]:
    return {
        yaml_key: float(getattr(args, arg_name))
        for yaml_key, arg_name in SORTFORMER_POSTPROCESSING_FIELDS
    }


def write_sortformer_postprocessing_yaml(path: Path, args) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = build_sortformer_postprocessing_parameters(args)
    lines = ["parameters:"]
    lines.extend(f"  {key}: {value}" for key, value in params.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def resolve_sortformer_postprocessing_yaml(
    args,
    run_dir: Path,
    cwd: Path | None = None,
    script_dir: Path | None = None,
) -> Path | None:
    requested = str(getattr(args, "sortformer_postprocessing_yaml", "") or "").strip()
    if requested:
        cwd = Path.cwd() if cwd is None else Path(cwd)
        script_dir = Path(__file__).resolve().parents[1] if script_dir is None else Path(script_dir)
        requested_path = Path(requested)
        candidates = [requested_path] if requested_path.is_absolute() else [cwd / requested_path, script_dir / requested_path]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"sortformer_postprocessing_yaml not found: {requested}. Searched: {searched}")

    if not bool(getattr(args, "sortformer_postprocessing", False)):
        return None

    return write_sortformer_postprocessing_yaml(
        Path(run_dir) / "01_diarization" / "sortformer_postprocessing.yaml",
        args,
    )


def apply_sortformer_streaming_config(diar_model, args, logger=None) -> dict[str, int]:
    if not bool(getattr(args, "sortformer_streaming_config", False)):
        return {}

    modules = getattr(diar_model, "sortformer_modules", None)
    if modules is None:
        if logger is not None:
            logger.warning("Sortformer model has no sortformer_modules; streaming config skipped.")
        return {}

    applied: dict[str, int] = {}
    for model_attr, arg_name in SORTFORMER_STREAMING_FIELDS:
        value = int(getattr(args, arg_name))
        setattr(modules, model_attr, value)
        applied[model_attr] = value

    if logger is not None:
        logger.info(f"Applied Sortformer streaming config: {applied}")
    return applied


def _as_embedding(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    embedding = np.asarray(value, dtype=np.float32)
    if embedding.size == 0:
        return None
    if embedding.ndim > 1:
        embedding = embedding.mean(axis=0)
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return None
    return embedding / norm


def _cosine_similarity(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return -1.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return -1.0
    return float(np.dot(vec_a, vec_b) / denom)


def _window_embedding(
    embedding_fn: Callable[[float, float], Any],
    start: float,
    end: float,
    min_duration: float,
) -> np.ndarray | None:
    if end - start < min_duration:
        return None
    return _as_embedding(embedding_fn(float(start), float(end)))


def _speaker_reference_embeddings(
    df: pd.DataFrame,
    embedding_fn: Callable[[float, float], Any],
    embedding_window: float,
    min_segment: float,
) -> dict[str, np.ndarray]:
    references: dict[str, list[np.ndarray]] = {}
    min_reference_duration = max(float(embedding_window), float(min_segment))

    for speaker, rows in df.groupby("speaker"):
        for _, row in rows.sort_values("start").iterrows():
            start = float(row["start"])
            end = float(row["end"])
            duration = end - start
            if duration < min_reference_duration:
                continue
            center = (start + end) / 2.0
            half_window = min(float(embedding_window), duration) / 2.0
            embedding = _window_embedding(
                embedding_fn,
                center - half_window,
                center + half_window,
                min_duration=min(float(embedding_window), duration) * 0.75,
            )
            if embedding is not None:
                references.setdefault(str(speaker), []).append(embedding)
            if len(references.get(str(speaker), [])) >= 6:
                break

    return {
        speaker: _as_embedding(np.mean(embeddings, axis=0))
        for speaker, embeddings in references.items()
        if embeddings
    }


def _boundary_score(
    embedding_fn: Callable[[float, float], Any],
    boundary: float,
    left_start: float,
    right_end: float,
    left_reference: np.ndarray,
    right_reference: np.ndarray,
    embedding_window: float,
) -> float | None:
    left_embedding = _window_embedding(
        embedding_fn,
        max(left_start, boundary - embedding_window),
        boundary,
        min_duration=embedding_window * 0.75,
    )
    right_embedding = _window_embedding(
        embedding_fn,
        boundary,
        min(right_end, boundary + embedding_window),
        min_duration=embedding_window * 0.75,
    )
    if left_embedding is None or right_embedding is None:
        return None
    return (
        _cosine_similarity(left_embedding, left_reference)
        + _cosine_similarity(right_embedding, right_reference)
        - _cosine_similarity(left_embedding, right_reference)
        - _cosine_similarity(right_embedding, left_reference)
    )


def _candidate_boundaries(
    original_boundary: float,
    lower: float,
    upper: float,
    step: float,
) -> list[float]:
    step = max(float(step), 0.001)
    values = {round(float(original_boundary), 3)}
    current = lower
    while current <= upper + 1e-9:
        values.add(round(float(current), 3))
        current += step
    return sorted(v for v in values if lower - 1e-9 <= v <= upper + 1e-9)


def refine_speaker_boundaries(
    df: pd.DataFrame,
    embedding_fn: Callable[[float, float], Any],
    max_shift: float = 0.4,
    step: float = 0.05,
    embedding_window: float = 0.4,
    min_segment: float = 0.6,
    max_gap: float = 0.35,
    min_improvement: float = 0.05,
    logger=None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Move close speaker-change boundaries toward the point that best matches each
    adjacent speaker's reference embedding.
    """
    if df is None or df.empty or len(df) < 2:
        return df, []

    refined = df.sort_values("start").reset_index(drop=True).copy()
    references = _speaker_reference_embeddings(
        refined,
        embedding_fn=embedding_fn,
        embedding_window=float(embedding_window),
        min_segment=float(min_segment),
    )
    if len(references) < 2:
        if logger is not None:
            logger.warning("Speaker boundary refinement skipped: not enough speaker references.")
        return refined, []

    adjustments: list[dict[str, Any]] = []
    for idx in range(len(refined) - 1):
        left = refined.loc[idx]
        right = refined.loc[idx + 1]
        left_speaker = str(left["speaker"])
        right_speaker = str(right["speaker"])
        if left_speaker == right_speaker:
            continue
        if left_speaker not in references or right_speaker not in references:
            continue

        left_start = float(left["start"])
        left_end = float(refined.loc[idx, "end"])
        right_start = float(refined.loc[idx + 1, "start"])
        right_end = float(right["end"])
        if left_end <= left_start or right_end <= right_start:
            continue
        gap = right_start - left_end
        if abs(gap) > float(max_gap):
            continue

        original_boundary = (left_end + right_start) / 2.0
        lower = max(original_boundary - float(max_shift), left_start + float(min_segment))
        upper = min(original_boundary + float(max_shift), right_end - float(min_segment))
        if lower > upper:
            continue

        original_score = _boundary_score(
            embedding_fn,
            original_boundary,
            left_start,
            right_end,
            references[left_speaker],
            references[right_speaker],
            float(embedding_window),
        )
        best_boundary = original_boundary
        best_score = original_score

        for candidate in _candidate_boundaries(original_boundary, lower, upper, float(step)):
            score = _boundary_score(
                embedding_fn,
                candidate,
                left_start,
                right_end,
                references[left_speaker],
                references[right_speaker],
                float(embedding_window),
            )
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_boundary = candidate
                best_score = score

        if best_score is None:
            continue
        score_before = float(original_score) if original_score is not None else -1.0
        improvement = float(best_score) - score_before
        if abs(best_boundary - original_boundary) < max(float(step) / 2.0, 0.001):
            continue
        if original_score is not None and improvement < float(min_improvement):
            continue

        best_boundary = round(float(best_boundary), 3)
        refined.loc[idx, "end"] = best_boundary
        refined.loc[idx + 1, "start"] = best_boundary
        adjustments.append(
            {
                "left_index": int(idx),
                "right_index": int(idx + 1),
                "left_speaker": left_speaker,
                "right_speaker": right_speaker,
                "old_left_end": round(left_end, 3),
                "old_right_start": round(right_start, 3),
                "old_boundary": round(original_boundary, 3),
                "new_boundary": best_boundary,
                "shift_seconds": round(best_boundary - original_boundary, 3),
                "score_before": round(score_before, 6),
                "score_after": round(float(best_score), 6),
                "score_improvement": round(improvement, 6),
            }
        )

    if logger is not None:
        logger.info(f"Speaker boundary refinement adjusted {len(adjustments)} boundaries.")
    return refined, adjustments


def write_boundary_refinement_report(
    run_dir: Path,
    adjustments: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> Path:
    out_path = Path(run_dir) / "01_diarization" / "boundary_refinements.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "stage": "speaker_boundary_refinement",
            "adjustment_count": len(adjustments),
            "parameters": parameters,
        },
        "adjustments": adjustments,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


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
