from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


SPEAKER_REFERENCE_QUALITY_WEIGHTS = {
    "naturally_clean_long": 1.0,
    "pure_chunk_long": 0.9,
    "naturally_clean_short": 0.65,
    "pure_chunk_short": 0.55,
    "original_overlap": 0.4,
    "original_longest": 0.3,
}


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


def _merged_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and merged[-1][1] >= start:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(
    start: float,
    end: float,
    mask: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    current_start = float(start)
    chunks: list[tuple[float, float]] = []
    for mask_start, mask_end in mask:
        if mask_end <= current_start:
            continue
        if mask_start >= end:
            break
        if mask_start > current_start:
            chunks.append((current_start, min(float(mask_start), float(end))))
        current_start = max(current_start, float(mask_end))
    if current_start < end:
        chunks.append((current_start, float(end)))
    return [(chunk_start, chunk_end) for chunk_start, chunk_end in chunks if chunk_end > chunk_start]


def _overlap_seconds(
    start: float,
    end: float,
    intervals: list[tuple[float, float]],
) -> float:
    total = 0.0
    for interval_start, interval_end in intervals:
        overlap = min(float(end), interval_end) - max(float(start), interval_start)
        if overlap > 0.0:
            total += overlap
    return total


def _candidate(
    *,
    start: float,
    end: float,
    candidate_type: str,
    overlap_ratio: float = 0.0,
) -> dict[str, Any]:
    duration = max(0.0, float(end) - float(start))
    return {
        "start": round(float(start), 6),
        "end": round(float(end), 6),
        "duration": round(duration, 6),
        "type": candidate_type,
        "weight": float(SPEAKER_REFERENCE_QUALITY_WEIGHTS[candidate_type]),
        "overlap_ratio": round(float(overlap_ratio), 6),
    }


def select_speaker_reference_candidates(
    df: pd.DataFrame,
    speaker: str,
    *,
    min_segment_duration: float = 2.0,
) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []

    speaker_key = str(speaker)
    rows = df.loc[df["speaker"].astype(str) == speaker_key].copy()
    if rows.empty:
        return []

    other_intervals = _merged_intervals(
        [
            (float(row["start"]), float(row["end"]))
            for _, row in df.iterrows()
            if str(row["speaker"]) != speaker_key
        ]
    )

    naturally_clean_long: list[dict[str, Any]] = []
    pure_chunk_long: list[dict[str, Any]] = []
    naturally_clean_short: list[dict[str, Any]] = []
    pure_chunk_short: list[dict[str, Any]] = []
    original_overlap: list[dict[str, Any]] = []
    original_longest: list[dict[str, Any]] = []

    min_duration = float(min_segment_duration)
    for _, row in rows.iterrows():
        start = float(row["start"])
        end = float(row["end"])
        duration = max(0.0, end - start)
        if duration <= 0.0:
            continue

        overlap = _overlap_seconds(start, end, other_intervals)
        overlap_ratio = overlap / duration if duration > 0.0 else 0.0
        chunks = _subtract_intervals(start, end, other_intervals)
        is_naturally_clean = len(chunks) == 1 and chunks[0][0] == start and chunks[0][1] == end

        for chunk_start, chunk_end in chunks:
            chunk_duration = chunk_end - chunk_start
            if is_naturally_clean and chunk_duration >= min_duration:
                naturally_clean_long.append(
                    _candidate(start=chunk_start, end=chunk_end, candidate_type="naturally_clean_long")
                )
            elif not is_naturally_clean and chunk_duration >= min_duration:
                pure_chunk_long.append(_candidate(start=chunk_start, end=chunk_end, candidate_type="pure_chunk_long"))
            elif is_naturally_clean:
                naturally_clean_short.append(
                    _candidate(start=chunk_start, end=chunk_end, candidate_type="naturally_clean_short")
                )
            else:
                pure_chunk_short.append(_candidate(start=chunk_start, end=chunk_end, candidate_type="pure_chunk_short"))

        if overlap > 0.0:
            original_overlap.append(
                _candidate(
                    start=start,
                    end=end,
                    candidate_type="original_overlap",
                    overlap_ratio=overlap_ratio,
                )
            )

        original_longest.append(
            _candidate(
                start=start,
                end=end,
                candidate_type="original_longest",
                overlap_ratio=overlap_ratio,
            )
        )

    naturally_clean_long.sort(key=lambda item: (-item["duration"], item["start"]))
    pure_chunk_long.sort(key=lambda item: (-item["duration"], item["start"]))
    naturally_clean_short.sort(key=lambda item: (-item["duration"], item["start"]))
    pure_chunk_short.sort(key=lambda item: (-item["duration"], item["start"]))
    original_overlap.sort(key=lambda item: (item["overlap_ratio"], -item["duration"], item["start"]))
    original_longest.sort(key=lambda item: (-item["duration"], item["overlap_ratio"], item["start"]))

    ordered: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for tier in (
        naturally_clean_long,
        pure_chunk_long,
        naturally_clean_short,
        pure_chunk_short,
        original_overlap,
        original_longest,
    ):
        for item in tier:
            key = (float(item["start"]), float(item["end"]))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
    return ordered


def build_weighted_reference_embedding(
    candidates: list[dict[str, Any]],
    *,
    embedding_fn: Callable[[float, float], Any],
    max_segments_per_speaker: int = 3,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    embeddings: list[np.ndarray] = []
    weights: list[float] = []
    used_segments: list[dict[str, Any]] = []

    for candidate in candidates:
        if len(embeddings) >= int(max_segments_per_speaker):
            break
        start = float(candidate["start"])
        end = float(candidate["end"])
        try:
            embedding = _as_embedding(embedding_fn(start, end))
        except Exception:
            embedding = None
        if embedding is None:
            continue

        weight = float(candidate.get("weight", 1.0))
        embeddings.append(embedding)
        weights.append(weight)
        used_segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(float(candidate.get("duration", end - start)), 3),
                "type": str(candidate.get("type", "unknown")),
                "weight": round(weight, 3),
                "overlap_ratio": round(float(candidate.get("overlap_ratio", 0.0)), 3),
            }
        )

    report = {
        "segment_count": len(embeddings),
        "quality_weight": round(float(np.mean(weights)) if weights else 0.0, 6),
        "segments": used_segments,
    }
    if not embeddings:
        return None, report

    weighted = np.average(np.asarray(embeddings), axis=0, weights=np.asarray(weights, dtype=np.float32))
    return _as_embedding(weighted), report
