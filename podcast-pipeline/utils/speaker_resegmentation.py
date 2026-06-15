from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.trace_artifacts import json_safe


@dataclass(frozen=True)
class AuditConfig:
    enabled: bool = True
    boundary_window: float = 1.5
    interior_min_duration: float = 8.0
    max_shift: float = 0.8
    max_extend: float = 0.8
    min_duration: float = 0.3
    min_overlap_duration: float = 0.2
    min_mapping_score: float = 0.7
    min_mapping_margin: float = 0.08


@dataclass(frozen=True)
class LocalActivity:
    local_speaker: str
    start: float
    end: float
    embedding: Any
    score: float | None = None


@dataclass(frozen=True)
class AuditRegion:
    kind: str
    start: float
    end: float
    left_index: int | None = None
    right_index: int | None = None
    segment_index: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class MappedActivity:
    local_speaker: str
    global_speaker: str | None
    start: float
    end: float
    score: float | None
    margin: float | None
    top2_speaker: str | None
    confidence: str


LocalActivityProvider = Callable[[AuditRegion], list[LocalActivity]]


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


def _cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return -1.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return -1.0
    return float(np.dot(left, right) / denom)


def _normalized_references(references: dict[str, Any]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for speaker, embedding in references.items():
        vector = _as_embedding(embedding)
        if vector is not None:
            normalized[str(speaker)] = vector
    return normalized


def build_global_references(
    df: pd.DataFrame,
    *,
    embedding_fn: Callable[[float, float], Any],
    min_segment_duration: float = 2.0,
    max_segments_per_speaker: int = 6,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    references: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    if df is None or df.empty:
        return references, report

    for speaker, rows in df.groupby("speaker"):
        speaker_key = str(speaker)
        candidates = rows.copy()
        candidates["duration"] = candidates["end"].astype(float) - candidates["start"].astype(float)
        candidates = candidates[candidates["duration"] >= float(min_segment_duration)]
        candidates = candidates.sort_values("duration", ascending=False)

        embeddings: list[np.ndarray] = []
        used_segments: list[dict[str, float]] = []
        for _, row in candidates.iterrows():
            if len(embeddings) >= int(max_segments_per_speaker):
                break
            start = float(row["start"])
            end = float(row["end"])
            try:
                embedding = _as_embedding(embedding_fn(start, end))
            except Exception:
                embedding = None
            if embedding is None:
                continue
            embeddings.append(embedding)
            used_segments.append({"start": round(start, 3), "end": round(end, 3)})

        report[speaker_key] = {
            "segment_count": len(embeddings),
            "segments": used_segments,
        }
        if embeddings:
            references[speaker_key] = _as_embedding(np.mean(embeddings, axis=0))

    return {
        speaker: embedding
        for speaker, embedding in references.items()
        if embedding is not None
    }, report


def find_audit_regions(df: pd.DataFrame, config: AuditConfig) -> list[AuditRegion]:
    if df is None or df.empty:
        return []

    ordered = df.sort_values("start").reset_index(drop=True)
    regions: list[AuditRegion] = []

    for idx in range(len(ordered) - 1):
        left = ordered.loc[idx]
        right = ordered.loc[idx + 1]
        if str(left["speaker"]) == str(right["speaker"]):
            continue
        left_end = float(left["end"])
        right_start = float(right["start"])
        center = (left_end + right_start) / 2.0
        regions.append(
            AuditRegion(
                kind="boundary",
                start=max(float(left["start"]), center - float(config.boundary_window)),
                end=min(float(right["end"]), center + float(config.boundary_window)),
                left_index=idx,
                right_index=idx + 1,
                reason="speaker_boundary",
            )
        )

    for idx, row in ordered.iterrows():
        duration = float(row["end"]) - float(row["start"])
        if duration >= float(config.interior_min_duration):
            regions.append(
                AuditRegion(
                    kind="interior",
                    start=float(row["start"]),
                    end=float(row["end"]),
                    segment_index=int(idx),
                    reason="long_segment",
                )
            )

    return regions


def map_local_activities_to_global(
    activities: list[LocalActivity],
    references: dict[str, Any],
    config: AuditConfig,
) -> list[MappedActivity]:
    normalized_refs = _normalized_references(references)
    mapped: list[MappedActivity] = []

    for activity in activities:
        embedding = _as_embedding(activity.embedding)
        scores = sorted(
            (
                (speaker, _cosine_similarity(embedding, reference))
                for speaker, reference in normalized_refs.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not scores:
            mapped.append(
                MappedActivity(
                    local_speaker=str(activity.local_speaker),
                    global_speaker=None,
                    start=float(activity.start),
                    end=float(activity.end),
                    score=None,
                    margin=None,
                    top2_speaker=None,
                    confidence="uncertain",
                )
            )
            continue

        top_speaker, top_score = scores[0]
        top2_speaker = scores[1][0] if len(scores) > 1 else None
        top2_score = scores[1][1] if len(scores) > 1 else -1.0
        margin = float(top_score - top2_score)
        confidence = (
            "high"
            if top_score >= float(config.min_mapping_score)
            and margin >= float(config.min_mapping_margin)
            else "uncertain"
        )
        mapped.append(
            MappedActivity(
                local_speaker=str(activity.local_speaker),
                global_speaker=top_speaker if confidence == "high" else None,
                start=round(float(activity.start), 3),
                end=round(float(activity.end), 3),
                score=round(float(top_score), 6),
                margin=round(float(margin), 6),
                top2_speaker=top2_speaker,
                confidence=confidence,
            )
        )

    return mapped


def decode_frame_states(
    start: float,
    end: float,
    mapped_activities: list[MappedActivity],
    *,
    primary_speakers: tuple[str, str],
    frame_step: float = 0.1,
) -> list[str]:
    if end <= start:
        return []
    step = max(float(frame_step), 0.001)
    frames: list[str] = []
    current = float(start)
    speaker_a, speaker_b = primary_speakers

    while current < float(end) - 1e-9:
        frame_end = min(float(end), current + step)
        active: set[str | None] = set()
        for activity in mapped_activities:
            overlap = min(frame_end, float(activity.end)) - max(current, float(activity.start))
            if overlap > 1e-9:
                active.add(activity.global_speaker)

        if not active:
            frames.append("NONE")
        elif None in active:
            frames.append("UNCERTAIN")
        elif speaker_a in active and speaker_b in active:
            frames.append("A_AND_B")
        elif active == {speaker_a}:
            frames.append("A_ONLY")
        elif active == {speaker_b}:
            frames.append("B_ONLY")
        else:
            frames.append("OTHER")
        current = frame_end

    return frames


def _region_payload(region: AuditRegion) -> dict[str, Any]:
    return {
        "kind": region.kind,
        "start": round(float(region.start), 3),
        "end": round(float(region.end), 3),
        "left_index": region.left_index,
        "right_index": region.right_index,
        "segment_index": region.segment_index,
        "reason": region.reason,
    }


def _mapped_payload(mapped: list[MappedActivity]) -> list[dict[str, Any]]:
    return [asdict(item) for item in mapped]


def _speaker_span(mapped: list[MappedActivity], speaker: str) -> tuple[float, float] | None:
    spans = [
        (float(item.start), float(item.end))
        for item in mapped
        if item.global_speaker == speaker and float(item.end) > float(item.start)
    ]
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans)


def _has_uncertain_mapping(mapped: list[MappedActivity]) -> bool:
    return any(item.global_speaker is None or item.confidence != "high" for item in mapped)


def _refresh_labels(df: pd.DataFrame) -> pd.DataFrame:
    refreshed = df.sort_values("start").reset_index(drop=True).copy()
    for idx in range(len(refreshed)):
        refreshed.loc[idx, "label"] = chr(ord("A") + idx % 26)
        refreshed.loc[idx, "segment"] = f"segment_{idx:05d}"
    return refreshed


def _drop_tiny_segments(df: pd.DataFrame, min_duration: float) -> pd.DataFrame:
    keep = (df["end"].astype(float) - df["start"].astype(float)) >= float(min_duration)
    return df.loc[keep].copy()


def _apply_boundary_region(
    refined: pd.DataFrame,
    region: AuditRegion,
    mapped: list[MappedActivity],
    config: AuditConfig,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if region.left_index is None or region.right_index is None:
        return refined, None

    left_idx = int(region.left_index)
    right_idx = int(region.right_index)
    if left_idx >= len(refined) or right_idx >= len(refined):
        return refined, None

    left_speaker = str(refined.loc[left_idx, "speaker"])
    right_speaker = str(refined.loc[right_idx, "speaker"])
    left_span = _speaker_span(mapped, left_speaker)
    right_span = _speaker_span(mapped, right_speaker)
    if left_span is None or right_span is None:
        return refined, None

    old_left_end = float(refined.loc[left_idx, "end"])
    old_right_start = float(refined.loc[right_idx, "start"])
    old_boundary = (old_left_end + old_right_start) / 2.0
    left_start, left_end = left_span
    right_start, right_end = right_span

    overlap_start = max(left_start, right_start)
    overlap_end = min(left_end, right_end)
    overlap_duration = overlap_end - overlap_start
    can_extend_left = 0.0 <= left_end - old_left_end <= float(config.max_extend)
    can_extend_right = 0.0 <= old_right_start - right_start <= float(config.max_extend)
    if (
        overlap_duration >= float(config.min_overlap_duration)
        and left_end > old_left_end
        and right_start < old_right_start
        and can_extend_left
        and can_extend_right
    ):
        updated = refined.copy()
        updated.loc[left_idx, "end"] = round(float(left_end), 3)
        updated.loc[right_idx, "start"] = round(float(right_start), 3)
        return updated, {
            "action": "mark_overlap",
            "region": _region_payload(region),
            "left_index": left_idx,
            "right_index": right_idx,
            "speakers": [left_speaker, right_speaker],
            "old_left_end": round(old_left_end, 3),
            "old_right_start": round(old_right_start, 3),
            "new_left_end": round(float(left_end), 3),
            "new_right_start": round(float(right_start), 3),
            "overlap_start": round(float(overlap_start), 3),
            "overlap_end": round(float(overlap_end), 3),
            "confidence": "high",
        }

    new_boundary = round(float((left_end + right_start) / 2.0), 3)
    if abs(new_boundary - old_boundary) > float(config.max_shift):
        return refined, None
    if new_boundary - float(refined.loc[left_idx, "start"]) < float(config.min_duration):
        return refined, None
    if float(refined.loc[right_idx, "end"]) - new_boundary < float(config.min_duration):
        return refined, None
    if abs(new_boundary - old_boundary) < 0.001:
        return refined, None

    updated = refined.copy()
    updated.loc[left_idx, "end"] = new_boundary
    updated.loc[right_idx, "start"] = new_boundary
    return updated, {
        "action": "shift",
        "region": _region_payload(region),
        "left_index": left_idx,
        "right_index": right_idx,
        "left_speaker": left_speaker,
        "right_speaker": right_speaker,
        "old_boundary": round(float(old_boundary), 3),
        "new_boundary": new_boundary,
        "shift_seconds": round(float(new_boundary - old_boundary), 3),
        "confidence": "high",
    }


def _apply_interior_region(
    refined: pd.DataFrame,
    region: AuditRegion,
    mapped: list[MappedActivity],
    config: AuditConfig,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if region.segment_index is None:
        return refined, None

    seg_idx = int(region.segment_index)
    if seg_idx >= len(refined):
        return refined, None

    original = refined.loc[seg_idx]
    original_speaker = str(original["speaker"])
    mapped = sorted(mapped, key=lambda item: (float(item.start), float(item.end)))
    usable = [
        item
        for item in mapped
        if item.global_speaker is not None
        and float(item.end) - float(item.start) >= float(config.min_duration)
    ]
    if not usable or all(item.global_speaker == original_speaker for item in usable):
        return refined, None

    new_rows: list[dict[str, Any]] = []
    for item in usable:
        row = original.to_dict()
        row["speaker"] = str(item.global_speaker)
        row["start"] = max(float(original["start"]), float(item.start))
        row["end"] = min(float(original["end"]), float(item.end))
        if row["end"] - row["start"] >= float(config.min_duration):
            new_rows.append(row)

    if len(new_rows) < 2:
        return refined, None

    updated_rows: list[dict[str, Any]] = []
    for idx, row in refined.iterrows():
        if int(idx) == seg_idx:
            updated_rows.extend(new_rows)
        else:
            updated_rows.append(row.to_dict())
    updated = pd.DataFrame(updated_rows, columns=refined.columns)
    return updated, {
        "action": "split",
        "region": _region_payload(region),
        "segment_index": seg_idx,
        "old_speaker": original_speaker,
        "new_speakers": [row["speaker"] for row in new_rows],
        "old_start": round(float(original["start"]), 3),
        "old_end": round(float(original["end"]), 3),
        "confidence": "high",
    }


def apply_resegmentation_audit(
    df: pd.DataFrame,
    *,
    references: dict[str, Any],
    local_activity_provider: LocalActivityProvider | None,
    config: AuditConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or AuditConfig()
    refined = df.sort_values("start").reset_index(drop=True).copy()
    report: dict[str, Any] = {
        "metadata": {
            "enabled": bool(config.enabled),
            "config": asdict(config),
            "candidate_count": 0,
            "adjustment_count": 0,
        },
        "adjustments": [],
        "skipped_regions": [],
    }

    if not bool(config.enabled) or refined.empty or local_activity_provider is None:
        return refined, report

    normalized_refs = _normalized_references(references)
    if len(normalized_refs) < 2:
        report["skipped_regions"].append({"reason": "not_enough_global_references"})
        return refined, report

    regions = find_audit_regions(refined, config)
    report["metadata"]["candidate_count"] = len(regions)

    for region in regions:
        activities = local_activity_provider(region)
        if not activities:
            report["skipped_regions"].append(
                {"reason": "no_local_activity", "region": _region_payload(region)}
            )
            continue

        mapped = map_local_activities_to_global(activities, normalized_refs, config)
        if _has_uncertain_mapping(mapped):
            report["skipped_regions"].append(
                {
                    "reason": "uncertain_local_mapping",
                    "region": _region_payload(region),
                    "local_activity": _mapped_payload(mapped),
                }
            )
            continue

        if region.kind == "boundary":
            refined, adjustment = _apply_boundary_region(refined, region, mapped, config)
        elif region.kind == "interior":
            refined, adjustment = _apply_interior_region(refined, region, mapped, config)
        else:
            adjustment = None

        if adjustment is not None:
            adjustment["local_activity"] = _mapped_payload(mapped)
            report["adjustments"].append(adjustment)

    refined = _drop_tiny_segments(refined, config.min_duration)
    refined = _refresh_labels(refined)
    report["metadata"]["adjustment_count"] = len(report["adjustments"])
    return refined, report


def write_resegmentation_report(run_dir: str | Path, report: dict[str, Any]) -> Path:
    out_path = Path(run_dir) / "01_diarization" / "speaker_resegmentation_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    metadata = dict(payload.get("metadata", {}))
    metadata["stage"] = "speaker_resegmentation_audit"
    payload["metadata"] = metadata
    out_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
