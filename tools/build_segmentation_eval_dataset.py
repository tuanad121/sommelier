from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("sommelier_batch_outputs /runs")
DEFAULT_OUTPUT_DIR = Path("datasets/segmentation_eval")

STAGE_DATA_RE = re.compile(r"const STAGE_DATA\s*=\s*(\{.*?\});\s*const VAD_CHUNKS", re.S)
TITLE_RE = re.compile(r"<h1>(.*?)</h1>", re.S)
VAD_CHUNKS_RE = re.compile(r"const VAD_CHUNKS\s*=\s*(\[.*?\]);", re.S)

FILLER_TEXTS = {
    "a",
    "à",
    "ạ",
    "ờ",
    "ừ",
    "ừm",
    "ừ nhỉ",
    "ờm",
    "ờ hả",
    "dạ",
    "vâng",
    "ừ ừ",
    "ờ ờ",
    "uk",
    "ừ ha",
}

SEGMENTATION_KEYWORDS = {
    "cut_missing_content": [
        "cắt thiếu",
        "thiếu đầu",
        "thiếu cuối",
        "mất đầu",
        "mất cuối",
        "thiếu chữ",
        "thiếu từ",
        "thiếu nội dung",
    ],
    "split_one_utterance": [
        "bị tách",
        "tách sai",
        "tách câu",
        "split",
    ],
    "merged_multiple_utterances": [
        "gộp",
        "dính",
        "ghép",
        "merge",
    ],
    "boundary_extra_content": [
        "thừa",
        "thừa chữ",
        "thừa từ",
    ],
    "duplication_or_repeat": [
        "lặp",
        "lặp từ",
        "lặp chữ",
    ],
}

NON_SEGMENTATION_KEYWORDS = {
    "speaker": ["speaker", "người nói"],
    "noise": ["ồn", "nhiễu", "tạp âm"],
    "english_codeswitch": ["tiếng anh", "english", "code-switch", "code switch"],
    "asr_text": ["speech to text", "asr", "text sai", "chuyển"],
}

VIDEO_TYPE_MAP = {
    "1. Podcast : interview": "podcast_interview",
    "2. News : formal speech": "news_formal",
    "3. Street : noisy interview": "street_noisy",
    "4. casual talkshow": "casual_talkshow",
    "5. Livestream ": "livestream_sales",
}


@dataclass
class HtmlPayload:
    path: Path
    title: str
    stage_data: dict[str, Any]
    vad_chunks: list[dict[str, Any]]


def load_html_payload(path: Path) -> HtmlPayload | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    stage_match = STAGE_DATA_RE.search(text)
    if not stage_match:
        return None
    title_match = TITLE_RE.search(text)
    vad_match = VAD_CHUNKS_RE.search(text)
    title = re.sub(r"<.*?>", "", title_match.group(1)).strip() if title_match else path.stem
    stage_data = json.loads(stage_match.group(1))
    vad_chunks = json.loads(vad_match.group(1)) if vad_match else []
    return HtmlPayload(path=path, title=title, stage_data=stage_data, vad_chunks=vad_chunks)


def count_annotations(segments: list[dict[str, Any]]) -> int:
    total = 0
    for segment in segments:
        if str(segment.get("note", "")).strip() or str(segment.get("correct_script", "")).strip():
            total += 1
    return total


def select_annotated_payload(run_dir: Path) -> HtmlPayload | None:
    best: HtmlPayload | None = None
    best_count = -1
    for path in sorted(run_dir.glob("index-*.html")):
        payload = load_html_payload(path)
        if not payload:
            continue
        segments = payload.stage_data.get("05_export", [])
        current_count = count_annotations(segments)
        if current_count > best_count:
            best = payload
            best_count = current_count
    return best if best_count > 0 else None


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def classify_note(note: str, duration: float, text: str) -> tuple[list[str], list[str], str]:
    lowered = note.lower()
    segmentation_tags: list[str] = []
    non_segmentation_tags: list[str] = []

    for tag, keywords in SEGMENTATION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            segmentation_tags.append(tag)

    for tag, keywords in NON_SEGMENTATION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            non_segmentation_tags.append(tag)

    normalized_text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]", "", text.lower()).strip()
    if duration <= 1.0:
        segmentation_tags.append("too_short")
    if duration <= 0.7:
        segmentation_tags.append("micro_segment")
    if duration >= 15.0:
        segmentation_tags.append("too_long")
    if normalized_text in FILLER_TEXTS and duration <= 1.2:
        segmentation_tags.append("backchannel_isolated")

    segmentation_tags = sorted(set(segmentation_tags))
    non_segmentation_tags = sorted(set(non_segmentation_tags))

    if segmentation_tags and note:
        relevance = "high"
    elif segmentation_tags:
        relevance = "possible"
    else:
        relevance = "low"
    return segmentation_tags, non_segmentation_tags, relevance


def resolve_full_audio(run_dir: Path, payload: HtmlPayload | None) -> str:
    if payload and payload.vad_chunks:
        raw_path = str(payload.vad_chunks[0].get("path", ""))
        marker = "/00_input/"
        if marker in raw_path:
            suffix = raw_path[raw_path.index(marker) + 1 :]
            candidate = run_dir / suffix
            if candidate.exists():
                return str(candidate.resolve())
    fallback = run_dir / "00_input" / "full.wav"
    return str(fallback.resolve()) if fallback.exists() else ""


def build_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            output_payload = load_html_payload(run_dir / "index.html")
            annotated_payload = select_annotated_payload(run_dir)
            if not output_payload or not annotated_payload:
                continue

            output_segments = output_payload.stage_data.get("05_export", [])
            annotated_segments = annotated_payload.stage_data.get("05_export", [])
            full_audio_path = resolve_full_audio(run_dir, output_payload)
            video_type = VIDEO_TYPE_MAP.get(category_dir.name, category_dir.name)

            for idx, output_segment in enumerate(output_segments):
                annotated_segment = annotated_segments[idx] if idx < len(annotated_segments) else {}
                note = normalize_text(annotated_segment.get("note"))
                correct_script = normalize_text(annotated_segment.get("correct_script"))
                pred_text = normalize_text(output_segment.get("text"))
                duration = float(output_segment.get("duration") or (output_segment.get("end", 0) - output_segment.get("start", 0)) or 0.0)
                segmentation_tags, non_segmentation_tags, relevance = classify_note(note, duration, pred_text)

                audio_rel = normalize_text(output_segment.get("audio_file"))
                segment_audio_path = str((run_dir / audio_rel).resolve()) if audio_rel else ""

                row = {
                    "dataset_version": "v1",
                    "task": "segmentation_vad_eval",
                    "video_type": video_type,
                    "category_folder": category_dir.name,
                    "run_id": run_dir.name,
                    "source_html": str(output_payload.path.resolve()),
                    "annotation_html": str(annotated_payload.path.resolve()),
                    "full_audio_path": full_audio_path,
                    "segment_audio_path": segment_audio_path,
                    "segment_index": idx,
                    "segment_id": f"{run_dir.name}:{idx:05d}",
                    "segment_label": normalize_text(output_segment.get("index")) or f"{idx:05d}",
                    "start": float(output_segment.get("start") or 0.0),
                    "end": float(output_segment.get("end") or 0.0),
                    "duration": duration,
                    "gap_from_prev": None,
                    "speaker": normalize_text(output_segment.get("speaker")),
                    "language": normalize_text(output_segment.get("language")) or "vi",
                    "transcript_pred": pred_text,
                    "transcript_whisper": normalize_text(output_segment.get("text_whisper")),
                    "transcript_phowhisper": normalize_text(output_segment.get("text_phowhisper")),
                    "transcript_chunkformer": normalize_text(output_segment.get("text_chunkformer")),
                    "correct_script": correct_script,
                    "note": note,
                    "has_note": bool(note),
                    "has_correct_script": bool(correct_script),
                    "is_reviewed_run": True,
                    "is_annotated_row": bool(note or correct_script),
                    "demucs": bool(output_segment.get("demucs")),
                    "is_separated": bool(output_segment.get("is_separated")),
                    "sepreformer": bool(output_segment.get("sepreformer")),
                    "asr_quality_source": normalize_text(output_segment.get("asr_quality_source")),
                    "asr_quality_actions": "|".join(output_segment.get("asr_quality_actions", []) or []),
                    "asr_context_pad_before": output_segment.get("asr_context_pad_before", 0),
                    "asr_context_pad_after": output_segment.get("asr_context_pad_after", 0),
                    "is_short_segment": duration < 1.0,
                    "is_micro_segment": duration < 0.7,
                    "is_long_segment": duration >= 15.0,
                    "is_backchannel_like": "backchannel_isolated" in segmentation_tags,
                    "segmentation_issue_tags": "|".join(segmentation_tags),
                    "non_segmentation_issue_tags": "|".join(non_segmentation_tags),
                    "segmentation_relevance": relevance,
                    "primary_eval_target": "segmentation" if relevance != "low" else "context_only",
                }
                if idx > 0:
                    prev_end = float(output_segments[idx - 1].get("end") or 0.0)
                    row["gap_from_prev"] = round(row["start"] - prev_end, 6)
                rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    by_video_type: dict[str, int] = {}
    by_relevance: dict[str, int] = {}
    annotated_rows = 0
    for row in rows:
        by_video_type[row["video_type"]] = by_video_type.get(row["video_type"], 0) + 1
        by_relevance[row["segmentation_relevance"]] = by_relevance.get(row["segmentation_relevance"], 0) + 1
        if row["is_annotated_row"]:
            annotated_rows += 1

    lines = [
        "# Segmentation Eval Dataset",
        "",
        f"- Rows: **{len(rows)}**",
        f"- Annotated rows (`note` or `correct_script`): **{annotated_rows}**",
        "",
        "## By video type",
        "",
    ]
    for key in sorted(by_video_type):
        lines.append(f"- `{key}`: **{by_video_type[key]}** rows")
    lines.extend(
        [
            "",
            "## By segmentation relevance",
            "",
        ]
    )
    for key in sorted(by_relevance):
        lines.append(f"- `{key}`: **{by_relevance[key]}** rows")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `segmentation_relevance=high` means note text directly suggests a boundary or segmentation problem.",
            "- `segmentation_relevance=possible` means the row looks risky because it is very short, very long, or a filler/backchannel segment.",
            "- `segmentation_relevance=low` means the row is preserved for context, but there is no direct segmentation signal yet.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_schema(path: Path) -> None:
    schema = """# Segmentation Eval Schema

- `dataset_version`: version label for reproducibility.
- `task`: fixed task name for downstream loaders.
- `video_type`: normalized video bucket.
- `category_folder`: original folder name under `runs`.
- `run_id`: run folder name.
- `source_html`: absolute path to the original `index.html`.
- `annotation_html`: absolute path to the selected annotated HTML.
- `full_audio_path`: absolute path to the full audio file for the run.
- `segment_audio_path`: absolute path to the exported segment audio clip.
- `segment_index`: zero-based row index in `05_export`.
- `segment_id`: stable synthetic identifier `run_id:index`.
- `segment_label`: original segment label from HTML/JSON.
- `start`, `end`, `duration`: segment timing in seconds.
- `gap_from_prev`: time gap from the previous segment.
- `speaker`: predicted speaker label.
- `language`: language tag from the pipeline.
- `transcript_pred`: final transcript shown in output.
- `transcript_whisper`, `transcript_phowhisper`, `transcript_chunkformer`: model alternatives.
- `correct_script`: manual correction from the note HTML.
- `note`: manual note from the note HTML.
- `has_note`, `has_correct_script`, `is_annotated_row`: manual supervision flags.
- `demucs`, `is_separated`, `sepreformer`: pipeline artifact flags.
- `asr_quality_source`, `asr_quality_actions`, `asr_context_pad_before`, `asr_context_pad_after`: ASR routing metadata.
- `is_short_segment`, `is_micro_segment`, `is_long_segment`, `is_backchannel_like`: heuristic timing flags.
- `segmentation_issue_tags`: heuristic tags tied to segmentation/VAD.
- `non_segmentation_issue_tags`: heuristic tags that likely belong to ASR/speaker/noise instead.
- `segmentation_relevance`: `high`, `possible`, or `low`.
- `primary_eval_target`: `segmentation` for likely segmentation rows, otherwise `context_only`.
"""
    path.write_text(schema, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a segmentation/VAD eval dataset from annotated Sommelier HTML runs.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.runs_root)
    if not rows:
        raise SystemExit("No annotated rows found. Check runs root and note HTML files.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "segmentation_eval_dataset.csv")
    write_jsonl(rows, args.output_dir / "segmentation_eval_dataset.jsonl")
    write_summary(rows, args.output_dir / "README.md")
    write_schema(args.output_dir / "SCHEMA.md")
    print(f"Wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
