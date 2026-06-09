from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("sommelier_batch_outputs /runs")
DEFAULT_DATASETS_ROOT = Path("datasets")

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
    "ờm",
    "dạ",
    "vâng",
    "uh",
    "ừ ừ",
    "ờ ờ",
}

VIDEO_TYPE_MAP = {
    "1. Podcast : interview": "podcast_interview",
    "2. News : formal speech": "news_formal",
    "3. Street : noisy interview": "street_noisy",
    "4. casual talkshow": "casual_talkshow",
    "5. Livestream ": "livestream_sales",
}

VAD_NOTE_TAG_RULES = {
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

DIARIZATION_NOTE_TAG_RULES = {
    "speaker_label_issue": [
        "speaker",
        "người nói",
    ],
    "speaker_missing": [
        "thiếu speaker",
        "thiếu người nói",
    ],
    "speaker_extra": [
        "thừa speaker",
    ],
    "speaker_wrong": [
        "sai speaker",
        "nhầm speaker",
    ],
    "backchannel_speaker_issue": [
        "backchannel",
        "filler",
    ],
}

PODCAST_ONLY_TYPE = "podcast_interview"

VAD_COLUMNS = [
    "sample_id",
    "start",
    "end",
    "duration",
    "gold_text",
]

DIARIZATION_COLUMNS = [
    "sample_id",
    "start",
    "end",
    "duration",
    "speaker",
    "gold_text",
]


@dataclass
class HtmlPayload:
    path: Path
    title: str
    stage_data: dict[str, Any]
    vad_chunks: list[dict[str, Any]]


def load_html_payload(path: Path) -> HtmlPayload | None:
    if not path.exists():
        return None
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
    return sum(bool(str(seg.get("note", "")).strip() or str(seg.get("correct_script", "")).strip()) for seg in segments)


def select_annotation_payload(run_dir: Path) -> HtmlPayload | None:
    best: HtmlPayload | None = None
    best_count = -1
    for path in sorted(run_dir.glob("index-*.html")):
        payload = load_html_payload(path)
        if not payload:
            continue
        count = count_annotations(payload.stage_data.get("05_export", []))
        if count > best_count:
            best = payload
            best_count = count
    return best if best_count > 0 else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def normalize_category(name: str) -> str:
    return VIDEO_TYPE_MAP.get(name, name.strip().lower().replace(" ", "_"))


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def compact_text(value: str) -> str:
    return re.sub(
        r"[^\w\sàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]",
        "",
        value.lower(),
    ).strip()


def derive_note_tags(note: str, rules: dict[str, list[str]]) -> list[str]:
    lowered = note.lower()
    tags: list[str] = []
    for tag, keywords in rules.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(tag)
    return sorted(set(tags))


def derive_vad_heuristic_tags(duration: float, text: str, gap_from_prev: float | None, gap_to_next: float | None) -> list[str]:
    tags: list[str] = []
    normalized = compact_text(text)
    if duration < 1.0:
        tags.append("too_short")
    if duration < 0.7:
        tags.append("micro_segment")
    if duration >= 15.0:
        tags.append("too_long")
    if normalized in FILLER_TEXTS and duration <= 1.2:
        tags.append("backchannel_isolated")
    if gap_from_prev is not None and gap_from_prev >= 1.0:
        tags.append("large_pre_gap")
    if gap_to_next is not None and gap_to_next >= 1.0:
        tags.append("large_post_gap")
    return sorted(set(tags))


def derive_diarization_heuristic_tags(
    duration: float,
    text: str,
    run_speaker_count: int,
    prev_speaker: str,
    speaker: str,
    next_speaker: str,
    video_type: str,
) -> list[str]:
    tags: list[str] = []
    normalized = compact_text(text)
    if normalized in FILLER_TEXTS and duration <= 1.2:
        tags.append("backchannel_segment")
    if prev_speaker and next_speaker and prev_speaker != speaker and next_speaker != speaker and prev_speaker == next_speaker:
        tags.append("isolated_speaker_flip")
    if video_type == "street_noisy" and run_speaker_count == 1:
        tags.append("undercluster_risk")
    if video_type in {"podcast_interview", "casual_talkshow", "livestream_sales"} and run_speaker_count >= 5:
        tags.append("overcluster_risk")
    if duration < 1.0 and prev_speaker and next_speaker and prev_speaker != next_speaker:
        tags.append("short_turn_boundary")
    return sorted(set(tags))


def resolve_full_audio(run_dir: Path, output_payload: HtmlPayload) -> str:
    for chunk in output_payload.vad_chunks:
        raw_path = str(chunk.get("path", ""))
        marker = "/00_input/"
        if marker in raw_path:
            suffix = raw_path[raw_path.index(marker) + 1 :]
            candidate = run_dir / suffix
            if candidate.exists():
                return str(candidate.resolve())
    fallback = run_dir / "00_input" / "full.wav"
    return str(fallback.resolve()) if fallback.exists() else ""


def resolve_segment_audio(run_dir: Path, audio_file: str) -> str:
    if not audio_file:
        return ""
    candidate = run_dir / "05_export" / "final" / audio_file.replace("05_export/final/", "")
    return str(candidate.resolve()) if candidate.exists() else ""


def select_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in columns} for row in rows]


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_rows(runs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    vad_rows: list[dict[str, Any]] = []
    diar_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    sample_counter = 0

    for category_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        video_type = normalize_category(category_dir.name)
        if video_type != PODCAST_ONLY_TYPE:
            continue
        for run_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            output_payload = load_html_payload(run_dir / "index.html")
            annotation_payload = select_annotation_payload(run_dir)
            diarization_json = load_json(run_dir / "01_diarization" / "diarization.json")
            if not output_payload or not annotation_payload or not diarization_json:
                continue

            diar_segments = diarization_json.get("segments", [])
            ann_segments = annotation_payload.stage_data.get("05_export", [])
            run_start_counter = sample_counter + 1

            for idx, seg in enumerate(diar_segments):
                sample_counter += 1
                ann = ann_segments[idx] if idx < len(ann_segments) else {}
                correct_script = normalize_text(ann.get("correct_script"))
                text = normalize_text(output_payload.stage_data.get("05_export", [])[idx].get("text") if idx < len(output_payload.stage_data.get("05_export", [])) else "")
                start = float(seg.get("start") or 0.0)
                end = float(seg.get("end") or 0.0)
                duration = float(seg.get("duration") or (end - start) or 0.0)
                speaker = normalize_text(seg.get("speaker"))

                common = {
                    "sample_id": f"podcast_{sample_counter:05d}",
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "speaker": speaker,
                    "gold_text": correct_script or text,
                    "_is_corrected": bool(correct_script),
                }

                vad_rows.append(
                    {
                        **common,
                    }
                )

                diar_rows.append(
                    {
                        **common,
                    }
                )

            if diar_segments:
                manifest_rows.append(
                    {
                        "run_name": run_dir.name,
                        "audio_stem": run_dir.name.replace("run_full_", "", 1),
                        "sample_id_start": f"podcast_{run_start_counter:05d}",
                        "sample_id_end": f"podcast_{sample_counter:05d}",
                        "row_start_index": run_start_counter - 1,
                        "row_end_index": sample_counter - 1,
                        "segment_count": len(diar_segments),
                    }
                )

    return vad_rows, diar_rows, manifest_rows


def write_dataset_readme(path: Path, title: str, rows: list[dict[str, Any]], corrected: int) -> None:
    lines = [
        f"# {title}",
        "",
        "- Scope: **podcast only**",
        f"- Rows: **{len(rows)}**",
        f"- Corrected rows (`correct_script` có dữ liệu): **{corrected}**",
        f"- Unchanged rows (không có `correct_script`): **{len(rows) - corrected}**",
        "",
        "## Rule",
        "",
        "- Không có `correct_script`: giữ nguyên transcript hiện tại làm gold.",
        "- Có `correct_script`: dùng bản sửa làm gold cuối cùng.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standardized VAD and speaker diarization datasets from annotated Sommelier runs.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    args = parser.parse_args()

    vad_rows, diar_rows, manifest_rows = build_rows(args.runs_root)
    if not vad_rows or not diar_rows:
        raise SystemExit("No annotated VAD/diarization rows found.")
    vad_corrected = sum(1 for row in vad_rows if row["_is_corrected"])
    diar_corrected = sum(1 for row in diar_rows if row["_is_corrected"])

    vad_dir = args.datasets_root / "vad_eval"
    diar_dir = args.datasets_root / "diarization_eval"

    vad_rows = select_columns(vad_rows, VAD_COLUMNS)
    diar_rows = select_columns(diar_rows, DIARIZATION_COLUMNS)

    write_csv(vad_rows, vad_dir / "vad_eval_dataset.csv")
    write_jsonl(vad_rows, vad_dir / "vad_eval_dataset.jsonl")
    write_dataset_readme(vad_dir / "README.md", "VAD Golden Dataset", vad_rows, vad_corrected)
    write_json(vad_dir / "golden_manifest.json", manifest_rows)
    write_text(
        vad_dir / "SCHEMA.md",
        """# VAD Golden Schema

- `sample_id`: mã định danh duy nhất cho mỗi dòng trong bản podcast-only.
- `start`, `end`, `duration`: mốc thời gian chuẩn của segment.
- `gold_text`: transcript cuối cùng dùng làm gold. Nếu có `correct_script` thì lấy bản sửa; nếu không có thì giữ transcript hiện tại.
""",
    )

    write_csv(diar_rows, diar_dir / "diarization_eval_dataset.csv")
    write_jsonl(diar_rows, diar_dir / "diarization_eval_dataset.jsonl")
    write_dataset_readme(diar_dir / "README.md", "Speaker Diarization Golden Dataset", diar_rows, diar_corrected)
    write_json(diar_dir / "golden_manifest.json", manifest_rows)
    write_text(
        diar_dir / "SCHEMA.md",
        """# Speaker Diarization Golden Schema

- `sample_id`: mã định danh duy nhất cho mỗi dòng trong bản podcast-only.
- `start`, `end`, `duration`: mốc thời gian chuẩn của segment.
- `speaker`: speaker hiện tại của segment trong pipeline.
- `gold_text`: transcript cuối cùng dùng làm gold. Nếu có `correct_script` thì lấy bản sửa; nếu không có thì giữ transcript hiện tại.
""",
    )

    print(f"Wrote VAD dataset: {len(vad_rows)} rows -> {vad_dir}")
    print(f"Wrote diarization dataset: {len(diar_rows)} rows -> {diar_dir}")


if __name__ == "__main__":
    main()
